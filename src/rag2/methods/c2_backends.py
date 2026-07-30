"""
RAG² C2 检索后端 — 三种实现，共享 C2 索引

按架构决策.md 第四节，三后端作 ablation：
  A 纯 LLM 判断：LLM 看所有 hypothetical_questions 选最相关（最准最贵）
  B 语义 embedding：对 hypothetical_questions/实体做 embedding 检索（非原文）
  C 混合：B 粗筛 top-K + LLM 精排（折中）

三后端共享同一份 (indices, graph)，可插拔对比。
"""
from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from rag2.gateway import ModelGateway
from rag2.methods.generative_index import GenerativeIndex, CorpusGraph
from rag2.methods.retriever import Retriever, _show_progress

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 抽象基类
# ─────────────────────────────────────────────────────────

class RetrievalBackend(ABC):
    """检索后端抽象：query → 排序后的 doc_id 列表。"""

    name: str = "base"

    def __init__(self, indices: list[GenerativeIndex], graph: CorpusGraph):
        self.indices = indices
        self.graph = graph
        self._id_to_idx = {idx.doc_id: idx for idx in indices}

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[int]:
        """返回排序后的 doc_id 列表（最相关在前）。"""
        ...


# ─────────────────────────────────────────────────────────
# 后端 A: 纯 LLM 判断
# ─────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are selecting documents relevant to a question from a semantic index.

Question: {question}

Below is a list of documents. Each has an id, title, hypothetical questions it can answer, and key entities.

{doc_list}

Select the {top_k} documents MOST relevant to answering the question. Consider both direct relevance and multi-hop relevance (a doc might be a useful intermediate step).

Call the select_documents tool with your chosen ids, most relevant first."""


JUDGE_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "select_documents",
            "description": "Select the most relevant document ids for the question",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Selected document ids, most relevant first",
                    },
                },
                "required": ["ids"],
            },
        },
    }
]


class LLMJudgeBackend(RetrievalBackend):
    """后端 A：LLM 看所有 hypothetical_questions + entities 选最相关。最准但每题贵。"""

    name = "A_llm_judge"

    def __init__(self, indices, graph, gateway: ModelGateway, role: str = "generator"):
        super().__init__(indices, graph)
        self.gw = gateway
        self.role = role

    def retrieve(self, query: str, top_k: int = 5) -> list[int]:
        # 构造文档列表（id + title + 前 3 个假设问题 + 前 5 个实体）
        lines = []
        for idx in self.indices:
            qs = " | ".join(idx.hypothetical_questions[:3])
            ents = ", ".join(e["entity"] for e in idx.entities[:5])
            lines.append(
                f"[{idx.doc_id}] {idx.title}\n"
                f"  Questions: {qs}\n"
                f"  Entities: {ents}\n"
                f"  Summary: {idx.summary_short}"
            )
        doc_list = "\n".join(lines)

        prompt = JUDGE_PROMPT.format(question=query, doc_list=doc_list, top_k=top_k)
        resp = self.gw.generate_complete(
            self.role, [{"role": "user", "content": prompt}],
            tools=JUDGE_TOOL_SCHEMA, role_tag="c2_backend_A",
            max_tokens=500, max_continuations=2,
        )
        return self._extract_ids(resp, top_k)

    @staticmethod
    def _extract_ids(resp, max_n: int) -> list[int]:
        """从工具调用提取 ids（不解析文本）。"""
        for tc in resp.tool_calls:
            if tc.get("function", {}).get("name") == "select_documents":
                try:
                    args = json.loads(tc["function"]["arguments"])
                    ids = args.get("ids", [])
                    return [int(i) for i in ids if str(i).lstrip("-").isdigit()][:max_n]
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass
        return []


# ─────────────────────────────────────────────────────────
# 后端 B: 语义 embedding（对 hypothetical-Q embedding，非原文）
# ─────────────────────────────────────────────────────────

class SemanticEmbeddingBackend(RetrievalBackend):
    """
    后端 B：对 hypothetical_questions 和实体名做 embedding 检索。

    关键创新：不是对原文 embedding（传统 RAG），而是对 LLM 生成的
    假设性问题 embedding——后者语义更聚焦、召回质量更高。
    """

    name = "B_semantic_embedding"

    def __init__(self, indices, graph, retriever: Retriever):
        super().__init__(indices, graph)
        self.retriever = retriever
        self._emb_index = None       # FAISS 索引（对 hypothetical-Q）
        self._q_to_doc = []          # 第 i 个 embedding 对应哪个 doc_id
        self._built = False

    def _build(self) -> None:
        """构建 hypothetical-Q 的 embedding 索引（一次性）。"""
        import faiss
        # 把所有 hypothetical_questions 摊平，记录每个 question → doc_id
        all_questions = []
        q_to_doc = []
        for idx in self.indices:
            for q in idx.hypothetical_questions:
                all_questions.append(q)
                q_to_doc.append(idx.doc_id)
        # 加上实体名作为补充锚点
        for idx in self.indices:
            for e in idx.entities:
                all_questions.append(f"entity: {e['entity']}")
                q_to_doc.append(idx.doc_id)

        if not all_questions:
            self._emb_index = None
            return

        # 用 retriever 的 embedder 编码（复用已加载模型，不重新初始化）
        emb = self.retriever.embedder.encode(
            all_questions, batch_size=64, show_progress_bar=_show_progress(),
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")

        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        self._emb_index = index
        self._q_to_doc = q_to_doc
        self._built = True
        logger.info("后端 B 索引构建: %d questions/entities → %d docs",
                    len(all_questions), len(self.indices))

    def retrieve(self, query: str, top_k: int = 5) -> list[int]:
        if not self._built:
            self._build()
        if self._emb_index is None:
            return []

        # query embedding
        q_emb = self.retriever.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
        ).astype("float32")

        # 检索 top top_k*3 个 question（因为多个 question 可能指向同一 doc）
        distances, indices = self._emb_index.search(q_emb, top_k * 3)

        # 聚合到 doc 级（同一 doc 取最高分）
        doc_scores: dict[int, float] = {}
        for dist, qi in zip(distances[0], indices[0]):
            if qi < 0:
                continue
            doc_id = self._q_to_doc[qi]
            if doc_id not in doc_scores or dist > doc_scores[doc_id]:
                doc_scores[doc_id] = float(dist)

        # 按分数排序，取 top_k
        ranked = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
        return [doc_id for doc_id, _ in ranked]


# ─────────────────────────────────────────────────────────
# 后端 C: 混合（B 粗筛 + LLM 精排）
# ─────────────────────────────────────────────────────────

RERANK_PROMPT = """Rank these candidate documents by relevance to the question.

Question: {question}

Candidates:
{candidates}

Select the top {top_k} most relevant ids. Call the select_documents tool with the ids, most relevant first."""


class HybridBackend(RetrievalBackend):
    """后端 C：B 粗筛 top-K + LLM 精排 top-k。折中方案。"""

    name = "C_hybrid"

    def __init__(self, indices, graph, retriever: Retriever,
                 gateway: ModelGateway, role: str = "generator",
                 filter_top_k: int = 20):
        super().__init__(indices, graph)
        self.backend_b = SemanticEmbeddingBackend(indices, graph, retriever)
        self.gw = gateway
        self.role = role
        self.filter_top_k = filter_top_k

    def retrieve(self, query: str, top_k: int = 5) -> list[int]:
        # 阶段 1: B 粗筛
        candidates = self.backend_b.retrieve(query, top_k=self.filter_top_k)
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return candidates

        # 阶段 2: LLM 精排（用工具调用，复用 JUDGE_TOOL_SCHEMA）
        cand_lines = []
        for cid in candidates:
            idx = self._id_to_idx.get(cid)
            if not idx:
                continue
            qs = " | ".join(idx.hypothetical_questions[:2])
            cand_lines.append(
                f"[{cid}] {idx.title}\n  Q: {qs}\n  E: {', '.join(e['entity'] for e in idx.entities[:3])}"
            )
        prompt = RERANK_PROMPT.format(
            question=query,
            candidates="\n".join(cand_lines),
            top_k=top_k,
        )
        resp = self.gw.generate_complete(
            self.role, [{"role": "user", "content": prompt}],
            tools=JUDGE_TOOL_SCHEMA, role_tag="c2_backend_C",
            max_tokens=500, max_continuations=2,
        )
        ids = LLMJudgeBackend._extract_ids(resp, top_k)
        return ids if ids else candidates[:top_k]


# ─────────────────────────────────────────────────────────
# 工厂
# ─────────────────────────────────────────────────────────

def build_backend(
    backend_name: str,
    indices: list[GenerativeIndex],
    graph: CorpusGraph,
    gateway: ModelGateway,
    retriever: Optional[Retriever] = None,
    role: str = "generator",
) -> RetrievalBackend:
    """按名字构造检索后端。"""
    if backend_name in ("A", "A_llm_judge"):
        return LLMJudgeBackend(indices, graph, gateway, role)
    elif backend_name in ("B", "B_semantic_embedding"):
        if retriever is None:
            raise ValueError("后端 B 需要传 retriever（提供 embedder）")
        return SemanticEmbeddingBackend(indices, graph, retriever)
    elif backend_name in ("C", "C_hybrid"):
        if retriever is None:
            raise ValueError("后端 C 需要传 retriever（B 部分用）")
        return HybridBackend(indices, graph, retriever, gateway, role)
    else:
        raise ValueError(f"未知后端: {backend_name}（可选 A/B/C）")
