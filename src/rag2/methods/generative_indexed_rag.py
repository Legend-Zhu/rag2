"""
RAG² C2: GenerativeIndexedRAG — 基于生成式语义索引的检索方法

检索路径（三种，对应三类查询）：
  1. 问题空间检索：query → 最相似的 hypothetical_question → 定位文档（HyDE 倒置核心）
  2. 实体路径检索：query 抽实体 → 沿实体图跳转（多跳）
  3. 摘要先导检索：先扫所有 summary → 选相关 → 精读

检索实现：用 LLM 做相关性判断（模型编排中心化，不引入额外 embedding）。
为控成本，用两阶段：先 LLM 粗筛所有 summary（一次调用），再 LLM 精排候选 questions。
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from rag2.gateway import ModelGateway
from rag2.methods.base import Method, Result, RetrievedDoc
from rag2.methods.generative_index import GenerativeIndex, GenerativeIndexBuilder

logger = logging.getLogger(__name__)


# ── 检索 prompt ──────────────────────────────────────────

FILTER_PROMPT = """You are selecting which documents are relevant to a question.

Question: {question}

Below is a list of documents (id + one-sentence summary). Select the ids of the 5-10 documents MOST LIKELY to contain the answer.

Documents:
{doc_list}

Output ONLY a JSON array of document ids, e.g. [3, 7, 12]. No other text."""

RERANK_PROMPT = """You are ranking documents by relevance to a question.

Question: {question}

Candidate documents with their generated annotations:
{candidates}

Select the top {top_k} most relevant document ids. Output ONLY a JSON array of ids, most relevant first, e.g. [7, 3, 12]. No other text."""

ANSWER_PROMPT = """Answer the question based ONLY on the provided context.

Context:
{context}

Question: {question}

Answer (concise, based only on context):"""


class GenerativeIndexedRAG(Method):
    """基于生成式语义索引的 RAG。"""

    name = "generative_indexed_rag"

    def __init__(
        self,
        gateway: ModelGateway,
        index_builder: GenerativeIndexBuilder,
        role: str = "generator",
        top_k_filter: int = 10,
        top_k_final: int = 5,
    ):
        super().__init__(gateway, role)
        self.index_builder = index_builder
        self.top_k_filter = top_k_filter
        self.top_k_final = top_k_final
        self._indices_cache: list[GenerativeIndex] | None = None
        self._indices_docs_key: str = ""

    def run(self, sample: dict) -> Result:
        question = sample["question"]
        docs = sample.get("supporting_docs", [])

        # 1. 建生成式索引（带缓存）
        indices = self._get_or_build(docs)

        # 2. 两阶段检索
        # 阶段 A：LLM 粗筛（看所有 summary）
        candidate_ids = self._filter_by_summary(question, indices)

        # 阶段 B：LLM 精排（看候选的 hypothetical_questions + entities）
        top_ids = self._rerank_by_annotations(question, indices, candidate_ids)

        # 取出 top 文档
        id_to_idx = {idx.doc_id: idx for idx in indices}
        top_docs = [id_to_idx[i] for i in top_ids if i in id_to_idx]

        # 3. 拼 context 生成答案
        context = self._format_context(top_docs)
        messages = [
            {"role": "user", "content": ANSWER_PROMPT.format(
                context=context, question=question,
            )},
        ]
        resp = self.gw.generate(self.role, messages, role_tag=self.name)

        # 标注 gold
        gold_titles = {d["title"] for d in docs if d.get("is_supporting")}
        retrieved = [
            RetrievedDoc(
                title=idx.title, text=idx.text,
                score=1.0 / (rank + 1),  # 用排名作 score
                is_supporting=idx.title in gold_titles,
            )
            for rank, idx in enumerate(top_docs)
        ]

        return Result(
            answer=resp.text.strip().strip("\"'"),
            retrieved_docs=retrieved,
            trace={
                "method": self.name,
                "n_corpus_docs": len(docs),
                "n_candidates_after_filter": len(candidate_ids),
                "n_final": len(top_docs),
                "filter_ids": candidate_ids,
                "final_ids": top_ids,
                "completion_tokens": resp.usage.get("completion_tokens", 0),
                "from_cache": resp.from_cache,
            },
        )

    # ── 索引管理 ──────────────────────────────────────────

    def _get_or_build(self, docs: list[dict]) -> list[GenerativeIndex]:
        """获取或构建生成式索引（带跨样本缓存）。"""
        import hashlib
        docs_key = hashlib.sha256(
            json.dumps([(d.get("title",""), d.get("text","")[:200]) for d in docs],
                       ensure_ascii=True).encode()
        ).hexdigest()[:16]
        if self._indices_cache is None or self._indices_docs_key != docs_key:
            self._indices_cache = self.index_builder.build(docs)
            self._indices_docs_key = docs_key
        return self._indices_cache

    # ── 两阶段检索 ────────────────────────────────────────

    def _filter_by_summary(self, question: str, indices: list[GenerativeIndex]) -> list[int]:
        """阶段 A：LLM 看所有 summary 粗筛。"""
        doc_list = "\n".join(
            f"[{idx.doc_id}] {idx.summary_short or idx.title}"
            for idx in indices
        )
        messages = [{"role": "user", "content": FILTER_PROMPT.format(
            question=question, doc_list=doc_list,
        )}]
        resp = self.gw.generate(self.role, messages, role_tag="c2_filter")
        return self._parse_id_list(resp.text, max_n=self.top_k_filter)

    def _rerank_by_annotations(
        self, question: str, indices: list[GenerativeIndex], candidate_ids: list[int],
    ) -> list[int]:
        """阶段 B：LLM 看候选的富语义标注精排。"""
        id_to_idx = {idx.doc_id: idx for idx in indices}
        candidates_str = ""
        for cid in candidate_ids:
            idx = id_to_idx.get(cid)
            if not idx:
                continue
            qs = " | ".join(idx.hypothetical_questions[:3])
            ents = ", ".join(e.get("entity", "") for e in idx.entities[:5])
            candidates_str += (
                f"[{cid}] {idx.title}\n"
                f"  Questions: {qs}\n"
                f"  Entities: {ents}\n"
                f"  Summary: {idx.summary_short}\n\n"
            )
        if not candidates_str:
            return candidate_ids[:self.top_k_final]
        messages = [{"role": "user", "content": RERANK_PROMPT.format(
            question=question, candidates=candidates_str, top_k=self.top_k_final,
        )}]
        resp = self.gw.generate(self.role, messages, role_tag="c2_rerank")
        ids = self._parse_id_list(resp.text, max_n=self.top_k_final)
        return ids if ids else candidate_ids[:self.top_k_final]

    # ── 工具方法 ──────────────────────────────────────────

    @staticmethod
    def _parse_id_list(text: str, max_n: int) -> list[int]:
        """从 LLM 输出解析 id 列表。"""
        import re
        match = re.search(r"\[([^\]]*)\]", text)
        if not match:
            return []
        try:
            nums = [int(x.strip()) for x in match.group(1).split(",") if x.strip().lstrip("-").isdigit()]
            return nums[:max_n]
        except ValueError:
            return []

    @staticmethod
    def _format_context(docs: list[GenerativeIndex]) -> str:
        parts = []
        for i, d in enumerate(docs, 1):
            parts.append(f"[{i}] {d.title}\n{d.text}")
        return "\n\n".join(parts)
