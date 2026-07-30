"""
RAG² TraditionalRAG — 经典 chunk+embed+top-k baseline

管道：检索(bge-m3+FAISS+rerank) → 拼 prompt → LLM 生成

这是 E1/E2/E4 的统一对照基线，代表"retrieval-centric"传统范式。
RAG² 的 thesis 正是要 challenge 这个范式。
"""
from __future__ import annotations

import logging

from rag2.gateway import ModelGateway
from rag2.methods.base import Method, Result, RetrievedDoc
from rag2.methods.retriever import Retriever

logger = logging.getLogger(__name__)


# ── Prompt 模板 ──────────────────────────────────────────

PROMPT_SYSTEM = (
    "You are a faithful question answering assistant. "
    "Answer the question based ONLY on the provided context. "
    "If the context does not contain the answer, say \"I don't know\". "
    "Be concise."
)

PROMPT_USER_TMPL = """Context:
{context}

Question: {question}

Answer (based only on the context above):"""


class TraditionalRAG(Method):
    """经典 RAG baseline：检索 → 拼接 → 生成。"""

    name = "traditional_rag"

    def __init__(
        self,
        gateway: ModelGateway,
        retriever: Retriever,
        role: str = "generator",
        top_k_recall: int = 10,
        top_k_rerank: int = 5,
    ):
        super().__init__(gateway, role)
        self.retriever = retriever
        self.top_k_recall = top_k_recall
        self.top_k_rerank = top_k_rerank

    def run(self, sample: dict) -> Result:
        question = sample["question"]
        docs = sample.get("supporting_docs", [])

        # 1. 检索
        hits = self.retriever.search(
            question, self.top_k_recall, self.top_k_rerank, rerank=True,
        )
        retrieved = [
            RetrievedDoc(
                title=h["title"], text=h["text"],
                score=h.get("rerank_score", h.get("score", 0.0)),
                is_supporting=h.get("is_supporting", False),
            )
            for h in hits
        ]

        # 2. 拼 context
        context = self._format_context(retrieved)

        # 3. 生成
        messages = [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": PROMPT_USER_TMPL.format(
                context=context, question=question,
            )},
        ]
        resp = self.gw.generate(self.role, messages, role_tag=self.name)

        # 4. 记录是否截断
        finish_reason = ""
        if resp.raw and '"finish_reason"' in str(resp.raw):
            # 粗解析；精确解析留给真实 API 返回结构
            import re
            m = re.search(r'"finish_reason"\s*:\s*"(\w+)"', str(resp.raw))
            finish_reason = m.group(1) if m else ""

        return Result(
            answer=resp.text.strip(),
            retrieved_docs=retrieved,
            trace={
                "method": self.name,
                "n_candidates": len(docs),
                "n_retrieved": len(retrieved),
                "prompt_tokens": resp.usage.get("prompt_tokens", 0),
                "completion_tokens": resp.usage.get("completion_tokens", 0),
                "finish_reason": finish_reason,
                "from_cache": resp.from_cache,
            },
            raw_response=resp.raw,
        )

    @staticmethod
    def _format_context(docs: list[RetrievedDoc]) -> str:
        """把检索结果拼成 context 文本。"""
        parts = []
        for i, d in enumerate(docs, 1):
            parts.append(f"[{i}] {d.title}\n{d.text}")
        return "\n\n".join(parts)
