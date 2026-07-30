"""
RAG² LongContext baseline — 全文塞窗口，不检索

管道：把检索池/全文整体拼入 1M 上下文窗口 → LLM 直接答

代表"长上下文取代 RAG"的范式。E0 拐点实验用它和 TraditionalRAG 对照，
验证 H1：access 是否仍是瓶颈。
"""
from __future__ import annotations

import logging

from rag2.gateway import ModelGateway, ContextOverflowError
from rag2.methods.base import Method, Result, RetrievedDoc

logger = logging.getLogger(__name__)


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


class LongContext(Method):
    """长上下文 baseline：全文塞窗口，无检索。"""

    name = "long_context"

    def __init__(self, gateway: ModelGateway, role: str = "generator"):
        super().__init__(gateway, role)

    def run(self, sample: dict) -> Result:
        question = sample["question"]
        docs = sample.get("supporting_docs", [])

        # 把所有文档拼成 context
        context = self._format_context(docs)

        messages = [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": PROMPT_USER_TMPL.format(
                context=context, question=question,
            )},
        ]

        try:
            resp = self.gw.generate(self.role, messages, role_tag=self.name)
            answer = resp.text.strip()
            overflow = False
        except ContextOverflowError as e:
            # 超窗口：这是 E0 实验的关键信号，不是错误
            # 记录下来供分析"长上下文何时失效"
            logger.warning("LongContext 超窗口（E0 拐点信号）: %s", e)
            answer = ""
            overflow = True
            resp = None

        # 所有文档都算"检索到"（语义上都在 context 里）
        retrieved = [
            RetrievedDoc(title=d.get("title", ""), text=d.get("text", ""),
                         is_supporting=d.get("is_supporting", False))
            for d in docs
        ]

        return Result(
            answer=answer,
            retrieved_docs=retrieved,
            trace={
                "method": self.name,
                "n_context_docs": len(docs),
                "context_chars": len(context),
                "prompt_tokens": resp.usage.get("prompt_tokens", 0) if resp else 0,
                "completion_tokens": resp.usage.get("completion_tokens", 0) if resp else 0,
                "context_overflow": overflow,
                "from_cache": resp.from_cache if resp else False,
            },
        )

    @staticmethod
    def _format_context(docs: list[dict]) -> str:
        parts = []
        for i, d in enumerate(docs, 1):
            parts.append(f"[{i}] {d.get('title','')}\n{d.get('text','')}")
        return "\n\n".join(parts)
