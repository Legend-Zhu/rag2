"""
RAG² C1: 推理驱动的检索编排器（v2 — 按架构决策.md 重构）

核心：模型推理出"需要什么事实" → 调 C2 后端检索 → 判断"够不够" → 循环。
不再是 grep/read 工具集（那是 coding agent 换皮），而是 reasoning-centric
的检索编排。

四种 loop 策略（共享推理骨架，差异在控制层）：
  S1 纯 ReAct：模型自主，无额外机制
  S2 +反思重规划：每 N 步反思
  S3 +并行分支：多路径探查
  S4 预设策略：按问题类型路由
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from rag2.gateway import ModelGateway
from rag2.methods.base import Method, Result, RetrievedDoc
from rag2.methods.generative_index import GenerativeIndex, CorpusGraph
from rag2.methods.c2_backends import RetrievalBackend

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 工具 schema：模型用工具提交子查询或最终答案（替代文本解析）
# ─────────────────────────────────────────────────────────

C1_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "issue_subqueries",
            "description": "Issue sub-queries to retrieve more evidence. Use when you need more information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "1-3 focused sub-queries to retrieve",
                    },
                },
                "required": ["queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit the final answer when you have enough evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "ONLY the answer entity/phrase, maximum 5 words. No explanation, no full sentence. Example: 'mitochondria' not 'The mitochondria is the powerhouse...'",
                    },
                },
                "required": ["answer"],
            },
        },
    },
]


# ─────────────────────────────────────────────────────────
# 推理步骤的数据结构
# ─────────────────────────────────────────────────────────

@dataclass
class ReasoningStep:
    """单步推理记录（供 trace 分析）。"""
    step: int
    thought: str = ""              # 模型的推理
    sub_queries: list[str] = field(default_factory=list)  # 本步要查的子查询
    retrieved_doc_ids: list[int] = field(default_factory=list)  # 检索到的文档
    evidence_summary: str = ""     # 本步获得的证据
    action: str = ""               # "retrieve" / "reflect" / "answer"
    elapsed_s: float = 0.0


# ─────────────────────────────────────────────────────────
# 推理循环骨架（所有策略共享）
# ─────────────────────────────────────────────────────────

class ReasoningOrchestrator(Method):
    """
    推理驱动的检索编排器基类。

    子类（S1/S2/S3/S4）实现 _build_messages 和 _should_continue 控制循环。
    所有策略共享：C2 后端检索 + 证据收集 + 最终答案生成。
    """

    name: str = "reasoning_orchestrator"

    def __init__(
        self,
        gateway: ModelGateway,
        backend: RetrievalBackend,
        role: str = "generator",
        max_steps: int = 6,
        top_k_per_query: int = 3,
    ):
        super().__init__(gateway, role)
        self.backend = backend
        self.max_steps = max_steps
        self.top_k_per_query = top_k_per_query

    def run(self, sample: dict) -> Result:
        """主循环：推理 → 检索 → 判断 → 循环。"""
        question = sample["question"]
        docs = sample.get("supporting_docs", [])
        t0 = time.time()

        steps: list[ReasoningStep] = []
        collected_doc_ids: list[int] = []   # 所有轮累积的 doc_id（去重）
        collected_set: set[int] = set()
        id_to_idx = self.backend._id_to_idx
        final_answer = ""

        for step_num in range(self.max_steps):
            step = ReasoningStep(step=step_num)
            step_t0 = time.time()

            # 1. 构造 prompt（子类决定怎么构造，含反思/并行等）
            messages = self._build_messages(
                question, steps, collected_doc_ids, id_to_idx, sample,
            )

            # 2. 调模型推理（用工具调用，替代文本解析）
            resp = self.gw.generate_complete(
                self.role, messages, role_tag=self.name,
                tools=C1_TOOL_SCHEMA,
                max_tokens=4096, max_continuations=3,
            )
            step.thought = resp.text

            # 3. 从工具调用提取：子查询 or 最终答案（不解析文本）
            sub_queries, is_final, answer = self._extract_action(resp)

            if is_final:
                final_answer = answer
                step.action = "answer"
                step.elapsed_s = time.time() - step_t0
                steps.append(step)
                break

            if not sub_queries:
                # 模型既没给子查询也没给答案 → 兜底强制答
                logger.warning("step %d 无工具调用，强制生成答案", step_num)
                final_answer = self._force_answer(question, collected_doc_ids, id_to_idx)
                step.action = "force_answer"
                step.elapsed_s = time.time() - step_t0
                steps.append(step)
                break

            # 4. 执行检索（对每个子查询调后端）
            step.sub_queries = sub_queries
            step.action = "retrieve"
            for sq in sub_queries:
                hits = self.backend.retrieve(sq, top_k=self.top_k_per_query)
                for h in hits:
                    if h not in collected_set:
                        collected_doc_ids.append(h)
                        collected_set.add(h)
                step.retrieved_doc_ids.extend(hits)

            step.elapsed_s = time.time() - step_t0
            steps.append(step)

            # 5. 控制层钩子：是否继续（子类可覆盖，如 S2 反思）
            if not self._should_continue(steps, collected_doc_ids):
                break
        else:
            # 用尽 max_steps，强制生成答案
            final_answer = self._force_answer(question, collected_doc_ids, id_to_idx)

        # 构造 retrieved_docs（累积的所有文档）
        gold_titles = {d["title"] for d in docs if d.get("is_supporting")}
        retrieved = []
        for did in collected_doc_ids:
            idx = id_to_idx.get(did)
            if idx:
                retrieved.append(RetrievedDoc(
                    title=idx.title, text=idx.text,
                    is_supporting=idx.title in gold_titles,
                ))

        return Result(
            answer=final_answer,
            retrieved_docs=retrieved,
            trace={
                "method": self.name,
                "backend": self.backend.name,
                "n_steps": len(steps),
                "n_collected_docs": len(collected_doc_ids),
                "n_corpus_docs": len(docs),
                "max_steps": self.max_steps,
                "steps": [
                    {
                        "step": s.step, "action": s.action,
                        "sub_queries": s.sub_queries,
                        "retrieved_doc_ids": s.retrieved_doc_ids,
                        "elapsed_s": s.elapsed_s,
                    } for s in steps
                ],
                "total_elapsed_s": time.time() - t0,
            },
        )

    # ── 子类必须实现 ──────────────────────────────────────

    @abstractmethod
    def _build_messages(
        self, question: str, steps: list[ReasoningStep],
        collected_doc_ids: list[int], id_to_idx: dict, sample: dict,
    ) -> list[dict]:
        """构造本步的 prompt。不同策略差异在此。"""
        ...

    def _should_continue(self, steps: list[ReasoningStep], collected_doc_ids: list[int]) -> bool:
        """是否继续循环。默认始终继续到 max_steps（S1）。S2 可加反思判断。"""
        # 通用早停：连续两步子查询高度重复 → 判定陷入循环
        if len(steps) >= 2:
            last_sq = set(q.lower().strip() for q in steps[-1].sub_queries)
            prev_sq = set(q.lower().strip() for q in steps[-2].sub_queries)
            if last_sq and prev_sq and last_sq == prev_sq:
                logger.info("早停：连续两步子查询完全相同，判定循环")
                return False
            # 检索结果也无新增（收集数未变）
            if (len(steps) >= 2
                and steps[-1].retrieved_doc_ids == steps[-2].retrieved_doc_ids):
                logger.info("早停：连续两步检索结果相同，无新证据")
                return False
        return True

    # ── 共享工具方法 ──────────────────────────────────────

    @staticmethod
    def _extract_action(resp) -> tuple[list[str], bool, str]:
        """
        从工具调用提取模型动作（不解析文本）。
        返回 (sub_queries, is_final, answer)。
        """
        for tc in resp.tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                continue
            if name == "submit_answer":
                answer = str(args.get("answer", "")).strip()[:300]
                return [], True, answer
            if name == "issue_subqueries":
                queries = args.get("queries", [])
                if isinstance(queries, list):
                    return [str(q).strip() for q in queries if str(q).strip()][:3], False, ""
        return [], False, ""

    def _force_answer(self, question: str, doc_ids: list[int], id_to_idx: dict) -> str:
        """用尽 max_steps 后强制生成答案。"""
        docs_text = "\n\n".join(
            f"[{i+1}] {id_to_idx[d].title}\n{id_to_idx[d].text[:1000]}"
            for i, d in enumerate(doc_ids[:5])
            if d in id_to_idx
        )
        prompt = (
            f"Based on the gathered evidence, answer the question concisely.\n\n"
            f"Evidence:\n{docs_text}\n\nQuestion: {question}\n\nFINAL ANSWER:"
        )
        resp = self.gw.generate(
            self.role, [{"role": "user", "content": prompt}],
            role_tag=self.name + "_force", max_tokens=2000,
        )
        if "FINAL ANSWER:" in resp.text:
            return resp.text.split("FINAL ANSWER:", 1)[1].strip().strip("\"'")[:300]
        return resp.text.strip()[:300]

    @staticmethod
    def _format_collected_evidence(doc_ids: list[int], id_to_idx: dict, max_docs: int = 5) -> str:
        """格式化已收集的证据（供 prompt 用）。"""
        parts = []
        for i, did in enumerate(doc_ids[-max_docs:], 1):  # 取最近 max_docs 个
            idx = id_to_idx.get(did)
            if idx:
                parts.append(f"[{did}] {idx.title}\n{idx.text[:600]}")
        return "\n\n".join(parts) if parts else "(暂无)"
