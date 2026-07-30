"""
RAG² C1: 四种 loop 策略实现

S1 纯 ReAct：模型自主，每步推理出子查询
S2 +反思重规划：每 N 步插入"已知/未知/下一步"反思
S3 +并行分支：同时启动多个子查询
S4 预设策略：按问题类型（单跳/多跳/实体）路由到固定流程
"""
from __future__ import annotations

import logging
from typing import Optional

from rag2.gateway import ModelGateway
from rag2.methods.c1_orchestrator import ReasoningOrchestrator, ReasoningStep
from rag2.methods.c2_backends import RetrievalBackend

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# S1: 纯 ReAct（基线策略）
# ─────────────────────────────────────────────────────────

S1_SYSTEM = """You are a reasoning-driven retrieval orchestrator. Answer questions by iteratively reasoning about what evidence you need and retrieving it.

Process:
1. REASON about the question: what facts do you need?
2. Call issue_subqueries with 1-3 focused sub-queries to retrieve evidence.
3. After receiving evidence, REASON whether you have enough.
4. If enough, call submit_answer with the concise answer.
5. If not enough, call issue_subqueries again with NEW queries for missing evidence.

CRITICAL RULES:
- Each sub-query should be a focused, searchable phrase (not the full question).
- For multi-hop questions, decompose into hops.
- BE EFFICIENT: call submit_answer as soon as you have direct evidence.
- DO NOT repeat sub-queries you already issued. Try a DIFFERENT angle if previous didn't help.
- If collected evidence already contains the answer, you MUST call submit_answer immediately."""

S1_USER_TMPL = """Question: {question}

{evidence_section}{history_section}

What sub-queries do you need now? (Or give FINAL ANSWER if you have enough evidence.)"""


class S1PureReAct(ReasoningOrchestrator):
    """S1：纯 ReAct。模型自主决定每步子查询。"""

    name = "S1_pure_react"

    def _build_messages(self, question, steps, collected_doc_ids, id_to_idx, sample):
        evidence = self._format_collected_evidence(collected_doc_ids, id_to_idx)
        history = self._format_history(steps)
        user = S1_USER_TMPL.format(
            question=question,
            evidence_section=f"Collected evidence:\n{evidence}\n" if collected_doc_ids else "",
            history_section=f"Previous reasoning:\n{history}\n" if history else "",
        )
        return [
            {"role": "system", "content": S1_SYSTEM},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _format_history(steps: list[ReasoningStep]) -> str:
        """格式化历史步骤（供模型回忆）。"""
        if not steps:
            return ""
        lines = []
        for s in steps[-3:]:  # 只保留最近 3 步，控 prompt 长度
            sq_str = ", ".join(s.sub_queries) if s.sub_queries else "(none)"
            lines.append(f"Step {s.step}: queried [{sq_str}] → got docs {s.retrieved_doc_ids}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# S2: ReAct + 反思重规划
# ─────────────────────────────────────────────────────────

S2_REFLECT_PROMPT = """You are reflecting on the retrieval progress so far.

Question: {question}

Steps taken:
{steps_summary}

Collected evidence summary:
{evidence}

Reflect:
1. KNOWN: What facts have you established?
2. UNKNOWN: What is still missing to answer the question?
3. NEXT: What specific sub-queries should you issue next? (Or state ANSWER_READY if sufficient)

Output format:
KNOWN: ...
UNKNOWN: ...
NEXT: SUBQUERIES: [...] OR ANSWER_READY"""


class S2ReflectReplan(ReasoningOrchestrator):
    """S2：每 reflect_every 步插入反思，重规划下一步。"""

    name = "S2_reflect"

    def __init__(self, *args, reflect_every: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.reflect_every = reflect_every

    def _build_messages(self, question, steps, collected_doc_ids, id_to_idx, sample):
        step_num = len(steps)

        # 触发反思：每 reflect_every 步且不是第一步
        if step_num > 0 and step_num % self.reflect_every == 0 and collected_doc_ids:
            return self._build_reflect_messages(question, steps, collected_doc_ids, id_to_idx)

        # 正常 ReAct 步骤（复用 S1 的 prompt）
        evidence = self._format_collected_evidence(collected_doc_ids, id_to_idx)
        history = S1PureReAct._format_history(steps)
        user = S1_USER_TMPL.format(
            question=question,
            evidence_section=f"Collected evidence:\n{evidence}\n" if collected_doc_ids else "",
            history_section=f"Previous reasoning:\n{history}\n" if history else "",
        )
        return [
            {"role": "system", "content": S1_SYSTEM + "\n\nYou may also reflect on progress periodically."},
            {"role": "user", "content": user},
        ]

    def _build_reflect_messages(self, question, steps, collected_doc_ids, id_to_idx):
        """构造反思 prompt。"""
        steps_summary = "\n".join(
            f"Step {s.step}: queried {s.sub_queries} → docs {s.retrieved_doc_ids}"
            for s in steps
        )
        evidence = self._format_collected_evidence(collected_doc_ids, id_to_idx, max_docs=3)
        user = S2_REFLECT_PROMPT.format(
            question=question, steps_summary=steps_summary, evidence=evidence,
        )
        return [{"role": "user", "content": user}]

    def _should_continue(self, steps, collected_doc_ids):
        """反思结果若 ANSWER_READY 则停止。"""
        if not steps:
            return True
        last = steps[-1]
        # 反思步的 thought 里若有 ANSWER_READY，标记停止
        if "ANSWER_READY" in (last.thought or ""):
            # 强制下一步生成答案
            return False
        return True


# ─────────────────────────────────────────────────────────
# S3: ReAct + 并行分支
# ─────────────────────────────────────────────────────────

S3_SYSTEM = S1_SYSTEM + """

You are running PARALLEL exploration. For each step, output MULTIPLE independent sub-queries that explore different aspects simultaneously:
SUBQUERIES: ["branch 1 query", "branch 2 query", "branch 3 query"]
Aim for 2-3 DIVERSE sub-queries per step to maximize coverage."""


class S3ParallelBranches(ReasoningOrchestrator):
    """S3：每步生成多个并行子查询，扩大覆盖。"""

    name = "S3_parallel"

    def __init__(self, *args, **kwargs):
        # S3 每步查更多，调小 top_k 控总量
        kwargs.setdefault("top_k_per_query", 2)
        super().__init__(*args, **kwargs)

    def _build_messages(self, question, steps, collected_doc_ids, id_to_idx, sample):
        evidence = self._format_collected_evidence(collected_doc_ids, id_to_idx)
        history = S1PureReact._format_history(steps)
        user = S1_USER_TMPL.format(
            question=question,
            evidence_section=f"Collected evidence:\n{evidence}\n" if collected_doc_ids else "",
            history_section=f"Previous reasoning:\n{history}\n" if history else "",
        )
        return [
            {"role": "system", "content": S3_SYSTEM},
            {"role": "user", "content": user},
        ]

    def _extract_action(self, resp):
        """S3 允许更多子查询（最多 3 个并行分支）。"""
        sqs, is_final, answer = super()._extract_action(resp)
        return sqs[:3], is_final, answer


# ─────────────────────────────────────────────────────────
# S4: 预设策略（按问题类型路由）
# ─────────────────────────────────────────────────────────

S4_CLASSIFY_PROMPT = """Classify this question into ONE type for retrieval strategy selection:

Question: {question}

Types:
- single_hop: direct factual question, answerable from one document
- multi_hop: requires chaining facts across 2+ documents (e.g. "Who is the spouse of the director of X?")
- entity_lookup: asks about a specific entity's attributes (e.g. "When was X born?")

Output ONLY the type name (single_hop / multi_hop / entity_lookup), nothing else."""


class S4PresetPolicy(ReasoningOrchestrator):
    """S4：先分类问题类型，按预设流程执行。"""

    name = "S4_preset"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._question_type: Optional[str] = None

    def _classify(self, question: str) -> str:
        """问题类型分类。"""
        resp = self.gw.generate(
            self.role,
            [{"role": "user", "content": S4_CLASSIFY_PROMPT.format(question=question)}],
            role_tag=self.name + "_classify", max_tokens=50,
        )
        qtype = resp.text.strip().lower()
        for t in ("single_hop", "multi_hop", "entity_lookup"):
            if t in qtype:
                return t
        return "single_hop"  # 默认

    def _build_messages(self, question, steps, collected_doc_ids, id_to_idx, sample):
        # 第一步：先分类
        if self._question_type is None:
            self._question_type = self._classify(question)
            logger.info("S4 分类: %s → %s", question[:40], self._question_type)

        evidence = self._format_collected_evidence(collected_doc_ids, id_to_idx)

        # 按类型给不同的 system prompt
        if self._question_type == "multi_hop":
            system = S1_SYSTEM + """

This is a MULTI-HOP question. Strategy:
1. First identify the linking entity/intermediate fact.
2. Query for the intermediate fact first.
3. Then query for the final answer using the intermediate result.
Decompose carefully: SUBQUERIES: ["hop 1 query", "hop 2 query"]"""
        elif self._question_type == "entity_lookup":
            system = S1_SYSTEM + """

This is an ENTITY LOOKUP question. Strategy:
1. Extract the core entity name.
2. Query directly for that entity's relevant attribute.
Usually 1 sub-query suffices."""
        else:  # single_hop
            system = S1_SYSTEM + """

This is a SINGLE-HOP question. A single focused sub-query should suffice.
Do not over-explore."""

        history = S1PureReact._format_history(steps)
        user = S1_USER_TMPL.format(
            question=question,
            evidence_section=f"Collected evidence:\n{evidence}\n" if collected_doc_ids else "",
            history_section=f"Previous reasoning:\n{history}\n" if history else "",
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ─────────────────────────────────────────────────────────
# 工厂
# ─────────────────────────────────────────────────────────

def build_strategy(
    strategy_name: str,
    gateway: ModelGateway,
    backend: RetrievalBackend,
    role: str = "generator",
    **kwargs,
) -> ReasoningOrchestrator:
    """按名字构造 C1 策略。"""
    strategies = {
        "S1": S1PureReAct,
        "S2": S2ReflectReplan,
        "S3": S3ParallelBranches,
        "S4": S4PresetPolicy,
    }
    name_key = strategy_name.upper().split("_")[0]  # "S1_pure_react" → "S1"
    cls = strategies.get(name_key)
    if cls is None:
        raise ValueError(f"未知策略: {strategy_name}（可选 S1/S2/S3/S4）")
    return cls(gateway, backend, role=role, **kwargs)
