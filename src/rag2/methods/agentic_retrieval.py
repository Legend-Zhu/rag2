"""
RAG² C1: AgenticRetrieval — 模型驱动自适应检索

核心思想：模型通过工具调用（grep/read/symbol_jump 等）在线、多轮、自适应地
探查原始语料，取消 chunk + embedding 预建步骤。这是 thesis H2 的实证。

ReAct 循环：
  Thought → Action(tool_call) → Observation(tool_result) → ... → Answer

控制：
  - max_steps：探查预算，防止无限循环（成本控制）
  - 每步的工具调用记入 trace，供论文分析
  - 收集所有 read 过的文档作为 retrieved_docs（供评测 recall@k）
"""
from __future__ import annotations

import json
import logging

from rag2.gateway import ModelGateway
from rag2.methods.base import Method, Result, RetrievedDoc
from rag2.methods.agentic_tools import (
    DocumentCorpus, build_tool_schemas, dispatch_tool,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a research assistant that answers questions by EXPLORING a document corpus using tools.

You have access to these tools:
- list_docs: list document titles and ids (start here to see what's available)
- grep <pattern>: search across all documents for keywords/regex
- read <doc_id>: read a specific document in full
- outline: get a compact outline (title + first sentence) of all docs
- symbol_jump <entity>: find all occurrences of an entity (for multi-hop reasoning)

CRITICAL EFFICIENCY RULES:
1. The grep tool returns snippets with context — often the answer is already visible in a snippet. READ THE SNIPPETS CAREFULLY before reading full documents.
2. Once you have found a document that directly answers the question, STOP exploring. Do not search further. Immediately give FINAL ANSWER.
3. Avoid "confirmation searching" — if doc A clearly states the answer, do not search doc B, C, D to double-check.
4. For multi-hop questions (e.g. "Who is the spouse of X?"), use symbol_jump to trace the linking entity across documents, then read only the documents that mention it.

STRATEGY:
1. grep for the most distinctive entity in the question.
2. READ the snippets returned — the answer is often there.
3. Only read full documents if snippets are insufficient.
4. Give FINAL ANSWER as soon as you have direct evidence.

OUTPUT FORMAT:
When you have the answer, respond with ONLY:
FINAL ANSWER: <concise answer, just the entity/phrase>

Do NOT explain your reasoning in the final answer."""

MAX_STEPS_DEFAULT = 12


class AgenticRetrieval(Method):
    """模型驱动自适应检索：模型通过工具在线探查语料。"""

    name = "agentic_retrieval"

    def __init__(
        self,
        gateway: ModelGateway,
        role: str = "generator",
        max_steps: int = MAX_STEPS_DEFAULT,
        max_tool_results_chars: int = 8000,
    ):
        super().__init__(gateway, role)
        self.max_steps = max_steps
        self.max_tool_results_chars = max_tool_results_chars
        self._tool_schemas = build_tool_schemas()

    def run(self, sample: dict) -> Result:
        question = sample["question"]
        docs = sample.get("supporting_docs", [])
        corpus = DocumentCorpus(docs=docs)

        # 对话历史：system + 用户问题
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}\n\nExplore the corpus and answer."},
        ]

        trace_steps = []          # 每步的 (thought/action/observation) 记录
        read_docs: list[dict] = []  # 所有 read 过的文档（用于评测 recall）
        final_answer = ""

        for step in range(self.max_steps):
            # 调模型，可能返回工具调用或最终答案
            resp = self.gw.generate(
                self.role, messages, tools=self._tool_schemas, role_tag=self.name,
            )

            # 情况 1：模型给出工具调用
            if resp.tool_calls:
                # 把模型的 tool_call 消息加入历史
                messages.append({
                    "role": "assistant",
                    "content": resp.text or "",
                    "tool_calls": resp.tool_calls,
                })
                # 执行每个工具调用
                tool_results_total_chars = 0
                for tc in resp.tool_calls:
                    fn = tc["function"]
                    name = fn["name"]
                    try:
                        args = json.loads(fn["arguments"]) if fn["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    result = dispatch_tool(corpus, name, args)
                    # 记录 read 过的文档
                    if name == "read" and "doc_id" in args:
                        doc = corpus._by_id.get(args["doc_id"])
                        if doc:
                            read_docs.append(doc)
                    # 累计结果长度，超限则截断
                    tool_results_total_chars += len(result)
                    if tool_results_total_chars > self.max_tool_results_chars:
                        result = result[:500] + "\n[... 结果过多，已截断]"
                    # 工具结果作为 tool 角色消息
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })
                    trace_steps.append({
                        "step": step, "tool": name, "args": args,
                        "result_preview": result[:200],
                    })
                    logger.debug("  step %d: %s(%s) → %d chars", step, name, args, len(result))
                continue

            # 情况 2：模型给出最终答案（无工具调用）
            text = resp.text.strip()
            # 提取 FINAL ANSWER
            if "FINAL ANSWER:" in text:
                final_answer = text.split("FINAL ANSWER:", 1)[1].strip()
            else:
                final_answer = text
            # 去掉可能的引号包裹
            final_answer = final_answer.strip("\"'")
            trace_steps.append({"step": step, "type": "final_answer", "text": final_answer[:200]})
            break
        else:
            # 用尽 max_steps 仍未给出答案
            final_answer = ""
            trace_steps.append({"step": self.max_steps, "type": "max_steps_exceeded"})

        # 构造 retrieved_docs（read 过的，标注是否 gold）
        gold_titles = {d["title"] for d in docs if d.get("is_supporting")}
        retrieved = [
            RetrievedDoc(
                title=d.get("title", ""), text=d.get("text", ""),
                is_supporting=d.get("title", "") in gold_titles,
            )
            for d in read_docs
        ]

        return Result(
            answer=final_answer,
            retrieved_docs=retrieved,
            trace={
                "method": self.name,
                "n_corpus_docs": len(docs),
                "n_read_docs": len(read_docs),
                "n_steps": len(trace_steps),
                "max_steps": self.max_steps,
                "steps": trace_steps,
                "completion_tokens": resp.usage.get("completion_tokens", 0),
                "from_cache": resp.from_cache,
            },
        )
