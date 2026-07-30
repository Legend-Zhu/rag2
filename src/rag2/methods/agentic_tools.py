"""
RAG² C1: Agentic 检索工具集

借鉴 coding agent（Cursor/Claude Code/Aider）的 grep/glob/符号跳转范式，
适配到通用非结构化文档语料。核心思想：让模型通过工具调用在线、多轮、
自适应地探查原始语料，取消 chunk + embedding 的预建步骤。

这是 thesis H2（模型即检索器）的实证基础。

工具集设计原则：
  1. 工具原语要"够用但不过多"——太多模型选择困难，太少探查能力不足
  2. 每个工具返回的信息量要适中——返回太少模型要多轮，返回太多浪费 token
  3. 工具要支持"渐进式聚焦"——先粗看结构，再精读相关段
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 语料抽象：把文档列表组织成可探查的结构
# ─────────────────────────────────────────────────────────

@dataclass
class DocumentCorpus:
    """
    文档语料的可探查视图。

    把 list[dict]（DataLoader 产出的 supporting_docs）组织成：
      - doc_id → {title, text}
      - 倒排索引（实体/关键词 → doc_ids）供 grep / symbol_jump 用
      - 文档大纲（首句/标题）供 list / outline 用
    """
    docs: list[dict] = field(default_factory=list)
    _by_id: dict[int, dict] = field(default_factory=dict, init=False)
    _title_to_ids: dict[str, list[int]] = field(default_factory=dict, init=False)

    def __post_init__(self):
        for i, d in enumerate(self.docs):
            self._by_id[i] = d
            title = d.get("title", "").strip().lower()
            self._title_to_ids.setdefault(title, []).append(i)

    def __len__(self):
        return len(self.docs)

    # ── 工具实现 ──────────────────────────────────────────

    def list_docs(self, limit: int = 20, offset: int = 0) -> str:
        """
        列出语料库中的文档标题（带 id），供模型粗览。
        类似 coding agent 的 list_dir。
        """
        lines = [f"共 {len(self.docs)} 篇文档。当前显示 {offset}–{min(offset+limit, len(self.docs))}："]
        for i in range(offset, min(offset + limit, len(self.docs))):
            d = self._by_id[i]
            title = d.get("title", "(无标题)")
            text = d.get("text", "")
            preview = text[:80].replace("\n", " ")
            lines.append(f"[doc {i}] {title} | {preview}...")
        return "\n".join(lines)

    def grep(self, pattern: str, max_hits: int = 10, context_chars: int = 150) -> str:
        """
        全文/正则检索：在所有文档里找匹配 pattern 的片段。
        类似 coding agent 的 grep。返回命中片段 + 来源 doc_id。

        Args:
            pattern: 关键词或正则表达式
            max_hits: 最多返回多少条命中
            context_chars: 每条命中前后保留多少字符
        """
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            # 非法正则，降级为字面量
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        hits = []
        for doc_id, d in self._by_id.items():
            text = d.get("text", "")
            for m in regex.finditer(text):
                start = max(0, m.start() - context_chars)
                end = min(len(text), m.end() + context_chars)
                snippet = text[start:end].replace("\n", " ")
                prefix = "..." if start > 0 else ""
                suffix = "..." if end < len(text) else ""
                hits.append({
                    "doc_id": doc_id,
                    "title": d.get("title", ""),
                    "snippet": f"{prefix}{snippet}{suffix}",
                })
                if len(hits) >= max_hits:
                    break
            if len(hits) >= max_hits:
                break

        if not hits:
            return f"未找到匹配 \"{pattern}\" 的内容。"
        lines = [f"找到 {len(hits)} 条匹配（限制 {max_hits}）："]
        for h in hits:
            lines.append(f"[doc {h['doc_id']}] {h['title']}: {h['snippet']}")
        return "\n".join(lines)

    def read(self, doc_id: int, max_chars: int = 3000) -> str:
        """
        读取指定文档全文（截断到 max_chars）。
        类似 coding agent 的 read。
        """
        if doc_id not in self._by_id:
            return f"错误：doc_id {doc_id} 不存在（共 {len(self.docs)} 篇）"
        d = self._by_id[doc_id]
        text = d.get("text", "")
        truncated = len(text) > max_chars
        content = text[:max_chars]
        if truncated:
            content += f"\n\n[... 截断，原文共 {len(text)} 字符，已显示前 {max_chars}]"
        return f"[doc {doc_id}] {d.get('title','')}\n\n{content}"

    def outline(self, limit: int = 30) -> str:
        """
        获取文档大纲：每篇文档的标题 + 首句。
        类似 coding agent 的符号大纲，帮助模型快速定位。
        """
        lines = [f"文档大纲（共 {len(self.docs)} 篇，显示前 {min(limit, len(self.docs))}）："]
        for i, d in list(self._by_id.items())[:limit]:
            title = d.get("title", "(无标题)")
            text = d.get("text", "")
            # 首句：到第一个句号/换行
            first_sent = re.split(r"[。.\n]", text, 1)[0][:100]
            lines.append(f"[doc {i}] {title} — {first_sent}")
        return "\n".join(lines)

    def symbol_jump(self, entity: str, max_hits: int = 10) -> str:
        """
        跨文档实体跳转：找所有提到某实体的文档。
        类似 coding agent 的 symbol jump（go-to-definition/find-references）。
        用于多跳推理的中间实体追踪。
        """
        return self.grep(re.escape(entity), max_hits=max_hits, context_chars=200)


# ─────────────────────────────────────────────────────────
# 工具注册：把 corpus 方法暴露成 OpenAI function calling 格式
# ─────────────────────────────────────────────────────────

def build_tool_schemas() -> list[dict]:
    """返回 OpenAI tools 格式的工具定义。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_docs",
                "description": "列出语料库中的文档标题和 id，用于粗览可用文档。先调此工具了解有哪些文档。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "返回文档数，默认 20", "default": 20},
                        "offset": {"type": "integer", "description": "偏移量，用于翻页", "default": 0},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "grep",
                "description": "在所有文档中检索关键词或正则表达式，返回命中片段。用于定位与问题相关的内容。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string", "description": "关键词或正则表达式"},
                        "max_hits": {"type": "integer", "description": "最大命中数，默认 10", "default": 10},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "读取指定 doc_id 的文档全文。grep/list_docs 定位到相关文档后用此工具精读。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "integer", "description": "文档 id（来自 list_docs/grep）"},
                        "max_chars": {"type": "integer", "description": "最大读取字符数，默认 3000", "default": 3000},
                    },
                    "required": ["doc_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "outline",
                "description": "获取所有文档的大纲（标题+首句），快速扫描定位。比 list_docs 信息更密。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "description": "返回文档数，默认 30", "default": 30},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "symbol_jump",
                "description": "跨文档查找某实体（人名/地名/概念）的所有出现位置，用于多跳推理的中间实体追踪。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "要查找的实体名"},
                        "max_hits": {"type": "integer", "description": "最大命中数，默认 10", "default": 10},
                    },
                    "required": ["entity"],
                },
            },
        },
    ]


def dispatch_tool(corpus: DocumentCorpus, name: str, args: dict) -> str:
    """执行工具调用，返回字符串结果。"""
    dispatch = {
        "list_docs": lambda: corpus.list_docs(**{k: v for k, v in args.items() if k in ("limit", "offset")}),
        "grep": lambda: corpus.grep(**{k: v for k, v in args.items() if k in ("pattern", "max_hits")}),
        "read": lambda: corpus.read(**{k: v for k, v in args.items() if k in ("doc_id", "max_chars")}),
        "outline": lambda: corpus.outline(**{k: v for k, v in args.items() if k in ("limit",)}),
        "symbol_jump": lambda: corpus.symbol_jump(**{k: v for k, v in args.items() if k in ("entity", "max_hits")}),
    }
    if name not in dispatch:
        return f"错误：未知工具 {name}"
    try:
        return dispatch[name]()
    except Exception as e:
        return f"工具 {name} 执行错误: {str(e)[:200]}"
