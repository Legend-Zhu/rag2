"""
RAG² C2: 生成式语义索引构建器（完整版 v2）

按架构决策.md 第六节规格：每文档生成
  - 假设性问题集（HyDE 倒置）
  - 实体-关系三元组（subject-relation-object，不止实体名）
  - 实体类型分类（person/location/org/date/concept）
  - 多粒度摘要（sentence / paragraph）
  - 相关性自述

跨文档共现图（CorpusGraph）：从所有文档的实体抽取结果聚合而成，
本地计算不调 LLM。支撑 C1 的多跳推理路径。

成本控制（API-only 必备）：
  - 批量化（一个 prompt 处理 batch_size 个文档）
  - 索引落盘缓存（一次性建，反复用）
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────

VALID_ENTITY_TYPES = {"person", "location", "organization", "date", "concept", "other"}


@dataclass
class GenerativeIndex:
    """单个文档的生成式索引条目（完整版）。"""
    doc_id: int
    title: str
    text: str
    hypothetical_questions: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)        # [{"entity": str, "type": str}]
    relations: list[dict] = field(default_factory=list)       # [{"subject": str, "relation": str, "object": str}]
    summary_short: str = ""        # sentence-level
    summary_long: str = ""         # paragraph-level
    relevance_self_desc: str = ""  # 何种查询下应被召回

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id, "title": self.title, "text": self.text,
            "hypothetical_questions": self.hypothetical_questions,
            "entities": self.entities, "relations": self.relations,
            "summary_short": self.summary_short, "summary_long": self.summary_long,
            "relevance_self_desc": self.relevance_self_desc,
        }

    @classmethod
    def from_dict(cls, d: dict, full_text: str = "") -> "GenerativeIndex":
        """从缓存 dict 重建；full_text 用原始完整 text（缓存里的可能截断）。"""
        return cls(
            doc_id=d.get("doc_id", 0),
            title=d.get("title", ""),
            text=full_text or d.get("text", ""),
            hypothetical_questions=d.get("hypothetical_questions", []),
            entities=d.get("entities", []),
            relations=d.get("relations", []),
            summary_short=d.get("summary_short", ""),
            summary_long=d.get("summary_long", ""),
            relevance_self_desc=d.get("relevance_self_desc", ""),
        )


@dataclass
class CorpusGraph:
    """
    跨文档实体共现图（本地计算，不调 LLM）。

    两种视图：
      - entity → docs：某实体出现在哪些文档（多跳跳转用）
      - doc → entities：某文档含哪些实体（实体路径检索用）
    另存关系三元组的全局索引：subject → relations
    """
    # entity（规范化小写）→ set(doc_id)
    entity_to_docs: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))
    # entity → type
    entity_to_type: dict[str, str] = field(default_factory=dict)
    # 全局关系三元组：[{subject, relation, object, doc_id}, ...]
    all_relations: list[dict] = field(default_factory=list)
    # subject（规范化）→ list of {relation, object, doc_id}
    subject_index: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))

    def add_document(self, doc_id: int, index: GenerativeIndex) -> None:
        """把一个文档的实体和关系加入图。"""
        for ent in index.entities:
            name = ent.get("entity", "").strip().lower()
            if not name:
                continue
            self.entity_to_docs[name].add(doc_id)
            etype = ent.get("type", "other")
            self.entity_to_type[name] = etype if etype in VALID_ENTITY_TYPES else "other"
        for rel in index.relations:
            entry = {
                "subject": rel.get("subject", "").strip().lower(),
                "relation": rel.get("relation", ""),
                "object": rel.get("object", "").strip().lower(),
                "doc_id": doc_id,
            }
            self.all_relations.append(entry)
            if entry["subject"]:
                self.subject_index[entry["subject"]].append(entry)

    def docs_for_entity(self, entity: str) -> list[int]:
        """查某实体出现在哪些文档（多跳跳转）。"""
        return sorted(self.entity_to_docs.get(entity.strip().lower(), set()))

    def relations_for_subject(self, subject: str) -> list[dict]:
        """查某主体的所有关系（多跳路径推理）。"""
        return self.subject_index.get(subject.strip().lower(), [])

    def multi_hop_path(self, start_entity: str, max_hops: int = 3) -> list[list[int]]:
        """
        从起始实体出发，找多跳文档路径（BFS）。
        返回 [[doc_id_chain], ...]，每条链是一串 doc_id。
        """
        start = start_entity.strip().lower()
        if start not in self.entity_to_docs:
            return []
        paths = []
        # BFS: (current_entity, visited_docs, hop)
        queue = [(start, [], 0)]
        seen_entities = {start}
        while queue:
            ent, doc_chain, hop = queue.pop(0)
            if hop >= max_hops:
                continue
            docs = self.entity_to_docs.get(ent, set())
            for d in docs:
                new_chain = doc_chain + [d]
                if len(new_chain) >= 2:
                    paths.append(new_chain)
                # 通过该文档的关系扩展到新实体
                for rel in self.subject_index.get(ent, []):
                    obj = rel.get("object", "")
                    if obj and obj not in seen_entities:
                        seen_entities.add(obj)
                        queue.append((obj, new_chain, hop + 1))
        return paths[:20]  # 限制返回数量

    def stats(self) -> dict:
        return {
            "n_entities": len(self.entity_to_docs),
            "n_relations": len(self.all_relations),
            "n_docs_with_entities": len(set().union(*self.entity_to_docs.values())) if self.entity_to_docs else 0,
        }


# ─────────────────────────────────────────────────────────
# 工具 schema（function calling，绕过文本解析）
# ─────────────────────────────────────────────────────────

INDEX_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "save_index_entries",
            "description": "Save the generated semantic index entries for the documents. Call this ONCE with ALL document entries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entries": {
                        "type": "array",
                        "description": "One entry per document, in the same order as the input documents.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "hypothetical_questions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "3-5 specific questions this document could answer",
                                },
                                "entities": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "entity": {"type": "string", "description": "exact name as in text"},
                                            "type": {"type": "string", "enum": ["person", "location", "organization", "date", "concept", "other"]},
                                        },
                                        "required": ["entity", "type"],
                                    },
                                    "description": "Named entities in the document",
                                },
                                "relations": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "subject": {"type": "string"},
                                            "relation": {"type": "string", "description": "snake_case verb phrase (e.g. born_in, spouse_of)"},
                                            "object": {"type": "string"},
                                        },
                                        "required": ["subject", "relation", "object"],
                                    },
                                    "description": "Explicit factual relations (subject-relation-object). Extract ONLY stated facts, no inference.",
                                },
                                "summary_short": {"type": "string", "description": "one concise sentence"},
                                "relevance_self_desc": {"type": "string", "description": "in one phrase, what queries should retrieve this doc"},
                            },
                            "required": ["hypothetical_questions", "entities", "relations", "summary_short", "relevance_self_desc"],
                        },
                    },
                },
                "required": ["entries"],
            },
        },
    }
]


INDEX_PROMPT_TMPL = """You are building a semantic index for a high-precision document retrieval system.

For EACH document below, generate structured annotations: hypothetical questions it can answer, named entities (with types), factual relations (subject-relation-object, snake_case relations), a one-sentence summary, and a relevance self-description.

After analyzing, call the save_index_entries tool with one entry per document (in the same order).

Guidelines for relations:
- Extract ONLY explicit factual relations stated in the text (do not infer).
- Use snake_case for relation names (e.g. born_in, spouse_of, member_of, located_in).
- Subject and object should be entity names that appear in the entities list.

Documents to index:
{documents}"""


# ─────────────────────────────────────────────────────────
# 构建器
# ─────────────────────────────────────────────────────────

class GenerativeIndexBuilder:
    """生成式索引构建器：批量给文档生成富语义标注 + 聚合共现图。"""

    def __init__(
        self,
        gateway,
        role: str = "generator",
        batch_size: int = 5,
        max_text_chars: int = 1500,
        cache_dir: str = "cache/generative_indices",
    ):
        self.gw = gateway
        self.role = role
        self.batch_size = batch_size
        self.max_text_chars = max_text_chars
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── 批量构建 ──────────────────────────────────────────

    def build(self, docs: list[dict], force_rebuild: bool = False) -> tuple[list[GenerativeIndex], CorpusGraph]:
        """
        给一批文档构建生成式索引 + 跨文档共现图。

        Args:
            docs: [{"title": str, "text": str, ...}]
            force_rebuild: 忽略缓存
        Returns:
            (indices, graph)：indices 与 docs 顺序对应；graph 是聚合后的共现图
        """
        cache_path = self._cache_path(docs)
        if not force_rebuild and cache_path.exists():
            logger.info("命中生成式索引缓存: %s", cache_path.name)
            indices = self._load_cache(cache_path, docs)
        else:
            indices = self._build_indices(docs, cache_path)

        # 聚合共现图（本地计算，不调 LLM）
        graph = CorpusGraph()
        for idx in indices:
            graph.add_document(idx.doc_id, idx)

        logger.info("生成式索引 + 共现图就绪: %d 文档, %s", len(indices), graph.stats())
        return indices, graph

    def _build_indices(self, docs: list[dict], cache_path: Path) -> list[GenerativeIndex]:
        """分 batch 调 LLM 构建索引。"""
        indices: list[GenerativeIndex] = []
        n_batches = (len(docs) + self.batch_size - 1) // self.batch_size
        t0 = time.time()

        for batch_idx in range(n_batches):
            start = batch_idx * self.batch_size
            batch = docs[start:start + self.batch_size]
            batch_indices = self._process_batch(batch, start)
            indices.extend(batch_indices)
            elapsed = time.time() - t0
            done = start + len(batch)
            logger.info("生成式索引进度: %d/%d 文档 (%.1fs, %.1fs/doc)",
                        done, len(docs), elapsed, elapsed / max(done, 1))

        self._save_cache(cache_path, indices)
        logger.info("生成式索引完成: %d 文档, 总耗时 %.1fs", len(indices), time.time() - t0)
        return indices

    def _process_batch(self, batch: list[dict], start_id: int) -> list[GenerativeIndex]:
        """处理一个 batch，返回索引条目。用 function calling 绕过文本解析。"""
        doc_strs = []
        for i, d in enumerate(batch):
            text = d.get("text", "")[:self.max_text_chars]
            doc_strs.append(f"[Doc {i}] Title: {d.get('title','')}\nText: {text}")
        documents_text = "\n\n".join(doc_strs)

        prompt = INDEX_PROMPT_TMPL.format(documents=documents_text)
        messages = [{"role": "user", "content": prompt}]

        try:
            # 用工具调用：模型把结构化结果作为 save_index_entries 的参数传入
            # 避免"自然语言→正则解析"的脆弱环节（思考过程可能含 JSON 片段干扰）
            resp = self.gw.generate_complete(
                self.role, messages, role_tag="c2_indexer",
                tools=INDEX_TOOL_SCHEMA,
                max_tokens=8192, max_continuations=4,
            )
            parsed = self._extract_tool_entries(resp, len(batch))
            if not any(parsed):
                logger.warning(
                    "batch %d 无工具调用结果，finish=%s, tool_calls=%d",
                    start_id, resp.finish_reason, len(resp.tool_calls),
                )
        except Exception as e:
            logger.error("batch %d 索引生成失败: %s，降级为空标注", start_id, str(e)[:100])
            parsed = [{} for _ in batch]

        results = []
        for i, d in enumerate(batch):
            ann = parsed[i] if i < len(parsed) else {}
            results.append(GenerativeIndex(
                doc_id=start_id + i,
                title=d.get("title", ""),
                text=d.get("text", ""),
                hypothetical_questions=self._clean_questions(ann.get("hypothetical_questions", [])),
                entities=self._clean_entities(ann.get("entities", [])),
                relations=self._clean_relations(ann.get("relations", [])),
                summary_short=str(ann.get("summary_short", ""))[:300],
                summary_long=str(ann.get("summary_long", ""))[:1000],
                relevance_self_desc=str(ann.get("relevance_self_desc", ""))[:200],
            ))
        return results

    @staticmethod
    def _extract_tool_entries(resp, expected_n: int) -> list[dict]:
        """从工具调用响应中提取 entries（不再做文本解析）。"""
        for tc in resp.tool_calls:
            if tc.get("function", {}).get("name") == "save_index_entries":
                try:
                    args = json.loads(tc["function"]["arguments"])
                    entries = args.get("entries", [])
                    if isinstance(entries, list):
                        while len(entries) < expected_n:
                            entries.append({})
                        return entries[:expected_n]
                except (json.JSONDecodeError, KeyError):
                    pass
        return [{} for _ in range(expected_n)]

        results = []
        for i, d in enumerate(batch):
            ann = parsed[i] if i < len(parsed) else {}
            results.append(GenerativeIndex(
                doc_id=start_id + i,
                title=d.get("title", ""),
                text=d.get("text", ""),
                hypothetical_questions=self._clean_questions(ann.get("hypothetical_questions", [])),
                entities=self._clean_entities(ann.get("entities", [])),
                relations=self._clean_relations(ann.get("relations", [])),
                summary_short=str(ann.get("summary_short", ""))[:300],
                summary_long=str(ann.get("summary_long", ""))[:1000],
                relevance_self_desc=str(ann.get("relevance_self_desc", ""))[:200],
            ))
        return results

    # ── 清洗（LLM 输出规范化）──────────────────────────────

    @staticmethod
    def _clean_questions(qs: list) -> list[str]:
        """清洗假设性问题：去空、去重、限长。"""
        seen = set()
        out = []
        for q in qs:
            q = str(q).strip()
            if q and q.lower() not in seen and len(q) < 300:
                seen.add(q.lower())
                out.append(q)
        return out[:10]

    @staticmethod
    def _clean_entities(ents: list) -> list[dict]:
        """清洗实体：规范化 type。"""
        out = []
        seen = set()
        for e in ents:
            if not isinstance(e, dict):
                continue
            name = str(e.get("entity", "")).strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())
            etype = str(e.get("type", "other")).strip().lower()
            if etype not in VALID_ENTITY_TYPES:
                etype = "other"
            out.append({"entity": name, "type": etype})
        return out[:30]  # 单文档实体上限

    @staticmethod
    def _clean_relations(rels: list) -> list[dict]:
        """清洗关系三元组。"""
        out = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            s = str(r.get("subject", "")).strip()
            rel = str(r.get("relation", "")).strip()
            o = str(r.get("object", "")).strip()
            if s and rel and o:
                out.append({"subject": s, "relation": rel, "object": o})
        return out[:20]  # 单文档关系上限

    # ── 响应解析（容错）────────────────────────────────────

    @staticmethod
    def _parse_response(text: str, expected_n: int) -> list[dict]:
        """解析 LLM 返回的 JSON 数组，容错处理截断。"""
        import re

        # 先尝试标准解析
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                arr = json.loads(match.group(0))
                if isinstance(arr, list):
                    while len(arr) < expected_n:
                        arr.append({})
                    return arr[:expected_n]
            except json.JSONDecodeError:
                pass

        # 截断容错：JSON 没闭合（thinking 模式下 max_tokens 不够）
        # 找最后一个完整对象（最后一个 "}"}），补 "]" 闭合
        start_match = re.search(r"\[", text)
        if start_match:
            fragment = text[start_match.start():]
            # 找最后一个完整的 } 作为对象边界
            last_obj_end = fragment.rfind("}")
            if last_obj_end > 0:
                candidate = fragment[:last_obj_end + 1] + "]"
                try:
                    arr = json.loads(candidate)
                    if isinstance(arr, list):
                        logger.info("JSON 截断容错：恢复 %d 个对象（原期望 %d）",
                                    len(arr), expected_n)
                        while len(arr) < expected_n:
                            arr.append({})
                        return arr[:expected_n]
                except json.JSONDecodeError:
                    pass
        return [{} for _ in range(expected_n)]

    # ── 缓存 ─────────────────────────────────────────────

    def _cache_path(self, docs: list[dict]) -> Path:
        import hashlib
        payload = json.dumps(
            [(d.get("title", ""), d.get("text", "")[:500]) for d in docs],
            ensure_ascii=True,
        )
        h = hashlib.sha256(payload.encode()).hexdigest()[:16]
        model = self.gw.resolve(self.role).model_name.replace("/", "_").replace(".", "_")
        return self.cache_dir / f"genidx_{model}_{h}.json"

    @staticmethod
    def _save_cache(path: Path, indices: list[GenerativeIndex]) -> None:
        data = [idx.to_dict() for idx in indices]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _load_cache(path: Path, docs: list[dict]) -> list[GenerativeIndex]:
        data = json.loads(path.read_text(encoding="utf-8"))
        indices = []
        for i, (d, ann) in enumerate(zip(docs, data)):
            indices.append(GenerativeIndex.from_dict(ann, full_text=d.get("text", "")))
        return indices
