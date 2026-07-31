"""
RAG² 递归字符分块。

策略：按分隔符层级递归切分，保留 heading 作为 chunk 上下文。
  层级：## 标题 → ### 子标题 → 段落 → 句子 → 字符

每个 chunk 可回溯到 parent_doc_id + 在原文中的位置。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from rag2.ingest.models import ParsedDoc, Chunk

logger = logging.getLogger(__name__)

# 默认分隔符（按优先级：标题 > 段落 > 句子）
DEFAULT_SEPARATORS = [
    "\n## ",    # markdown 二级标题
    "\n### ",   # markdown 三级标题
    "\n#### ",  # markdown 四级标题
    "\n\n",     # 段落
    "\n",       # 行
    ". ",       # 句子
    " ",        # 词
]


class RecursiveChunker:
    """递归字符分块器。

    用法：
        chunker = RecursiveChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.chunk(parsed_doc)
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        """将 ParsedDoc 分块为 Chunk 列表。"""
        text = doc.text
        if not text.strip():
            return []

        # 先按 sections 建立标题映射（char offset → heading）
        heading_map = self._build_heading_map(doc)

        # 递归切分
        splits = self._split_text(text, self.separators)

        # 合并 splits 为 chunks（控制在 chunk_size 附近）
        raw_chunks = self._merge_splits(splits, text)

        # 构建 Chunk 对象
        chunks = []
        base_meta = {
            "filename": doc.filename,
            "mime_type": doc.mime_type,
            "title": doc.title,
            "source_path": doc.source_path,
        }

        for i, (chunk_text, char_start, char_end) in enumerate(raw_chunks):
            # 从 heading_map 找这个 chunk 属于哪个标题
            heading = self._find_heading(heading_map, char_start)
            page = self._find_page(doc.sections, char_start, text)

            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_{i:04d}",
                parent_doc_id=doc.doc_id,
                text=chunk_text.strip(),
                heading=heading,
                page=page,
                char_start=char_start,
                char_end=char_end,
                metadata={**base_meta},
            ))

        logger.info("分块 %s: %d → %d chunks (size=%d, overlap=%d)",
                     doc.filename, len(text), len(chunks), self.chunk_size, self.chunk_overlap)
        return chunks

    def chunk_many(self, docs: list[ParsedDoc]) -> list[Chunk]:
        """批量分块多个文档。"""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk(doc))
        return all_chunks

    # ── 内部方法 ──────────────────────────────────────

    def _split_text(self, text: str, separators: list[str]) -> list[tuple[str, int, int]]:
        """递归按分隔符切分。返回 [(text, char_start, char_end), ...]。"""
        if len(text) <= self.chunk_size:
            return [(text, 0, len(text))]

        # 尝试每个分隔符
        for i, sep in enumerate(separators):
            if sep not in text:
                continue

            parts = []
            pos = 0
            while pos < len(text):
                # 找下一个分隔符位置
                idx = text.find(sep, pos + 1)
                if idx == -1:
                    idx = len(text)

                part = text[pos:idx]
                if part:
                    parts.append((part, pos, idx))

                pos = idx

            # 如果切分后每段都 <= chunk_size，直接返回
            if all(len(p[0]) <= self.chunk_size * 1.5 for p in parts):
                return parts

            # 否则递归切分超长的段
            refined = []
            next_seps = separators[i + 1:] if i + 1 < len(separators) else [" "]
            for part_text, start, end in parts:
                if len(part_text) <= self.chunk_size * 1.5:
                    refined.append((part_text, start, end))
                else:
                    refined.extend(self._split_text(part_text, next_seps))
            return refined

        # 所有分隔符都没命中，强制按长度切
        return self._force_split(text)

    def _force_split(self, text: str) -> list[tuple[str, int, int]]:
        """强制按 chunk_size 切分（最后手段）。"""
        parts = []
        for i in range(0, len(text), self.chunk_size):
            part = text[i:i + self.chunk_size]
            parts.append((part, i, i + len(part)))
        return parts

    def _merge_splits(self, splits: list[tuple[str, int, int]],
                      full_text: str) -> list[tuple[str, int, int]]:
        """合并相邻的小段，使每个 chunk 接近 chunk_size。"""
        if not splits:
            return []

        chunks = []
        current_text = ""
        current_start = splits[0][1] if splits else 0

        for part_text, start, end in splits:
            # 如果加上这段不超过 chunk_size，合并
            if len(current_text) + len(part_text) <= self.chunk_size:
                current_text += part_text
            else:
                # 保存当前 chunk
                if current_text.strip():
                    chunks.append((current_text, current_start, current_start + len(current_text)))

                # 开始新 chunk（带 overlap）
                if self.chunk_overlap > 0 and len(current_text) > self.chunk_overlap:
                    overlap = current_text[-self.chunk_overlap:]
                    current_text = overlap + part_text
                    current_start = start - self.chunk_overlap
                else:
                    current_text = part_text
                    current_start = start

        # 最后一个 chunk
        if current_text.strip():
            chunks.append((current_text, current_start, current_start + len(current_text)))

        return chunks

    def _build_heading_map(self, doc: ParsedDoc) -> list[tuple[int, str]]:
        """构建 char_offset → heading 映射。

        返回 [(char_offset, heading_str), ...] 按 offset 排序。
        """
        heading_map = []
        text = doc.text

        for section in doc.sections:
            heading = section.get("heading", "")
            if not heading:
                continue
            # 在全文中找 heading 的位置
            pos = text.find(heading)
            if pos >= 0:
                heading_map.append((pos, heading))

        heading_map.sort(key=lambda x: x[0])
        return heading_map

    @staticmethod
    def _find_heading(heading_map: list[tuple[int, str]], char_pos: int) -> str:
        """找到 char_pos 所属的最近标题。"""
        result = ""
        for offset, heading in heading_map:
            if offset <= char_pos:
                result = heading
            else:
                break
        return result

    @staticmethod
    def _find_page(sections: list[dict], char_pos: int, full_text: str) -> Optional[int]:
        """尝试找 char_pos 对应的页码。"""
        # 从 sections 里找有 page 信息的
        for section in sections:
            page = section.get("page")
            if page is None:
                continue
            heading = section.get("heading", "")
            if heading:
                pos = full_text.find(heading)
                if pos >= 0 and pos <= char_pos:
                    return page
        return None
