"""
RAG² 文档入库数据结构。

ParsedDoc: 解析后的文档（原始文件 → 结构化文本 + metadata）
Chunk: 分块后的检索单元（带 parent_doc 回溯）
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def file_hash(file_path: Path) -> str:
    """文件内容 SHA-256 前 16 位。用于 doc_id 和去重。"""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


@dataclass
class ParsedDoc:
    """解析后的文档。

    由 DocumentParser.parse() 产出。包含全文文本 + 结构化段落 + 元数据。
    """
    doc_id: str                          # 文件内容哈希（去重用）
    source_path: str                     # 原始文件路径
    filename: str                        # 文件名（含扩展名）
    mime_type: str                       # MIME 类型
    title: str                           # 文档标题（从内容或文件名提取）
    text: str                            # 全文（用于分块和搜索）
    sections: list[dict] = field(default_factory=list)
    # sections: [{"heading": str, "text": str, "page": int|None, "level": int}]
    metadata: dict = field(default_factory=dict)
    # metadata: {"author", "created", "modified", "pages", "n_chars", "parser"}

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source_path": self.source_path,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "title": self.title,
            "text": self.text,
            "sections": self.sections,
            "metadata": self.metadata,
        }


@dataclass
class Chunk:
    """分块后的检索单元。

    由 RecursiveChunker.chunk() 产出。每个 chunk 可回溯到源文档。
    """
    chunk_id: str                        # f"{parent_doc_id}_{i:04d}"
    parent_doc_id: str                   # 回溯到 ParsedDoc.doc_id
    text: str                            # chunk 文本
    heading: str = ""                    # 所属标题（从 sections 提取）
    page: Optional[int] = None           # PDF 页码（其他格式 None）
    char_start: int = 0                  # 在原文中的起始位置
    char_end: int = 0                    # 结束位置
    metadata: dict = field(default_factory=dict)
    # metadata: 继承自 ParsedDoc（filename, mime_type, title 等）

    @property
    def title_for_index(self) -> str:
        """用于 embedding 索引的标题（heading + filename 上下文）。"""
        parts = []
        if self.heading:
            parts.append(self.heading)
        fname = self.metadata.get("filename", "")
        if fname and fname != self.heading:
            parts.append(fname)
        return " | ".join(parts) if parts else self.metadata.get("title", "")

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "parent_doc_id": self.parent_doc_id,
            "text": self.text,
            "heading": self.heading,
            "page": self.page,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": self.metadata,
        }
