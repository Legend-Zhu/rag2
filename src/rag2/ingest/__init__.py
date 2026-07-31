"""RAG² 文档入库：解析 + 分块 + 索引注册。"""
from rag2.ingest.models import ParsedDoc, Chunk
from rag2.ingest.parser import DocumentParser
from rag2.ingest.chunker import RecursiveChunker
from rag2.ingest.pipeline import IngestPipeline

__all__ = ["ParsedDoc", "Chunk", "DocumentParser", "RecursiveChunker", "IngestPipeline"]
