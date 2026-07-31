"""
RAG² 端到端入库管线。

目录 → 解析 → 分块 → 建索引 → 注册 IndexManager → 记录 Metrics
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from rag2.ingest.models import ParsedDoc, Chunk
from rag2.ingest.parser import DocumentParser
from rag2.ingest.chunker import RecursiveChunker

logger = logging.getLogger(__name__)


class IngestPipeline:
    """端到端文档入库管线。

    用法：
        pipeline = IngestPipeline()
        version = pipeline.ingest_dir(Path("./documents"), "my_corpus")
    """

    def __init__(
        self,
        parser: DocumentParser | None = None,
        chunker: RecursiveChunker | None = None,
        retriever=None,
        index_manager=None,
        metrics=None,
    ):
        self.parser = parser or DocumentParser()
        self.chunker = chunker or RecursiveChunker()
        self.retriever = retriever
        self.index_manager = index_manager
        self.metrics = metrics

    def ingest_dir(
        self,
        dir_path: Path,
        corpus_id: str,
        build_index: bool = True,
    ) -> dict:
        """目录 → 解析 → 分块 → 建索引 → 注册。

        Returns:
            {"corpus_id", "version_id", "n_files", "n_chunks", "parse_time_s",
             "chunk_time_s", "index_time_s", "total_time_s"}
        """
        dir_path = Path(dir_path)
        t_total = time.time()

        # Phase 1: 解析
        t0 = time.time()
        docs = self.parser.parse_dir(dir_path)
        parse_time = time.time() - t0
        logger.info("Phase 1 解析: %d 文档, %.1fs", len(docs), parse_time)

        if not docs:
            return {"corpus_id": corpus_id, "error": "no documents parsed", "n_files": 0}

        # Phase 2: 分块
        t0 = time.time()
        all_chunks = self.chunker.chunk_many(docs)
        chunk_time = time.time() - t0
        logger.info("Phase 2 分块: %d chunks, %.1fs", len(all_chunks), chunk_time)

        # Phase 3: 建索引
        index_time = 0.0
        if build_index and self.retriever:
            t0 = time.time()
            chunk_docs = [{"title": c.title_for_index, "text": c.text} for c in all_chunks]
            self.retriever.build_index(chunk_docs)
            index_time = time.time() - t0
            logger.info("Phase 3 索引: %.1fs", index_time)

        # Phase 4: 注册 IndexManager
        version_id = ""
        if self.index_manager:
            corpus_dict = {c.chunk_id: {"title": c.title_for_index, "text": c.text} for c in all_chunks}
            version_id = self.index_manager.register(
                corpus_id=corpus_id,
                corpus=corpus_dict,
                embed_dim=1024,
                extra={
                    "n_files": len(docs),
                    "n_chunks": len(all_chunks),
                    "chunk_config": {
                        "chunk_size": self.chunker.chunk_size,
                        "chunk_overlap": self.chunker.chunk_overlap,
                    },
                    "source_files": [{"filename": d.filename, "doc_id": d.doc_id,
                                       "mime_type": d.mime_type, "n_chars": len(d.text)}
                                      for d in docs],
                    "chunk_map": {c.chunk_id: {"parent_doc_id": c.parent_doc_id,
                                                "heading": c.heading, "page": c.page,
                                                "char_start": c.char_start, "char_end": c.char_end}
                                   for c in all_chunks},
                },
            )
            logger.info("Phase 4 注册: version=%s", version_id)

        # Phase 5: 记录 Metrics
        total_time = time.time() - t_total
        if self.metrics:
            self.metrics.record_ingestion(
                corpus_id=corpus_id, n_files=len(docs), n_chunks=len(all_chunks),
                parse_time_s=parse_time, chunk_time_s=chunk_time,
                index_time_s=index_time, total_time_s=total_time,
            )

        result = {
            "corpus_id": corpus_id,
            "version_id": version_id,
            "n_files": len(docs),
            "n_chunks": len(all_chunks),
            "parse_time_s": round(parse_time, 2),
            "chunk_time_s": round(chunk_time, 2),
            "index_time_s": round(index_time, 2),
            "total_time_s": round(total_time, 2),
        }
        logger.info("入库完成 %s: %s", corpus_id, {k: v for k, v in result.items() if k != "corpus_id"})
        return result

    def ingest_file(self, file_path: Path, corpus_id: str) -> dict:
        """单文件增量入库。"""
        file_path = Path(file_path)
        doc = self.parser.parse(file_path)
        chunks = self.chunker.chunk(doc)

        if self.retriever:
            chunk_docs = [{"title": c.title_for_index, "text": c.text} for c in chunks]
            self.retriever.build_index(chunk_docs)  # 注意：这里重建，生产环境应增量

        return {
            "corpus_id": corpus_id,
            "filename": file_path.name,
            "doc_id": doc.doc_id,
            "n_chunks": len(chunks),
        }

    def _record_ingestion(self, **kwargs):
        """记录入库指标到 MetricsCollector。"""
        if not self.metrics:
            return
        try:
            import sqlite3
            from datetime import datetime
            ts = datetime.utcnow().isoformat()
            db = self.metrics.db_path
            with sqlite3.connect(str(db)) as conn:
                conn.execute(
                    "INSERT INTO ingestion_events (timestamp, corpus_id, n_files, n_chunks, "
                    "parse_time_s, chunk_time_s, index_time_s, total_time_s) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (ts, kwargs.get("corpus_id", ""), kwargs.get("n_files", 0),
                     kwargs.get("n_chunks", 0), kwargs.get("parse_time", 0),
                     kwargs.get("chunk_time", 0), kwargs.get("index_time", 0),
                     kwargs.get("total_time", 0)),
                )
                conn.commit()
        except Exception as e:
            logger.warning("记录入库指标失败: %s", e)
