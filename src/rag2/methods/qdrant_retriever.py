"""
Qdrant 混合检索器：dense HNSW + sparse + RRF 融合 + CrossEncoder 重排。

替代 FAISS Retriever + FusionRetriever（grep 层）。
- dense: bge-m3 1024 维 → Qdrant HNSW（亚线性搜索）
- sparse: bge-m3 词项权重 → Qdrant sparse inverted index
- 融合: Qdrant Query API RRF
- 重排: bge-reranker-v2-m3 CrossEncoder（外部，top-30 → top-k）

支持百万级文档、实时 upsert、payload 过滤。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class QdrantRetriever:
    """Qdrant 混合检索器。

    用法：
        retriever = QdrantRetriever(collection_name="my_corpus")
        retriever.build_index(chunks)          # 建集合 + upsert
        results = retriever.search("query")    # 混合搜索 + 重排
    """

    DENSE_DIM = 1024  # bge-m3 输出维度

    def __init__(
        self,
        collection_name: str = "rag2",
        host: str = "localhost",
        port: int = 6333,
        embedder=None,             # BGEM3Encoder 实例（外部注入或懒加载）
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        device: str = "mps",
        use_embedded: bool = False,  # True = 无需 Docker，嵌入式本地存储
        embedded_path: str = "qdrant_data",
    ):
        self.collection_name = collection_name
        self.cross_encoder_model = cross_encoder_model
        self.device = device

        # 连接 Qdrant
        from qdrant_client import QdrantClient
        if use_embedded:
            self.client = QdrantClient(path=embedded_path)
        else:
            self.client = QdrantClient(host=host, port=port)

        # 编码器（懒加载）
        self._embedder = embedder
        self._cross_encoder = None

    @property
    def embedder(self):
        if self._embedder is None:
            from rag2.methods.bge_m3_encoder import BGEM3Encoder
            self._embedder = BGEM3Encoder(device=self.device)
        return self._embedder

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            os.environ.setdefault("TQDM_DISABLE", "1")
            t0 = time.time()
            self._cross_encoder = CrossEncoder(self.cross_encoder_model, device=self.device)
            logger.info("加载 CrossEncoder (%.1fs)", time.time() - t0)
        return self._cross_encoder

    # ── 集合管理 ────────────────────────────────────────

    def build_index(self, chunks: list, force_rebuild: bool = False) -> str:
        """建集合 + 批量 upsert chunks。

        Args:
            chunks: list[Chunk]（rag2.ingest.models.Chunk）
            force_rebuild: 删除已有集合重建

        Returns:
            collection_name
        """
        from qdrant_client.models import (
            Distance, VectorParams, SparseVectorParams, SparseIndexParams,
            PointStruct, NamedSparseVector,
        )

        # 删除已有集合
        if force_rebuild:
            self.client.delete_collection(self.collection_name)

        # 创建集合（dense HNSW + sparse inverted index）
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.DENSE_DIM,
                        distance=Distance.DOT,  # 内积（配合 L2 归一化 = 余弦）
                    ),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(),
                    ),
                },
            )
            logger.info("创建 Qdrant 集合: %s", self.collection_name)

        # 批量 upsert
        self.upsert_chunks(chunks)
        logger.info("索引完成: %s (%d chunks)", self.collection_name, len(chunks))
        return self.collection_name

    def upsert_chunks(self, chunks: list) -> int:
        """增量 upsert chunks（实时，无需全量重建）。

        Returns:
            upsert 的 chunk 数
        """
        from qdrant_client.models import PointStruct, SparseVector

        if not chunks:
            return 0

        # 编码所有 chunk 文本
        texts = [c.text for c in chunks]
        encoded = self.embedder.encode(texts)

        # 构建 points
        points = []
        for i, chunk in enumerate(chunks):
            # 确保是 int ID（Qdrant 需要）
            point_id = self._hash_to_int(chunk.chunk_id)

            sparse_vec = encoded["sparse"][i]
            points.append(PointStruct(
                id=point_id,
                vector={
                    "dense": encoded["dense"][i].tolist(),
                    "sparse": SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values()),
                    ),
                },
                payload={
                    "chunk_id": chunk.chunk_id,
                    "parent_doc_id": chunk.parent_doc_id,
                    "heading": chunk.heading,
                    "page": chunk.page,
                    "text": chunk.text,
                    "title": chunk.title_for_index,
                    "filename": chunk.metadata.get("filename", ""),
                    "mime_type": chunk.metadata.get("mime_type", ""),
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                },
            ))

        # 分批 upsert
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i:i + batch_size],
            )

        logger.info("Upsert %d chunks to %s", len(points), self.collection_name)
        return len(points)

    def delete_chunks(self, doc_ids: list[str]) -> int:
        """删除指定文档的所有 chunk。

        Args:
            doc_ids: parent_doc_id 列表
        Returns:
            删除的点数（估计值）
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        deleted = 0
        for doc_id in doc_ids:
            count_before = self.client.count(
                self.collection_name,
                count_filter=Filter(
                    must=[FieldCondition(key="parent_doc_id", match=MatchValue(value=doc_id))]
                ),
            ).count
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[FieldCondition(key="parent_doc_id", match=MatchValue(value=doc_id))]
                ),
            )
            deleted += count_before
        logger.info("删除 %d 点 (doc_ids: %s)", deleted, doc_ids)
        return deleted

    # ── 搜索 ────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_rerank: bool = True,
        use_sparse: bool = True,
    ) -> list[dict]:
        """混合搜索：dense + sparse → RRF 融合 → CrossEncoder 重排。

        Args:
            query: 查询文本
            top_k: 返回结果数
            use_rerank: 是否用 CrossEncoder 重排
            use_sparse: 是否启用 sparse 混合（False = 纯 dense）

        Returns:
            [{"chunk_id", "parent_doc_id", "heading", "page", "text", "title",
              "score", "source"}, ...]
        """
        # 编码查询
        encoded = self.embedder.encode_single(query)
        query_dense = encoded["dense"].tolist()
        query_sparse = encoded["sparse"]

        from qdrant_client.models import (
            SparseVector,
        )

        if use_sparse and query_sparse:
            # 混合搜索：dense prefetch + sparse prefetch → RRF
            from qdrant_client.models import Prefetch, FusionQuery, Fusion

            dense_prefetch = Prefetch(
                query=query_dense,
                using="dense",
                limit=top_k * 3,
            )
            sparse_prefetch = Prefetch(
                query=SparseVector(
                    indices=list(query_sparse.keys()),
                    values=list(query_sparse.values()),
                ),
                using="sparse",
                limit=top_k * 3,
            )

            results = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[dense_prefetch, sparse_prefetch],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=top_k * 3 if use_rerank else top_k,
                with_payload=True,
            ).points
        else:
            # 纯 dense 搜索
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_dense,
                using="dense",
                limit=top_k * 3 if use_rerank else top_k,
                with_payload=True,
            ).points

        # 转换结果
        candidates = []
        for r in results:
            p = r.payload or {}
            candidates.append({
                "chunk_id": p.get("chunk_id", str(r.id)),
                "parent_doc_id": p.get("parent_doc_id", ""),
                "heading": p.get("heading", ""),
                "page": p.get("page"),
                "text": p.get("text", ""),
                "title": p.get("title", ""),
                "score": float(r.score),
                "source": "dense+sparse" if use_sparse else "dense",
            })

        if not use_rerank or len(candidates) <= top_k:
            return candidates[:top_k]

        # CrossEncoder 重排
        pairs = [[query, f"{c['title']}: {c['text']}"] for c in candidates]
        rerank_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]

    def search_dense(self, query: str, top_k: int = 10) -> list[dict]:
        """纯 dense 搜索（baseline）。"""
        return self.search(query, top_k=top_k, use_rerank=False, use_sparse=False)

    def search_sparse(self, query: str, top_k: int = 10) -> list[dict]:
        """纯 sparse 搜索（baseline）。"""
        encoded = self.embedder.encode_single(query)
        query_sparse = encoded["sparse"]
        if not query_sparse:
            return []

        from qdrant_client.models import SparseVector

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=SparseVector(
                indices=list(query_sparse.keys()),
                values=list(query_sparse.values()),
            ),
            using="sparse",
            limit=top_k,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id": (r.payload or {}).get("chunk_id", str(r.id)),
                "parent_doc_id": (r.payload or {}).get("parent_doc_id", ""),
                "text": (r.payload or {}).get("text", ""),
                "title": (r.payload or {}).get("title", ""),
                "score": float(r.score),
                "source": "sparse",
            }
            for r in results
        ]

    def read_doc(self, doc_id: str) -> dict | None:
        """按 chunk_id 读取单个 chunk。"""
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="chunk_id", match=MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=True,
        )
        if results[0]:
            p = results[0][0].payload
            return {
                "doc_id": p.get("chunk_id", ""),
                "parent_doc_id": p.get("parent_doc_id", ""),
                "heading": p.get("heading", ""),
                "page": p.get("page"),
                "text": p.get("text", ""),
                "title": p.get("title", ""),
            }
        return None

    def count(self) -> int:
        """返回集合中的点数。"""
        try:
            return self.client.count(self.collection_name).count
        except Exception:
            return 0

    # ── 工具方法 ──────────────────────────────────────

    @staticmethod
    def _hash_to_int(s: str) -> int:
        """字符串哈希 → 正整数（Qdrant point ID 要求）。"""
        import hashlib
        h = hashlib.md5(s.encode()).hexdigest()
        return int(h[:15], 16)  # 60 位正整数，Qdrant 支持 uint64
