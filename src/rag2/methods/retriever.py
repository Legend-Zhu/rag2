"""
RAG² 检索器 — bge-m3 + FAISS + bge-reranker

职责：
  - 对文档集编码建索引（FAISS），落盘缓存复用
  - 查询时 top-k 召回 + 重排
  - 被 TraditionalRAG baseline 复用，也被 E2 对照实验复用

性能（M4 实测）：
  - 编码 1000 chunk: 30.7s (31ms/chunk, MPS 加速)
  - FAISS 建索引: <0.01s
  - 单次检索: ~334ms（含 query 编码）
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 强制 HF 镜像（实测直连 timeout）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def _show_progress() -> bool:
    """是否显示 tqdm 进度条。重定向到文件时关闭（避免 \\r 污染日志）。"""
    if os.environ.get("RAG2_NO_PROGRESS", ""):
        return False
    # stdout/stderr 不是 tty（重定向到文件）时关闭
    if not (sys.stderr.isatty() and sys.stdout.isatty()):
        # 同时设全局禁用开关，拦 sentence_transformers 内部的 batch 进度条
        os.environ["TQDM_DISABLE"] = "1"
        return False
    return True


class Retriever:
    """bge-m3 embedding + FAISS + bge-reranker。"""

    def __init__(
        self,
        embed_model: str = "BAAI/bge-m3",
        rerank_model: str = "BAAI/bge-reranker-v2-m3",
        device: str = "mps",
        cache_dir: str = "cache/indices",
    ):
        self.embed_model_name = embed_model
        self.rerank_model_name = rerank_model
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 懒加载（避免 import 时即下载模型）
        self._embedder = None
        self._reranker = None

        # 当前索引状态
        self._docs: list[dict] = []          # [{"title","text"}]
        self._index = None                    # FAISS 索引
        self._corpus_key: str = ""            # 当前语料的哈希标识

    # ── 模型懒加载 ────────────────────────────────────────

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            t0 = time.time()
            self._embedder = SentenceTransformer(self.embed_model_name, device=self.device)
            logger.info("加载 embedder %s (%.1fs)", self.embed_model_name, time.time() - t0)
        return self._embedder

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            t0 = time.time()
            self._reranker = CrossEncoder(self.rerank_model_name, device=self.device)
            logger.info("加载 reranker %s (%.1fs)", self.rerank_model_name, time.time() - t0)
        return self._reranker

    # ── 语料标识 ──────────────────────────────────────────

    @staticmethod
    def _corpus_hash(docs: list[dict]) -> str:
        """语料内容哈希，用于索引缓存 key。"""
        payload = json.dumps(
            [(d.get("title", ""), d.get("text", "")) for d in docs],
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def _index_path(self, corpus_key: str) -> Path:
        return self.cache_dir / f"idx_{corpus_key}.npz"

    # ── 建索引 ────────────────────────────────────────────

    def build_index(self, docs: list[dict], force_rebuild: bool = False) -> str:
        """
        对文档集建 FAISS 索引，落盘缓存。

        Args:
            docs: [{"title": str, "text": str}, ...]
            force_rebuild: 忽略缓存重建
        Returns:
            corpus_key（用于后续查询绑定）
        """
        import faiss

        corpus_key = self._corpus_hash(docs)
        cache_path = self._index_path(corpus_key)

        # 命中缓存：直接加载
        if not force_rebuild and cache_path.exists():
            logger.info("命中索引缓存: %s", cache_path.name)
            data = np.load(cache_path, allow_pickle=False)
            self._docs = docs
            dim = int(data["dim"].item() if data["dim"].ndim > 0 else data["dim"])
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(data["embeddings"])
            self._corpus_key = corpus_key
            return corpus_key

        # 编码
        texts = [f"{d.get('title','')}: {d.get('text','')}" for d in docs]
        logger.info("编码 %d 文档建索引...", len(texts))
        t0 = time.time()
        if _show_progress():
            emb = self.embedder.encode(
                texts, batch_size=32, show_progress_bar=True,
                convert_to_numpy=True, normalize_embeddings=True,
            )
        else:
            # 静默编码：临时吞 stderr，避开 sentence_transformers 的 Batches 进度条
            emb = self._encode_silenced(texts)
        elapsed = time.time() - t0
        logger.info("编码完成 %.1fs (%.0f ms/doc)", elapsed,
                    elapsed / max(len(texts), 1) * 1000)

        # 建索引（内积 + L2 归一化 = 余弦）
        dim = emb.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(emb.astype("float32"))
        self._docs = docs
        self._corpus_key = corpus_key

        # 落盘
        np.savez(cache_path, embeddings=emb.astype("float32"), dim=np.array([dim]))
        logger.info("索引落盘: %s", cache_path.name)

        return corpus_key

    # ── 查询 ──────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k_recall: int = 10,
        top_k_rerank: int = 5,
        rerank: bool = True,
    ) -> list[dict]:
        """
        查询：召回 top_k_recall → (可选)重排 → 返回 top_k_rerank。

        Returns:
            [{"title","text","score","rank"}, ...]  按相关性降序
        """
        if self._index is None:
            raise RuntimeError("未建索引，先调 build_index()")

        # 1. 召回
        if _show_progress():
            q_emb = self.embedder.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True,
            ).astype("float32")
        else:
            q_emb = self._encode_silenced([query]).astype("float32")
        distances, indices = self._index.search(q_emb, top_k_recall)

        candidates = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                continue
            doc = self._docs[idx]
            candidates.append({
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "score": float(dist),
                "recall_rank": rank,
            })

        if not rerank or len(candidates) <= top_k_rerank:
            return candidates[:top_k_rerank]

        # 2. 重排（reranker.predict 内部 sentence_transformers 打 Batches 进度条到 stderr，
        #    重定向到文件时强制吞掉，避免 \r 污染日志）
        pairs = [[query, f"{c['title']}: {c['text']}"] for c in candidates]
        if _show_progress():
            rerank_scores = self.reranker.predict(pairs)
        else:
            rerank_scores = self._predict_silenced(pairs)
        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        for i, c in enumerate(candidates):
            c["rank"] = i
        return candidates[:top_k_rerank]

    def _predict_silenced(self, pairs):
        """reranker.predict 时临时把 stderr 重定向到 devnull，吞掉 Batches 进度条。

        用 os.dup2 在 fd 层操作（不用 with，因为 os.open 返回 int 不是 context manager）。
        """
        import os as _os
        fd = sys.stderr.fileno()
        saved_fd = _os.dup(fd)
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        try:
            _os.dup2(devnull_fd, fd)
            scores = self.reranker.predict(pairs)
        finally:
            _os.dup2(saved_fd, fd)
            _os.close(devnull_fd)
            _os.close(saved_fd)
        return scores

    def _encode_silenced(self, texts):
        """embedder.encode 时同时吞掉 stdout 和 stderr 的进度条。"""
        import os as _os
        out_fd, err_fd = sys.stdout.fileno(), sys.stderr.fileno()
        saved_out, saved_err = _os.dup(out_fd), _os.dup(err_fd)
        devnull_fd = _os.open(_os.devnull, _os.O_WRONLY)
        try:
            _os.dup2(devnull_fd, out_fd)
            _os.dup2(devnull_fd, err_fd)
            emb = self.embedder.encode(
                texts, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True,
            )
        finally:
            _os.dup2(saved_out, out_fd)
            _os.dup2(saved_err, err_fd)
            _os.close(devnull_fd)
            _os.close(saved_out)
            _os.close(saved_err)
        return emb
