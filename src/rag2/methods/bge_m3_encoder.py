"""
bge-m3 编码器：输出 dense (1024) + sparse (词项→权重)。

bge-m3 原生支持三种输出：
  - dense: 1024 维稠密向量（语义相似度）
  - sparse: 词项 ID → 权重（词面匹配，替代 grep 倒排索引）
  - colbert: token 级向量（多向量交互，可选）

本编码器只用 dense + sparse，用于 Qdrant 混合检索。
"""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


class BGEM3Encoder:
    """bge-m3 dense + sparse 编码器。

    用法：
        encoder = BGEM3Encoder()
        output = encoder.encode(["hello world", "foo bar"])
        # output["dense"]: np.array (N, 1024)
        # output["sparse"]: list[dict[int, float]]  词项ID→权重
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "mps",
        use_fp16: bool = True,
        batch_size: int = 32,
    ):
        from FlagEmbedding import BGEM3FlagModel

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        t0 = time.time()
        self._model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            device=device,
        )
        logger.info("加载 bge-m3 编码器 (%.1fs)", time.time() - t0)

    def encode(self, texts: list[str], batch_size: int | None = None) -> dict:
        """编码文本，返回 dense + sparse。

        Args:
            texts: 文本列表
            batch_size: 批量大小（默认用初始化值）

        Returns:
            {
                "dense": np.ndarray (N, 1024),       # L2 归一化
                "sparse": list[dict[int, float]],     # 每个文本的稀疏表示
            }
        """
        bs = batch_size or self.batch_size

        output = self._model.encode(
            texts,
            batch_size=bs,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense = np.array(output["dense_vecs"], dtype="float32")
        # L2 归一化（Qdrant 用 DotProduct = 归一化后的余弦）
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        dense = dense / norms

        sparse = output["lexical_weights"]
        # 确保是 list[dict]
        if isinstance(sparse, dict):
            sparse = [sparse]

        # 转换 key 为 int
        sparse_list = []
        for s in sparse:
            sparse_list.append({int(k): float(v) for k, v in s.items()})

        return {"dense": dense, "sparse": sparse_list}

    def encode_single(self, text: str) -> dict:
        """编码单条文本。"""
        result = self.encode([text])
        return {
            "dense": result["dense"][0],
            "sparse": result["sparse"][0],
        }
