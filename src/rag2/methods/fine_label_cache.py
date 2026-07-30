"""
RAG² 精标注缓存（渐进式索引基础设施）

设计：
  - 精标注按 corpus_id 独立存储，跨查询持久化
  - 查询时：粗筛 → 查缓存 → miss 的才标注 → 写入缓存
  - 越用越快越完整（渐进式构建）

存储结构:
  cache/fine_labels/{corpus_id}.json  每个文档一个文件
    {
      "corpus_id": "...",
      "title": "...",
      "hypothetical_questions": [...],
      "entities": [...],
      "relations": [...],
      "summary_short": "...",
      "relevance_self_desc": "...",
      "model": "qwen3.8 / k3",
      "timestamp": ...
    }

也支持批量读取/写入，避免逐文件 IO。
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class FineLabelCache:
    """精标注持久化缓存。按 corpus_id 存取。"""

    def __init__(self, cache_dir: str = "cache/fine_labels"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._mem_cache: dict[str, dict] = {}  # 进程内缓存，避免重复读盘
        self._loaded = False

    def _path(self, corpus_id: str) -> Path:
        # corpus_id 可能含特殊字符，转安全文件名
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(corpus_id))
        return self.cache_dir / f"{safe}.json"

    def get(self, corpus_id: str) -> Optional[dict]:
        """读取单个文档的精标注。命中返回 dict，miss 返回 None。"""
        cid = str(corpus_id)
        if cid in self._mem_cache:
            return self._mem_cache[cid]
        p = self._path(cid)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            self._mem_cache[cid] = data
            return data
        return None

    def get_many(self, corpus_ids: list[str]) -> dict[str, Optional[dict]]:
        """批量读取。返回 {corpus_id: label_or_None}。"""
        return {cid: self.get(cid) for cid in corpus_ids}

    def put(self, corpus_id: str, label: dict) -> None:
        """写入单个文档的精标注（持久化）。"""
        cid = str(corpus_id)
        label = {**label, "corpus_id": cid, "timestamp": time.time()}
        self._mem_cache[cid] = label
        self._path(cid).write_text(
            json.dumps(label, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def put_many(self, labels: dict[str, dict]) -> None:
        """批量写入。labels: {corpus_id: label}。"""
        for cid, label in labels.items():
            self.put(cid, label)

    def get_or_compute(
        self,
        corpus_ids: list[str],
        compute_fn,
    ) -> dict[str, dict]:
        """
        批量获取，miss 的用 compute_fn 计算并缓存。

        Args:
            corpus_ids: 要获取精标注的文档 ID 列表
            compute_fn: Callable[list[corpus_id], dict[corpus_id, label]]
                        只对 miss 的调用，返回它们的精标注
        Returns:
            {corpus_id: label} 全部结果（含缓存命中 + 新计算）
        """
        cached = {}
        missing = []
        for cid in corpus_ids:
            label = self.get(cid)
            if label is not None:
                cached[cid] = label
            else:
                missing.append(cid)

        if missing:
            logger.info("精标注缓存: %d 命中, %d miss（需计算）",
                        len(cached), len(missing))
            computed = compute_fn(missing)
            self.put_many(computed)
            cached.update(computed)
        else:
            logger.info("精标注缓存: %d 全部命中", len(cached))

        return cached

    def stats(self) -> dict:
        """缓存统计。"""
        n_files = len(list(self.cache_dir.glob("*.json")))
        return {"cached_docs": n_files, "mem_cached": len(self._mem_cache)}
