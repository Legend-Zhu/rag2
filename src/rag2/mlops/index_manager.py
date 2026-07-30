"""
RAG² MLOps: 索引版本化管理。

管理多语料多版本索引的元数据：embedding NPZ、grep 倒排索引、HyDE 缓存。
支持版本注册、激活、切换、增量更新。

元数据存储在 cache/index_manifest.json。
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class IndexManager:
    """索引版本管理器。

    管理 corpus_id -> versions 的映射，每个 version 关联：
      - embedding NPZ 文件
      - grep 倒排索引
      - HyDE 缓存
      - 元数据（文档数、维度、创建时间、corpus hash）
    """

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "index_manifest.json"
        self.manifest: dict = self._load_manifest()

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text())
        return {}

    def _save_manifest(self):
        self.manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _corpus_hash(corpus: dict[str, dict]) -> str:
        """计算语料哈希（与 Retriever._corpus_hash 一致）。"""
        items = sorted(
            (cid, d.get("title", ""), d.get("text", "")[:500])
            for cid, d in corpus.items()
        )
        raw = json.dumps(items, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def register(
        self,
        corpus_id: str,
        corpus: dict[str, dict],
        index_file: str = "",
        grep_index: str = "",
        hyde_cache: str = "",
        embed_dim: int = 0,
        extra: dict | None = None,
    ) -> str:
        """注册一个新索引版本。

        Args:
            corpus_id: 语料标识（如 "scifact", "arxiv_2026"）
            corpus: {cid: {"title":..., "text":...}}
            index_file: embedding NPZ 文件路径
            grep_index: grep 倒排索引文件路径
            hyde_cache: HyDE 改写缓存文件路径
            embed_dim: embedding 维度
            extra: 额外元数据

        Returns:
            version_id（如 "v1_20260729_153000"）
        """
        chash = self._corpus_hash(corpus)
        ts = datetime.utcnow()
        version_id = f"v1_{ts.strftime('%Y%m%d_%H%M%S')}"

        # 检查是否已有相同 hash 的版本
        if corpus_id in self.manifest:
            for vid, vinfo in self.manifest[corpus_id].get("versions", {}).items():
                if vinfo.get("corpus_hash") == chash:
                    logger.info("语料 %s 已有相同版本 %s，跳过注册", corpus_id, vid)
                    return vid

        version_info = {
            "corpus_hash": chash,
            "created": ts.isoformat(),
            "n_docs": len(corpus),
            "embed_dim": embed_dim,
            "index_file": index_file,
            "grep_index": grep_index,
            "hyde_cache": hyde_cache,
            "active": False,
            **(extra or {}),
        }

        if corpus_id not in self.manifest:
            self.manifest[corpus_id] = {"versions": {}, "active_version": None}

        self.manifest[corpus_id]["versions"][version_id] = version_info

        # 如果是第一个版本，自动激活
        if self.manifest[corpus_id]["active_version"] is None:
            self.manifest[corpus_id]["active_version"] = version_id
            version_info["active"] = True

        self._save_manifest()
        logger.info("注册语料 %s 版本 %s (%d 文档)", corpus_id, version_id, len(corpus))
        return version_id

    def get_active(self, corpus_id: str) -> dict | None:
        """获取当前活跃版本的元数据。"""
        info = self.manifest.get(corpus_id)
        if not info:
            return None
        vid = info.get("active_version")
        if not vid:
            return None
        return info["versions"].get(vid)

    def activate(self, corpus_id: str, version_id: str) -> bool:
        """切换活跃版本。"""
        info = self.manifest.get(corpus_id)
        if not info or version_id not in info.get("versions", {}):
            return False
        # 取消旧活跃
        old = info.get("active_version")
        if old and old in info["versions"]:
            info["versions"][old]["active"] = False
        # 激活新版本
        info["active_version"] = version_id
        info["versions"][version_id]["active"] = True
        self._save_manifest()
        logger.info("切换语料 %s 活跃版本 -> %s", corpus_id, version_id)
        return True

    def list_versions(self, corpus_id: str | None = None) -> dict:
        """列出所有语料的版本（或指定语料）。"""
        if corpus_id:
            return self.manifest.get(corpus_id, {})
        return self.manifest

    def list_corpora(self) -> list[str]:
        """列出所有已注册的语料。"""
        return list(self.manifest.keys())

    def add_documents(
        self, corpus_id: str, new_docs: dict[str, dict]
    ) -> str | None:
        """增量更新：记录新增文档（实际重建索引需调用方执行）。

        Returns:
            新版本 ID（如果有变化），否则 None
        """
        active = self.get_active(corpus_id)
        if not active:
            logger.warning("语料 %s 无活跃版本，无法增量更新", corpus_id)
            return None

        old_n = active.get("n_docs", 0)
        new_n = old_n + len(new_docs)
        ts = datetime.utcnow()
        version_id = f"v{len(self.manifest[corpus_id]['versions']) + 1}_{ts.strftime('%Y%m%d_%H%M%S')}"

        version_info = {
            **active,
            "created": ts.isoformat(),
            "n_docs": new_n,
            "active": False,
            "incremental_from": self.manifest[corpus_id]["active_version"],
            "added_docs": len(new_docs),
        }

        self.manifest[corpus_id]["versions"][version_id] = version_info
        self._save_manifest()
        logger.info("增量更新语料 %s: +%d 文档 -> 版本 %s", corpus_id, len(new_docs), version_id)
        return version_id

    def remove_corpus(self, corpus_id: str):
        """删除语料的所有版本记录。"""
        if corpus_id in self.manifest:
            del self.manifest[corpus_id]
            self._save_manifest()
            logger.info("删除语料 %s", corpus_id)
