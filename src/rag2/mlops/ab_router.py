"""
RAG² MLOps: A/B 测试路由。

将查询路由到不同检索策略（vanilla vs fusion），记录结果用于对比。
路由方式：round_robin / random / hash（确定性）。
"""
from __future__ import annotations

import hashlib
import logging
import random
import time
from typing import Callable, Optional

from rag2.mlops.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class ABRouter:
    """A/B 测试路由器。

    用法：
        router = ABRouter({"vanilla": fn1, "fusion": fn2}, mc)
        result = router.search(query)  # 自动路由 + 记录
    """

    def __init__(
        self,
        strategies: dict[str, Callable],
        metrics: MetricsCollector | None = None,
        router_type: str = "round_robin",
        corpus_id: str = "",
    ):
        """
        Args:
            strategies: {name: search_fn} 搜索函数，签名为 fn(query, top_k) -> list[dict]
            metrics: 指标采集器（可选）
            router_type: "round_robin" | "random" | "hash"
            corpus_id: 语料标识（记录到 metrics）
        """
        self.strategies = strategies
        self.metrics = metrics
        self.router_type = router_type
        self.corpus_id = corpus_id
        self._rr_counter = 0
        self._strategy_names = list(strategies.keys())

    def route(self, query: str) -> str:
        """决定用哪个策略处理这个查询。"""
        if len(self._strategy_names) == 1:
            return self._strategy_names[0]

        if self.router_type == "round_robin":
            name = self._strategy_names[self._rr_counter % len(self._strategy_names)]
            self._rr_counter += 1
            return name

        if self.router_type == "hash":
            h = int(hashlib.md5(query.encode()).hexdigest(), 16)
            return self._strategy_names[h % len(self._strategy_names)]

        # random
        return random.choice(self._strategy_names)

    def search(
        self,
        query: str,
        top_k: int = 3,
        strategy: str | None = None,
        gold_cids: set[str] | None = None,
        verdict: str = "",
        correct: bool | None = None,
    ) -> tuple[str, list[dict], float]:
        """执行搜索（自动路由或指定策略），记录指标。

        Args:
            query: 查询文本
            top_k: 返回文档数
            strategy: 指定策略（None = 自动路由）
            gold_cids: gold 文档 ID 集合（用于计算 recall）
            verdict: 验证结果（SUPPORTED/REFUTED/NEI）
            correct: 是否正确

        Returns:
            (strategy_name, results, latency_s)
        """
        strat = strategy or self.route(query)
        fn = self.strategies.get(strat)
        if not fn:
            raise ValueError(f"未知策略: {strat}")

        t0 = time.time()
        results = fn(query, top_k=top_k)
        latency = time.time() - t0

        # 记录检索指标
        if self.metrics:
            top_cids = [r.get("cid", r.get("title", "")) for r in results[:top_k]]
            gold_found = None
            if gold_cids is not None:
                gold_found = bool(gold_cids & set(top_cids))

            self.metrics.record_retrieval(
                query=query, strategy=strat, corpus_id=self.corpus_id,
                n_results=len(results), latency_s=latency,
                top_cids=top_cids, gold_found=gold_found,
            )

            # 记录 A/B 结果
            if verdict or correct is not None:
                self.metrics.record_ab_outcome(
                    strategy=strat, query=query, verdict=verdict,
                    correct=correct, latency_s=latency,
                )

        return strat, results, latency

    def get_results(self) -> list[dict]:
        """获取 A/B 测试汇总结果。"""
        if self.metrics:
            return self.metrics.get_ab_results()
        return []
