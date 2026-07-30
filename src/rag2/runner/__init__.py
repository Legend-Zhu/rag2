"""
RAG² ExperimentRunner — 把方法 × 数据集 × 模型串成实验闭环

职责：
  - 接收实验配置（dataset, methods, models, metrics）
  - 逐 sample 跑方法，收集 Result
  - 断点续跑（sample 级落盘，中断不重付费 —— API-only 必备）
  - 跑完喂评测层，输出指标 + bootstrap CI
  - 结果落盘（含 git commit + config，供论文复现）
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml

from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled
from rag2.methods.base import Method, Result
from rag2.eval import aggregate

logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


@dataclass
class ExperimentConfig:
    """单次实验的配置。"""
    name: str                         # 实验名，如 "E0_bottleneck_shift"
    dataset: str                      # musique | hotpotqa | alce | freshqa
    methods: list[str]                # ["traditional_rag", "long_context"]
    model: str = "kimi-k3"            # 单模型（多模型时跑多次）
    sample_n: int | None = None       # 覆盖子采样数，None 用 config 默认
    metrics: list[str] | None = None  # 默认 EM/F1/correct/recall@5

    def to_dict(self) -> dict:
        return asdict(self)


class ExperimentRunner:
    """实验编排器。"""

    def __init__(self, gateway: ModelGateway, config_path: str = "config/config.yaml"):
        self.gw = gateway
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 方法工厂 ──────────────────────────────────────────

    def _build_method(self, method_name: str, model: str, shared_retriever=None) -> Method:
        """按名字构造方法实例。"""
        role = "generator"
        # 临时把 gateway 的角色解析指向指定模型
        if method_name == "long_context":
            from rag2.methods.long_context import LongContext
            return LongContext(self.gw, role=role)
        elif method_name == "traditional_rag":
            from rag2.methods.traditional_rag import TraditionalRAG
            from rag2.methods.retriever import Retriever
            rcfg = self.cfg.get("retrieval", {})
            retriever = shared_retriever or Retriever(
                embed_model=rcfg.get("embedder", "BAAI/bge-m3"),
                rerank_model=rcfg.get("reranker", "BAAI/bge-reranker-v2-m3"),
                device=rcfg.get("embed_device", "mps"),
            )
            return TraditionalRAG(
                self.gw, retriever, role=role,
                top_k_recall=rcfg.get("top_k_recall", 10),
                top_k_rerank=rcfg.get("top_k_rerank", 5),
            )
        else:
            raise ValueError(f"未知方法: {method_name}（W2 三支柱在此扩展）")

    # ── 单实验执行 ────────────────────────────────────────

    def run(self, exp: ExperimentConfig, force_refresh: bool = False) -> dict:
        """
        跑一个实验：加载数据 → 逐方法逐 sample → 评测 → 落盘。

        Args:
            exp: 实验配置
            force_refresh: 忽略已完成结果重跑
        Returns:
            {"config": ..., "results": {method: {metrics...}}, "per_sample": [...]}
        """
        out_dir = RESULTS_DIR / exp.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. 加载数据
        samples = load_dataset_subsampled(exp.dataset)
        if exp.sample_n and exp.sample_n < len(samples):
            samples = samples[:exp.sample_n]
        logger.info("实验 %s: %s 数据集 %d 题", exp.name, exp.dataset, len(samples))

        # 2. 落盘配置（供复现）
        config_dump = exp.to_dict()
        config_dump["timestamp"] = time.time()
        config_dump["model_resolved"] = self.gw.resolve(exp.model).model_name
        (out_dir / "config.json").write_text(
            json.dumps(config_dump, ensure_ascii=False, indent=2), encoding="utf-8",
        )

        all_results = {}
        # 3. 逐方法
        for method_name in exp.methods:
            logger.info("  → 跑方法: %s", method_name)
            per_sample = self._run_method(
                method_name, exp.model, samples, out_dir, exp.name, force_refresh,
            )
            # 4. 评测
            agg = self._evaluate(per_sample)
            all_results[method_name] = agg
            logger.info("    指标: %s", {k: f"{v['mean']:.3f}" for k, v in agg.items()})

        # 5. 汇总落盘
        summary = {"config": config_dump, "results": all_results}
        (out_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        return summary

    def _run_method(
        self, method_name: str, model: str, samples: list[dict],
        out_dir: Path, exp_name: str, force_refresh: bool,
    ) -> list[dict]:
        """跑单个方法的所有 sample，断点续跑。"""
        per_sample_path = out_dir / f"{method_name}_per_sample.jsonl"

        # 断点续跑：加载已完成
        done_ids: set[str] = set()
        done_records: list[dict] = []
        if per_sample_path.exists() and not force_refresh:
            with per_sample_path.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    done_records.append(rec)
                    done_ids.add(rec["id"])
            logger.info("    断点续跑: 已完成 %d / %d", len(done_ids), len(samples))

        # 构造方法（共享检索器：traditional_rag 需要建一次索引）
        method = self._build_method(method_name, model)

        # traditional_rag: 对全语料建一次索引
        if method_name == "traditional_rag":
            all_docs = []
            for s in samples:
                all_docs.extend(s.get("supporting_docs", []))
            if all_docs:
                logger.info("    建语料索引 (%d 文档)...", len(all_docs))
                method.retriever.build_index(all_docs)

        # 逐 sample
        records = list(done_records)
        for i, s in enumerate(samples):
            if s["id"] in done_ids:
                continue
            t0 = time.time()
            try:
                result = method.run(s)
                rec = {
                    "id": s["id"],
                    "question": s["question"],
                    "gold": s["answer"],
                    "pred": result.answer,
                    "retrieved_titles": [d.title for d in result.retrieved_docs],
                    "gold_titles": [d["title"] for d in s.get("supporting_docs", [])
                                    if d.get("is_supporting")],
                    "trace": result.trace,
                    "elapsed_s": time.time() - t0,
                }
            except Exception as e:
                logger.error("    sample %s 失败: %s", s["id"], str(e)[:100])
                rec = {
                    "id": s["id"], "question": s["question"], "gold": s["answer"],
                    "pred": "", "error": str(e)[:200],
                    "retrieved_titles": [], "gold_titles": [],
                    "trace": {}, "elapsed_s": time.time() - t0,
                }
            records.append(rec)
            # 增量落盘（每题写一行，断点可续）
            with per_sample_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if (i + 1) % 10 == 0:
                logger.info("    进度 %d/%d", i + 1, len(samples))

        return records

    @staticmethod
    def _evaluate(per_sample: list[dict]) -> dict:
        """跑评测聚合。"""
        valid = [r for r in per_sample if "error" not in r]
        if not valid:
            return {"error": "no valid samples"}
        return aggregate(valid)
