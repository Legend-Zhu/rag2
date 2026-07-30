"""
RAG² E0 实验脚本 —— 长上下文 vs 传统 RAG 拐点（H1 立论）

目的：验证 H1——长上下文 + 廉价 token 已使 RAG 的瓶颈从 access 转移到 grounding。
做法：在 10³/10⁴/10⁵/10⁶ 四档检索池规模下，对比 LongContext（全文塞窗口）和
      TraditionalRAG（检索 top-k）的答案准确率。
预期：小池子下两者接近；池子增大后 LongContext 保持/上升，TraditionalRAG 因
      检索噪声略降——拐点位置即 "access 不再是瓶颈" 的实证。

注意：thinking 模式强制开启，单题 completion ~100 tokens。22:00-08:00 享 0.2 折。
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled
from rag2.methods.retriever import Retriever
from rag2.methods.long_context import LongContext
from rag2.methods.traditional_rag import TraditionalRAG
from rag2.eval import aggregate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# E0 检索池构造
# ─────────────────────────────────────────────────────────

def build_pool_at_scale(all_docs: list[dict], gold_docs: list[dict],
                         target_tokens: int, seed: int = 42) -> list[dict]:
    """
    构造指定 token 规模的检索池，保证 gold 文档在内。

    Args:
        all_docs: 全量候选文档（干扰池）
        gold_docs: 必须包含的 gold 支撑文档
        target_tokens: 目标池规模（token 数，按 ~3 字符/token 估）
        seed: 随机种子
    """
    import random
    rng = random.Random(seed)

    # 先放 gold
    pool = list(gold_docs)
    pool_chars = sum(len(d["text"]) for d in pool)
    target_chars = target_tokens * 3

    # 候选干扰文档（排除已在 pool 里的）
    pool_keys = {(d.get("title", ""), d.get("text", "")[:50]) for d in pool}
    candidates = [d for d in all_docs
                  if (d.get("title", ""), d.get("text", "")[:50]) not in pool_keys]
    rng.shuffle(candidates)

    # 填充到目标规模
    for d in candidates:
        if pool_chars >= target_chars:
            break
        pool.append(d)
        pool_chars += len(d["text"])

    actual_tokens = pool_chars // 3
    logger.info("  池规模目标 %d tokens, 实际 %d tokens (%d 文档, gold %d)",
                target_tokens, actual_tokens, len(pool), len(gold_docs))
    return pool


# ─────────────────────────────────────────────────────────
# 主实验
# ─────────────────────────────────────────────────────────

def run_e0(model: str = "qwen3.8", n_questions: int = 20,
           scales: list[int] = None, dry_run: bool = False):
    """
    跑 E0 拐点实验。

    Args:
        model: 用哪个模型（当前只有 qwen3.8 接入了 API）
        n_questions: 评测题数（节省成本，先小规模验证）
        scales: 检索池规模档（token 数）
        dry_run: 只跑最小规模验证连通性
    """
    if scales is None:
        scales = [1_000, 10_000, 100_000] if dry_run else [1_000, 10_000, 100_000, 1_000_000]

    import torch

    gw = ModelGateway()
    samples = load_dataset_subsampled("musique")[:n_questions]
    logger.info("E0: %s, %d 题, 规模档 %s", model, n_questions, scales)

    # 收集所有可用干扰文档（从全量子采样里抽）
    all_docs = []
    for s in load_dataset_subsampled("musique"):
        all_docs.extend(s.get("supporting_docs", []))
    logger.info("干扰池总量: %d 文档", len(all_docs))

    # 关键修复：Retriever 全局单例，复用 embedder/reranker，避免 MPS 显存累积爆内存
    shared_retriever = Retriever()
    # 预热加载（一次性，之后所有 scale 复用同一模型实例）
    logger.info("预热加载 embedder/reranker（单例复用）...")
    _ = shared_retriever.embedder
    _ = shared_retriever.reranker

    lc_method = LongContext(gw, role=model)

    results = {}
    out_dir = Path("results/E0_bottleneck_shift")
    out_dir.mkdir(parents=True, exist_ok=True)

    for scale in scales:
        logger.info("\n===== 规模档: 10^%d tokens =====", len(str(scale)) - 1)
        scale_results = {"long_context": [], "traditional_rag": []}

        for i, s in enumerate(samples):
            gold = [d for d in s["supporting_docs"] if d.get("is_supporting")]
            # 构造该规模的池子
            pool = build_pool_at_scale(all_docs, gold, scale)

            # 方法 1: LongContext（全文塞窗口）
            sample_lc = {**s, "supporting_docs": pool}
            try:
                r_lc = lc_method.run(sample_lc)
                scale_results["long_context"].append({
                    "id": s["id"], "pred": r_lc.answer, "gold": s["answer"],
                    "overflow": r_lc.trace.get("context_overflow", False),
                    "completion_tokens": r_lc.trace.get("completion_tokens", 0),
                })
            except Exception as e:
                logger.error("  LC 题 %s 失败: %s", s["id"], str(e)[:80])

            # 方法 2: TraditionalRAG（复用 shared_retriever，只重建索引）
            try:
                shared_retriever.build_index(pool, force_rebuild=True)
                rag = TraditionalRAG(gw, shared_retriever, role=model)
                r_rag = rag.run(sample_lc)  # 用同一池子，公平对照
                scale_results["traditional_rag"].append({
                    "id": s["id"], "pred": r_rag.answer, "gold": s["answer"],
                    "retrieved_titles": [d.title for d in r_rag.retrieved_docs],
                    "gold_titles": [d["title"] for d in gold],
                    "completion_tokens": r_rag.trace.get("completion_tokens", 0),
                })
            except Exception as e:
                logger.error("  RAG 题 %s 失败: %s", s["id"], str(e)[:80])

            # 清理 MPS 缓存（防累积）
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

            logger.info("  题 %d/%d done (LC: %s)", i + 1, n_questions,
                        (scale_results["long_context"][-1].get("pred", "")[:30]
                         if scale_results["long_context"] else "(空)"))

        # 评测
        for method in list(scale_results.keys()):
            recs = scale_results[method]
            agg = aggregate(recs)
            logger.info("  [%s] %s", method, {k: f"{v['mean']:.2f}" for k, v in agg.items() if isinstance(v, dict) and "mean" in v})
            scale_results[method] = {"per_sample": recs, "metrics": agg}

        results[f"scale_{scale}"] = scale_results

        # 增量落盘
        (out_dir / f"E0_{model}_n{n_questions}.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3.8")
    p.add_argument("--n", type=int, default=20, help="评测题数（控制成本）")
    p.add_argument("--dry-run", action="store_true", help="只跑小规模验证连通")
    args = p.parse_args()

    t0 = time.time()
    results = run_e0(model=args.model, n_questions=args.n, dry_run=args.dry_run)
    logger.info("\n===== E0 完成，总耗时 %.1f 分钟 =====", (time.time() - t0) / 60)
