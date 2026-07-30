"""
RAG² C1×C2 ablation 实验（论文核心表）

4 策略（S1/S2/S3/S4）× 3 后端（A/B/C）= 12 组合（可选取关键组合跑）
共享 C2 索引（建一次复用），每组合跑同样 n 题。

输出：每组合的 EM/F1/recall/步数/cost，落盘 JSON 便于画论文表格。
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled
from rag2.methods.generative_index import GenerativeIndexBuilder
from rag2.methods.retriever import Retriever
from rag2.methods.c2_backends import build_backend
from rag2.methods.c1_strategies import build_strategy
from rag2.eval import aggregate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/c1_c2_ablation")


def run_ablation(
    model: str = "qwen3.8",
    n_questions: int = 20,
    strategies: list[str] = None,
    backends: list[str] = None,
    n_corpus_docs: int = 50,
    max_steps: int = 5,
    dataset: str = "sciq",
):
    """跑 C1×C2 ablation。"""
    if strategies is None:
        strategies = ["S1", "S2", "S3", "S4"]
    if backends is None:
        backends = ["A", "B", "C"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gw = ModelGateway()

    # 1. 准备语料（取若干题的所有文档合并作共享语料）
    samples = load_dataset_subsampled(dataset)[:n_questions]
    all_docs = []
    for s in load_dataset_subsampled(dataset)[:max(n_questions * 2, 10)]:
        all_docs.extend(s.get("supporting_docs", []))
    corpus = all_docs[:n_corpus_docs]
    logger.info("数据集 %s: 共享语料 %d 文档，评测 %d 题", dataset, len(corpus), n_questions)

    # 2. 建 C2 索引（一次性，所有组合复用）
    logger.info("建 C2 生成式索引...")
    indices, graph = GenerativeIndexBuilder(gw, role=model).build(corpus)
    logger.info("C2 索引就绪: %s", graph.stats())

    # 3. 准备 Retriever（后端 B/C 需要 embedder）
    ret = Retriever()
    _ = ret.embedder  # 预热

    # 4. 跑所有 策略×后端 组合
    all_results = {}
    out_file = RESULTS_DIR / f"ablation_{model}_n{n_questions}.json"

    for strat_name in strategies:
        for backend_name in backends:
            combo_key = f"{strat_name}_{backend_name}"
            logger.info("\n===== 组合: %s =====", combo_key)

            try:
                backend = build_backend(backend_name, indices, graph, gw, retriever=ret, role=model)
                strategy = build_strategy(
                    strat_name, gw, backend, role=model, max_steps=max_steps,
                )
            except Exception as e:
                logger.error("组合 %s 初始化失败: %s", combo_key, str(e)[:100])
                continue

            per_sample = []
            for i, s in enumerate(samples):
                # 构造 sample：用共享语料（不是单题文档），公平对照
                test_sample = {**s, "supporting_docs": corpus}
                t0 = time.time()
                try:
                    result = strategy.run(test_sample)
                    rec = {
                        "id": s["id"], "pred": result.answer, "gold": s["answer"],
                        "retrieved_titles": [d.title for d in result.retrieved_docs],
                        "gold_titles": [d["title"] for d in s["supporting_docs"]
                                        if d.get("is_supporting")],
                        "n_steps": result.trace.get("n_steps", 0),
                        "n_collected": result.trace.get("n_collected_docs", 0),
                        "elapsed_s": time.time() - t0,
                    }
                    correct = (result.answer.lower().strip(".\"'") in s["answer"].lower()
                               or s["answer"].lower() in result.answer.lower())
                    logger.info("  题 %d/%d: %s (%s, %d步)",
                                i+1, n_questions, result.answer[:30],
                                "YES" if correct else "NO",
                                result.trace.get("n_steps", 0))
                except Exception as e:
                    logger.error("  题 %d 失败: %s", i+1, str(e)[:80])
                    rec = {"id": s["id"], "pred": "", "gold": s["answer"], "error": str(e)[:100]}
                per_sample.append(rec)

            # 评测
            valid = [r for r in per_sample if "error" not in r]
            metrics = aggregate(valid) if valid else {"error": "all failed"}
            avg_steps = sum(r.get("n_steps", 0) for r in valid) / max(len(valid), 1)
            avg_elapsed = sum(r.get("elapsed_s", 0) for r in valid) / max(len(valid), 1)

            all_results[combo_key] = {
                "strategy": strat_name, "backend": backend_name,
                "metrics": metrics,
                "avg_steps": round(avg_steps, 2),
                "avg_elapsed_s": round(avg_elapsed, 1),
                "n_valid": len(valid),
                "per_sample": per_sample,
            }
            logger.info("[%s] EM=%.2f F1=%.2f recall=%.2f avg_steps=%.1f avg_time=%.1fs",
                        combo_key,
                        metrics.get("em", {}).get("mean", 0),
                        metrics.get("f1", {}).get("mean", 0),
                        metrics.get("recall@5", {}).get("mean", 0),
                        avg_steps, avg_elapsed)

            # 增量落盘
            out_file.write_text(json.dumps(all_results, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    # 5. 汇总表
    logger.info("\n" + "=" * 70)
    logger.info("C1×C2 Ablation 汇总（n=%d）", n_questions)
    logger.info("=" * 70)
    logger.info("%-12s %6s %6s %9s %8s %9s %10s",
                "组合", "EM", "F1", "contains", "correct", "recall@5", "avg_steps")
    for combo_key, r in all_results.items():
        m = r.get("metrics", {})
        if isinstance(m, dict) and "em" in m:
            logger.info("%-12s %6.2f %6.2f %9.2f %8.2f %9.2f %10.1f",
                        combo_key,
                        m["em"]["mean"], m["f1"]["mean"],
                        m.get("contains", {}).get("mean", 0),
                        m["correct"]["mean"],
                        m.get("recall@5", {}).get("mean", 0),
                        r["avg_steps"])

    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="kimi-k3")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--dataset", default="sciq", help="数据集: sciq/musique/hotpotqa")
    p.add_argument("--strategies", nargs="+", default=["S1"])
    p.add_argument("--backends", nargs="+", default=["B"])
    p.add_argument("--corpus-docs", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=4)
    args = p.parse_args()

    t0 = time.time()
    run_ablation(
        model=args.model, n_questions=args.n,
        strategies=args.strategies, backends=args.backends,
        n_corpus_docs=args.corpus_docs, max_steps=args.max_steps,
        dataset=args.dataset,
    )
    logger.info("\n总耗时 %.1f 分钟", (time.time() - t0) / 60)
