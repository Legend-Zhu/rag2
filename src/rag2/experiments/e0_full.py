"""
RAG² E0 完整版 —— 双模型 × 四规模档 × n=100

晚上 22:00-08:00 跑（Qwen3.8 享 0.2 折）。
K3 温度强制 1.0，Qwen3.8 温度 0.0。
跑完产出双模型拐点曲线对比。

用法:
  python3 src/rag2/experiments/e0_full.py --model kimi-k3 --n 100
  python3 src/rag2/experiments/e0_full.py --model qwen3.8 --n 100
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rag2.experiments.e0_bottleneck import run_e0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["kimi-k3", "qwen3.8"],
                   help="跑哪个模型")
    p.add_argument("--n", type=int, default=100, help="评测题数")
    p.add_argument("--start-scale", type=int, default=None,
                   help="从哪个规模档开始（断点续跑，如 100000）")
    args = p.parse_args()

    logger.info("=" * 60)
    logger.info("E0 完整版: model=%s n=%d", args.model, args.n)
    logger.info("=" * 60)

    scales = [1_000, 10_000, 100_000, 1_000_000]
    if args.start_scale:
        scales = [s for s in scales if s >= args.start_scale]
        logger.info("断点续跑，从 %d 开始", args.start_scale)

    t0 = time.time()
    results = run_e0(model=args.model, n_questions=args.n, scales=scales)
    elapsed_min = (time.time() - t0) / 60
    logger.info("E0 完成，总耗时 %.1f 分钟", elapsed_min)

    # 打印汇总表
    logger.info("\n" + "=" * 60)
    logger.info("汇总: %s (n=%d)", args.model, args.n)
    logger.info("=" * 60)
    logger.info("%-12s %-16s %6s %6s %8s %10s",
                "scale", "method", "EM", "F1", "correct", "recall@5")
    for scale_key, scale_data in results.items():
        scale = scale_key.replace("scale_", "")
        for method, data in scale_data.items():
            m = data.get("metrics", {})
            if isinstance(m, dict) and "em" in m:
                logger.info("%-12s %-16s %6.2f %6.2f %8.2f %10.2f",
                            scale, method,
                            m["em"]["mean"], m["f1"]["mean"],
                            m["correct"]["mean"],
                            m.get("recall@5", {}).get("mean", 0))


if __name__ == "__main__":
    main()
