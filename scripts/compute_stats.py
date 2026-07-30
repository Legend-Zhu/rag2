#!/usr/bin/env python3
"""
从 A/B/C 实验结果 JSON 生成论文用统计 CSV（取代手填）。

输出：
  paper/data/accuracy.csv  —— 各条件 accuracy + 95%CI + gold recall
  paper/data/paired.csv    —— 配对比较 rescued/hurt + McNemar 精确 p

用法：
  python scripts/compute_stats.py [results/ab_scale100.json] [paper/data]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 让脚本能 import src/rag2
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rag2.eval.metrics import bootstrap_ci_supported, paired_comparison  # noqa: E402

CONDITIONS = [
    ("A", "No Retrieval"),
    ("B1", "Vanilla RAG"),
    ("B2", "RAG^2 Fusion"),
    ("B3", "Long-context"),
    ("C", "Single-doc Oracle"),
]
# 配对比较：("表格标签", new, base) —— "X vs Y" 对应 new=X, base=Y
PAIRINGS = [
    ("B2 vs A", "B2", "A"),
    ("B2 vs B1", "B2", "B1"),
    ("B3 vs B2", "B3", "B2"),
    ("C vs B2", "C", "B2"),
]
GOLD = "SUPPORTED"


def pct(x: float) -> int:
    """比例(0-1) -> 整数百分比。"""
    return int(round(x * 100))


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "ab_scale100.json"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "paper" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(results_path.read_text(encoding="utf-8"))
    res = data["results"]
    gold_ret = data.get("gold_retrieved", {})
    n = data.get("n", len(next(iter(res.values()))))

    print(f"# {results_path}  (n={n})")

    # ── accuracy.csv ──
    acc_rows = []
    print("\n== Accuracy (vs SUPPORTED) ==")
    for cond, label in CONDITIONS:
        verdicts = res[cond]
        point, lo, hi = bootstrap_ci_supported(verdicts, gold=GOLD, n_resample=10000, seed=42)
        # gold recall：检索条件才有
        gr = ""
        if cond in gold_ret:
            gr_list = gold_ret[cond]
            gr = pct(sum(1 for g in gr_list if g) / len(gr_list)) if gr_list else ""
        acc_rows.append((cond, label, pct(point), pct(lo), pct(hi), gr))
        gr_str = f", gold_recall={gr}%" if gr != "" else ""
        print(f"  {cond:3s} {label:18s} acc={pct(point):3d}%  CI=[{pct(lo)},{pct(hi)}]{gr_str}")

    acc_csv = out_dir / "accuracy.csv"
    with acc_csv.open("w", encoding="utf-8") as f:
        f.write("cond,label,acc,ci_lo,ci_hi,gold_recall\n")
        for cond, label, acc, lo, hi, gr in acc_rows:
            f.write(f"{cond},{label},{acc},{lo},{hi},{gr}\n")
    print(f"\n-> wrote {acc_csv}")

    # ── paired.csv ──
    pair_rows = []
    print("\n== Paired comparison ==")
    for label, new, base in PAIRINGS:
        pc = paired_comparison(res[new], res[base], gold=GOLD)
        pair_rows.append((label, pc["rescued"], pc["hurt"], pc["net"], pc["p_two_sided"]))
        print(
            f"  {label:10s} rescued={pc['rescued']:3d}  hurt={pc['hurt']:3d}  "
            f"net={pc['net']:+d}  McNemar p={pc['p_two_sided']:.4f}  "
            f"(discordant={pc['discordant']})"
        )

    pair_csv = out_dir / "paired.csv"
    with pair_csv.open("w", encoding="utf-8") as f:
        f.write("comparison,rescued,hurt,net,mcnemar_p_two_sided\n")
        for label, rescued, hurt, net, p in pair_rows:
            f.write(f"{label},{rescued},{hurt},{net},{p:.6f}\n")
    print(f"\n-> wrote {pair_csv}")


if __name__ == "__main__":
    main()
