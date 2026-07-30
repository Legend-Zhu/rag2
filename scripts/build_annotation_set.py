#!/usr/bin/env python3
"""
生成人工标注文件：合并 claim / 改写 claim / 摘要 / 各条件 verdict，供人工填 human_label。

输出：
  data/annotation/claims_to_annotate.csv  -- 逐条标注表（human_label/confidence/notes 待填）
  data/annotation/all_claims_verdicts.csv  -- 全量 verdict 参考（n=100 当前 run）

标注集 = 全部 n 条 claim（n=100 时即全部；扩到 n≥300 后从中固定 seed 抽样）。
is_disputed=True 的为优先标注（oracle≠SUPPORTED 的争议样本）。

用法：
  python scripts/build_annotation_set.py [results/ab_scale100.json] [data/annotation]
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = "SUPPORTED"

HEADERS = [
    "idx", "pid", "to_annotate", "is_disputed",
    "original_claim", "reformulated_claim", "title", "abstract",
    "verdict_A", "verdict_B1", "verdict_B2", "verdict_B3", "verdict_C",
    "gold_recall_B1", "gold_recall_B2", "gold_recall_B3",
    "human_label", "confidence", "notes",
]


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "ab_scale100.json"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "data" / "annotation"
    out_dir.mkdir(parents=True, exist_ok=True)

    claims = json.loads((ROOT / "data" / "arxiv_2026_claims.json").read_text(encoding="utf-8"))
    corpus = json.loads((ROOT / "data" / "arxiv_2026_corpus.json").read_text(encoding="utf-8"))
    reform_path = ROOT / "cache" / "arxiv_reformulated_claims.json"
    reform = json.loads(reform_path.read_text(encoding="utf-8")) if reform_path.exists() else {}
    data = json.loads(results_path.read_text(encoding="utf-8"))
    res = data["results"]
    gold_ret = data.get("gold_retrieved", {})

    n = len(next(iter(res.values())))
    test_pids = list(claims.keys())[:n]

    rows = []
    n_disputed = 0
    for i, pid in enumerate(test_pids):
        c_verdict = res["C"][i]
        is_disputed = c_verdict != GOLD
        if is_disputed:
            n_disputed += 1
        gr = {k: gold_ret.get(k, [None] * n)[i] for k in ("B1", "B2", "B3")}
        rows.append({
            "idx": i,
            "pid": pid,
            "to_annotate": "True",  # n=100 时全标；扩样后改为抽样
            "is_disputed": str(is_disputed),
            "original_claim": claims.get(pid, ""),
            "reformulated_claim": reform.get(pid, claims.get(pid, "")),
            "title": corpus.get(pid, {}).get("title", ""),
            "abstract": corpus.get(pid, {}).get("text", ""),
            "verdict_A": res["A"][i],
            "verdict_B1": res["B1"][i],
            "verdict_B2": res["B2"][i],
            "verdict_B3": res["B3"][i],
            "verdict_C": c_verdict,
            "gold_recall_B1": gr["B1"],
            "gold_recall_B2": gr["B2"],
            "gold_recall_B3": gr["B3"],
            "human_label": "",
            "confidence": "",
            "notes": "",
        })

    # 标注文件（含空 human_label 等待填）
    ann_csv = out_dir / "claims_to_annotate.csv"
    with ann_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        # 争议样本置顶
        rows_sorted = sorted(rows, key=lambda r: (r["is_disputed"] != "True", r["idx"]))
        w.writerows(rows_sorted)

    # 全量 verdict 参考
    all_csv = out_dir / "all_claims_verdicts.csv"
    with all_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADERS)
        w.writeheader()
        w.writerows(rows)

    print(f"# {results_path}  (n={n}, disputed={n_disputed})")
    print(f"-> wrote {ann_csv}  ({len(rows)} claims, disputed first)")
    print(f"-> wrote {all_csv}")
    print("\nDisputed claims (oracle != SUPPORTED):")
    for r in rows:
        if r["is_disputed"] == "True":
            print(f"  idx={r['idx']:3d} pid={r['pid']}  C={r['verdict_C']:18s} "
                  f"B1={r['verdict_B1']:18s} B2={r['verdict_B2']:18s} B3={r['verdict_B3']}")


if __name__ == "__main__":
    main()
