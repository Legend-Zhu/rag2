#!/usr/bin/env python3
"""
P0-2 标注整合分析：人工 ground truth vs LLM-judge。

输入：
  data/annotation/claims_to_annotate.csv  (human_label 已填)
  results/ab_scale100.json 或 ab_scale466.json (各条件 verdict)

输出（打印 + 可选写 results/annotation_analysis.json）：
  - 人工 label 分布
  - LLM-judge(C) vs 人工 一致率 + Cohen's κ
  - 人工判定的"坏 claim"（human != SUPPORTED）
  - 敏感性：坏 claim 剔除前后各条件 accuracy 对比
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ['SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO']


def cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float]:
    """返回 (agreement, kappa)。a/b 为两个 rater 的 label 列表。"""
    n = len(a)
    agree = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a)
    cb = Counter(b)
    expected = sum(ca[l] / n * cb[l] / n for l in LABELS)
    kappa = (agree - expected) / (1 - expected) if expected < 1 else 1.0
    return agree, kappa


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'results' / 'ab_scale100.json'
    ann_path = ROOT / 'data' / 'annotation' / 'claims_to_annotate.csv'

    rows = list(csv.DictReader(ann_path.open(encoding='utf-8')))
    labeled = [r for r in rows if r['human_label'].strip()]
    if not labeled:
        print('无已标注行（human_label 全空），退出'); return
    print(f'已标注 {len(labeled)}/{len(rows)} 条')

    # 人工分布
    hd = Counter(r['human_label'] for r in labeled)
    print(f'人工分布: {dict(hd)}')

    # 载入系统 verdict（按 idx 对齐）
    data = json.loads(results_path.read_text(encoding='utf-8'))
    res = data['results']
    gold_ret = data.get('gold_retrieved', {})
    # 标注行可能有 disputed 置顶，按 idx 排回系统顺序
    by_idx = {int(r['idx']): r for r in labeled}
    idxs = sorted(by_idx)
    n = len(idxs)
    human = [by_idx[i]['human_label'] for i in idxs]

    # C (oracle) vs 人工
    c_ver = [res['C'][i] for i in idxs]
    agree_c, kappa_c = cohen_kappa(c_ver, human)
    print(f'\nLLM-judge(C oracle) vs 人工: agreement={agree_c:.1%}  Cohen κ={kappa_c:.3f} (n={n})')

    # 各条件 vs 人工（系统判断与人工的一致率，衡量各条件贴近人工真值程度）
    print('\n各条件 vs 人工(agreement):')
    for cond in ['A', 'B1', 'B2', 'B3', 'C']:
        if cond not in res: continue
        v = [res[cond][i] for i in idxs]
        ag, _ = cohen_kappa(v, human)
        print(f'  {cond:3s} {ag:.1%}')

    # 坏 claim（人工 != SUPPORTED）
    bad = [(i, human[k]) for k, i in enumerate(idxs) if human[k] != 'SUPPORTED']
    print(f'\n人工判定的坏 claim (human != SUPPORTED): {len(bad)} 条')
    for i, lab in bad:
        r = by_idx[i]
        print(f"  idx={i:3d} pid={r['pid']} human={lab} C={r['verdict_C']:18s} "
              f"B1={r['verdict_B1']:18s} B2={r['verdict_B2']:18s} B3={r['verdict_B3']:18s}")
        if r['notes'].strip():
            print(f'      notes: {r["notes"][:100]}')

    # 敏感性：剔除坏 claim 前后 accuracy 对比
    def acc(verdicts, keep):
        sel = [v for v, k in zip(verdicts, keep) if k]
        return sum(1 for v in sel if v == 'SUPPORTED') / len(sel) if sel else 0.0

    keep_all = [True] * n
    keep_valid = [human[k] == 'SUPPORTED' for k in range(n)]
    print('\n敏感性（accuracy = 判为 SUPPORTED 比例，gold=SUPPORTED）:')
    print(f'  {"cond":4s} {"全部 n="+str(n):>12} {"剔坏 n="+str(sum(keep_valid)):>16}')
    for cond in ['A', 'B1', 'B2', 'B3', 'C']:
        if cond not in res: continue
        v = [res[cond][i] for i in idxs]
        a_all = acc(v, keep_all)
        a_val = acc(v, keep_valid)
        print(f'  {cond:4s} {a_all:>11.1%} {a_val:>15.1%}')

    # 落盘
    out = ROOT / 'results' / 'annotation_analysis.json'
    out.write_text(json.dumps({
        'results_source': str(results_path),
        'n_labeled': n,
        'human_dist': dict(hd),
        'c_vs_human_agreement': agree_c,
        'c_vs_human_kappa': kappa_c,
        'bad_claims': [{'idx': i, 'human': lab} for i, lab in bad],
        'cond_vs_human': {c: cohen_kappa([res[c][i] for i in idxs], human)[0]
                          for c in ['A', 'B1', 'B2', 'B3', 'C'] if c in res},
    }, ensure_ascii=False, indent=2))
    print(f'\n-> wrote {out}')


if __name__ == '__main__':
    main()
