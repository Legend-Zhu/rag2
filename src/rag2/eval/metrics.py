"""
RAG² 评测层

指标：
  - EM / F1（标准化后的 token 级，HotpotQA 官方风格）
  - Recall@k（检索质量）
  - RAGAS faithfulness（接地性，W2 接入完整版）
  - ALCE citation F1（W2 接入）
  - claim_level_citation_granularity / cross_source_contradiction_detection_rate（W2 自定义）

统计：
  - bootstrap 重采样 95% CI（n=1000）
"""
from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Callable

import numpy as np


# ─────────────────────────────────────────────────────────
# 文本归一化（HotpotQA 官方风格）
# ─────────────────────────────────────────────────────────

def _normalize_answer(s: str) -> str:
    """小写、去标点、去冠词、去多余空白。"""
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _tokenize(s: str) -> list[str]:
    return _normalize_answer(s).split()


# ─────────────────────────────────────────────────────────
# 单题指标
# ─────────────────────────────────────────────────────────

def exact_match(pred: str, gold: str) -> float:
    """EM: 归一化后完全相等得 1，否则 0。"""
    return float(_normalize_answer(pred) == _normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    """token 级 F1（HotpotQA 风格）。"""
    pred_toks = _tokenize(pred)
    gold_toks = _tokenize(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved_titles: list[str], gold_titles: set[str], k: int | None = None) -> float:
    """检索 Recall@k：gold 文档在前 k 个检索结果中的命中率。"""
    if not gold_titles:
        return 0.0
    tops = retrieved_titles[:k] if k else retrieved_titles
    hit = len(set(tops) & gold_titles)
    return hit / len(gold_titles)


def contains_match(pred: str, gold: str) -> float:
    """
    包含匹配：gold 是 pred 的子串，或 pred 是 gold 的子串（归一化后）。
    比 EM 宽松（容忍冗余解释），比 F1 公平（不因答案长短惩罚）。
    适合"答案实体正确但模型多说了几句"的场景。
    """
    p = _normalize_answer(pred)
    g = _normalize_answer(gold)
    if not p or not g:
        return 0.0
    # 完全相等
    if p == g:
        return 1.0
    # gold 是 pred 子串（模型答了 gold + 多余解释）
    if g in p:
        return 1.0
    # pred 是 gold 子串（模型答了部分）
    if p in g:
        return 0.5
    # 词级包含：gold 所有词都在 pred 里
    g_words = set(g.split())
    p_words = set(p.split())
    if g_words and g_words.issubset(p_words):
        return 1.0
    return 0.0


def is_correct(pred: str, gold: str) -> float:
    """复合正确性：EM 或 F1 > 0.5 算对（多跳 QA 常用宽松判定）。"""
    return float(exact_match(pred, gold) > 0 or f1_score(pred, gold) > 0.5)


# ─────────────────────────────────────────────────────────
# 批量指标 + bootstrap CI
# ─────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: list[float],
    n_resample: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """bootstrap 重采样估均值 + 置信区间。返回 (mean, lo, hi)。"""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.array([
        rng.choice(arr, size=n, replace=True).mean()
        for _ in range(n_resample)
    ])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(means, [alpha * 100, (1 - alpha) * 100])
    return float(arr.mean()), float(lo), float(hi)


def bootstrap_ci_supported(
    verdicts: list[str],
    gold: str = "SUPPORTED",
    n_resample: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    claim 验证准确率的 bootstrap CI。

    准确率 = 判为 gold 的比例（默认 gold=SUPPORTED）。用固定 seed 保可复现。
    返回 (point, lo, hi)，单位为比例（0-1）。
    """
    n = len(verdicts)
    if n == 0:
        return 0.0, 0.0, 0.0
    correct = np.array([1.0 if v == gold else 0.0 for v in verdicts])
    point = float(correct.mean())
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(correct, size=n, replace=True).mean() for _ in range(n_resample)])
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(boots, [alpha * 100, (1 - alpha) * 100])
    return point, float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> dict:
    """
    McNemar 精确检验（配对二分类）。

    Args:
        b: 新系统对、基线错的不一致对数（rescued）
        c: 新系统错、基线对的不一致对数（hurt）
    Returns:
        {"discordant": b+c, "p_two_sided": float, "p_one_sided": float}
    """
    d = b + c
    if d == 0:
        return {"discordant": 0, "p_two_sided": 1.0, "p_one_sided": 1.0}
    # H0: b ~ Binom(d, 0.5)
    pmf = [math.comb(d, k) * (0.5 ** d) for k in range(d + 1)]
    k_obs = max(b, c)
    p_one = sum(pmf[k_obs:])
    p_two = min(1.0, 2 * p_one)
    return {"discordant": d, "p_two_sided": p_two, "p_one_sided": p_one}


def paired_comparison(
    verdicts_new: list[str],
    verdicts_base: list[str],
    gold: str = "SUPPORTED",
) -> dict:
    """
    配对比较：new 相对 base 的 rescued/hurt + McNemar 精确 p。

    "rescued" = new 对而 base 错；"hurt" = new 错而 base 对。
    表格标签 "X vs Y" 对应 new=X, base=Y。
    """
    assert len(verdicts_new) == len(verdicts_base), "配对需等长"
    new_ok = [v == gold for v in verdicts_new]
    base_ok = [v == gold for v in verdicts_base]
    b = sum(n and not bb for n, bb in zip(new_ok, base_ok))  # rescued
    c = sum(bb and not n for n, bb in zip(new_ok, base_ok))  # hurt
    mc = mcnemar_exact(b, c)
    return {"rescued": b, "hurt": c, "net": b - c, **mc}


def aggregate(
    per_sample: list[dict],
    metric_fns: dict[str, Callable[..., float]] | None = None,
    n_resample: int = 1000,
    ci: float = 0.95,
) -> dict:
    """
    聚合每题指标，返回均值 + bootstrap CI。

    Args:
        per_sample: 每题的 {"pred": str, "gold": str, "retrieved_titles": [...], ...}
        metric_fns: 自定义指标函数，默认 EM/F1/Recall@5/correct
    Returns:
        {metric_name: {"mean": float, "ci_lo": float, "ci_hi": float}}
    """
    if metric_fns is None:
        metric_fns = {
            "em": lambda s: exact_match(s["pred"], s["gold"]),
            "f1": lambda s: f1_score(s["pred"], s["gold"]),
            "contains": lambda s: contains_match(s["pred"], s["gold"]),
            "correct": lambda s: is_correct(s["pred"], s["gold"]),
            "recall@5": lambda s: recall_at_k(s.get("retrieved_titles", []),
                                               set(s.get("gold_titles", [])), k=5),
        }

    results = {}
    for mname, fn in metric_fns.items():
        try:
            values = [fn(s) for s in per_sample]
        except KeyError:
            continue
        mean, lo, hi = _bootstrap_ci(values, n_resample, ci)
        results[mname] = {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n": len(values)}
    return results
