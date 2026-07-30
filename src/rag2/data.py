"""
RAG² DataLoader — 四数据集统一加载 + 固定子采样

职责：
  - 加载 MuSiQue / HotpotQA / ALCE / FreshQA
  - 统一成内部 Sample schema
  - 固定 seed 子采样（跨实验共用同一批题，保可比 + 可复现）
  - 子采样结果落盘，后续模块直接读

内部 Sample schema:
  {
    "id": str,
    "question": str,
    "answer": str,            # gold
    "supporting_docs": [{"title": str, "text": str}],
    "metadata": {...}
  }

注意：所有数据源走 HuggingFace，必须设 HF_ENDPOINT 镜像（实测直连 timeout）。
"""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Iterator

import yaml

logger = logging.getLogger(__name__)

# 强制 HF 镜像（Mac mini 实测直连 timeout，必须走 hf-mirror）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


SUBSAMPLE_DIR = Path("data_raw/subsamples")


def _load_cfg(path: str = "config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_subsample(name: str, samples: list[dict], seed: int, n: int) -> Path:
    """子采样结果落盘，保证可复现。"""
    SUBSAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out = SUBSAMPLE_DIR / f"{name}_seed{seed}_n{n}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("子采样落盘: %s (%d 题)", out, len(samples))
    return out


def _load_subsample(name: str, seed: int, n: int) -> list[dict] | None:
    out = SUBSAMPLE_DIR / f"{name}_seed{seed}_n{n}.jsonl"
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    return None


# ─────────────────────────────────────────────────────────
# 各数据集适配器（原始格式 → 内部 schema）
# ─────────────────────────────────────────────────────────

def _adapt_musique(row: dict) -> dict:
    """MuSiQue: 每题带支撑+干扰段落列表（dgslibisey/MuSiQue, validation split）。"""
    docs = []
    paras = row.get("paragraphs") or []
    for p in paras:
        if isinstance(p, dict):
            docs.append({
                "title": p.get("title", ""),
                "text": p.get("paragraph_text", ""),
                "is_supporting": p.get("is_supporting", False),
            })
    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "supporting_docs": docs,
        "metadata": {
            "dataset": "musique",
            "answerable": row.get("answerable", True),
            "question_decomposition": row.get("question_decomposition", []),
        },
    }


def _adapt_hotpotqa(row: dict) -> dict:
    """HotpotQA: context 字段含 (title, [sents]) 对。"""
    docs = []
    ctx = row.get("context") or {"title": [], "sentences": []}
    titles = ctx.get("title", [])
    sents_lists = ctx.get("sentences", [])
    for t, sents in zip(titles, sents_lists):
        docs.append({"title": t, "text": " ".join(sents)})
    return {
        "id": row.get("_id", row.get("id", "")),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "supporting_docs": docs,
        "metadata": {
            "dataset": "hotpotqa",
            "type": row.get("type", ""),        # bridge / comparison
            "level": row.get("level", ""),
        },
    }


def _adapt_alce(row: dict) -> dict:
    """ALCE: 带 doc 列表，输出格式偏 citation。"""
    docs = []
    for d in row.get("docs", []) or []:
        docs.append({"title": d.get("title", ""), "text": d.get("text", "")})
    return {
        "id": row.get("id", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "supporting_docs": docs,
        "metadata": {"dataset": "alce", "output": row.get("output", "")},
    }


def _adapt_freshqa(row: dict) -> dict:
    """FreshQA: 多种类型，answer 可能多值。"""
    return {
        "id": str(row.get("id", row.get("question", "")[:20])),
        "question": row.get("question", ""),
        "answer": str(row.get("answer", "")),
        "supporting_docs": [],                  # FreshQA 通常无文档，靠时效检索
        "metadata": {
            "dataset": "freshqa",
            "answer_type": row.get("answer_type", ""),
            "search_required": row.get("search_required", False),
        },
    }


def _adapt_sciq(row: dict) -> dict:
    """
    SciQ: 科学问答，每题带 support（支撑科学事实）+ 3 个干扰答案。

    support 作 gold 文档；3 个 distractor 答案文本作干扰。
    纯科学内容（生物/化学/物理），审查风险极低。
    """
    support = row.get("support", "")
    docs = []
    if support:
        docs.append({
            "title": "Supporting science fact",
            "text": support,
            "is_supporting": True,
        })
    for i in range(1, 4):
        dist = row.get(f"distractor{i}", "")
        if dist:
            docs.append({
                "title": f"Distractor {i}",
                "text": dist,
                "is_supporting": False,
            })
    return {
        "id": str(row.get("qid", row.get("question", "")[:20])),
        "question": row.get("question", ""),
        "answer": str(row.get("correct_answer", "")),
        "supporting_docs": docs,
        "metadata": {"dataset": "sciq"},
    }


# ─────────────────────────────────────────────────────────
# SciFact: 独立语料库 + claim 验证（BEIR 标准格式）
# ─────────────────────────────────────────────────────────

_SCIFACT_CACHE: dict = {}  # 进程内缓存（corpus/qrels 加载一次）


def _load_scifact_artifacts():
    """加载 SciFact 的 corpus + queries + qrels（带进程内缓存）。"""
    if _SCIFACT_CACHE:
        return _SCIFACT_CACHE
    import pandas as pd

    base = Path("data_raw/scifact")
    corpus_df = pd.read_parquet(base / "corpus.parquet")
    queries_df = pd.read_parquet(base / "queries.parquet")

    # corpus: _id → {title, text}
    corpus = {}
    for _, r in corpus_df.iterrows():
        corpus[str(r["_id"])] = {"title": str(r.get("title", "")), "text": str(r.get("text", ""))}

    # queries: _id → text
    queries = {}
    for _, r in queries_df.iterrows():
        queries[str(r["_id"])] = str(r.get("text", ""))

    # qrels: query_id → set(corpus_id)
    qrels = {}
    qrels_path = base / "qrels_test.tsv"
    if qrels_path.exists():
        with qrels_path.open("r", encoding="utf-8") as f:
            next(f)  # 跳表头
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    qid, cid = str(parts[0]), str(parts[1])
                    qrels.setdefault(qid, set()).add(cid)

    _SCIFACT_CACHE.update({"corpus": corpus, "queries": queries, "qrels": qrels})
    logger.info("SciFact 加载: %d 文档, %d queries, %d qrels",
                len(corpus), len(queries), len(qrels))
    return _SCIFACT_CACHE


def _adapt_scifact(row: dict) -> dict:
    """
    SciFact: 科学 claim 验证。每条 claim 是一个科学断言，要在语料库里找证据。

    特殊处理：supporting_docs 不是每题独立，而是从全局 corpus 里按 qrels 取 gold。
    但 DataLoader 的设计是"每题带自己的 docs"，所以这里只放 gold 文档标记，
    真实的全语料检索由上层（ablation 脚本）共享 corpus 处理。

    为兼容现有 pipeline，这里把 gold 文档放进 supporting_docs（标记 is_supporting），
    同时把 claim 文本同时作为 question 和 answer（claim 验证任务的"答案"是
    SUPPORT/REFUTE，但我们的 C1 是生成式，需要适配）。
    """
    artifacts = _load_scifact_artifacts()
    qid = str(row.get("_id", ""))
    claim = row.get("text", "")
    gold_cids = artifacts["qrels"].get(qid, set())

    # gold 文档
    docs = []
    for cid in gold_cids:
        c = artifacts["corpus"].get(cid)
        if c:
            docs.append({
                "title": c["title"][:80],
                "text": c["text"],
                "is_supporting": True,
                "corpus_id": cid,
            })

    return {
        "id": f"scifact-{qid}",
        "question": f"Scientific claim: {claim}\n\nFind evidence in the corpus that supports or refutes this claim. Summarize the key finding.",
        "answer": claim,  # claim 本身作为 reference（评测用 contains）
        "supporting_docs": docs,
        "metadata": {
            "dataset": "scifact",
            "claim": claim,
            "gold_corpus_ids": list(gold_cids),
        },
    }


# 真实 HF repo 名（已验证，2026-07-26）
# - MuSiQue: dgslibisey/MuSiQue, validation split, 2417 rows, 字段含 is_supporting
# - SciQ: allenai/sciq, parquet 直连下载（绕开 load_dataset 的镜像重定向问题）
# - SciFact: BeIR/scifact（BEIR 标准格式，corpus + queries + qrels 分离）
ADAPTERS = {
    "musique": ("dgslibisey/MuSiQue", "validation", _adapt_musique),
    "hotpotqa": ("hotpot_qa", "validation", _adapt_hotpotqa),
    "alce": ("princeton-nlp/ALCE", "data", _adapt_alce),
    "freshqa": ("freshqa", "default", _adapt_freshqa),
    "sciq": ("allenai/sciq", "train", _adapt_sciq),
    "scifact": ("BeIR/scifact", "test", _adapt_scifact),
}


def _fetch_raw(name: str, split: str):
    """从 HuggingFace 拉取原始数据集。"""
    from datasets import load_dataset
    repo, _, _ = ADAPTERS[name]
    if name == "hotpotqa":
        ds = load_dataset(repo, "distractor", split=split, trust_remote_code=True)
    elif name == "sciq":
        # SciQ 走 parquet 直读（hf-mirror 对 resolve 重定向导致 load_dataset 失败）
        import pandas as pd
        parquet_path = Path("data_raw/sciq") / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"SciQ parquet 未下载: {parquet_path}。"
                f"先运行: curl -sL https://huggingface.co/datasets/allenai/sciq/resolve/main/data/{split}-00000-of-00001.parquet -o {parquet_path}"
            )
        df = pd.read_parquet(parquet_path)
        ds = df.to_dict("records")
    elif name == "scifact":
        # SciFact 走本地 parquet（已下载 corpus/queries + qrels TSV）
        import pandas as pd
        base = Path("data_raw/scifact")
        qrels = _load_scifact_artifacts()["qrels"]
        # 只返回有 qrels 的 queries（test set）
        query_ids = set(qrels.keys())
        queries_df = pd.read_parquet(base / "queries.parquet")
        ds = queries_df[queries_df["_id"].astype(str).isin(query_ids)].to_dict("records")
    else:
        ds = load_dataset(repo, split=split, trust_remote_code=True)
    return ds


def load_dataset_subsampled(name: str, force_refresh: bool = False) -> list[dict]:
    """
    加载某数据集的固定子采样。

    Args:
        name: musique | hotpotqa | alce | freshqa
        force_refresh: 忽略缓存重新采样
    Returns:
        统一 schema 的样本列表
    """
    cfg = _load_cfg()
    if name not in cfg["datasets"]:
        raise KeyError(f"配置中无数据集: {name}")
    ds_cfg = cfg["datasets"][name]
    seed, n, split = ds_cfg["seed"], ds_cfg["n"], ds_cfg.get("split", "dev")

    # 1. 命中已落盘子采样
    if not force_refresh:
        cached = _load_subsample(name, seed, n)
        if cached is not None:
            logger.info("命中已落盘子采样: %s (%d 题)", name, len(cached))
            return cached

    # 2. 拉取 + 适配 + 子采样
    _, _, adapter = ADAPTERS[name]
    logger.info("从 HuggingFace 拉取 %s (split=%s)...", name, split)
    raw = _fetch_raw(name, split)

    # 适配成内部 schema
    adapted = [adapter(row) for row in raw]
    logger.info("%s 原始 %d 题，适配后 %d 题", name, len(raw), len(adapted))

    # 固定 seed 子采样
    rng = random.Random(seed)
    if len(adapted) > n:
        samples = rng.sample(adapted, n)
    else:
        samples = adapted                      # 不足则全取
        logger.warning("%s 原始 %d < 目标 %d，全取", name, len(adapted), n)

    # 落盘
    _save_subsample(name, samples, seed, len(samples))
    return samples


def load_all(force_refresh: bool = False) -> dict[str, list[dict]]:
    """加载全部四个数据集的子采样。"""
    return {name: load_dataset_subsampled(name, force_refresh)
            for name in ["musique", "hotpotqa", "alce", "freshqa"]}


def iter_samples(name: str, force_refresh: bool = False) -> Iterator[dict]:
    """迭代器形式。"""
    yield from load_dataset_subsampled(name, force_refresh)
