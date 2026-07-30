"""
RAG² C2: 融合检索器 — 三层递进召回

架构：
  Query → HyDE×3 改写 → embedding 搜索 (top-20) ──┐
         ↓                                          ├─ 合并池 → CrossEncoder 重排 → top-10
         提取关键词 → grep MAX IDF (top-10) ────────┘

三层各自解决不同失败模式：
  1. HyDE×3     — 查询-文档词汇鸿沟（同义不同词）         81%→87% (+6pt)
  2. grep IDF   — 顺带提及盲区（gold 只顺带提某词一次）     87%→91% (+4pt)
  3. CrossEncoder — 语义消歧（多文档都含某词，哪个真相关）   91%→92% (+1pt)

性能（M4 实测, SciFact 5183 篇, n=50）：
  recall@10: 92% (49/53)
  延迟: ~16s/query（grep 全库扫描是瓶颈，预建倒排索引后 <2s）
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

STOPWORDS = frozenset({
    'the','a','an','and','or','but','in','on','at','to','of','for','with','from',
    'by','as','is','are','was','were','be','been','being','have','has','had','do',
    'does','did','will','would','can','could','should','may','might','must','shall',
    'this','that','these','those','it','its','they','them','their','there','here',
    'which','who','whom','whose','what','when','where','why','how','than','then',
    'so','if','because','while','during','between','within','without','about','into',
    'through','after','before','more','less','most','least','very','much','many',
    'some','any','all','both','each','other','such','same','own','new','one','two',
    'also','not','no','nor','only','just','very','too','either','neither','whether',
})


class FusionRetriever:
    """三层融合检索器：HyDE + grep + CrossEncoder。

    依赖 Retriever（bge-m3 + FAISS）做 embedding 召回，独立做 grep 和 CrossEncoder 重排。
    HyDE 改写缓存到 cache/hyde_rewrites.json，首次需 LLM 生成。
    """

    def __init__(
        self,
        retriever,                       # Retriever 实例（已建索引）
        corpus: dict[str, dict],         # {cid: {"title":..., "text":...}}
        cross_encoder_model: str = "BAAI/bge-reranker-v2-m3",
        device: str = "mps",
        hyde_cache_path: str = "cache/hyde_rewrites.json",
    ):
        self.retriever = retriever
        self.corpus = corpus
        self.title_to_cid = {c["title"]: cid for cid, c in corpus.items()}
        self.cid_to_title = {cid: c["title"] for cid, c in corpus.items()}
        self.corpus_texts = {cid: c.get("text", "") for cid, c in corpus.items()}
        self.N = len(corpus)

        self._ce = None
        self._ce_model = cross_encoder_model
        self._device = device

        self.hyde_cache: dict[str, list[str]] = {}
        p = Path(hyde_cache_path)
        if p.exists():
            self.hyde_cache = json.loads(p.read_text())

        # grep 倒排索引（懒建，加速重复 grep）
        self._inverted: dict[str, set[str]] | None = None

    # ── CrossEncoder 懒加载 ──────────────────────────────

    @property
    def ce(self):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            os.environ.setdefault("TQDM_DISABLE", "1")
            t0 = time.time()
            self._ce = CrossEncoder(self._ce_model, device=self._device)
            logger.info("加载 CrossEncoder %s (%.1fs)", self._ce_model, time.time() - t0)
        return self._ce

    # ── 关键词提取（启发式，无 LLM 依赖）──────────────────

    @staticmethod
    def extract_terms(claim: str, max_terms: int = 6) -> list[str]:
        """启发式提取搜索词：分词 → 去停用词 → 保留原始词形（小写）。"""
        words = re.findall(r"[a-zA-Z]{4,}", claim.lower())
        terms, seen = [], set()
        for w in words:
            if w in STOPWORDS or len(w) < 3:
                continue
            if w not in seen:
                seen.add(w)
                terms.append(w)
        return terms[:max_terms]

    # ── grep: 倒排索引查表（子串匹配，等效 regex）─────────

    @property
    def inverted_index(self) -> dict[str, set[str]]:
        """term → set(cid) 倒排索引。懒建 + 磁盘缓存。

        建一次 0.2s，查表 <0.01s/term，比正则扫描快 32x。
        磁盘缓存到 cache/grep_inverted_index.json（6.3MB）。
        """
        if self._inverted is None:
            idx_path = Path("cache/grep_inverted_index.json")
            if idx_path.exists():
                t0 = time.time()
                raw = json.loads(idx_path.read_text())
                self._inverted = {k: set(v) for k, v in raw.items()}
                logger.info("加载 grep 倒排索引 (%.1fs, %d 词)", time.time() - t0, len(self._inverted))
            else:
                t0 = time.time()
                self._inverted = {}
                for cid, text in self.corpus_texts.items():
                    for w in set(re.findall(r"[a-zA-Z]{3,}", text.lower())):
                        self._inverted.setdefault(w, set()).add(cid)
                logger.info("建 grep 倒排索引 (%.1fs, %d 词)", time.time() - t0, len(self._inverted))
                # 落盘缓存
                idx_path.parent.mkdir(parents=True, exist_ok=True)
                idx_path.write_text(json.dumps(
                    {k: sorted(v) for k, v in self._inverted.items()}, ensure_ascii=False
                ))
        return self._inverted

    def grep_term(self, term: str) -> set[str]:
        """查倒排索引。子串匹配（等效 regex）+ 单数回退。

        "body" 匹配 "body", "bodies", "antibody"（子串）
        "venule" 匹配 "venule", "venules", "venulectomy"
        "venules" 也匹配 "venule"（单数回退）
        """
        if not term or len(term) < 3:
            return set()
        inv = self.inverted_index
        matched = set()
        for w, cids in inv.items():
            if term in w:
                matched |= cids
        # 单数回退
        if term.endswith('s') and len(term) > 3:
            singular = term[:-1]
            for w, cids in inv.items():
                if singular in w:
                    matched |= cids
        return matched

    def grep_max_idf(self, query: str, top_k: int = 10) -> list[str]:
        """grep 所有搜索词，按 MAX IDF 排序返回 top-k cid。

        MAX IDF：取文档匹配词中最高的 IDF 作为主分。
        区别于 SUM IDF（奖励匹配多常见词），MAX IDF 奖励匹配最稀有词。
        """
        terms = self.extract_terms(query)
        doc_max = defaultdict(float)
        for term in terms:
            matched = self.grep_term(term)
            if not matched:
                continue
            idf = math.log(self.N / len(matched))
            for cid in matched:
                if idf > doc_max[cid]:
                    doc_max[cid] = idf
        ranked = sorted(doc_max.items(), key=lambda x: -x[1])
        return [cid for cid, _ in ranked[:top_k]]

    # ── embedding 召回（HyDE×3 RRF）──────────────────────

    def emb_scores(self, query: str, top_n: int = 20) -> dict[str, float]:
        """HyDE×3 改写 + embedding 召回，RRF 融合分数。"""
        rewrites = self.hyde_cache.get(query, [query])
        scores: dict[str, float] = defaultdict(float)
        for rw in rewrites:
            results = self.retriever.search(
                rw, top_k_recall=top_n, top_k_rerank=top_n, rerank=False
            )
            for rank, h in enumerate(results):
                cid = self.title_to_cid.get(h.get("title", ""), "")
                if cid:
                    scores[cid] += 1.0 / (60 + rank + 1)
        return dict(scores)

    # ── 融合检索主入口 ────────────────────────────────────

    def search(self, query: str, top_k: int = 10, use_grep: bool = True,
               use_rerank: bool = True) -> list[dict]:
        """三层融合检索。

        Args:
            query: 查询文本（claim / question）
            top_k: 返回文档数
            use_grep: 是否启用 grep 补盲区（默认 True）
            use_rerank: 是否启用 CrossEncoder 重排（默认 True）

        Returns:
            [{"cid","title","text","score","source"}, ...] 按相关性降序
        """
        # 1. embedding 召回 top-20
        es = self.emb_scores(query, top_n=20)
        emb_cids = [c for c, _ in sorted(es.items(), key=lambda x: -x[1])[:20]]

        if use_grep:
            # 2. grep MAX IDF 补盲区 top-10
            grep_cids = self.grep_max_idf(query, top_k=10)
            # 3. 合并去重
            pool = list(dict.fromkeys(emb_cids + grep_cids))
        else:
            pool = emb_cids

        if not pool:
            return []

        if not use_rerank:
            # 不重排，按 embedding RRF 排序
            ranked = sorted(pool, key=lambda c: -es.get(c, 0.0))
            return [self._format(c, es.get(c, 0.0), "emb") for c in ranked[:top_k]]

        # 4. CrossEncoder 重排
        pairs = [
            [query, f"{self.cid_to_title.get(c, '')}: {self.corpus_texts.get(c, '')}"]
            for c in pool
        ]
        scores = self.ce.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(pool, [float(s) for s in scores]), key=lambda x: -x[1])

        results = []
        for cid, score in ranked[:top_k]:
            source = "emb+grep" if (cid in emb_cids and cid in grep_cids) else \
                     "grep" if (use_grep and cid in grep_cids) else "emb"
            results.append(self._format(cid, score, source))
        return results

    def _format(self, cid: str, score: float, source: str) -> dict:
        return {
            "cid": cid,
            "title": self.cid_to_title.get(cid, ""),
            "text": self.corpus_texts.get(cid, ""),
            "score": score,
            "source": source,
        }
