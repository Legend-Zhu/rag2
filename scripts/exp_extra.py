#!/usr/bin/env python3
"""
主表之外的补充实验（复用主跑 test_ab_scale100.py 的缓存：index/reformulation/HyDE）。

子命令：
  forced      P1-1 强制猜测基线：A_forced + B2_forced（禁 NOT_ENOUGH_INFO，分离拒答效应）
  b3decouple  P1-3 B3 解耦：B3r(emb top-20 + rerank) 与 B2p(fusion top-20)，隔离 rerank/文档数
  truncation  P2-1 截断扫描：B2 在 500/1K/2K/4K/8K 字符截断下的准确率曲线

用法：
  python scripts/exp_extra.py forced     [--n 466]
  python scripts/exp_extra.py b3decouple [--n 466]
  python scripts/exp_extra.py truncation [--n 200] [--lengths 500,1000,2000,4000,8000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, 'src')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pathlib import Path

from rag2.gateway import ModelGateway
from rag2.methods.retriever import Retriever
from rag2.methods.fusion_retriever import FusionRetriever
from rag2.prompts import verify_claim as _verify

MODEL = 'deepseek-v4-flash'


# ── 共享配置（复用主跑缓存）──────────────────────────────────
class Ctx:
    def __init__(self, n: int):
        self.corpus = json.loads(Path('data/arxiv_2026_corpus.json').read_text())
        self.claims_data = json.loads(Path('data/arxiv_2026_claims.json').read_text())
        self.test_pids = list(self.claims_data.keys())[:n]
        self.n = len(self.test_pids)
        self.gw = ModelGateway()
        self.reform = json.loads(Path('cache/arxiv_reformulated_claims.json').read_text())
        all_docs = [{'title': d['title'], 'text': d['text']} for d in self.corpus.values()]
        self.ret = Retriever()
        self.ret.build_index(all_docs)  # 命中缓存（466 篇索引已建）
        self.fr = FusionRetriever(retriever=self.ret, corpus=self.corpus)
        _ = self.fr.inverted_index
        # 注入 HyDE 缓存（主跑已覆盖全部 466）
        self.fr.hyde_cache = json.loads(Path('cache/arxiv_hyde_rewrites.json').read_text())
        self.title_to_pid = {d['title']: pid for pid, d in self.corpus.items()}
        print(f'语料 {len(self.corpus)} 篇, 测试 {self.n} 篇, HyDE 缓存 {len(self.fr.hyde_cache)}', flush=True)

    def claim(self, pid):
        return self.reform.get(pid, self.claims_data.get(pid, ''))

    def verify(self, claim, context=None, forced=False):
        return _verify(self.gw, MODEL, claim, context=context, forced=forced)


def make_context(doc_dicts, corpus, max_docs=3, trunc_chars=None):
    """构造检索上下文；trunc_chars 非 None 时把每篇截到该字符数。

    text/title 缺省时按 cid 回退到 corpus（fr.search 结果可能只带 cid/title）。
    """
    lines = []
    for j, d in enumerate(doc_dicts[:max_docs]):
        cid = d.get('cid', '')
        title = d.get('title') or corpus.get(cid, {}).get('title', '')
        text = d.get('text') or corpus.get(cid, {}).get('text', '')
        if trunc_chars is not None:
            text = text[:trunc_chars]
        lines.append(f'[Paper {j+1}] {title}\n{text}')
    return '\n\n'.join(lines)


def fusion3_dicts(ctx, claim):
    """B2 融合检索 top-3 -> [{cid,title,text}]"""
    res = ctx.fr.search(claim, top_k=3, use_grep=True, use_rerank=True)
    return [{'cid': r.get('cid', ''), 'title': r.get('title', ''), 'text': r.get('text', '')} for r in res]


# ── P1-1 强制猜测 ─────────────────────────────────────────
def run_forced(ctx, out_dir):
    """A_forced / B2_forced：禁 NEI，强制二选一。"""
    res = {'A_forced': [], 'B2_forced': []}
    t0 = time.time()
    out = out_dir / 'exp_forced_guess.json'
    for i, pid in enumerate(ctx.test_pids):
        claim = ctx.claim(pid)
        res['A_forced'].append(ctx.verify(claim, context=None, forced=True))
        b2 = fusion3_dicts(ctx, claim)
        ctx_b2 = make_context(b2, ctx.corpus, max_docs=3)
        res['B2_forced'].append(ctx.verify(claim, context=ctx_b2, forced=True))
        if (i + 1) % 10 == 0:
            print(f'  [{i+1:3d}/{ctx.n}] ({time.time()-t0:.0f}s) '
                  f'A_f={acc(res["A_forced"]):.0%} B2_f={acc(res["B2_forced"]):.0%}', flush=True)
            out.write_text(json.dumps({'n': i+1, 'results': res}, ensure_ascii=False, indent=2))
    out.write_text(json.dumps({'n': ctx.n, 'results': res}, ensure_ascii=False, indent=2))
    print(f'保存: {out} | A_forced={acc(res["A_forced"]):.0%} B2_forced={acc(res["B2_forced"]):.0%}', flush=True)


# ── P1-3 B3 解耦 ─────────────────────────────────────────
def run_b3decouple(ctx, out_dir):
    """B3r=emb top-20+rerank；B2p=fusion top-20。配合主跑 B1/B2/B3 构成 2x2 归因。"""
    res = {'B3r': [], 'B2p': []}
    t0 = time.time()
    out = out_dir / 'exp_b3_decouple.json'
    for i, pid in enumerate(ctx.test_pids):
        claim = ctx.claim(pid)
        # B3r: emb top-20 + rerank（Retriever）
        emb20r = ctx.ret.search(claim, top_k_recall=20, top_k_rerank=20, rerank=True)
        b3r = [{'cid': ctx.title_to_pid.get(r['title'], ''), 'title': r['title'],
                'text': ctx.corpus.get(ctx.title_to_pid.get(r['title'], ''), {}).get('text', '')}
               for r in emb20r]
        res['B3r'].append(ctx.verify(claim, context=make_context(b3r, ctx.corpus, max_docs=20)))
        # B2p: fusion top-20（FusionRetriever）
        fus20 = ctx.fr.search(claim, top_k=20, use_grep=True, use_rerank=True)
        b2p = [{'cid': r.get('cid', ''), 'title': r.get('title', ''), 'text': r.get('text', '')} for r in fus20]
        res['B2p'].append(ctx.verify(claim, context=make_context(b2p, ctx.corpus, max_docs=20)))
        if (i + 1) % 10 == 0:
            print(f'  [{i+1:3d}/{ctx.n}] ({time.time()-t0:.0f}s) '
                  f'B3r={acc(res["B3r"]):.0%} B2p={acc(res["B2p"]):.0%}', flush=True)
            out.write_text(json.dumps({'n': i+1, 'results': res}, ensure_ascii=False, indent=2))
    out.write_text(json.dumps({'n': ctx.n, 'results': res}, ensure_ascii=False, indent=2))
    print(f'保存: {out} | B3r={acc(res["B3r"]):.0%} B2p={acc(res["B2p"]):.0%}', flush=True)


# ── P2-1 截断扫描 ─────────────────────────────────────────
def run_truncation(ctx, out_dir, lengths):
    """B2 在各截断长度下的准确率。full(=None) 与主跑 B2 一致（有缓存）。"""
    res = {f't{lc}': [] for lc in lengths}
    t0 = time.time()
    out = out_dir / 'exp_truncation.json'
    for i, pid in enumerate(ctx.test_pids):
        claim = ctx.claim(pid)
        b2 = fusion3_dicts(ctx, claim)
        for lc in lengths:
            res[f't{lc}'].append(ctx.verify(claim, context=make_context(b2, ctx.corpus, max_docs=3, trunc_chars=lc)))
        if (i + 1) % 10 == 0:
            line = ' '.join(f'{lc}={acc(res[f"t{lc}"]):.0%}' for lc in lengths)
            print(f'  [{i+1:3d}/{ctx.n}] ({time.time()-t0:.0f}s) {line}', flush=True)
            out.write_text(json.dumps({'n': i+1, 'lengths': lengths, 'results': res}, ensure_ascii=False, indent=2))
    out.write_text(json.dumps({'n': ctx.n, 'lengths': lengths, 'results': res}, ensure_ascii=False, indent=2))
    print(f'保存: {out} | ' + ' '.join(f'{lc}c={acc(res[f"t{lc}"]):.0%}' for lc in lengths), flush=True)


def acc(verdicts):
    return sum(1 for v in verdicts if v == 'SUPPORTED') / len(verdicts) if verdicts else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exp', choices=['forced', 'b3decouple', 'truncation'])
    ap.add_argument('--n', type=int, default=466)
    ap.add_argument('--lengths', type=str, default='500,1000,2000,4000,8000')
    args = ap.parse_args()

    out_dir = Path('results')
    out_dir.mkdir(exist_ok=True)
    ctx = Ctx(args.n)

    if args.exp == 'forced':
        run_forced(ctx, out_dir)
    elif args.exp == 'b3decouple':
        run_b3decouple(ctx, out_dir)
    elif args.exp == 'truncation':
        lengths = [int(x) for x in args.lengths.split(',')]
        run_truncation(ctx, out_dir, lengths)


if __name__ == '__main__':
    main()
