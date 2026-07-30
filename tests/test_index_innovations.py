"""索引创新四方向 + baseline：统一 recall 评测。

全量 SciFact 5183，n=50 claim，recall@10。
每个方向独立评测，结果和 baseline(81%) 直接对比。
"""
import sys, time, json
sys.path.insert(0, 'src')
from pathlib import Path
from collections import defaultdict
from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.retriever import Retriever

art = _load_scifact_artifacts()
corpus = art['corpus']
samples = load_dataset_subsampled('scifact')[:50]
gw = ModelGateway()

all_docs = [{'_id': cid, **c} for cid, c in corpus.items()]
title_to_cid = {c['title']: cid for cid, c in corpus.items()}
cid_to_title = {cid: c['title'] for cid, c in corpus.items()}

def eval_recall(retrieve_fn, name):
    hits = 0; total = 0
    t0 = time.time()
    for s in samples:
        claim = s['metadata']['claim']
        gold_cids = set(s['metadata']['gold_corpus_ids'])
        retrieved = retrieve_fn(claim, top_k=10)
        n_hit = len(set(retrieved) & gold_cids)
        hits += n_hit; total += len(gold_cids)
    dt = time.time() - t0
    r = hits / max(total, 1)
    print(f'  [{name}] recall@10: {hits}/{total} = {r:.0%} ({dt:.0f}s)', flush=True)
    return r


# ══════════════════════════════════════════════
# baseline: 原文 embedding
# ══════════════════════════════════════════════
print('=== baseline: 原文 embedding ===', flush=True)
ret_orig = Retriever()
ret_orig.build_index([{'title': d['title'], 'text': d['text']} for d in all_docs])

def retrieve_baseline(query, top_k=10):
    results = ret_orig.search(query, top_k_recall=top_k, top_k_rerank=top_k, rerank=False)
    return [title_to_cid.get(h['title'], '') for h in results]

r_baseline = eval_recall(retrieve_baseline, '原文 embedding')


# ══════════════════════════════════════════════
# 方向 B: 混合检索（embedding + BM25 倒排）
# ══════════════════════════════════════════════
print('\n=== 方向 B: 混合检索（embedding + BM25）===', flush=True)
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import tempfile, shutil, math

# 建 BM25 倒排索引
bm25_dir = Path('cache/bm25_index')
if not bm25_dir.exists() or not exists_in(str(bm25_dir)):
    bm25_dir.mkdir(parents=True, exist_ok=True)
    schema = Schema(doc_id=ID(stored=True), title=TEXT(stored=True), text=TEXT(stored=True))
    ix = create_in(str(bm25_dir), schema)
    writer = ix.writer()
    for d in all_docs:
        writer.add_document(doc_id=str(d['_id']), title=d['title'], text=d['text'])
    writer.commit()
    print('  BM25 倒排索引建好', flush=True)
else:
    print('  BM25 倒排索引命中缓存', flush=True)

ix = open_dir(str(bm25_dir))

def bm25_search(query_str, top_k=10):
    """BM25 检索，返回 [(corpus_id, score), ...]。"""
    with ix.searcher() as searcher:
        qp = QueryParser("text", ix.schema)
        try:
            q = qp.parse(query_str)
        except:
            return []
        results = searcher.search(q, limit=top_k)
        return [(r['doc_id'], r.score) for r in results]

def retrieve_hybrid(query, top_k=10):
    """混合：embedding 取 top-20 ∪ BM25 取 top-20，按归一化分数融合排序。"""
    # embedding
    emb_results = ret_orig.search(query, top_k_recall=20, top_k_rerank=20, rerank=False)
    emb_cids = [title_to_cid.get(h['title'], '') for h in emb_results]
    emb_scores = {title_to_cid.get(h['title'], ''): h['score'] for h in emb_results}

    # BM25
    bm25_results = bm25_search(query, top_k=20)

    # 融合（RRF: Reciprocal Rank Fusion，不依赖分数尺度）
    rrf_k = 60
    fused = defaultdict(float)
    for rank, cid in enumerate(emb_cids):
        if cid:
            fused[cid] += 1.0 / (rrf_k + rank + 1)
    for rank, (cid, score) in enumerate(bm25_results):
        fused[cid] += 1.0 / (rrf_k + rank + 1)

    ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
    return [cid for cid, _ in ranked]

r_hybrid = eval_recall(retrieve_hybrid, '混合 embedding+BM25 (RRF)')


# ══════════════════════════════════════════════
# 方向 A: HyDE 双向（query 改写为假设性描述 + 检索）
# ══════════════════════════════════════════════
print('\n=== 方向 A: HyDE（query 改写后检索）===', flush=True)
HYDE_TOOL = [{'type':'function','function':{'name':'rewrite','parameters':{'type':'object','properties':{'queries':{'type':'array','items':{'type':'string'},'description':'3 query rewrites'}}}}}]

# 缓存 HyDE 改写（claim 数量有限）
hyde_cache = {}
hyde_cache_file = Path('cache/hyde_rewrites.json')
if hyde_cache_file.exists():
    hyde_cache = json.loads(hyde_cache_file.read_text())

def hyde_rewrite(claim):
    if claim in hyde_cache:
        return hyde_cache[claim]
    prompt = f'Rewrite this scientific claim into 3 different search queries that would find supporting evidence. Call rewrite tool.\n\nClaim: {claim}'
    resp = gw.generate('deepseek-v4-flash', [{'role':'user','content':prompt}],
                      tools=HYDE_TOOL, role_tag='hyde', max_tokens=500)
    rewrites = [claim]  # 包含原文
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'rewrite':
            try: rewrites.extend(json.loads(tc['function']['arguments']).get('queries',[]))
            except: pass
    hyde_cache[claim] = rewrites
    hyde_cache_file.write_text(json.dumps(hyde_cache, ensure_ascii=False))
    return rewrites

def retrieve_hyde(query, top_k=10):
    rewrites = hyde_rewrite(query)
    # 每个改写检索 top-10，合并去重取 top
    fused = defaultdict(float)
    for rw in rewrites:
        results = ret_orig.search(rw, top_k_recall=10, top_k_rerank=10, rerank=False)
        for rank, h in enumerate(results):
            cid = title_to_cid.get(h['title'], '')
            if cid:
                fused[cid] += 1.0 / (60 + rank + 1)
    ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
    return [cid for cid, _ in ranked]

r_hyde = eval_recall(retrieve_hyde, 'HyDE query 改写')


# ══════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════
print(f'\n{"="*50}', flush=True)
print('索引创新 recall@10 对比（n=50, SciFact 5183）', flush=True)
print(f'{"="*50}', flush=True)
print(f'  baseline 原文 embedding:    {r_baseline:.0%}', flush=True)
print(f'  方向B 混合 embedding+BM25:  {r_hybrid:.0%}', flush=True)
print(f'  方向A HyDE query 改写:      {r_hyde:.0%}', flush=True)
print(f'\n  vs baseline 提升:', flush=True)
print(f'  方向B: {(r_hybrid-r_baseline)*100:+.0f}pt', flush=True)
print(f'  方向A: {(r_hyde-r_baseline)*100:+.0f}pt', flush=True)
