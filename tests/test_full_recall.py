"""修正版：用 embedding 对粗索引 summary 检索（非关键词过滤），再 LLM judge。

之前 0% 是因为关键词过滤太严。改用 embedding 语义检索粗索引。
"""
import sys, time, json
sys.path.insert(0, 'src')
from pathlib import Path
from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.retriever import Retriever

art = _load_scifact_artifacts()
corpus = art['corpus']
samples = load_dataset_subsampled('scifact')[:50]

# 加载已建好的粗索引
coarse = json.loads(Path('cache/full_coarse_index.json').read_text())
print(f'粗索引: {len(coarse)} 篇', flush=True)

# 把粗索引的 summary 当文档，建 embedding 索引
print('\n建粗索引 embedding（用 summary 当文本）...', flush=True)
ret = Retriever()
coarse_docs = [{'_id': cid, 'title': c.get('summary','')[:80], 'text': c.get('summary','') + ' ' + ' '.join(c.get('keywords',[]))}
               for cid, c in coarse.items()]
ret.build_index([{'title': d['title'], 'text': d['text']} for d in coarse_docs])
print(f'粗索引 embedding 建好: {len(coarse_docs)} 篇', flush=True)

# recall: 用 claim 检索粗索引 summary
print('\n=== embedding 粗索引 recall@10 ===', flush=True)
hits = 0; total = 0
for s in samples:
    claim = s['metadata']['claim']
    gold_cids = set(s['metadata']['gold_corpus_ids'])
    results = ret.search(claim, top_k_recall=10, top_k_rerank=10, rerank=False)[:10]
    # 映射回 corpus_id（通过 summary 标题）
    hit_summaries = {h['title'] for h in results}
    gold_summaries = {coarse.get(gid,{}).get('summary','')[:80] for gid in gold_cids if gid in coarse}
    n_hit = len(hit_summaries & gold_summaries)
    hits += n_hit; total += len(gold_cids)
print(f'粗索引 embedding recall@10: {hits}/{total} = {hits/max(total,1):.0%}', flush=True)

# 对比：原文 embedding（之前 BM25 测过 81%）
print('\n=== 原文 embedding recall@10（对比）===', flush=True)
ret2 = Retriever()
all_docs = [{'title': c['title'], 'text': c['text']} for c in corpus.values()]
ret2.build_index(all_docs)
hits2 = 0; total2 = 0
for s in samples:
    claim = s['metadata']['claim']
    gold_cids = set(s['metadata']['gold_corpus_ids'])
    results = ret2.search(claim, top_k_recall=10, top_k_rerank=10, rerank=False)[:10]
    hit_titles = {h['title'] for h in results}
    gold_titles = {corpus[gid]['title'] for gid in gold_cids if gid in corpus}
    n_hit = len(hit_titles & gold_titles)
    hits2 += n_hit; total2 += len(gold_cids)
print(f'原文 embedding recall@10: {hits2}/{total2} = {hits2/max(total2,1):.0%}', flush=True)

print(f'\n=== 对比 ===')
print(f'原文 embedding: {hits2/max(total2,1):.0%}')
print(f'粗索引(summary) embedding: {hits/max(total,1):.0%}')
