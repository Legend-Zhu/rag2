"""SciFact 最小验证：100 篇 corpus + 5 个 claim，验证 C2+C1 端到端。"""
import sys, time
sys.path.insert(0, 'src')
from rag2.gateway import ModelGateway
from rag2.data import _load_scifact_artifacts, load_dataset_subsampled
from rag2.methods.generative_index import GenerativeIndexBuilder
from rag2.methods.retriever import Retriever
from rag2.methods.c2_backends import build_backend
from rag2.methods.c1_strategies import build_strategy

print('=== 1. 加载 SciFact 共享语料（前 100 篇 + gold）===', flush=True)
art = _load_scifact_artifacts()
corpus = art['corpus']

# 取前 5 个 claim，把它们的 gold 文档加进 corpus（保证 gold 在池子里）
samples = load_dataset_subsampled('scifact')[:5]
gold_ids = set()
for s in samples:
    gold_ids.update(s['metadata']['gold_corpus_ids'])

# 构造 100 篇语料：gold 文档 + 其他随机文档凑到 100
docs = []
seen = set(gold_ids)
# 先放 gold
for gid in gold_ids:
    c = corpus.get(gid)
    if c:
        docs.append({'_id': gid, 'title': c['title'], 'text': c['text']})
# 再补其他文档到 30
for cid, c in corpus.items():
    if cid in seen: continue
    docs.append({'_id': cid, 'title': c['title'], 'text': c['text']})
    seen.add(cid)
    if len(docs) >= 30: break

print(f'语料: {len(docs)} 篇（含 {len(gold_ids)} 个 gold）', flush=True)

print('\n=== 2. 建 C2 索引（K3）===', flush=True)
gw = ModelGateway()
t0 = time.time()
indices, graph = GenerativeIndexBuilder(gw, role='qwen3.8', batch_size=5).build(docs)
print(f'耗时{time.time()-t0:.0f}s, 图:{graph.stats()}', flush=True)

print('\n=== 3. 三后端 recall 对比 ===', flush=True)
ret = Retriever()
_ = ret.embedder  # 预热

# 建立 doc_id → corpus_id 映射
id_to_corpus = {idx.doc_id: docs[i].get('_id', str(i)) for i, idx in enumerate(indices)}

for bn in ['A', 'B', 'C']:
    backend = build_backend(bn, indices, graph, gw, retriever=ret, role='qwen3.8')
    total_hit = 0
    total_gold = 0
    for s in samples:
        gold_cids = set(s['metadata']['gold_corpus_ids'])
        claim = s['metadata']['claim']
        hits = backend.retrieve(claim, top_k=5)
        hit_cids = {id_to_corpus.get(h) for h in hits}
        n_hit = len(hit_cids & gold_cids)
        total_hit += n_hit
        total_gold += len(gold_cids)
    recall = total_hit / max(total_gold, 1)
    print(f'[后端{bn}] recall@5 = {total_hit}/{total_gold} = {recall:.0%}', flush=True)
