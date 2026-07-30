"""C2 三后端对比测试脚本。"""
import sys
sys.path.insert(0, 'src')
from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled
from rag2.methods.generative_index import GenerativeIndexBuilder
from rag2.methods.retriever import Retriever
from rag2.methods.c2_backends import build_backend

samples = load_dataset_subsampled('musique')[:3]
all_docs = []
for s in samples:
    all_docs.extend(s['supporting_docs'])
docs = all_docs[:20]

print(f'建索引 {len(docs)} 文档', flush=True)
gw = ModelGateway()
builder = GenerativeIndexBuilder(gw, role='qwen3.8', batch_size=5)
indices, graph = builder.build(docs)
print(f'图: {graph.stats()}', flush=True)

ret = Retriever()
q = samples[0]['question']
gold = samples[0]['answer']
gold_titles = {d['title'] for d in samples[0]['supporting_docs'] if d.get('is_supporting')}
id_to_title = {idx.doc_id: idx.title for idx in indices}

print(f'\nQ: {q[:60]}', flush=True)
print(f'gold: {gold} | gold titles: {gold_titles}\n', flush=True)

for bn in ['A', 'B', 'C']:
    backend = build_backend(bn, indices, graph, gw, retriever=ret, role='qwen3.8')
    hits = backend.retrieve(q, top_k=5)
    ht = [id_to_title.get(h, '?') for h in hits]
    gcount = sum(1 for t in ht if t in gold_titles)
    print(f'[后端{bn}] {backend.name} 命中gold {gcount}/{len(gold_titles)}', flush=True)
    for h, t in zip(hits, ht):
        mark = 'HIT' if t in gold_titles else '   '
        print(f'  {mark} [{h}] {t[:45]}', flush=True)
    print(flush=True)
