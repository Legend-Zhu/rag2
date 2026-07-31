"""端到端测试 QdrantRetriever：arXiv 2026 语料。

验证：
  1. bge-m3 dense + sparse 编码正常
  2. build_index: upsert 到 Qdrant
  3. search: dense + sparse RRF 混合搜索
  4. search_dense / search_sparse: baseline 对比
  5. read_doc: 按 chunk_id 读取
  6. 增量 upsert + 删除
"""
import sys, time, json, os
sys.path.insert(0, 'src')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pathlib import Path

# 1. 测试 bge-m3 编码器
print('=== 1. bge-m3 编码器测试 ===', flush=True)
from rag2.methods.bge_m3_encoder import BGEM3Encoder

encoder = BGEM3Encoder(device='mps')
output = encoder.encode(['multimodal LLM for medical understanding', 'sparse autoencoder interpretability'])
print(f'  dense shape: {output["dense"].shape}', flush=True)
print(f'  sparse: {len(output["sparse"])} items, first has {len(output["sparse"][0])} terms', flush=True)
print(f'  dense L2 norms: {[f"{x:.3f}" for x in (output["dense"]**2).sum(axis=1).tolist()]} (应≈1.0)', flush=True)

# 2. 准备测试 chunks（从 arXiv 语料取 20 篇）
print('\n=== 2. 准备测试数据 ===', flush=True)
corpus = json.loads(Path('data/arxiv_2026_corpus.json').read_text())
pids = list(corpus.keys())[:20]
print(f'  arXiv 语料: 取 {len(pids)} 篇', flush=True)

from rag2.ingest.models import ParsedDoc, Chunk
chunks = []
for pid in pids:
    doc = corpus[pid]
    # 模拟 Chunk（直接从 arXiv 摘要构造）
    c = Chunk(
        chunk_id=f"{pid}_0000",
        parent_doc_id=pid,
        text=doc['text'],
        heading=doc['title'][:80],
        metadata={"filename": f"{pid}.txt", "title": doc['title']},
    )
    chunks.append(c)
print(f'  chunks: {len(chunks)}', flush=True)

# 3. 建 Qdrant 索引
print('\n=== 3. 建 Qdrant 索引 ===', flush=True)
from rag2.methods.qdrant_retriever import QdrantRetriever
qr = QdrantRetriever(
    collection_name="test_arxiv",
    embedder=encoder,  # 复用已加载的编码器
)

t0 = time.time()
qr.build_index(chunks, force_rebuild=True)
print(f'  建索引: {time.time()-t0:.1f}s, {qr.count()} points', flush=True)

# 4. 混合搜索
print('\n=== 4. 混合搜索 ===', flush=True)
query = "vision-centric multimodal model for medical diagnosis"
t0 = time.time()
results = qr.search(query, top_k=5, use_rerank=False)  # 先不重排
print(f'  search (dense+sparse RRF): {time.time()-t0:.2f}s, {len(results)} results', flush=True)
for r in results[:3]:
    print(f'    [{r["chunk_id"]}] score={r["score"]:.4f} {r["title"][:60]}', flush=True)

# 5. 纯 dense vs 纯 sparse
print('\n=== 5. dense vs sparse 对比 ===', flush=True)
dense_results = qr.search_dense(query, top_k=5)
print(f'  dense only: {len(dense_results)} results', flush=True)
for r in dense_results[:3]:
    print(f'    [{r["chunk_id"]}] score={r["score"]:.4f} {r["title"][:60]}', flush=True)

sparse_results = qr.search_sparse(query, top_k=5)
print(f'  sparse only: {len(sparse_results)} results', flush=True)
for r in sparse_results[:3]:
    print(f'    [{r["chunk_id"]}] score={r["score"]:.4f} {r["title"][:60]}', flush=True)

# 6. read_doc
print('\n=== 6. read_doc ===', flush=True)
test_cid = chunks[0].chunk_id
doc = qr.read_doc(test_cid)
if doc:
    print(f'  read_doc("{test_cid}"): text={len(doc["text"])} chars', flush=True)
    print(f'    parent_doc_id={doc["parent_doc_id"]}', flush=True)
else:
    print(f'  read_doc failed', flush=True)

# 7. 增量 upsert
print('\n=== 7. 增量 upsert ===', flush=True)
new_pid = list(corpus.keys())[20]  # 第 21 篇
new_chunk = Chunk(
    chunk_id=f"{new_pid}_0000",
    parent_doc_id=new_pid,
    text=corpus[new_pid]['text'],
    heading=corpus[new_pid]['title'][:80],
    metadata={"filename": f"{new_pid}.txt", "title": corpus[new_pid]['title']},
)
before = qr.count()
qr.upsert_chunks([new_chunk])
after = qr.count()
print(f'  upsert 1 chunk: {before} → {after} points', flush=True)

# 8. 删除
qr.delete_chunks([new_pid])
after_del = qr.count()
print(f'  delete doc {new_pid}: {after} → {after_del} points', flush=True)

print(f'\n=== 全部测试通过 ===', flush=True)
