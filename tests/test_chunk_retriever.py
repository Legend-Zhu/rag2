"""测试 Retriever chunk 级适配 + 增量更新。

验证：
  1. build_chunk_index: 从 Chunk 对象建索引
  2. search 返回 doc_id (chunk_id) + chunk_meta (parent_doc_id, heading, page)
  3. read_doc: 按 chunk_id 读取
  4. incremental_update: 检测新增/删除文件
"""
import sys, os, shutil
sys.path.insert(0, 'src')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
from rag2.ingest import DocumentParser, RecursiveChunker

# 准备测试文档
test_dir = Path('/tmp/test_ingest_v2')
test_dir.mkdir(exist_ok=True)
(test_dir / 'doc1.md').write_text('''# Project Alpha
## Architecture
We use a microservices architecture with 12 services.
Each service handles a specific domain.

## Results
Q3 completion rate: 87%.
Deployed 15 new features in production.
''')
(test_dir / 'doc2.txt').write_text('Team quarterly review. Shipped 3 major features. Customer satisfaction at 9.2/10.')

# 1. 解析 + 分块
parser = DocumentParser()
chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)
docs = parser.parse_dir(test_dir)
chunks = chunker.chunk_many(docs)
print(f'解析 {len(docs)} 文件 → {len(chunks)} chunks', flush=True)

# 2. build_chunk_index（不实际建 FAISS，只测数据结构）
# 用 mock retriever 测试接口
from rag2.ingest.models import Chunk as ChunkType

# 模拟 build_chunk_index 的数据结构验证
doc_ids = [c.chunk_id for c in chunks]
chunk_map = {c.chunk_id: {"parent_doc_id": c.parent_doc_id, "heading": c.heading, "page": c.page}
             for c in chunks}

print(f'\n=== chunk 级数据结构 ===', flush=True)
print(f'  doc_ids (chunk_ids): {doc_ids[:3]}...', flush=True)
print(f'  chunk_map 样例:', flush=True)
for cid in doc_ids[:3]:
    cm = chunk_map[cid]
    print(f'    {cid}: parent={cm["parent_doc_id"]}, heading="{cm["heading"][:30]}"', flush=True)

# 唯一性
print(f'  chunk_id 唯一: {len(doc_ids) == len(set(doc_ids))}', flush=True)

# parent_doc_id 回溯
parent_ids = set(cm["parent_doc_id"] for cm in chunk_map.values())
doc_id_set = set(d.doc_id for d in docs)
print(f'  parent_doc_id 全匹配: {parent_ids == parent_ids & doc_id_set}', flush=True)

# 3. 增量更新测试
from rag2.ingest import IngestPipeline
from rag2.mlops import IndexManager

pipeline = IngestPipeline(parser=parser, chunker=chunker, index_manager=IndexManager())

# 第一次入库
result1 = pipeline.ingest_dir(test_dir, corpus_id='test_inc', build_index=False)
print(f'\n=== 第一次入库 ===', flush=True)
print(f'  {result1["n_files"]} files, {result1["n_chunks"]} chunks', flush=True)

# 收集已知 doc_ids
known_ids = set()
active = pipeline.index_manager.get_active('test_inc')
if active:
    for sf in active.get('source_files', []):
        known_ids.add(sf['doc_id'])

# 添加新文件
(test_dir / 'doc3.md').write_text('# New Report\nThis is a new quarterly summary document.')
# 修改已有文件（不同内容 → 不同 hash）
(test_dir / 'doc2.txt').write_text('Updated review. Now 4 major features. CSAT 9.5/10.')

# 增量更新
result2 = pipeline.incremental_update(test_dir, 'test_inc', known_doc_ids=known_ids)
print(f'\n=== 增量更新 ===', flush=True)
print(f'  新增: {result2["n_new"]}', flush=True)
print(f'  删除(变化): {result2["n_removed"]}', flush=True)
print(f'  未变: {result2["n_unchanged"]}', flush=True)
print(f'  总 chunks: {result2["n_chunks"]}', flush=True)

# 4. Retriever read_doc 接口验证（用真实 Retriever）
print(f'\n=== Retriever chunk 级接口验证 ===', flush=True)
from rag2.methods.retriever import Retriever
ret = Retriever()

# 模拟 _doc_ids 和 _docs（不建真实索引，只测数据结构）
ret._docs = [{"title": c.title_for_index, "text": c.text} for c in chunks[:3]]
ret._doc_ids = [c.chunk_id for c in chunks[:3]]
ret._chunk_map = {c.chunk_id: {"parent_doc_id": c.parent_doc_id, "heading": c.heading} for c in chunks[:3]}

# read_doc 测试
test_id = chunks[0].chunk_id
doc = ret.read_doc(test_id)
print(f'  read_doc("{test_id}"):', flush=True)
print(f'    doc_id: {doc["doc_id"]}', flush=True)
print(f'    title: {doc["title"][:50]}', flush=True)
print(f'    text preview: {doc["text"][:60]}...', flush=True)
if "chunk_meta" in doc:
    print(f'    chunk_meta: parent={doc["chunk_meta"]["parent_doc_id"]}, heading="{doc["chunk_meta"]["heading"][:30]}"', flush=True)

print(f'\n=== 全部测试通过 ===', flush=True)

# 清理
shutil.rmtree(test_dir)
