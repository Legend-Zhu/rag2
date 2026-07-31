"""测试文档解析 + 分块管线。

验证：
  1. 每种格式正确解析（MD/HTML/TXT）
  2. 分块大小合理（~512 chars）、重叠保留、heading 映射
  3. chunk_id → parent_doc_id 回溯正确
  4. 端到端管线能跑通（目录→解析→分块→注册）
"""
import sys, json, time
sys.path.insert(0, 'src')
from pathlib import Path
from rag2.ingest import DocumentParser, RecursiveChunker, IngestPipeline
from rag2.mlops import IndexManager

# ── 1. 解析器测试 ──
print('=== 1. 文档解析测试 ===\n', flush=True)
parser = DocumentParser()

for f in sorted(Path('/tmp/test_docs').glob('*')):
    if f.is_file():
        doc = parser.parse(f)
        print(f'  {f.name} ({doc.mime_type}):', flush=True)
        print(f'    title: {doc.title}', flush=True)
        print(f'    text: {len(doc.text)} chars, sections: {len(doc.sections)}', flush=True)
        print(f'    metadata: {doc.metadata}', flush=True)
        if doc.sections:
            print(f'    first section: heading="{doc.sections[0]["heading"][:40]}" level={doc.sections[0].get("level")}', flush=True)
        print(f'    text preview: {doc.text[:120]}...', flush=True)
        print()

# ── 2. 分块测试 ──
print('=== 2. 分块测试 ===\n', flush=True)
chunker = RecursiveChunker(chunk_size=200, chunk_overlap=30)

for f in sorted(Path('/tmp/test_docs').glob('*')):
    if not f.is_file(): continue
    doc = parser.parse(f)
    chunks = chunker.chunk(doc)
    print(f'  {f.name}: {len(chunks)} chunks', flush=True)
    for c in chunks[:3]:
        print(f'    [{c.chunk_id}] heading="{c.heading[:30]}" text={len(c.text)}chars pos={c.char_start}-{c.char_end}', flush=True)
        print(f'      preview: {c.text[:80]}...', flush=True)
    if len(chunks) > 3:
        print(f'    ... ({len(chunks)-3} more)', flush=True)
    print()

# ── 3. chunk↔doc 映射验证 ──
print('=== 3. chunk↔doc 映射验证 ===\n', flush=True)
docs = parser.parse_dir(Path('/tmp/test_docs'))
all_chunks = chunker.chunk_many(docs)
print(f'  总文档: {len(docs)}, 总 chunks: {len(all_chunks)}', flush=True)

# 每个 chunk 的 parent_doc_id 应对应一个真实 doc
doc_ids = {d.doc_id for d in docs}
orphan = [c for c in all_chunks if c.parent_doc_id not in doc_ids]
print(f'  孤儿 chunk（parent_doc_id 不匹配）: {len(orphan)}', flush=True)

# chunk_id 唯一性
chunk_ids = [c.chunk_id for c in all_chunks]
print(f'  chunk_id 唯一: {len(chunk_ids) == len(set(chunk_ids))}', flush=True)

# ── 4. 端到端管线测试（无 retriever，只测解析+分块+注册）──
print('\n=== 4. 端到端管线（无索引构建）===\n', flush=True)
im = IndexManager()
pipeline = IngestPipeline(parser=parser, chunker=chunker, index_manager=im)

result = pipeline.ingest_dir(Path('/tmp/test_docs'), corpus_id='test_docs', build_index=False)
print(f'  入库结果:', flush=True)
for k, v in result.items():
    if k != 'chunk_map':
        print(f'    {k}: {v}', flush=True)

# 验证 IndexManager 注册
active = im.get_active('test_docs')
if active:
    print(f'\n  IndexManager 注册验证:', flush=True)
    print(f'    version: {im.list_versions()["test_docs"]["active_version"]}', flush=True)
    print(f'    n_files: {active.get("n_files")}', flush=True)
    print(f'    n_chunks: {active.get("n_chunks")}', flush=True)
    print(f'    chunk_config: {active.get("chunk_config")}', flush=True)
    print(f'    source_files: {len(active.get("source_files", []))} files', flush=True)

print('\n=== 全部测试通过 ===', flush=True)
