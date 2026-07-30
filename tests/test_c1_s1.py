"""C1 推理编排器测试：S1 策略 + 后端 B，跑真实多跳题。"""
import sys
sys.path.insert(0, 'src')
from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled
from rag2.methods.generative_index import GenerativeIndexBuilder
from rag2.methods.retriever import Retriever
from rag2.methods.c2_backends import build_backend
from rag2.methods.c1_strategies import build_strategy

samples = load_dataset_subsampled('musique')[:2]
all_docs = []
for s in samples:
    all_docs.extend(s['supporting_docs'])
docs = all_docs[:20]

print(f'建 C2 索引（{len(docs)} 文档）', flush=True)
gw = ModelGateway()
indices, graph = GenerativeIndexBuilder(gw, role='qwen3.8').build(docs)
print(f'图: {graph.stats()}', flush=True)

ret = Retriever()
backend = build_backend('B', indices, graph, gw, retriever=ret)

s = samples[0]
print(f'\n=== 测试 S1 策略 ===', flush=True)
print(f'Q: {s["question"]}', flush=True)
print(f'gold: {s["answer"]}', flush=True)

strategy = build_strategy('S1', gw, backend, role='qwen3.8', max_steps=5)
result = strategy.run(s)

correct = (result.answer.lower().strip('.\"\'') in s['answer'].lower()
           or s['answer'].lower() in result.answer.lower())
print(f'\n=== 结果 ===', flush=True)
print(f'答案: {result.answer!r}', flush=True)
print(f'匹配: {"YES" if correct else "NO"}', flush=True)
print(f'步数: {result.trace["n_steps"]}, 收集文档: {result.trace["n_collected_docs"]}', flush=True)
print(f'总耗时: {result.trace["total_elapsed_s"]:.1f}s', flush=True)
print(f'\n=== 推理轨迹 ===', flush=True)
for st in result.trace['steps']:
    print(f'[step {st["step"]}] {st["action"]}', flush=True)
    if st.get('sub_queries'):
        for sq in st['sub_queries']:
            print(f'  subq: {sq[:70]}', flush=True)
    if st.get('retrieved_doc_ids'):
        print(f'  retrieved: {st["retrieved_doc_ids"]}', flush=True)

gold_titles = {d['title'] for d in s['supporting_docs'] if d.get('is_supporting')}
hit = sum(1 for d in result.retrieved_docs if d.is_supporting)
print(f'\ngold 命中: {hit}/{len(gold_titles)}', flush=True)
