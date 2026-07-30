"""
RAG² 对照实验：C2 检索的真实贡献

三组对照（同一批 SciFact claim）:
  A. 无检索 baseline：模型直接答，不给文档
  B. C2 检索 + C1 推理：完整 RAG² 管道
  C. gold 文档直塞：oracle 上限

关键对比:
  B - A = C2 检索的真实贡献
  C - B = 检索/推理的提升空间
"""
import sys, time, json
sys.path.insert(0, 'src')
from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.generative_index import GenerativeIndexBuilder
from rag2.methods.retriever import Retriever
from rag2.methods.c2_backends import build_backend
from rag2.methods.c1_strategies import build_strategy
from rag2.eval import aggregate, exact_match, f1_score, contains_match

# 准备数据
art = _load_scifact_artifacts()
samples = load_dataset_subsampled('scifact')[:5]
gold_ids = set()
for s in samples:
    gold_ids.update(s['metadata']['gold_corpus_ids'])

# 30 文档语料（含 gold，同 test_scifact）
docs = []
seen = set(gold_ids)
for gid in gold_ids:
    c = art['corpus'].get(gid)
    if c: docs.append({'_id': gid, 'title': c['title'], 'text': c['text']})
for cid, c in art['corpus'].items():
    if cid in seen: continue
    docs.append({'_id': cid, 'title': c['title'], 'text': c['text']})
    seen.add(cid)
    if len(docs) >= 30: break

gw = ModelGateway()
print(f'=== 数据: {len(samples)} claim, {len(docs)} 文档语料 ===\n', flush=True)

# 复用缓存索引
indices, graph = GenerativeIndexBuilder(gw, role='qwen3.8').build(docs)
ret = Retriever()
_ = ret.embedder
id_to_corpus = {idx.doc_id: docs[i].get('_id', str(i)) for i, idx in enumerate(indices)}

# ── 组 A: 无检索 baseline ──────────────────────────────
print('=== 组 A: 无检索 baseline（模型直接答）===', flush=True)
A_recs = []
for i, s in enumerate(samples):
    claim = s['metadata']['claim']
    prompt = f"""You are a scientific claim verifier. Without any external documents, based ONLY on your own knowledge, verify this claim.

Claim: {claim}

Respond with: VERDICT: [SUPPORTED/REFUTED/NOT_ENOUGH_INFO]
Then briefly explain (1-2 sentences). Call submit_verdict tool."""
    from rag2.methods.c1_orchestrator import C1_TOOL_SCHEMA
    # 简单调用，不用 C1 循环
    resp = gw.generate('qwen3.8', [{'role':'user','content':prompt}],
                       tools=[{'type':'function','function':{'name':'submit_verdict','parameters':{'type':'object','properties':{'verdict':{'type':'string','enum':['SUPPORTED','REFUTED','NOT_ENOUGH_INFO']},'explanation':{'type':'string'}},'required':['verdict']}}}],
                       role_tag='baseline_A', max_tokens=2000)
    # 提取 verdict
    verdict = 'UNKNOWN'
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'submit_verdict':
            try:
                args = json.loads(tc['function']['arguments'])
                verdict = args.get('verdict', 'UNKNOWN')
            except: pass
    print(f'  claim {i+1}: {verdict} | {claim[:50]}', flush=True)
    A_recs.append({'id': s['id'], 'pred': verdict, 'gold': s['answer'],
                   'claim': claim, 'gold_corpus_ids': list(s['metadata']['gold_corpus_ids'])})

# ── 组 B: C2 检索 + C1 推理 ────────────────────────────
print(f'\n=== 组 B: C2 检索 + C1 推理（完整 RAG²）===', flush=True)
backend = build_backend('B', indices, graph, gw, retriever=ret, role='qwen3.8')
strategy = build_strategy('S1', gw, backend, role='qwen3.8', max_steps=4)
B_recs = []
for i, s in enumerate(samples):
    test_sample = {**s, 'supporting_docs': docs}
    result = strategy.run(test_sample)
    # C1 的答案是文本，提取 verdict
    pred = result.answer[:200]
    print(f'  claim {i+1}: {pred[:60]}...', flush=True)
    B_recs.append({'id': s['id'], 'pred': pred, 'gold': s['answer'],
                   'claim': s['metadata']['claim'],
                   'gold_corpus_ids': list(s['metadata']['gold_corpus_ids']),
                   'retrieved_doc_ids': result.trace.get('steps',[])})

# ── 组 C: gold 文档直塞（oracle）────────────────────────
print(f'\n=== 组 C: gold 文档直塞（oracle 上限）===', flush=True)
C_recs = []
for i, s in enumerate(samples):
    claim = s['metadata']['claim']
    gold_docs = s['supporting_docs']  # gold 文档
    gold_text = '\n\n'.join(f'[{j+1}] {d["title"]}\n{d["text"]}' for j,d in enumerate(gold_docs))
    prompt = f"""Based on the provided evidence, verify this scientific claim.

Evidence:
{gold_text}

Claim: {claim}

Call submit_verdict tool with verdict (SUPPORTED/REFUTED/NOT_ENOUGH_INFO) and explanation."""
    resp = gw.generate('qwen3.8', [{'role':'user','content':prompt}],
                       tools=[{'type':'function','function':{'name':'submit_verdict','parameters':{'type':'object','properties':{'verdict':{'type':'string','enum':['SUPPORTED','REFUTED','NOT_ENOUGH_INFO']},'explanation':{'type':'string'}},'required':['verdict']}}}],
                       role_tag='oracle_C', max_tokens=2000)
    verdict = 'UNKNOWN'
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'submit_verdict':
            try:
                args = json.loads(tc['function']['arguments'])
                verdict = args.get('verdict', 'UNKNOWN')
            except: pass
    print(f'  claim {i+1}: {verdict} | {claim[:50]}', flush=True)
    C_recs.append({'id': s['id'], 'pred': verdict, 'gold': s['answer'],
                   'claim': claim, 'gold_corpus_ids': list(s['metadata']['gold_corpus_ids'])})

# ── 汇总 ───────────────────────────────────────────────
print(f'\n{"="*60}', flush=True)
print(f'对照汇总（n={len(samples)}）', flush=True)
print(f'{"="*60}', flush=True)
print(f'{"组":<8} {"verdict 分布":<40}', flush=True)
for name, recs in [('A 无检索', A_recs), ('B RAG²', B_recs), ('C Oracle', C_recs)]:
    verdicts = [r['pred'] if name != 'B RAG²' else 'TEXT' for r in recs]
    from collections import Counter
    dist = dict(Counter(verdicts))
    print(f'{name:<8} {dist}', flush=True)

print(f'\n=== 逐条对比 ===', flush=True)
for i in range(len(samples)):
    print(f'claim {i+1}: {samples[i]["metadata"]["claim"][:50]}', flush=True)
    print(f'  A(无检索): {A_recs[i]["pred"]}', flush=True)
    print(f'  B(RAG²):  {B_recs[i]["pred"][:40]}', flush=True)
    print(f'  C(oracle):{C_recs[i]["pred"]}', flush=True)
