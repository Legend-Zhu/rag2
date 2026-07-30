"""代码库语料实验：Graphify-Labs/graphify（2026-04 创建，274 .py 文件）。

第二类"模型不知道的语料"：代码库。
和 arXiv 实验同样的 A/B/C 协议，但语料是源代码+文档。

流程：
  1. 加载 .py + .md 文件作为文档
  2. LLM 从文档提取 claim
  3. LLM 改写 claim
  4. A/B/C 对照（无检索 / vanilla RAG / RAG² / oracle）
"""
import sys, time, json, re, os
sys.path.insert(0, 'src')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pathlib import Path
from collections import defaultdict
from rag2.gateway import ModelGateway
from rag2.methods.retriever import Retriever
from rag2.methods.fusion_retriever import FusionRetriever

REPO = Path('/tmp/graphify')
gw = ModelGateway()
MODEL = 'deepseek-v4-flash'

# ── Step 1: 加载代码库文件 ──
print('=== Step 1: 加载代码库文件 ===', flush=True)
docs = {}
for f in REPO.rglob('*'):
    if '.git' in str(f): continue
    if f.suffix in ('.py', '.md', '.rst', '.txt'):
        try:
            text = f.read_text(errors='ignore').strip()
            if len(text) < 50: continue  # 太短跳过
            # 截断超长文件
            if len(text) > 4000: text = text[:4000]
            rel = str(f.relative_to(REPO))
            docs[rel] = {'title': rel, 'text': text}
        except: pass

print(f'  加载: {len(docs)} 个文件', flush=True)
py_count = sum(1 for k in docs if k.endswith('.py'))
md_count = sum(1 for k in docs if k.endswith('.md'))
print(f'  .py: {py_count}, .md: {md_count}, other: {len(docs)-py_count-md_count}', flush=True)

# ── Step 2: LLM 从文档提取 claim ──
print('\n=== Step 2: LLM 提取 claim ===', flush=True)

EXTRACT_TOOL = [{'type':'function','function':{'name':'extract_claims','parameters':{
    'type':'object','properties':{
        'claims':{'type':'array','items':{'type':'object','properties':{
            'claim':{'type':'string'},
            'source_file':{'type':'string'}
        }}}
    }}}}]

claims_file = Path('data/graphify_claims.json')
if claims_file.exists():
    all_claims = json.loads(claims_file.read_text())
    print(f'  缓存命中: {len(all_claims)} claims', flush=True)
else:
    all_claims = []
    # 从 .md 文件提取（文档更容易提取 claim）
    md_files = [(k, v) for k, v in docs.items() if k.endswith('.md') and len(v['text']) > 200]
    print(f'  从 {len(md_files)} 个 .md 文件提取...', flush=True)
    for fname, doc in md_files[:20]:  # 最多 20 个文件
        prompt = (f'Extract 2-3 verifiable technical claims from this documentation. '
                  f'Each claim should be a factual statement about the project that can be verified from the codebase. '
                  f'Call extract_claims tool.\n\nFile: {fname}\nContent:\n{doc["text"][:2000]}')
        resp = gw.generate(MODEL, [{'role':'user','content':prompt}],
                          tools=EXTRACT_TOOL, role_tag='extract_claims', max_tokens=1000)
        for tc in resp.tool_calls:
            if tc['function']['name'] == 'extract_claims':
                try:
                    items = json.loads(tc['function']['arguments']).get('claims', [])
                    for item in items:
                        c = item.get('claim', '').strip()
                        if c and len(c) > 20:
                            all_claims.append({'claim': c, 'source_file': item.get('source_file', fname)})
                except: pass
    claims_file.write_text(json.dumps(all_claims, ensure_ascii=False, indent=2))
    print(f'  提取: {len(all_claims)} claims', flush=True)

if len(all_claims) < 10:
    print(f'  claims 太少 ({len(all_claims)}), 跳过', flush=True)
    sys.exit(0)

# 取前 30 个
test_claims = all_claims[:30]
print(f'  测试: {len(test_claims)} claims\n', flush=True)

# ── Step 3: LLM 改写 claim ──
print('=== Step 3: 改写 claim ===', flush=True)
REFORM_TOOL = [{'type':'function','function':{'name':'reformulate','parameters':{
    'type':'object','properties':{'claim':{'type':'string'}}}}}]
reform_file = Path('data/graphify_reformulated.json')
reform_cache = json.loads(reform_file.read_text()) if reform_file.exists() else {}

for i, tc in enumerate(test_claims):
    orig = tc['claim']
    if orig in reform_cache: continue
    prompt = (f'Reformulate this technical claim using DIFFERENT vocabulary. Keep the same meaning. '
              f'Call reformulate tool.\n\n{orig}')
    resp = gw.generate(MODEL, [{'role':'user','content':prompt}],
                      tools=REFORM_TOOL, role_tag='reform', max_tokens=500)
    new = orig
    for tc2 in resp.tool_calls:
        if tc2['function']['name'] == 'reformulate':
            try: new = json.loads(tc2['function']['arguments']).get('claim', orig)
            except: pass
    reform_cache[orig] = new
    if (i+1) % 10 == 0: print(f'  改写 {i+1}/{len(test_claims)}', flush=True)
reform_file.write_text(json.dumps(reform_cache, ensure_ascii=False, indent=2))
print(f'  完成: {len(test_claims)} claims\n', flush=True)

# ── Step 4: HyDE 改写 ──
print('=== Step 4: HyDE ===', flush=True)
HYDE_TOOL = [{'type':'function','function':{'name':'rewrite','parameters':{
    'type':'object','properties':{'queries':{'type':'array','items':{'type':'string'}}}}}}]
hyde_file = Path('data/graphify_hyde.json')
hyde_cache = json.loads(hyde_file.read_text()) if hyde_file.exists() else {}

for i, tc in enumerate(test_claims):
    claim = reform_cache.get(tc['claim'], tc['claim'])
    if claim in hyde_cache: continue
    prompt = f'Rewrite into 3 search queries to find source code/docs. Call rewrite tool.\n\n{claim}'
    resp = gw.generate(MODEL, [{'role':'user','content':prompt}],
                      tools=HYDE_TOOL, role_tag='hyde', max_tokens=500)
    rws = [claim]
    for tc2 in resp.tool_calls:
        if tc2['function']['name'] == 'rewrite':
            try: rws = json.loads(tc2['function']['arguments']).get('queries', [claim])
            except: pass
    hyde_cache[claim] = rws
    if (i+1) % 10 == 0: print(f'  HyDE {i+1}/{len(test_claims)}', flush=True)
hyde_file.write_text(json.dumps(hyde_cache, ensure_ascii=False, indent=2))
print(f'  完成\n', flush=True)

# ── Step 5: 建检索器 ──
print('=== Step 5: 建索引 ===', flush=True)
all_docs_list = [{'title': d['title'], 'text': d['text']} for d in docs.values()]
ret = Retriever()
ret.build_index(all_docs_list)
fr = FusionRetriever(retriever=ret, corpus=docs)
fr.hyde_cache = {}
for tc in test_claims:
    claim = reform_cache.get(tc['claim'], tc['claim'])
    fr.hyde_cache[claim] = hyde_cache.get(claim, [claim])
print(f'  索引建好: {len(docs)} 文件\n', flush=True)

# ── Step 6: A/B/C 对照 ──
print('=== Step 6: A/B1/B2/C 对照 ===', flush=True)

VERIFY_SYSTEM = """You verify technical claims about a software project. Given a claim and evidence from source code/documentation, determine:
- SUPPORTED: evidence clearly supports the claim
- REFUTED: evidence clearly contradicts
- NOT_ENOUGH_INFO: insufficient evidence
Call verify tool."""
VERIFY_TOOL = [{'type':'function','function':{'name':'verify','parameters':{
    'type':'object','properties':{
        'verdict':{'type':'string','enum':['SUPPORTED','REFUTED','NOT_ENOUGH_INFO']}
    },'required':['verdict']}}}]

def verify(claim, context=None):
    msg = f'Claim: {claim}'
    if context:
        msg += f'\n\n--- Evidence ---\n{context}\n--- End ---\n\nBased ONLY on evidence, verify.'
    else:
        msg += '\n\nBased on your knowledge of this project, verify. If unknown, answer NOT_ENOUGH_INFO.'
    resp = gw.generate(MODEL, [{'role':'system','content':VERIFY_SYSTEM},{'role':'user','content':msg}],
                      tools=VERIFY_TOOL, role_tag='verify', max_tokens=500)
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'verify':
            try: return json.loads(tc['function']['arguments']).get('verdict','UNKNOWN')
            except: pass
    text = (resp.text or '').upper()
    for v in ['NOT_ENOUGH_INFO','SUPPORTED','REFUTED']:
        if v in text: return v
    return 'UNKNOWN'

results = {'A':[], 'B1':[], 'B2':[], 'C':[]}
title_to_fid = {d['title']: fid for fid, d in docs.items()}

for i, tc in enumerate(test_claims):
    claim = reform_cache.get(tc['claim'], tc['claim'])
    source_file = tc['source_file']
    gold_text = docs.get(source_file, {}).get('text', tc['claim'])

    # A: 无检索
    v_a = verify(claim, context=None)

    # B1: vanilla RAG
    emb_r = ret.search(claim, top_k_recall=3, top_k_rerank=3, rerank=False)
    emb_fids = [title_to_fid.get(r['title'],'') for r in emb_r]
    ctx_b1 = '\n\n'.join(f'[{fid}]\n{docs[fid]["text"][:1500]}' for fid in emb_fids[:3] if fid in docs)
    v_b1 = verify(claim, context=ctx_b1)

    # B2: RAG² 融合
    fusion_r = fr.search(claim, top_k=3, use_grep=True, use_rerank=True)
    ctx_b2 = '\n\n'.join(f'[{r.get("cid","")}]\n{r["text"][:1500]}' for r in fusion_r[:3])
    v_b2 = verify(claim, context=ctx_b2)

    # C: oracle
    ctx_c = f'[{source_file}]\n{gold_text[:2000]}'
    v_c = verify(claim, context=ctx_c)

    results['A'].append(v_a)
    results['B1'].append(v_b1)
    results['B2'].append(v_b2)
    results['C'].append(v_c)

    if (i+1) % 5 == 0:
        def acc(k): return sum(1 for v in results[k][:i+1] if v=='SUPPORTED')/(i+1)
        print(f'  [{i+1:2d}/{len(test_claims)}] A={acc("A"):.0%} B1={acc("B1"):.0%} B2={acc("B2"):.0%} C={acc("C"):.0%}', flush=True)

# ── 统计 ──
print(f'\n{"="*60}', flush=True)
print(f'代码库语料 A/B/C 对照（Graphify, {len(docs)} 文件, n={len(test_claims)}）', flush=True)
print(f'{"="*60}', flush=True)
for cond in ['A','B1','B2','C']:
    n = len(results[cond])
    s = sum(1 for v in results[cond] if v=='SUPPORTED')
    print(f'  {cond}: {s}/{n} = {s/n:.0%}  ({", ".join(results[cond][:10])}...)', flush=True)

# 保存
Path('results/ab_code_repo.json').write_text(json.dumps({
    'corpus': f'Graphify-Labs/graphify ({len(docs)} files, created 2026-04)',
    'n': len(test_claims),
    'results': results,
}, ensure_ascii=False, indent=2))
