"""代码库实验 v2：绕过 Retriever，直接用 SentenceTransformer + FAISS。

Retriever 类卡死原因不明，直接用底层组件更快。
"""
import sys, time, json, re, os, hashlib
sys.path.insert(0, 'src')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pathlib import Path
from collections import defaultdict
import numpy as np
import faiss

# ── 加载文件 ──
REPO = Path('/tmp/graphify')
docs = {}
for f in REPO.rglob('*'):
    if '.git' in str(f): continue
    if f.suffix in ('.py','.md','.rst','.txt'):
        try:
            text = f.read_text(errors='ignore').strip()
            if len(text) < 50: continue
            if len(text) > 2000: text = text[:2000]
            rel = str(f.relative_to(REPO))
            docs[rel] = {'title': rel, 'text': text}
        except: pass
print(f'{len(docs)} files', flush=True)

# ── 直接编码建索引 ──
print('encoding...', flush=True)
from sentence_transformers import SentenceTransformer
t0 = time.time()
model = SentenceTransformer('BAAI/bge-m3', device='mps')
texts = [f"{d['title']}: {d['text']}" for d in docs.values()]
embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                     show_progress_bar=False, batch_size=16).astype('float32')
print(f'encoded {len(texts)} docs in {time.time()-t0:.1f}s, dim={embs.shape[1]}', flush=True)

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
print(f'FAISS index built', flush=True)

# ── 加载已缓存的 claims + reformulation + hyde ──
claims = json.loads(Path('data/graphify_claims.json').read_text())
reform = json.loads(Path('data/graphify_reformulated.json').read_text())
hyde = json.loads(Path('data/graphify_hyde.json').read_text())
test_claims = claims[:30]
print(f'{len(test_claims)} test claims', flush=True)

# ── 检索函数 ──
doc_keys = list(docs.keys())
def emb_search(query, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False).astype('float32')
    scores, indices = index.search(q_emb, top_k)
    return [(doc_keys[i], float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]

def emb_search_hyde(query, top_k=3):
    """HyDE×3 RRF"""
    rws = hyde.get(query, [query])
    fused = defaultdict(float)
    for rw in rws:
        q_emb = model.encode([rw], convert_to_numpy=True, normalize_embeddings=True,
                             show_progress_bar=False).astype('float32')
        scores, indices = index.search(q_emb, 20)
        for rank, (i, s) in enumerate(zip(indices[0], scores[0])):
            if i >= 0:
                fused[doc_keys[i]] += 1.0 / (60 + rank + 1)
    return [(k, v) for k, v in sorted(fused.items(), key=lambda x: -x[1])[:top_k]]

# grep 倒排索引
print('building grep index...', flush=True)
inverted = defaultdict(set)
for fid, d in docs.items():
    for w in set(re.findall(r'[a-zA-Z]{3,}', d['text'].lower())):
        inverted[w].add(fid)
inverted = dict(inverted)
print(f'grep index: {len(inverted)} terms', flush=True)

import math
N = len(docs)
def grep_max_idf(query, top_k=10):
    terms = re.findall(r'[a-zA-Z]{4,}', query.lower())
    terms = [w for w in terms if w not in {'the','and','for','with','that','this','from','are','was','were','have','been','will','would','can','could','should','may','might','must','shall','into','through','about','which','their','there','here','when','where','while'}]
    doc_max = defaultdict(float)
    for term in terms[:6]:
        matched = set()
        for w, cids in inverted.items():
            if term in w:
                matched |= cids
        if not matched: continue
        idf = math.log(N / len(matched))
        for cid in matched:
            if idf > doc_max[cid]: doc_max[cid] = idf
    return [k for k, _ in sorted(doc_max.items(), key=lambda x: -x[1])[:top_k]]

# CrossEncoder
print('loading cross-encoder...', flush=True)
from sentence_transformers import CrossEncoder
ce = CrossEncoder('BAAI/bge-reranker-v2-m3', device='mps')
print(f'ce loaded', flush=True)

def fusion_search(query, top_k=3):
    emb_cids = [k for k, _ in emb_search_hyde(query, top_k=20)]
    grep_cids = grep_max_idf(query, top_k=10)
    pool = list(dict.fromkeys(emb_cids + grep_cids))
    if not pool: return []
    pairs = [[query, f"{docs[c]['title']}: {docs[c]['text']}"] for c in pool]
    scores = ce.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(pool, [float(s) for s in scores]), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_k]]

# ── LLM verify ──
from rag2.gateway import ModelGateway
gw = ModelGateway()
MODEL_NAME = 'deepseek-v4-flash'
VERIFY_SYS = "Verify technical claims about a software project. SUPPORTED/REFUTED/NOT_ENOUGH_INFO. Call verify tool."
VERIFY_TOOL = [{'type':'function','function':{'name':'verify','parameters':{
    'type':'object','properties':{'verdict':{'type':'string','enum':['SUPPORTED','REFUTED','NOT_ENOUGH_INFO']}},
    'required':['verdict']}}}]

def verify(claim, context=None):
    msg = f'Claim: {claim}'
    if context: msg += f'\n\n--- Evidence ---\n{context}\n--- End ---\n\nBased ONLY on evidence, verify.'
    else: msg += '\n\nBased on your knowledge, verify. If unknown, NOT_ENOUGH_INFO.'
    resp = gw.generate(MODEL_NAME, [{'role':'system','content':VERIFY_SYS},{'role':'user','content':msg}],
                      tools=VERIFY_TOOL, role_tag='verify', max_tokens=500)
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'verify':
            try: return json.loads(tc['function']['arguments']).get('verdict','UNKNOWN')
            except: pass
    text = (resp.text or '').upper()
    for v in ['NOT_ENOUGH_INFO','SUPPORTED','REFUTED']:
        if v in text: return v
    return 'UNKNOWN'

# ── A/B/C ──
print('\n=== A/B1/B2/C ===', flush=True)
results = {'A':[], 'B1':[], 'B2':[], 'C':[]}
t_start = time.time()

for i, tc in enumerate(test_claims):
    claim = reform.get(tc['claim'], tc['claim'])
    gold_file = tc['source_file']
    gold_text = docs.get(gold_file, {}).get('text', tc['claim'])

    # A
    v_a = verify(claim)

    # B1: vanilla
    emb_r = emb_search(claim, top_k=3)
    ctx = '\n\n'.join(f'[{k}]\n{docs[k]["text"][:1500]}' for k, _ in emb_r if k in docs)
    v_b1 = verify(claim, context=ctx)

    # B2: fusion
    fus_r = fusion_search(claim, top_k=3)
    ctx2 = '\n\n'.join(f'[{k}]\n{docs[k]["text"][:1500]}' for k in fus_r if k in docs)
    v_b2 = verify(claim, context=ctx2)

    # C: oracle
    ctx_c = f'[{gold_file}]\n{gold_text[:2000]}'
    v_c = verify(claim, context=ctx_c)

    results['A'].append(v_a)
    results['B1'].append(v_b1)
    results['B2'].append(v_b2)
    results['C'].append(v_c)

    if (i+1) % 5 == 0:
        def acc(k): return sum(1 for v in results[k][:i+1] if v=='SUPPORTED')/(i+1)
        print(f'  [{i+1:2d}/30] ({time.time()-t_start:.0f}s) A={acc("A"):.0%} B1={acc("B1"):.0%} B2={acc("B2"):.0%} C={acc("C"):.0%}', flush=True)

# ── 结果 ──
print(f'\n{"="*60}', flush=True)
print(f'代码库语料 A/B/C（Graphify, {len(docs)} 文件, n=30）', flush=True)
print(f'{"="*60}', flush=True)
for cond in ['A','B1','B2','C']:
    n = len(results[cond])
    s = sum(1 for v in results[cond] if v=='SUPPORTED')
    print(f'  {cond}: {s}/{n} = {s/n:.0%}', flush=True)

Path('results/ab_code_repo.json').write_text(json.dumps({
    'corpus': f'Graphify-Labs/graphify ({len(docs)} files, 2026-04)',
    'n': 30, 'results': results,
}, ensure_ascii=False, indent=2))
