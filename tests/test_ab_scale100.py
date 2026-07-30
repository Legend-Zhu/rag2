"""扩大实验：n=100，5 条件 A/B1/B2/B3/C + bootstrap CI。

条件：
  A   无检索：模型凭记忆
  B1  vanilla RAG：embedding top-3（无 HyDE/grep/CrossEncoder）
  B2  RAG² 融合：HyDE+grep+CrossEncoder top-3
  B3  long-context：embedding top-20 全给（不重排）
  C   oracle：直接给 gold 论文

输出：每条件准确率 + 95% bootstrap CI
"""
import sys, time, json, re, os, math, random
sys.path.insert(0, 'src')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pathlib import Path
from collections import defaultdict
from rag2.gateway import ModelGateway
from rag2.methods.retriever import Retriever
from rag2.methods.fusion_retriever import FusionRetriever
from rag2.prompts import verify_claim as _verify_claim_impl

# ── 加载语料 ──
corpus = json.loads(Path('data/arxiv_2026_corpus.json').read_text())
claims_data = json.loads(Path('data/arxiv_2026_claims.json').read_text())
print(f'arXiv 2026 语料: {len(corpus)} 篇, claims: {len(claims_data)}', flush=True)

N_TEST = len(claims_data)  # 用全部已下载语料（当前 466 篇）
test_pids = list(claims_data.keys())[:N_TEST]
print(f'测试集: {N_TEST} 篇\n', flush=True)

gw = ModelGateway()
MODEL = 'deepseek-v4-flash'

# ── 建检索器 ──
all_docs = [{'title': d['title'], 'text': d['text']} for d in corpus.values()]
ret = Retriever()
ret.build_index(all_docs)
fr = FusionRetriever(retriever=ret, corpus=corpus)
_ = fr.inverted_index
title_to_pid = {d['title']: pid for pid, d in corpus.items()}

# ── Step 1: LLM 改写 claim（复用缓存）─────────────────
print('=== Step 1: 改写 claim ===', flush=True)
REFORM_TOOL = [{'type':'function','function':{'name':'reformulate','parameters':{
    'type':'object','properties':{'claim':{'type':'string'}}}}}]
reform_file = Path('cache/arxiv_reformulated_claims.json')
reform_cache = json.loads(reform_file.read_text()) if reform_file.exists() else {}

def reformulate_claim(pid, original):
    if pid in reform_cache: return reform_cache[pid]
    prompt = (f'Reformulate this scientific claim using DIFFERENT vocabulary and sentence structure. '
              f'Keep the same meaning but change the wording completely. Call reformulate tool.\n\n{original}')
    resp = gw.generate(MODEL, [{'role':'user','content':prompt}],
                      tools=REFORM_TOOL, role_tag='reformulate', max_tokens=500)
    new = original
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'reformulate':
            try: new = json.loads(tc['function']['arguments']).get('claim', original)
            except: pass
    reform_cache[pid] = new
    reform_file.write_text(json.dumps(reform_cache, ensure_ascii=False, indent=2))
    return new

reformulated = {}
for i, pid in enumerate(test_pids):
    reformulated[pid] = reformulate_claim(pid, claims_data[pid])
    if (i+1) % 20 == 0:
        print(f'  改写 {i+1}/{N_TEST}', flush=True)
print(f'  完成: {len(reformulated)} 篇\n', flush=True)

# ── Step 2: HyDE 改写（复用缓存）──────────────────────
print('=== Step 2: HyDE 改写 ===', flush=True)
HYDE_TOOL = [{'type':'function','function':{'name':'rewrite','parameters':{
    'type':'object','properties':{'queries':{'type':'array','items':{'type':'string'}}}}}}]
hyde_file = Path('cache/arxiv_hyde_rewrites.json')
hyde_cache = json.loads(hyde_file.read_text()) if hyde_file.exists() else {}

def hyde_rewrite(claim):
    if claim in hyde_cache: return hyde_cache[claim]
    prompt = (f'Rewrite this scientific claim into 3 DIFFERENT search queries. '
              f'Call rewrite tool.\n\nClaim: {claim}')
    resp = gw.generate(MODEL, [{'role':'user','content':prompt}],
                      tools=HYDE_TOOL, role_tag='hyde', max_tokens=500)
    rws = [claim]
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'rewrite':
            try: rws = json.loads(tc['function']['arguments']).get('queries', [claim])
            except: pass
    hyde_cache[claim] = rws
    hyde_file.write_text(json.dumps(hyde_cache, ensure_ascii=False, indent=2))
    return rws

fr.hyde_cache = {}
for i, pid in enumerate(test_pids):
    claim = reformulated[pid]
    fr.hyde_cache[claim] = hyde_rewrite(claim)
    if (i+1) % 20 == 0:
        print(f'  HyDE {i+1}/{N_TEST}', flush=True)
print(f'  完成: {len(fr.hyde_cache)} 篇\n', flush=True)

# ── Step 3: 五条件对照 ────────────────────────────────
print('=== Step 3: A/B1/B2/B3/C 对照 ===', flush=True)

# 统一验证 prompt（src/rag2/prompts/verify.py），取代此前内联版本。
# n=30/n=100 旧 prompt 已合并为唯一来源；forced=True 走强制猜测变体（P1-1）。
def verify_claim(claim, context=None, forced=False):
    return _verify_claim_impl(gw, MODEL, claim, context=context, forced=forced)

def make_context(results, corpus, max_docs=3, full_text=True):
    """构造检索上下文。"""
    lines = []
    for j, r in enumerate(results[:max_docs]):
        cid = r.get('cid') or r.get('title','')
        title = r.get('title', corpus.get(cid,{}).get('title',''))
        text = r.get('text', corpus.get(cid,{}).get('text',''))
        if not full_text: text = text[:500]
        lines.append(f'[Paper {j+1}] {title}\n{text}')
    return '\n\n'.join(lines)

# 逐 claim 跑 5 条件
results = {k: [] for k in ['A','B1','B2','B3','C']}
gold_retrieved = {'B1': [], 'B2': [], 'B3': []}
results_file = Path(f'results/ab_scale{N_TEST}.json')
t_start = time.time()

for i, pid in enumerate(test_pids):
    claim = reformulated[pid]
    gold_abstract = corpus[pid]['text']
    gold_title = corpus[pid]['title']

    # A: 无检索
    v_a = verify_claim(claim, context=None)
    results['A'].append(v_a)

    # B1: vanilla RAG (embedding top-3, no HyDE/grep/rerank)
    emb_results = ret.search(claim, top_k_recall=3, top_k_rerank=3, rerank=False)
    emb_cids = [title_to_pid.get(r['title'],'') for r in emb_results]
    ctx_b1 = make_context(
        [{'cid':c, 'title':corpus[c]['title'], 'text':corpus[c]['text']} for c in emb_cids if c],
        corpus, max_docs=3)
    v_b1 = verify_claim(claim, context=ctx_b1)
    results['B1'].append(v_b1)
    gold_retrieved['B1'].append(pid in emb_cids[:3])

    # B2: RAG² 融合 (HyDE+grep+CrossEncoder top-3)
    fusion_results = fr.search(claim, top_k=3, use_grep=True, use_rerank=True)
    ctx_b2 = make_context(fusion_results, corpus, max_docs=3)
    v_b2 = verify_claim(claim, context=ctx_b2)
    results['B2'].append(v_b2)
    gold_retrieved['B2'].append(pid in [r.get('cid','') for r in fusion_results[:3]])

    # B3: long-context (embedding top-20, no rerank, full text)
    emb20 = ret.search(claim, top_k_recall=20, top_k_rerank=20, rerank=False)
    emb20_cids = [title_to_pid.get(r['title'],'') for r in emb20]
    ctx_b3 = make_context(
        [{'cid':c, 'title':corpus[c]['title'], 'text':corpus[c]['text']} for c in emb20_cids if c],
        corpus, max_docs=20)
    v_b3 = verify_claim(claim, context=ctx_b3)
    results['B3'].append(v_b3)
    gold_retrieved['B3'].append(pid in emb20_cids[:20])

    # C: oracle
    ctx_c = f'{gold_title}\n{gold_abstract}'
    v_c = verify_claim(claim, context=ctx_c)
    results['C'].append(v_c)

    # 增量保存（每 10 个）
    if (i+1) % 10 == 0:
        dt = time.time() - t_start
        # 临时统计
        def acc(key): return sum(1 for v in results[key][:i+1] if v == 'SUPPORTED') / (i+1)
        print(f'  [{i+1:3d}/{N_TEST}] ({dt:.0f}s) '
              f'A={acc("A"):.0%} B1={acc("B1"):.0%} B2={acc("B2"):.0%} '
              f'B3={acc("B3"):.0%} C={acc("C"):.0%}', flush=True)
        results_file.write_text(json.dumps({
            'n': i+1, 'results': {k: v[:i+1] for k,v in results.items()},
            'gold_retrieved': {k: v[:i+1] for k,v in gold_retrieved.items()},
        }, ensure_ascii=False, indent=2))

# ── 统计 + bootstrap CI ───────────────────────────────
print(f'\n{"="*70}', flush=True)
print(f'A/B/C 对照（arXiv 2026, n={N_TEST}, "模型不知道的语料"）', flush=True)
print(f'{"="*70}', flush=True)

def bootstrap_ci(verdicts, n_boot=10000, ci=0.95):
    """bootstrap 95% CI for SUPPORTED accuracy."""
    n = len(verdicts)
    correct = [1 if v == 'SUPPORTED' else 0 for v in verdicts]
    point = sum(correct) / n
    boots = []
    for _ in range(n_boot):
        sample = [correct[random.randint(0, n-1)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int((1-ci)/2 * n_boot)]
    hi = boots[int((1-(1-ci)/2) * n_boot)]
    return point, lo, hi

cond_names = {'A':'无检索', 'B1':'vanilla RAG', 'B2':'RAG²融合', 'B3':'long-context', 'C':'Oracle'}
print(f'\n{"条件":<16} {"准确率":>8} {"95% CI":>16} {"gold召回":>10}', flush=True)
print('-' * 55, flush=True)

for cond in ['A','B1','B2','B3','C']:
    n = len(results[cond])
    correct = sum(1 for v in results[cond] if v == 'SUPPORTED')
    point, lo, hi = bootstrap_ci(results[cond])
    gr = ''
    if cond in gold_retrieved:
        gr_f = sum(1 for g in gold_retrieved[cond] if g) / len(gold_retrieved[cond])
        gr = f'{gr_f:.0%}'
    print(f'  {cond} {cond_names[cond]:<12} {point:.0%} ({correct}/{n})  [{lo:.0%}, {hi:.0%}]  {gr:>8}', flush=True)

# 逐 claim 对比
print(f'\n逐 claim 对比（前 20）:', flush=True)
print(f'  {"#":>3} {"A":>18} {"B1":>18} {"B2":>18} {"B3":>18} {"C":>18}', flush=True)
for i in range(min(20, N_TEST)):
    row = f'  {i+1:3d}'
    for cond in ['A','B1','B2','B3','C']:
        v = results[cond][i]
        row += f' {v:>18}'
    print(row, flush=True)

# 保存最终结果
results_file.write_text(json.dumps({
    'n': N_TEST,
    'corpus': f'arXiv 2026 (cs.CL+cs.AI, {len(corpus)} papers, Jul 2026)',
    'results': results,
    'gold_retrieved': gold_retrieved,
    'accuracy': {cond: {
        'point': sum(1 for v in results[cond] if v=='SUPPORTED')/len(results[cond]),
        **dict(zip(['lo','hi'], bootstrap_ci(results[cond])[1:]))
    } for cond in results},
}, ensure_ascii=False, indent=2))
print(f'\n保存: {results_file}', flush=True)
print(f'总耗时: {time.time()-t_start:.0f}s', flush=True)
