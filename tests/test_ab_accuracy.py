"""A/B/C 准确率对照：在 arXiv 2026"模型不知道的语料"上证明 RAG² 答得更准。

实验设计：
  1. 从 30 篇 arXiv 2026 论文摘要提取 claim
  2. LLM 改写 claim（词面不同于摘要，让检索非平凡）
  3. A（无检索）：模型凭记忆判断 SUPPORTED/REFUTED/NOT_ENOUGH_INFO
  4. B（RAG²）：融合检索 top-5 → 给上下文 → 模型判断
  5. C（oracle）：直接给 gold 论文 → 模型判断
  6. 比 A vs B vs C 准确率

核心假设：在"模型不知道的语料"上，A 大量答 NOT_ENOUGH_INFO/猜错，
        B 靠检索答对，B 准确率 >> A。
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
from rag2.data import _load_scifact_artifacts

# ── 加载 arXiv 2026 语料 ──
corpus = json.loads(Path('data/arxiv_2026_corpus.json').read_text())
print(f'arXiv 2026 语料: {len(corpus)} 篇', flush=True)

# ── 加载已提取的 claims ──
claims_data = json.loads(Path('data/arxiv_2026_claims.json').read_text())
print(f'已提取 claims: {len(claims_data)} 篇', flush=True)

# 取前 30 篇作为测试集
test_pids = list(claims_data.keys())[:30]
print(f'测试集: {len(test_pids)} 篇\n', flush=True)

gw = ModelGateway()
MODEL = 'deepseek-v4-flash'

# ── 建融合检索器 ──
all_docs = [{'title': d['title'], 'text': d['text']} for d in corpus.values()]
ret = Retriever()
ret.build_index(all_docs)
fr = FusionRetriever(retriever=ret, corpus=corpus)
_ = fr.inverted_index

# ── Step 1: LLM 改写 claim（词面不同于摘要）─────────
print('=== Step 1: LLM 改写 claim ===', flush=True)

REFORM_TOOL = [{'type':'function','function':{'name':'reformulate','parameters':{
    'type':'object','properties':{
        'claim':{'type':'string','description':'reformulated claim in different words, same meaning'}
    }}}}]

reform_file = Path('cache/arxiv_reformulated_claims.json')
reform_cache = json.loads(reform_file.read_text()) if reform_file.exists() else {}

def reformulate_claim(pid, original_claim):
    if pid in reform_cache:
        return reform_cache[pid]
    prompt = (f'Reformulate this scientific claim using DIFFERENT vocabulary and sentence structure. '
              f'Keep the same meaning but change the wording completely. Do NOT use phrases from the original. '
              f'Call reformulate tool.\n\nOriginal claim: {original_claim}')
    resp = gw.generate(MODEL, [{'role':'user','content':prompt}],
                      tools=REFORM_TOOL, role_tag='reformulate', max_tokens=500)
    new_claim = original_claim
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'reformulate':
            try:
                new_claim = json.loads(tc['function']['arguments']).get('claim', original_claim)
            except:
                pass
    reform_cache[pid] = new_claim
    reform_file.write_text(json.dumps(reform_cache, ensure_ascii=False, indent=2))
    return new_claim

reformulated = {}
for i, pid in enumerate(test_pids):
    reformulated[pid] = reformulate_claim(pid, claims_data[pid])
    if i < 3:
        print(f'  [{pid}] 原: {claims_data[pid][:70]}', flush=True)
        print(f'         改: {reformulated[pid][:70]}', flush=True)
print(f'  改写完成: {len(reformulated)} 篇\n', flush=True)

# ── Step 2: HyDE 改写（生成检索查询）────────────────
print('=== Step 2: HyDE 改写 ===', flush=True)

HYDE_TOOL = [{'type':'function','function':{'name':'rewrite','parameters':{
    'type':'object','properties':{'queries':{'type':'array','items':{'type':'string'}}}}}}]

hyde_file = Path('cache/arxiv_hyde_rewrites.json')
hyde_cache = json.loads(hyde_file.read_text()) if hyde_file.exists() else {}

def hyde_rewrite(claim):
    if claim in hyde_cache:
        return hyde_cache[claim]
    prompt = (f'Rewrite this scientific claim into 3 DIFFERENT search queries to find the source paper. '
              f'Use different angles: the method, the result, the domain. '
              f'Call rewrite tool.\n\nClaim: {claim}')
    resp = gw.generate(MODEL, [{'role':'user','content':prompt}],
                      tools=HYDE_TOOL, role_tag='hyde', max_tokens=500)
    rewrites = [claim]
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'rewrite':
            try:
                rewrites = json.loads(tc['function']['arguments']).get('queries', [claim])
            except:
                pass
    hyde_cache[claim] = rewrites
    hyde_file.write_text(json.dumps(hyde_cache, ensure_ascii=False, indent=2))
    return rewrites

# 为改写后的 claim 生成 HyDE
fr.hyde_cache = {}  # 用新的 cache
for i, pid in enumerate(test_pids):
    claim = reformulated[pid]
    rewrites = hyde_rewrite(claim)
    fr.hyde_cache[claim] = rewrites
print(f'  HyDE 改写完成: {len(fr.hyde_cache)} 篇\n', flush=True)

# ── Step 3: A/B/C 准确率对照 ──────────────────────────
print('=== Step 3: A/B/C 准确率对照 ===', flush=True)

VERIFY_SYSTEM = """You are a scientific claim verifier. Given a scientific claim, determine if it is:
- SUPPORTED: the evidence clearly supports the claim
- REFUTED: the evidence clearly contradicts the claim
- NOT_ENOUGH_INFO: insufficient evidence to determine

Call the verify tool with your verdict. Be rigorous: only SUPPORTED if the evidence directly supports the claim."""

VERIFY_TOOL = [{'type':'function','function':{'name':'verify','parameters':{
    'type':'object','properties':{
        'verdict':{'type':'string','enum':['SUPPORTED','REFUTED','NOT_ENOUGH_INFO']},
        'reason':{'type':'string'}
    },
    'required':['verdict']}}}]

def verify_claim(claim, context=None):
    """让模型验证 claim。context=None 时凭记忆，否则用检索结果。"""
    user_msg = f'Claim: {claim}'
    if context:
        user_msg += f'\n\n--- Evidence ---\n{context}\n--- End Evidence ---\n\nBased ONLY on the evidence above, verify the claim.'
    else:
        user_msg += '\n\nBased on your knowledge, verify this claim. If you do not have sufficient information, answer NOT_ENOUGH_INFO.'

    resp = gw.generate(MODEL,
        [{'role':'system','content':VERIFY_SYSTEM},
         {'role':'user','content':user_msg}],
        tools=VERIFY_TOOL, role_tag='verify', max_tokens=500)

    for tc in resp.tool_calls:
        if tc['function']['name'] == 'verify':
            try:
                return json.loads(tc['function']['arguments']).get('verdict', 'UNKNOWN')
            except:
                pass
    # 兜底：从文本解析（模型可能没调工具）
    text = (resp.text or '').upper()
    for v in ['NOT_ENOUGH_INFO', 'SUPPORTED', 'REFUTED']:
        if v in text:
            return v
    return 'UNKNOWN'

results = {'A': [], 'B': [], 'C': []}

for i, pid in enumerate(test_pids):
    claim = reformulated[pid]
    gold_abstract = corpus[pid]['text']
    gold_title = corpus[pid]['title']

    # A: 无检索（凭记忆）
    verdict_a = verify_claim(claim, context=None)

    # B: RAG² 检索（完整摘要，不截断；top-3 减噪声）
    retrieved = fr.search(claim, top_k=3, use_grep=True, use_rerank=True)
    retrieved_context = '\n\n'.join(
        f'[Paper {j+1}] {r["title"]}\n{r["text"]}'
        for j, r in enumerate(retrieved[:3])
    )
    verdict_b = verify_claim(claim, context=retrieved_context)
    gold_in_retrieved = pid in [r.get('cid','') for r in retrieved[:3]]

    # C: oracle（直接给 gold 论文）
    oracle_context = f'{gold_title}\n{gold_abstract}'
    verdict_c = verify_claim(claim, context=oracle_context)

    results['A'].append(verdict_a)
    results['B'].append(verdict_b)
    results['C'].append(verdict_c)

    status = '✓' if verdict_b == 'SUPPORTED' else '✗'
    print(f'  [{i+1:2d}/{len(test_pids)}] {status} A={verdict_a:18s} B={verdict_b:18s} C={verdict_c:18s} '
          f'gold_retrieved={gold_in_retrieved}  {gold_title[:40]}', flush=True)

# ── 统计 ──────────────────────────────────────────────
print(f'\n{"="*60}', flush=True)
print('A/B/C 准确率对照（arXiv 2026, n=30, "模型不知道的语料"）', flush=True)
print(f'{"="*60}', flush=True)

# 所有 claim 都是 SUPPORTED（从论文摘要提取的，论文支持这些 claim）
# 所以正确答案 = SUPPORTED
for cond in ['A', 'B', 'C']:
    cond_name = {'A':'无检索', 'B':'RAG²', 'C':'Oracle'}
    correct = sum(1 for v in results[cond] if v == 'SUPPORTED')
    nei = sum(1 for v in results[cond] if v == 'NOT_ENOUGH_INFO')
    wrong = sum(1 for v in results[cond] if v == 'REFUTED')
    other = sum(1 for v in results[cond] if v not in ('SUPPORTED','NOT_ENOUGH_INFO','REFUTED'))
    total = len(results[cond])
    print(f'  {cond} ({cond_name[cond]}): SUPPORTED={correct}/{total}={correct/total:.0%}  '
          f'NEI={nei}  REFUTED={wrong}  OTHER={other}', flush=True)

cond_name = {'A':'无检索', 'B':'RAG²', 'C':'Oracle'}
a_acc = sum(1 for v in results['A'] if v == 'SUPPORTED') / len(results['A'])
b_acc = sum(1 for v in results['B'] if v == 'SUPPORTED') / len(results['B'])
c_acc = sum(1 for v in results['C'] if v == 'SUPPORTED') / len(results['C'])

print(f'\n  A (无检索) 准确率:  {a_acc:.0%}', flush=True)
print(f'  B (RAG²)  准确率:  {b_acc:.0%}  (+{(b_acc-a_acc)*100:.0f}pt vs A)', flush=True)
print(f'  C (Oracle)准确率:  {c_acc:.0%}  (+{(c_acc-a_acc)*100:.0f}pt vs A)', flush=True)

if b_acc > a_acc:
    print(f'\n  ✓ RAG² 在"模型不知道的语料"上答得更准！(+{(b_acc-a_acc)*100:.0f}pt)', flush=True)
else:
    print(f'\n  ✗ RAG² 未提升准确率，需分析原因', flush=True)

# 保存结果
Path('results/ab_accuracy_arxiv2026.json').write_text(json.dumps({
    'n': len(test_pids),
    'corpus': 'arXiv 2026 (cs.CL + cs.AI, 466 papers)',
    'results': results,
    'accuracy': {'A': a_acc, 'B': b_acc, 'C': c_acc},
}, ensure_ascii=False, indent=2))
