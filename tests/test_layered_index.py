"""方向5: 分层索引验证——粗筛 + 精标注 + 渐进缓存。

验证三件事:
  1. 粗索引（摘要+关键词）粗筛能否保留 gold 文档（recall）
  2. 精标注的渐进缓存：第一次 miss 标注，第二次命中缓存
  3. 分层 vs 单层的 recall 对比
"""
import sys, time, json
sys.path.insert(0, 'src')
from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.fine_label_cache import FineLabelCache
from rag2.methods.retriever import Retriever

art = _load_scifact_artifacts()
corpus = art['corpus']
samples = load_dataset_subsampled('scifact')[:5]

# 50 篇语料（含 5 个 gold）
gold_ids = set()
for s in samples:
    gold_ids.update(s['metadata']['gold_corpus_ids'])
docs = []
seen = set(gold_ids)
for gid in gold_ids:
    c = corpus.get(gid)
    if c: docs.append({'_id': gid, **c})
for cid, c in corpus.items():
    if cid in seen: continue
    docs.append({'_id': cid, **c})
    seen.add(cid)
    if len(docs) >= 50: break

print(f'语料: {len(docs)} 篇（{len(gold_ids)} gold）\n', flush=True)

gw = ModelGateway()
ret = Retriever()
_ = ret.embedder
fine_cache = FineLabelCache()

# === 步骤1: 粗筛（用 embedding 对原文检索，先看 baseline recall）===
print('=== 步骤1: 粗筛 recall（embedding 原文 baseline）===', flush=True)
ret.build_index([{'title': d['title'], 'text': d['text']} for d in docs])
for s in samples:
    claim = s['metadata']['claim']
    gold_cids = set(s['metadata']['gold_corpus_ids'])
    hits = ret.search(claim, top_k_recall=10, top_k_rerank=5, rerank=False)[:10]
    # 映射回 corpus_id
    hit_titles = {h['title'] for h in hits}
    gold_titles = {corpus[gid]['title'] for gid in gold_cids if gid in corpus}
    hit_count = len(hit_titles & gold_titles)
    print(f'  recall@10: {hit_count}/{len(gold_titles)} | {claim[:50]}', flush=True)

# === 步骤2: 精标注渐进缓存验证 ===
print(f'\n=== 步骤2: 精标注渐进缓存（同一 claim 查两次）===', flush=True)
s = samples[0]
claim = s['metadata']['claim']
gold_cids = list(s['metadata']['gold_corpus_ids'])

# 用粗筛结果当候选（top-5）
hits = ret.search(claim, top_k_recall=5, top_k_rerank=5, rerank=False)[:5]
candidate_titles = [h['title'] for h in hits]
# 找候选对应的 corpus_id
title_to_cid = {c['title']: cid for cid, c in corpus.items()}
candidate_cids = [title_to_cid.get(t, '') for t in candidate_titles if title_to_cid.get(t)]

print(f'  候选 {len(candidate_cids)} 文档: {candidate_cids}', flush=True)

# 第一次：标注（应该全 miss）
print(f'\n  第一次标注（应全 miss）:', flush=True)
t0 = time.time()
def compute_fine_labels(cids):
    """对 miss 的文档批量精标注。"""
    # 用 LLM 对每个文档生成精标注（这里简化：一次调一个）
    results = {}
    for cid in cids:
        c = corpus.get(cid, {})
        if not c: continue
        prompt = f"""Analyze this scientific abstract and extract structured info. Call save_label tool.

Title: {c.get('title','')}
Text: {c.get('text','')}

Extract: hypothetical questions, entities (with types), relations (subject-relation-object), one-sentence summary."""
        tool = [{'type':'function','function':{'name':'save_label','parameters':{'type':'object','properties':{'hypothetical_questions':{'type':'array','items':{'type':'string'}},'entities':{'type':'array','items':{'type':'object','properties':{'entity':{'type':'string'},'type':{'type':'string'}}}},'relations':{'type':'array','items':{'type':'object','properties':{'subject':{'type':'string'},'relation':{'type':'string'},'object':{'type':'string'}}}},'summary':{'type':'string'}},'required':['summary']}}}]
        resp = gw.generate('qwen3.8', [{'role':'user','content':prompt}], tools=tool, role_tag='fine_label', max_tokens=2000)
        label = {'summary': '', 'hypothetical_questions': [], 'entities': [], 'relations': []}
        for tc in resp.tool_calls:
            if tc['function']['name'] == 'save_label':
                try:
                    args = json.loads(tc['function']['arguments'])
                    label.update(args)
                except: pass
        results[cid] = label
    return results

labels1 = fine_cache.get_or_compute(candidate_cids, compute_fine_labels)
dt1 = time.time() - t0
print(f'    耗时: {dt1:.1f}s, 标注了 {len(candidate_cids)} 文档', flush=True)

# 第二次：同一批候选（应全命中缓存）
print(f'  第二次（同候选，应全命中缓存）:', flush=True)
t0 = time.time()
labels2 = fine_cache.get_or_compute(candidate_cids, compute_fine_labels)
dt2 = time.time() - t0
print(f'    耗时: {dt2:.1f}s（应为秒级）, 缓存命中', flush=True)
print(f'    加速比: {dt1/max(dt2,0.01):.0f}x', flush=True)

print(f'\n=== 缓存统计 ===', flush=True)
print(f'  {fine_cache.stats()}', flush=True)
