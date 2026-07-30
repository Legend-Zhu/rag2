"""融合检索：embedding+HyDE×3 (87%) + grep 词面扫描（补 13% 盲区）

假设：embedding 漏召的 13% 是因为 gold 文档只是"顺带提及"claim 中的实体。
     grep 该实体能直接命中（词面匹配不依赖语义、不依赖词频），补上 embedding 盲区。
     与 BM25 的关键区别：BM25 按 TF-IDF 排序取 top-10，顺带提及的词频太低排不进；
     grep 返回所有匹配文档，不管词频。

验证：
  Step 1 诊断: 对每个 miss case，grep claim 实体能否找到 gold？
  Step 2 融合: embedding+HyDE×3 + grep → RRF 融合 → recall@10 目标 90%+
"""
import sys, time, json, re
sys.path.insert(0, 'src')
from pathlib import Path
from collections import defaultdict
from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.retriever import Retriever

art = _load_scifact_artifacts()
corpus = art['corpus']
samples = load_dataset_subsampled('scifact')[:50]
gw = ModelGateway()

all_docs = [{'_id': cid, **c} for cid, c in corpus.items()]
title_to_cid = {c['title']: cid for cid, c in corpus.items()}
cid_to_title = {cid: c['title'] for cid, c in corpus.items()}

# 预建文本索引（grep 用）
corpus_texts = {d['_id']: d['text'] for d in all_docs}

def eval_recall(retrieve_fn, name):
    hits = 0; total = 0; t0 = time.time()
    miss_cases = []
    for i, s in enumerate(samples):
        claim = s['metadata']['claim']
        gold_cids = set(s['metadata']['gold_corpus_ids'])
        retrieved = retrieve_fn(claim, top_k=10)
        n_hit = len(set(retrieved) & gold_cids)
        hits += n_hit; total += len(gold_cids)
        if n_hit < len(gold_cids):
            miss_cases.append((i, claim, gold_cids, retrieved, n_hit))
    dt = time.time() - t0
    r = hits / max(total, 1)
    print(f'  [{name}] recall@10: {hits}/{total} = {r:.0%} ({dt:.0f}s)', flush=True)
    return r, miss_cases

# ══════════════════════════════════════════════
# baseline: embedding + HyDE×3 (复用缓存)
# ══════════════════════════════════════════════
print('=== baseline: embedding + HyDE×3 ===', flush=True)
ret_orig = Retriever()
ret_orig.build_index([{'title': d['title'], 'text': d['text']} for d in all_docs])
hyde_cache = json.loads(Path('cache/hyde_rewrites.json').read_text())

def retrieve_hyde3(query, top_k=10):
    rewrites = hyde_cache.get(query, [query])
    fused = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=top_k, top_k_rerank=top_k, rerank=False)):
            cid = title_to_cid.get(h['title'], '')
            if cid: fused[cid] += 1.0/(60+rank+1)
    return [c for c,_ in sorted(fused.items(), key=lambda x:-x[1])[:top_k]]

r_baseline, miss_cases = eval_recall(retrieve_hyde3, 'HyDE×3 baseline')

# ══════════════════════════════════════════════
# Step 1: 诊断 — miss cases 能否被 grep 救回
# ══════════════════════════════════════════════
print(f'\n=== Step 1: 诊断 {len(miss_cases)} 个 miss case 能否被 grep 救回 ===', flush=True)

# 启发式提取关键词（不依赖 LLM，可复现）
STOPWORDS = {
    'the','a','an','and','or','but','in','on','at','to','of','for','with','from',
    'by','as','is','are','was','were','be','been','being','have','has','had','do',
    'does','did','will','would','can','could','should','may','might','must','shall',
    'this','that','these','those','it','its','they','them','their','there','here',
    'which','who','whom','whose','what','when','where','why','how','than','then',
    'so','if','because','while','during','between','within','without','about','into',
    'through','after','before','more','less','most','least','very','much','many',
    'some','any','all','both','each','other','such','same','own','new','one','two',
    'also','not','no','nor','only','just','very','too','either','neither','whether',
}

def extract_terms(claim):
    """启发式：分词 → 去停用词 → 去短词 → 取词干（strip trailing 's'）。"""
    words = re.findall(r"[a-zA-Z]{4,}", claim.lower())
    terms = []
    seen = set()
    for w in words:
        stem = w.rstrip('s')  # venules → venule, arterioles → arteriole
        if stem in STOPWORDS or len(stem) < 3:
            continue
        if stem not in seen:
            seen.add(stem)
            terms.append(stem)
    return terms[:6]  # 最多 6 个词

def grep_term(term, corpus_texts):
    """正则扫描全库，返回匹配的 cid set。"""
    if not term or len(term) < 3: return set()
    try: regex = re.compile(re.escape(term), re.IGNORECASE)
    except: return set()
    return {cid for cid, text in corpus_texts.items() if regex.search(text)}

grep_rescued = 0
grep_partial = 0
for idx, claim, gold_cids, emb_retrieved, n_hit in miss_cases:
    terms = extract_terms(claim)
    grep_cids = set()
    term_hits = {}
    for t in terms:
        hits = grep_term(t, corpus_texts)
        term_hits[t] = len(hits)
        grep_cids |= hits
    gold_in_grep = gold_cids & grep_cids
    rescued = len(gold_in_grep)
    if rescued > 0:
        grep_rescued += 1
        if rescued > n_hit: grep_partial += 1
    print(f'  miss {idx}: terms={terms} hits={term_hits} → grep {len(grep_cids)} docs, '
          f'gold found: {rescued}/{len(gold_cids)} {"✓ RESCUED" if rescued>n_hit else ""}', flush=True)

print(f'\n  grep 救回（找到 embedding 漏掉的 gold）: {grep_rescued}/{len(miss_cases)} miss cases', flush=True)

# ══════════════════════════════════════════════
# Step 2: 融合检索 — 三种策略对比
# ══════════════════════════════════════════════
import math
N_CORPUS = len(corpus_texts)
print(f'\n=== Step 2: 融合检索（3 种策略）===', flush=True)

# ── 策略 A: IDF 加权 RRF ──
# grep 每个 term，用 IDF(log(N/df)) 加权。
# 稀有词（df=3, IDF=7.45）权重大，常见词（df=2792, IDF=0.6）接近零。
# alpha=0.007: 一个稀有词匹配(IDF~6) ≈ embedding top-10 RRF(~0.043)
ALPHA = 0.007

def retrieve_fusion_idf(query, top_k=10):
    rewrites = hyde_cache.get(query, [query])
    scores = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=20, top_k_rerank=20, rerank=False)):
            cid = title_to_cid.get(h['title'],'')
            if cid: scores[cid] += 1.0/(60+rank+1)
    terms = extract_terms(query)
    for term in terms:
        matched = grep_term(term, corpus_texts)
        if not matched: continue
        idf = math.log(N_CORPUS / len(matched))
        for cid in matched:
            scores[cid] += idf * ALPHA
    return [c for c,_ in sorted(scores.items(), key=lambda x:-x[1])[:top_k]]

r_idf, _ = eval_recall(retrieve_fusion_idf, 'A: IDF 加权 RRF')

# ── 策略 B: 稀有词补集（df<100 的词才 grep，补 embedding 之外） ──
def retrieve_supplement(query, top_k=10):
    emb_results = retrieve_hyde3(query, top_k=top_k)
    emb_set = set(emb_results)
    terms = extract_terms(query)
    grep_scores = defaultdict(int)
    for term in terms:
        matched = grep_term(term, corpus_texts)
        if len(matched) >= 100: continue  # 跳过常见词
        for cid in matched:
            if cid not in emb_set:
                grep_scores[cid] += 1  # 匹配的稀有词数
    result = list(emb_results)
    for cid, sc in sorted(grep_scores.items(), key=lambda x:-x[1]):
        if len(result) >= top_k: break
        if cid not in result: result.append(cid)
    return result[:top_k]

r_supp, _ = eval_recall(retrieve_supplement, 'B: 稀有词补集(df<100)')

# ── 策略 C: 多稀有词匹配（匹配 2+ 稀有词才纳入，高精度） ──
def retrieve_multi_rare(query, top_k=10):
    rewrites = hyde_cache.get(query, [query])
    scores = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=20, top_k_rerank=20, rerank=False)):
            cid = title_to_cid.get(h['title'],'')
            if cid: scores[cid] += 1.0/(60+rank+1)
    terms = extract_terms(query)
    rare_term_matches = defaultdict(int)  # cid → 匹配的稀有词数
    for term in terms:
        matched = grep_term(term, corpus_texts)
        if len(matched) >= 100: continue  # 只看稀有词
        for cid in matched:
            rare_term_matches[cid] += 1
    # 匹配 2+ 稀有词的 doc: 给强加分（相当于 embedding top-1）
    for cid, n_rare in rare_term_matches.items():
        if n_rare >= 2:
            scores[cid] += 0.05 * n_rare  # 每个稀有词 +0.05
        else:
            scores[cid] += 0.02  # 单稀有词匹配轻量加分
    return [c for c,_ in sorted(scores.items(), key=lambda x:-x[1])[:top_k]]

r_multi, _ = eval_recall(retrieve_multi_rare, 'C: 多稀有词加权(2+强)')

# ══════════════════════════════════════════════
print(f'\n{"="*55}', flush=True)
print('融合检索 recall@10 对比（n=50, 5183 篇）', flush=True)
print(f'{"="*55}', flush=True)
print(f'  baseline HyDE×3:       {r_baseline:.0%}', flush=True)
print(f'  A IDF 加权 RRF:         {r_idf:.0%}  ({(r_idf-r_baseline)*100:+.0f}pt)', flush=True)
print(f'  B 稀有词补集(df<100):   {r_supp:.0%}  ({(r_supp-r_baseline)*100:+.0f}pt)', flush=True)
print(f'  C 多稀有词加权:         {r_multi:.0%}  ({(r_multi-r_baseline)*100:+.0f}pt)', flush=True)
print(f'\n  grep 诊断: {grep_rescued}/{len(miss_cases)} miss case 可被 grep 找到 gold', flush=True)
