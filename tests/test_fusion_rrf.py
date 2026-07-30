"""MAX IDF 加权 RRF：替代固定 5+5 分割。

固定分割的问题：grep 占 5 槽，把 embedding 命中的 gold 挤出（miss 23, 47）。
RRF 融合：所有文档在一个池子里按合并分排序，"同时在 emb+grep"的文档自然最高。

grep 分 = MAX(匹配词 IDF) * alpha
emb 分 = HyDE×3 RRF（不变）
合并分 = emb 分 + grep 分

调 alpha：
  alpha=0.005: MAX IDF 7.45 → 0.037（≈emb top-5，保守，少误挤）
  alpha=0.007: MAX IDF 7.45 → 0.052（≈emb top-1，平衡）
  alpha=0.010: MAX IDF 7.45 → 0.075（>emb top-1，激进，多救回但多误挤）
  alpha=0.012: MAX IDF 7.45 → 0.089（最强 grep 推力）
"""
import sys, time, json, re, math
sys.path.insert(0, 'src')
from pathlib import Path
from collections import defaultdict
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.retriever import Retriever

art = _load_scifact_artifacts()
corpus = art['corpus']
samples = load_dataset_subsampled('scifact')[:50]

all_docs = [{'_id': cid, **c} for cid, c in corpus.items()]
title_to_cid = {c['title']: cid for cid, c in corpus.items()}
corpus_texts = {d['_id']: d['text'] for d in all_docs}
N_CORPUS = len(corpus_texts)

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
    words = re.findall(r"[a-zA-Z]{4,}", claim.lower())
    terms, seen = [], set()
    for w in words:
        stem = w.rstrip('s')
        if stem in STOPWORDS or len(stem) < 3: continue
        if stem not in seen: seen.add(stem); terms.append(stem)
    return terms[:6]

def grep_term(term):
    if not term or len(term) < 3: return set()
    try: regex = re.compile(re.escape(term), re.IGNORECASE)
    except: return set()
    return {cid for cid, text in corpus_texts.items() if regex.search(text)}

ret_orig = Retriever()
ret_orig.build_index([{'title': d['title'], 'text': d['text']} for d in all_docs])
hyde_cache = json.loads(Path('cache/hyde_rewrites.json').read_text())

def emb_scores(query, top_n=20):
    rewrites = hyde_cache.get(query, [query])
    scores = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=top_n, top_k_rerank=top_n, rerank=False)):
            cid = title_to_cid.get(h['title'],'')
            if cid: scores[cid] += 1.0/(60+rank+1)
    return scores

def grep_max_idf(query):
    """返回 {cid: max_idf} —— 每个匹配文档的最稀有词 IDF。"""
    terms = extract_terms(query)
    doc_max = defaultdict(float)
    for term in terms:
        matched = grep_term(term)
        if not matched: continue
        idf = math.log(N_CORPUS / len(matched))
        for cid in matched:
            if idf > doc_max[cid]:
                doc_max[cid] = idf
    return doc_max

def eval_recall(retrieve_fn, name):
    hits = 0; total = 0; t0 = time.time()
    miss_detail = []
    for i, s in enumerate(samples):
        claim = s['metadata']['claim']
        gold_cids = set(s['metadata']['gold_corpus_ids'])
        retrieved = retrieve_fn(claim, top_k=10)
        n_hit = len(set(retrieved) & gold_cids)
        hits += n_hit; total += len(gold_cids)
        if n_hit < len(gold_cids):
            miss_detail.append((i, n_hit, len(gold_cids)))
    dt = time.time() - t0
    r = hits / max(total, 1)
    print(f'  [{name}] recall@10: {hits}/{total} = {r:.0%} ({dt:.0f}s)  miss={miss_detail}', flush=True)
    return r

print('=== MAX IDF 加权 RRF（调 alpha）===', flush=True)

def make_rrf(alpha):
    def retrieve(query, top_k=10):
        es = emb_scores(query, top_n=20)
        gs = grep_max_idf(query)
        merged = defaultdict(float)
        for cid, sc in es.items(): merged[cid] += sc
        for cid, sc in gs.items(): merged[cid] += sc * alpha
        return [c for c,_ in sorted(merged.items(), key=lambda x:-x[1])[:top_k]]
    return retrieve

for alpha in [0.005, 0.007, 0.010, 0.012]:
    r = eval_recall(make_rrf(alpha), f'RRF alpha={alpha}')

# 对照：固定 5+5（已知 91%）
def retrieve_fixed55(query, top_k=10):
    gs = grep_max_idf(query)
    grep_top = [c for c,_ in sorted(gs.items(), key=lambda x:-x[1])[:5]]
    es = emb_scores(query, top_n=15)
    grep_set = set(grep_top)
    emb_top = [c for c,_ in sorted(es.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

r_fixed = eval_recall(retrieve_fixed55, '固定 grep5+emb5 (control)')

# ── 自适应：根据最稀有词的 df 决定 grep 槽数 ──
def retrieve_adaptive(query, top_k=10):
    gs = grep_max_idf(query)
    if not gs:
        es = emb_scores(query, top_n=top_k)
        return [c for c,_ in sorted(es.items(), key=lambda x:-x[1])[:top_k]]
    # 最稀有词的 df → 决定 grep 槽数
    max_idf = max(gs.values())
    # IDF > 6 (df<8): 极稀有，grep 7 槽
    # IDF > 4 (df<95): 稀有，grep 5 槽
    # IDF > 2 (df<700): 适中，grep 3 槽
    # else: 不用 grep
    if max_idf > 6: n_grep = 7
    elif max_idf > 4: n_grep = 5
    elif max_idf > 2: n_grep = 3
    else: n_grep = 0

    if n_grep == 0:
        es = emb_scores(query, top_n=top_k)
        return [c for c,_ in sorted(es.items(), key=lambda x:-x[1])[:top_k]]

    grep_top = [c for c,_ in sorted(gs.items(), key=lambda x:-x[1])[:n_grep]]
    es = emb_scores(query, top_n=top_k + n_grep)
    grep_set = set(grep_top)
    emb_top = [c for c,_ in sorted(es.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

r_adapt = eval_recall(retrieve_adaptive, '自适应槽位(IDF分级)')

print(f'\n{"="*60}', flush=True)
print('MAX IDF RRF 全对比（n=50, 5183 篇）', flush=True)
print(f'{"="*60}', flush=True)
print(f'  baseline HyDE×3:           87%', flush=True)
print(f'  固定 grep5+emb5:           91%  (+4pt) [control]', flush=True)
print(f'  自适应槽位:                {r_adapt:.0%}  ({(r_adapt-0.87)*100:+.0f}pt)', flush=True)
