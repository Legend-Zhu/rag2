"""交叉编码器重排——直接用 CrossEncoder，不走 _predict_silenced。

修正：之前 _predict_silenced 的 os.dup2 在后台进程死锁。
现在直接 ce.predict，加逐 query 进度输出。
"""
import sys, time, json, re, math, os
sys.path.insert(0, 'src')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pathlib import Path
from collections import defaultdict
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.retriever import Retriever
from sentence_transformers import CrossEncoder

art = _load_scifact_artifacts()
corpus = art['corpus']
samples = load_dataset_subsampled('scifact')[:50]

all_docs = [{'_id': cid, **c} for cid, c in corpus.items()]
title_to_cid = {c['title']: cid for cid, c in corpus.items()}
cid_to_title = {cid: c['title'] for cid, c in corpus.items()}
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

def grep_max_idf(query):
    terms = extract_terms(query)
    doc_max = defaultdict(float)
    for term in terms:
        matched = grep_term(term)
        if not matched: continue
        idf = math.log(N_CORPUS / len(matched))
        for cid in matched:
            if idf > doc_max[cid]: doc_max[cid] = idf
    return doc_max

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

# 直接加载 CrossEncoder
print('加载 CrossEncoder...', flush=True)
t0 = time.time()
ce = CrossEncoder('BAAI/bge-reranker-v2-m3', device='mps')
print(f'  加载完成: {time.time()-t0:.1f}s\n', flush=True)

def retrieve_fusion_rerank(query, top_k=10):
    es = emb_scores(query, top_n=20)
    emb_cids = [c for c, _ in sorted(es.items(), key=lambda x:-x[1])[:20]]
    gs = grep_max_idf(query)
    grep_cids = [c for c, _ in sorted(gs.items(), key=lambda x:-x[1])[:10]]
    pool = list(dict.fromkeys(emb_cids + grep_cids))
    pairs = [[query, f"{cid_to_title.get(c,'')}: {corpus_texts.get(c,'')}"] for c in pool]
    scores = ce.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(pool, [float(s) for s in scores]), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_k]]

def retrieve_emb_rerank(query, top_k=10):
    es = emb_scores(query, top_n=20)
    pool = [c for c, _ in sorted(es.items(), key=lambda x:-x[1])[:20]]
    pairs = [[query, f"{cid_to_title.get(c,'')}: {corpus_texts.get(c,'')}"] for c in pool]
    scores = ce.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(pool, [float(s) for s in scores]), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_k]]

# 逐 query 运行 + 进度
print('=== 交叉编码器重排 ===', flush=True)
for name, fn in [('emb20+grep10 → rerank', retrieve_fusion_rerank),
                  ('emb20 → rerank', retrieve_emb_rerank)]:
    hits = 0; total = 0; t0 = time.time()
    miss_detail = []
    for i, s in enumerate(samples):
        claim = s['metadata']['claim']
        gold_cids = set(s['metadata']['gold_corpus_ids'])
        retrieved = fn(claim, top_k=10)
        n_hit = len(set(retrieved) & gold_cids)
        hits += n_hit; total += len(gold_cids)
        if n_hit < len(gold_cids):
            miss_detail.append((i, n_hit, len(gold_cids)))
        if i % 10 == 9:
            print(f'  {name}: {i+1}/50 done, running recall={hits}/{total}', flush=True)
    dt = time.time() - t0
    r = hits / max(total, 1)
    print(f'  [{name}] recall@10: {hits}/{total} = {r:.0%} ({dt:.0f}s)  miss={miss_detail}', flush=True)

print(f'\n{"="*60}', flush=True)
print('交叉编码器重排 recall@10（n=50, 5183 篇）', flush=True)
print(f'{"="*60}', flush=True)
print(f'  baseline HyDE×3:           87%', flush=True)
print(f'  MAX IDF grep5+emb5:         91%  (+4pt)', flush=True)
print(f'  emb20+grep10 → CrossEncoder:  见上', flush=True)
print(f'  emb20 → CrossEncoder:         见上', flush=True)
