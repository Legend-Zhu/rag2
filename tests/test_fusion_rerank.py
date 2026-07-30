"""交叉编码器重排合并池：embedding + grep 候选 → reranker 语义重排 → top-10。

之前所有实验 rerank=False。现在：
  1. embedding top-20（语义召回）
  2. grep MAX IDF top-10（词面补盲区）
  3. 合并池（~25-30 篇）→ bge-reranker 交叉编码器重排
  4. 取 top-10

交叉编码器能语义区分"ITAM 磷酸化阻止转移"(gold) vs 其他 ITAM 文档，
解决 miss 14（gold 在池子里但排名低）。同时重排避免固定分配误挤（miss 23, 47）。
"""
import sys, time, json, re, math, os
sys.path.insert(0, 'src')
os.environ['TQDM_DISABLE'] = '1'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pathlib import Path
from collections import defaultdict
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.retriever import Retriever

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

# ── 交叉编码器重排合并池 ──
def retrieve_fusion_rerank(query, top_k=10):
    # 1. embedding top-20
    es = emb_scores(query, top_n=20)
    emb_cids = [c for c, _ in sorted(es.items(), key=lambda x:-x[1])[:20]]

    # 2. grep MAX IDF top-10
    gs = grep_max_idf(query)
    grep_cids = [c for c, _ in sorted(gs.items(), key=lambda x:-x[1])[:10]]

    # 3. 合并去重
    pool = list(dict.fromkeys(emb_cids + grep_cids))  # 保序去重

    # 4. 交叉编码器重排
    pairs = [[query, f"{cid_to_title.get(c,'')}: {corpus_texts.get(c,'')}"] for c in pool]
    scores = ret_orig._predict_silenced(pairs)
    ranked = sorted(zip(pool, scores), key=lambda x: -float(x[1]))
    return [c for c, _ in ranked[:top_k]]

print('=== 交叉编码器重排合并池 ===', flush=True)
print('(加载 reranker 首次调用会慢，后续快)', flush=True)
r_rerank = eval_recall(retrieve_fusion_rerank, 'emb20+grep10 → reranker')

# 对照组：纯 embedding rerank（不用 grep）
def retrieve_emb_rerank(query, top_k=10):
    es = emb_scores(query, top_n=20)
    pool = [c for c, _ in sorted(es.items(), key=lambda x:-x[1])[:20]]
    pairs = [[query, f"{cid_to_title.get(c,'')}: {corpus_texts.get(c,'')}"] for c in pool]
    scores = ret_orig._predict_silenced(pairs)
    ranked = sorted(zip(pool, scores), key=lambda x: -float(x[1]))
    return [c for c, _ in ranked[:top_k]]

r_emb_rr = eval_recall(retrieve_emb_rerank, 'emb20 → reranker (no grep)')

# 对照组：MAX IDF grep5 + emb5（已知 91%）
def retrieve_fixed55(query, top_k=10):
    gs = grep_max_idf(query)
    grep_top = [c for c, _ in sorted(gs.items(), key=lambda x:-x[1])[:5]]
    es = emb_scores(query, top_n=15)
    grep_set = set(grep_top)
    emb_top = [c for c, _ in sorted(es.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

r_fixed = eval_recall(retrieve_fixed55, 'grep5+emb5 (control)')

print(f'\n{"="*60}', flush=True)
print('交叉编码器重排 recall@10 对比（n=50, 5183 篇）', flush=True)
print(f'{"="*60}', flush=True)
print(f'  baseline HyDE×3:           87%', flush=True)
print(f'  grep5+emb5 (无重排):       91%  (+4pt)', flush=True)
print(f'  emb20 → reranker:          {r_emb_rr:.0%}  ({(r_emb_rr-0.87)*100:+.0f}pt)', flush=True)
print(f'  emb20+grep10 → reranker:   {r_rerank:.0%}  ({(r_rerank-0.87)*100:+.0f}pt)', flush=True)
