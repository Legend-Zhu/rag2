"""融合检索调试 + 新策略。

关键问题：诊断证明 grep 在 7/7 miss case 找到 gold，但 3 种融合策略都 87% 不变。
原因猜测：常见词匹配的文档也叠加 IDF 分，把 gold 挤出 top-10。
本脚本：逐 case 调试 gold 排名 + 尝试 grep-first 策略。
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
    terms = []
    seen = set()
    for w in words:
        stem = w.rstrip('s')
        if stem in STOPWORDS or len(stem) < 3: continue
        if stem not in seen:
            seen.add(stem)
            terms.append(stem)
    return terms[:6]

def grep_term(term, corpus_texts):
    if not term or len(term) < 3: return set()
    try: regex = re.compile(re.escape(term), re.IGNORECASE)
    except: return set()
    return {cid for cid, text in corpus_texts.items() if regex.search(text)}

# ── 复用 baseline ──
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

# ══════════════════════════════════════════════
# 调试：逐 miss case 看 gold 在 IDF 融合里的排名
# ══════════════════════════════════════════════
ALPHA = 0.007

print('=== 调试：miss case 的 gold 在 IDF 融合里的排名 ===\n', flush=True)

for i, s in enumerate(samples):
    claim = s['metadata']['claim']
    gold_cids = set(s['metadata']['gold_corpus_ids'])
    emb_top10 = retrieve_hyde3(claim, top_k=10)
    if gold_cids & set(emb_top10): continue  # 只看 miss case

    # IDF 融合分数
    rewrites = hyde_cache.get(claim, [claim])
    scores = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=20, top_k_rerank=20, rerank=False)):
            cid = title_to_cid.get(h['title'],'')
            if cid: scores[cid] += 1.0/(60+rank+1)
    terms = extract_terms(claim)
    term_dfs = {}
    for term in terms:
        matched = grep_term(term, corpus_texts)
        if not matched: continue
        idf = math.log(N_CORPUS / len(matched))
        term_dfs[term] = (len(matched), idf)
        for cid in matched:
            scores[cid] += idf * ALPHA

    sorted_docs = sorted(scores.items(), key=lambda x:-x[1])
    top10 = [c for c,_ in sorted_docs[:10]]

    for gc in gold_cids:
        rank = next((r+1 for r, (c,_) in enumerate(sorted_docs) if c == gc), -1)
        gold_score = scores.get(gc, 0)
        emb_score = sum(1.0/(60+r+1) for rw in rewrites for r, h in
                        enumerate(ret_orig.search(rw, top_k_recall=20, top_k_rerank=20, rerank=False))
                        if title_to_cid.get(h['title']) == gc)
        grep_score = gold_score - emb_score
        print(f'miss {i}: gold={gc} rank={rank}/{len(sorted_docs)} score={gold_score:.4f} '
              f'(emb={emb_score:.4f} grep={grep_score:.4f})', flush=True)
        print(f'  terms: {[(t, df, f"{idf:.2f}") for t,(df,idf) in term_dfs.items()]}', flush=True)
        if rank > 10 or rank == -1:
            # 打印 top-10 和 gold 周围的文档
            start = max(0, rank-3) if rank > 0 else 0
            for r, (c, sc) in enumerate(sorted_docs[start:start+13], start):
                marker = ' <<<' if c in gold_cids else ''
                print(f'    rank {r+1}: {c} score={sc:.4f}{marker}', flush=True)
        print()

# ══════════════════════════════════════════════
# 新策略：grep-first（稀有词 IDF 排序取 top-N + embedding 补齐）
# ══════════════════════════════════════════════
print('=== 新策略 ===', flush=True)

def eval_recall(retrieve_fn, name):
    hits = 0; total = 0; t0 = time.time()
    for s in samples:
        claim = s['metadata']['claim']
        gold_cids = set(s['metadata']['gold_corpus_ids'])
        retrieved = retrieve_fn(claim, top_k=10)
        hits += len(set(retrieved) & gold_cids); total += len(gold_cids)
    dt = time.time() - t0
    r = hits / max(total, 1)
    print(f'  [{name}] recall@10: {hits}/{total} = {r:.0%} ({dt:.0f}s)', flush=True)
    return r

# 策略 D: grep-first 5 + embedding 5
# grep 所有词按 IDF 排序，取 top-5 最具区分度的文档，embedding 补 5
def retrieve_grep_first(query, top_k=10):
    terms = extract_terms(query)
    grep_scores = defaultdict(float)
    for term in terms:
        matched = grep_term(term, corpus_texts)
        if not matched: continue
        idf = math.log(N_CORPUS / len(matched))
        for cid in matched:
            grep_scores[cid] += idf  # 纯 IDF 累加，不乘 alpha
    grep_top = [c for c,_ in sorted(grep_scores.items(), key=lambda x:-x[1])[:5]]
    # embedding 补齐
    rewrites = hyde_cache.get(query, [query])
    emb_scores = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=15, top_k_rerank=15, rerank=False)):
            cid = title_to_cid.get(h['title'],'')
            if cid: emb_scores[cid] += 1.0/(60+rank+1)
    grep_set = set(grep_top)
    emb_top = [c for c,_ in sorted(emb_scores.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

r_d = eval_recall(retrieve_grep_first, 'D: grep-first5 + emb5')

# 策略 E: grep-first 3 + embedding 7（更保守，减少对 embedding 的干扰）
def retrieve_grep_first3(query, top_k=10):
    terms = extract_terms(query)
    grep_scores = defaultdict(float)
    for term in terms:
        matched = grep_term(term, corpus_texts)
        if not matched: continue
        idf = math.log(N_CORPUS / len(matched))
        for cid in matched:
            grep_scores[cid] += idf
    grep_top = [c for c,_ in sorted(grep_scores.items(), key=lambda x:-x[1])[:3]]
    rewrites = hyde_cache.get(query, [query])
    emb_scores = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=15, top_k_rerank=15, rerank=False)):
            cid = title_to_cid.get(h['title'],'')
            if cid: emb_scores[cid] += 1.0/(60+rank+1)
    grep_set = set(grep_top)
    emb_top = [c for c,_ in sorted(emb_scores.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

r_e = eval_recall(retrieve_grep_first3, 'E: grep-first3 + emb7')

# 策略 F: 自适应——只在有高 IDF 稀有词时用 grep
# 如果某词 df<20（极稀有），grep 命中的 doc 直接进 top-3，否则全用 embedding
def retrieve_adaptive(query, top_k=10):
    terms = extract_terms(query)
    rare_docs = defaultdict(int)  # cid → 匹配的极稀有词数
    for term in terms:
        matched = grep_term(term, corpus_texts)
        if len(matched) > 0 and len(matched) <= 20:  # 极稀有词
            for cid in matched:
                rare_docs[cid] += 1
    # 匹配 2+ 极稀有词的 doc → 高置信，直接进 top
    confident = [c for c, n in sorted(rare_docs.items(), key=lambda x:-x[1]) if n >= 2][:3]
    # 也加上匹配 1 个极稀有词的 top-2
    single_rare = [c for c, n in sorted(rare_docs.items(), key=lambda x:-x[1]) if n == 1][:2]
    grep_top = confident + [c for c in single_rare if c not in confident]
    # embedding 补齐
    rewrites = hyde_cache.get(query, [query])
    emb_scores = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=15, top_k_rerank=15, rerank=False)):
            cid = title_to_cid.get(h['title'],'')
            if cid: emb_scores[cid] += 1.0/(60+rank+1)
    grep_set = set(grep_top)
    emb_top = [c for c,_ in sorted(emb_scores.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

r_f = eval_recall(retrieve_adaptive, 'F: 自适应(极稀有词 df≤20)')

print(f'\n{"="*55}', flush=True)
print('融合策略 recall@10 对比（n=50, 5183 篇）', flush=True)
print(f'{"="*55}', flush=True)
print(f'  baseline HyDE×3:           87%', flush=True)
print(f'  D grep-first5 + emb5:      {r_d:.0%}  ({(r_d-0.87)*100:+.0f}pt)', flush=True)
print(f'  E grep-first3 + emb7:      {r_e:.0%}  ({(r_e-0.87)*100:+.0f}pt)', flush=True)
print(f'  F 自适应(极稀有词):        {r_f:.0%}  ({(r_f-0.87)*100:+.0f}pt)', flush=True)
