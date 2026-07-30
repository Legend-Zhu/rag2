"""MAX IDF grep 排序：优先最稀有词匹配的文档。

SUM IDF 的问题：匹配 3 个常见词(各 IDF~2) = 6 > 匹配 1 个稀有词(IDF=7)。
gold 只顺带匹配 1 个稀有词，被匹配多常见词的文档淹没。

MAX IDF：取匹配词中最高的 IDF 作为主分。匹配 "venule"(IDF=7.45) 的文档
排第一，不管它还匹配多少常见词。

变体：
  G: MAX IDF grep-first5 + emb5
  H: MAX IDF + 0.3*second_max（奖励也匹配第二稀有词）
  I: 只 grep 最稀有 2 词(df<50) + MAX IDF 排序 + emb 补齐
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

def get_term_idf(term):
    matched = grep_term(term, corpus_texts)
    if not matched: return 0, set()
    return math.log(N_CORPUS / len(matched)), matched

ret_orig = Retriever()
ret_orig.build_index([{'title': d['title'], 'text': d['text']} for d in all_docs])
hyde_cache = json.loads(Path('cache/hyde_rewrites.json').read_text())

def emb_scores(query, top_n=15):
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
    print(f'  [{name}] recall@10: {hits}/{total} = {r:.0%} ({dt:.0f}s)', flush=True)
    if miss_detail:
        print(f'    miss: {miss_detail}', flush=True)
    return r

# ══════════════════════════════════════════════
# G: MAX IDF grep-first5 + emb5
# ══════════════════════════════════════════════
def retrieve_max_idf(query, top_k=10):
    terms = extract_terms(query)
    # 计算每个 term 的 IDF 和匹配集
    term_info = []
    for term in terms:
        idf, matched = get_term_idf(term)
        if idf > 0:
            term_info.append((term, idf, matched))

    # 对每个匹配的 doc，取 MAX IDF 作为主分
    doc_max_idf = defaultdict(float)
    doc_second_idf = defaultdict(float)
    for term, idf, matched in term_info:
        for cid in matched:
            if idf > doc_max_idf[cid]:
                doc_second_idf[cid] = doc_max_idf[cid]
                doc_max_idf[cid] = idf
            elif idf > doc_second_idf[cid]:
                doc_second_idf[cid] = idf

    # 排序：MAX IDF 降序
    grep_ranked = sorted(doc_max_idf.items(), key=lambda x: -x[1])
    grep_top = [c for c, _ in grep_ranked[:5]]

    # embedding 补齐
    es = emb_scores(query, top_n=15)
    grep_set = set(grep_top)
    emb_top = [c for c, _ in sorted(es.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

print('=== MAX IDF 策略 ===', flush=True)
r_g = eval_recall(retrieve_max_idf, 'G: MAX IDF grep5 + emb5')

# ══════════════════════════════════════════════
# H: MAX + 0.3*second_max（奖励也匹配第二稀有词）
# ══════════════════════════════════════════════
def retrieve_max_second(query, top_k=10):
    terms = extract_terms(query)
    term_info = []
    for term in terms:
        idf, matched = get_term_idf(term)
        if idf > 0:
            term_info.append((term, idf, matched))

    doc_score = defaultdict(float)
    for term, idf, matched in term_info:
        for cid in matched:
            # 暂存所有匹配的 IDF
            doc_score[cid]  # 确保 key 存在
            if not hasattr(doc_score, '_terms'):
                pass

    # 重新计算：每 doc 的所有匹配 IDF 列表
    doc_idfs = defaultdict(list)
    for term, idf, matched in term_info:
        for cid in matched:
            doc_idfs[cid].append(idf)

    # score = max + 0.3 * second_max
    doc_final = {}
    for cid, idfs in doc_idfs.items():
        idfs_sorted = sorted(idfs, reverse=True)
        max_idf = idfs_sorted[0]
        second = idfs_sorted[1] if len(idfs_sorted) > 1 else 0
        doc_final[cid] = max_idf + 0.3 * second

    grep_ranked = sorted(doc_final.items(), key=lambda x: -x[1])
    grep_top = [c for c, _ in grep_ranked[:5]]

    es = emb_scores(query, top_n=15)
    grep_set = set(grep_top)
    emb_top = [c for c, _ in sorted(es.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

r_h = eval_recall(retrieve_max_second, 'H: MAX+0.3*2nd grep5 + emb5')

# ══════════════════════════════════════════════
# I: 只 grep 最稀有 2 词(df<50) + MAX IDF + emb 补齐
# ══════════════════════════════════════════════
def retrieve_rare2(query, top_k=10):
    terms = extract_terms(query)
    # 计算 IDF，只保留 df<50 的稀有词
    term_info = []
    for term in terms:
        idf, matched = get_term_idf(term)
        if idf > 0 and len(matched) < 50:
            term_info.append((term, idf, matched))

    # 按 IDF 降序，取最稀有 2 词
    term_info.sort(key=lambda x: -x[1])
    term_info = term_info[:2]

    if not term_info:
        # 无稀有词，纯 embedding
        es = emb_scores(query, top_n=top_k)
        return [c for c, _ in sorted(es.items(), key=lambda x:-x[1])[:top_k]]

    # 合并最稀有 2 词的匹配集，按 MAX IDF 排序
    doc_max = defaultdict(float)
    doc_second = defaultdict(float)
    for term, idf, matched in term_info:
        for cid in matched:
            if idf > doc_max[cid]:
                doc_second[cid] = doc_max[cid]
                doc_max[cid] = idf
            elif idf > doc_second[cid]:
                doc_second[cid] = idf

    doc_final = {cid: doc_max[cid] + 0.3 * doc_second[cid] for cid in doc_max}
    grep_ranked = sorted(doc_final.items(), key=lambda x: -x[1])
    grep_top = [c for c, _ in grep_ranked[:5]]

    es = emb_scores(query, top_n=15)
    grep_set = set(grep_top)
    emb_top = [c for c, _ in sorted(es.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

r_i = eval_recall(retrieve_rare2, 'I: 稀有2词(df<50) MAX+2nd grep5')

# ══════════════════════════════════════════════
# J: MAX IDF grep-first3 + emb7（减少对 embedding 的干扰）
# ══════════════════════════════════════════════
def retrieve_max_idf3(query, top_k=10):
    terms = extract_terms(query)
    term_info = []
    for term in terms:
        idf, matched = get_term_idf(term)
        if idf > 0:
            term_info.append((term, idf, matched))

    doc_max = defaultdict(float)
    doc_second = defaultdict(float)
    for term, idf, matched in term_info:
        for cid in matched:
            if idf > doc_max[cid]:
                doc_second[cid] = doc_max[cid]
                doc_max[cid] = idf
            elif idf > doc_second[cid]:
                doc_second[cid] = idf

    doc_final = {cid: doc_max[cid] + 0.3 * doc_second[cid] for cid in doc_max}
    grep_ranked = sorted(doc_final.items(), key=lambda x: -x[1])
    grep_top = [c for c, _ in grep_ranked[:3]]

    es = emb_scores(query, top_n=15)
    grep_set = set(grep_top)
    emb_top = [c for c, _ in sorted(es.items(), key=lambda x:-x[1]) if c not in grep_set]
    return grep_top + emb_top[:top_k-len(grep_top)]

r_j = eval_recall(retrieve_max_idf3, 'J: MAX+2nd grep3 + emb7')

print(f'\n{"="*55}', flush=True)
print('MAX IDF 策略 recall@10 对比（n=50, 5183 篇）', flush=True)
print(f'{"="*55}', flush=True)
print(f'  baseline HyDE×3:           87%', flush=True)
print(f'  D SUM IDF grep5 + emb5:    89%  (+2pt)', flush=True)
print(f'  G MAX IDF grep5 + emb5:    {r_g:.0%}  ({(r_g-0.87)*100:+.0f}pt)', flush=True)
print(f'  H MAX+0.3*2nd grep5:       {r_h:.0%}  ({(r_h-0.87)*100:+.0f}pt)', flush=True)
print(f'  I 稀有2词(df<50) MAX:      {r_i:.0%}  ({(r_i-0.87)*100:+.0f}pt)', flush=True)
print(f'  J MAX+2nd grep3 + emb7:    {r_j:.0%}  ({(r_j-0.87)*100:+.0f}pt)', flush=True)
