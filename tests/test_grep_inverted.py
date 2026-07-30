"""grep 性能优化：预建倒排索引。

当前：每次 grep 都正则扫描 5183 篇文档文本（~6s/query）
优化：预建 term → doc_ids 倒排索引，grep 变查表（<0.1s）

验证：
  1. 建倒排索引耗时
  2. grep 有索引 vs 无索引耗时对比
  3. 结果一致性验证（两组 grep 结果完全相同）
"""
import sys, time, json, re
sys.path.insert(0, 'src')
from pathlib import Path
from collections import defaultdict
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts

art = _load_scifact_artifacts()
corpus = art['corpus']
samples = load_dataset_subsampled('scifact')[:50]

all_docs = [{'_id': cid, **c} for cid, c in corpus.items()]
corpus_texts = {d['_id']: d['text'] for d in all_docs}
N_CORPUS = len(corpus_texts)

STOPWORDS = frozenset({
    'the','a','an','and','or','but','in','on','at','to','of','for','with','from',
    'by','as','is','are','was','were','be','been','being','have','has','had','do',
    'does','did','will','would','can','could','should','may','might','must','shall',
    'this','that','these','those','it','its','they','them','their','there','here',
    'which','who','whom','whose','what','when','where','why','how','than','then',
    'so','if','because','while','during','between','within','without','about','into',
    'through','after','before','more','less','most','least','very','much','many',
    'some','any','all','both','each','other','such','same','own','new','one','two',
    'also','not','no','nor','only','just','very','too','either','neither','whether',
})

def extract_terms(claim):
    words = re.findall(r"[a-zA-Z]{4,}", claim.lower())
    terms, seen = [], set()
    for w in words:
        if w in STOPWORDS or len(w) < 3: continue
        if w not in seen: seen.add(w); terms.append(w)
    return terms[:6]

# ── 无索引 grep（当前方案，正则全库扫描）───────────────
def grep_term_regex(term, corpus_texts):
    if not term or len(term) < 3: return set()
    try: regex = re.compile(re.escape(term), re.IGNORECASE)
    except: return set()
    return {cid for cid, text in corpus_texts.items() if regex.search(text)}

# ── 预建倒排索引 ─────────────────────────────────────
def build_inverted_index(corpus_texts):
    """term → set(cid) 倒排索引。存原始词形（小写），不词干化。"""
    index = defaultdict(set)
    for cid, text in corpus_texts.items():
        words = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
        for w in words:
            index[w].add(cid)
    return dict(index)

def grep_term_inverted(term, inverted):
    """查倒排索引。子串匹配（等效 regex）+ 单数回退。

    "body" 匹配 "body", "bodies", "antibody"（子串）
    "venule" 匹配 "venule", "venules", "venulectomy"
    "venules" 也匹配 "venule"（单数回退）
    """
    if not term or len(term) < 3: return set()
    matched = set()
    for w, cids in inverted.items():
        if term in w:
            matched |= cids
    # 单数回退：去掉末尾 's' 再查（"venules" → "venule"）
    if term.endswith('s') and len(term) > 3:
        singular = term[:-1]
        for w, cids in inverted.items():
            if singular in w:
                matched |= cids
    return matched

print(f'=== 预建倒排索引（{N_CORPUS} 篇）===', flush=True)
t0 = time.time()
inverted = build_inverted_index(corpus_texts)
build_time = time.time() - t0
print(f'  建索引耗时: {build_time:.1f}s, {len(inverted)} 个词', flush=True)

# 保存索引
idx_file = Path('cache/grep_inverted_index.json')
t0 = time.time()
idx_file.write_text(json.dumps({k: sorted(v) for k, v in inverted.items()}, ensure_ascii=False))
save_time = time.time() - t0
print(f'  保存索引: {idx_file.stat().st_size/1024/1024:.1f}MB, {save_time:.1f}s', flush=True)

# 加载索引
t0 = time.time()
loaded = json.loads(idx_file.read_text())
loaded = {k: set(v) for k, v in loaded.items()}
load_time = time.time() - t0
print(f'  加载索引: {load_time:.1f}s\n', flush=True)

# ── 性能对比 + 一致性验证 ────────────────────────────
print('=== 性能对比 + 一致性验证（50 claims × 6 terms）===\n', flush=True)

# 收集所有要 grep 的 term
all_queries = []
for s in samples:
    claim = s['metadata']['claim']
    all_queries.append(claim)

# 无索引 grep
t0 = time.time()
regex_results = {}
for claim in all_queries:
    terms = extract_terms(claim)
    for term in terms:
        regex_results[(claim, term)] = grep_term_regex(term, corpus_texts)
regex_time = time.time() - t0

# 有索引 grep
t0 = time.time()
inverted_results = {}
for claim in all_queries:
    terms = extract_terms(claim)
    for term in terms:
        inverted_results[(claim, term)] = grep_term_inverted(term, loaded)
inverted_time = time.time() - t0

# 一致性验证
n_terms = len(regex_results)
exact_match = 0
superset = 0  # inverted 结果 ⊇ regex 结果（倒排索引可能多匹配，但不该少）
missing = 0   # inverted 结果 ⊄ regex 结果（丢了文档，不可接受）
for key in regex_results:
    r_set = regex_results[key]
    i_set = inverted_results[key]
    if r_set == i_set:
        exact_match += 1
    elif r_set <= i_set:
        superset += 1  # 倒排多匹配了（前缀匹配），可以接受
    else:
        missing += 1
        if missing <= 3:
            diff = r_set - i_set
            print(f'  不一致: {key[1]} regex={len(r_set)} inverted={len(i_set)} 丢了 {len(diff)} 个', flush=True)

print(f'  无索引 grep（正则扫描）: {regex_time:.1f}s', flush=True)
print(f'  有索引 grep（倒排查表）: {inverted_time:.1f}s', flush=True)
print(f'  加速比: {regex_time/max(inverted_time,0.001):.0f}x', flush=True)
print(f'\n  一致性: {exact_match}/{n_terms} 完全相同, {superset} 超集, {missing} 丢失', flush=True)

# 预估查询延迟
avg_terms = sum(len(extract_terms(c)) for c in all_queries) / len(all_queries)
per_query_regex = regex_time / len(all_queries)
per_query_inverted = inverted_time / len(all_queries)
print(f'\n  每 query 平均 {avg_terms:.1f} 个 term', flush=True)
print(f'  每 query grep 耗时: 无索引 {per_query_regex:.2f}s → 有索引 {per_query_inverted:.3f}s', flush=True)
