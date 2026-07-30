"""索引创新深挖：HyDE 强化 + 关系图 + HyDE 精排。

baseline 81% → HyDE 87%。本脚本测三种强化能否继续推高。
"""
import sys, time, json
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

def eval_recall(retrieve_fn, name):
    hits = 0; total = 0; t0 = time.time()
    for s in samples:
        gold_cids = set(s['metadata']['gold_corpus_ids'])
        retrieved = retrieve_fn(s['metadata']['claim'], top_k=10)
        hits += len(set(retrieved) & gold_cids); total += len(gold_cids)
    dt = time.time() - t0
    r = hits / max(total, 1)
    print(f'  [{name}] recall@10: {hits}/53 = {r:.0%} ({dt:.0f}s)', flush=True)
    return r

# ── 复用 baseline 索引 ──────────────────────────────────
ret_orig = Retriever()
ret_orig.build_index([{'title': d['title'], 'text': d['text']} for d in all_docs])

# BM25 索引
from whoosh.index import create_in, open_dir, exists_in
bm25_dir = Path('cache/bm25_index')
ix = open_dir(str(bm25_dir))
def bm25_search(query_str, top_k=20):
    from whoosh.qparser import QueryParser
    with ix.searcher() as searcher:
        try: q = QueryParser("text", ix.schema).parse(query_str)
        except: return []
        return [(r['doc_id'], r.score) for r in searcher.search(q, limit=top_k)]

# 复用 HyDE 缓存
hyde_cache = json.loads(Path('cache/hyde_rewrites.json').read_text())


# ══════════════════════════════════════════════
# 强化 1: HyDE 多角度（5 个改写）+ 改写也走 BM25
# ══════════════════════════════════════════════
print('=== 强化 1: HyDE 强化（5角度 + BM25 融合）===', flush=True)
HYDE5_TOOL = [{'type':'function','function':{'name':'rewrite','parameters':{'type':'object','properties':{'queries':{'type':'array','items':{'type':'string'}}}}}}]
hyde5_cache = {}
hyde5_file = Path('cache/hyde5_rewrites.json')
if hyde5_file.exists():
    hyde5_cache = json.loads(hyde5_file.read_text())

def hyde5_rewrite(claim):
    if claim in hyde5_cache: return hyde5_cache[claim]
    prompt = f'Rewrite this scientific claim into 5 DIFFERENT search queries to find evidence. Use different angles: the main entity, the mechanism, the outcome, related methods, opposing findings. Call rewrite tool.\n\nClaim: {claim}'
    resp = gw.generate('deepseek-v4-flash', [{'role':'user','content':prompt}],
                      tools=HYDE5_TOOL, role_tag='hyde5', max_tokens=500)
    rws = [claim]
    for tc in resp.tool_calls:
        if tc['function']['name'] == 'rewrite':
            try: rws.extend(json.loads(tc['function']['arguments']).get('queries',[]))
            except: pass
    hyde5_cache[claim] = rws
    hyde5_file.write_text(json.dumps(hyde5_cache, ensure_ascii=False))
    return rws

def retrieve_hyde5_bm25(query, top_k=10):
    rewrites = hyde5_rewrite(query)
    fused = defaultdict(float)
    for rw in rewrites:
        # embedding
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=10, top_k_rerank=10, rerank=False)):
            cid = title_to_cid.get(h['title'], '')
            if cid: fused[cid] += 1.0/(60+rank+1)
        # BM25
        for rank, (cid, sc) in enumerate(bm25_search(rw, top_k=10)):
            fused[cid] += 1.0/(60+rank+1)
    return [c for c,_ in sorted(fused.items(), key=lambda x:-x[1])[:top_k]]

r_hyde5 = eval_recall(retrieve_hyde5_bm25, 'HyDE×5 + BM25 融合')


# ══════════════════════════════════════════════
# 强化 2: 关系图多跳检索
# ══════════════════════════════════════════════
print('\n=== 强化 2: 关系图多跳检索 ===', flush=True)
# 用全量粗索引里的实体图（需先建）。这里简化：
# 先 embedding 找 top-3 种子文档，再通过共享实体扩展候选
# 加载已建的关系图（从 fine_labels 或粗索引）
# 简化版：用关键词重叠做"伪关系图"（共享专业术语的文档相关）
from collections import Counter
import re

# 提取每篇文档的关键术语（简化：高频 capitalized 词）
doc_terms = {}
for d in all_docs:
    terms = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', d['title'] + ' ' + d['text']))
    doc_terms[d['_id']] = terms

# 建倒排：term → 文档集
term_to_docs = defaultdict(set)
for cid, terms in doc_terms.items():
    for t in terms:
        term_to_docs[t].add(cid)

def retrieve_graph(query, top_k=10):
    # 种子：embedding top-5
    seed_results = ret_orig.search(query, top_k_recall=5, top_k_rerank=5, rerank=False)
    seed_cids = [title_to_cid.get(h['title'],'') for h in seed_results if title_to_cid.get(h['title'])]
    # 扩展：通过共享实体找相关文档
    fused = defaultdict(float)
    for rank, cid in enumerate(seed_cids):
        fused[cid] += 1.0/(60+rank+1)  # 种子权重高
        # 找共享实体的文档
        seed_terms = doc_terms.get(cid, set())
        for term in seed_terms:
            for related_cid in term_to_docs.get(term, set()):
                if related_cid != cid and related_cid not in [c for c,_ in sorted(fused.items(), key=lambda x:-x[1])[:5]]:
                    fused[related_cid] += 0.3/(60+1)  # 扩展权重低
    return [c for c,_ in sorted(fused.items(), key=lambda x:-x[1])[:top_k]]

r_graph = eval_recall(retrieve_graph, '关系图（共享实体扩展）')


# ══════════════════════════════════════════════
# 强化 3: HyDE(3) + LLM 精排
# ══════════════════════════════════════════════
print('\n=== 强化 3: HyDE + LLM 精排 ===', flush=True)
RERANK_TOOL = [{'type':'function','function':{'name':'rank','parameters':{'type':'object','properties':{'ids':{'type':'array','items':{'type':'string'}}}}}}]

def retrieve_hyde_rerank(query, top_k=10):
    # HyDE 粗筛 top-20
    rewrites = hyde_cache.get(query, [query])
    fused = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=10, top_k_rerank=10, rerank=False)):
            cid = title_to_cid.get(h['title'],'')
            if cid: fused[cid] += 1.0/(60+rank+1)
    candidates = [c for c,_ in sorted(fused.items(), key=lambda x:-x[1])][:20]
    if not candidates: return []
    # LLM 精排
    cand_text = '\n'.join(f'[{c}] {cid_to_title.get(c,"")[:80]}' for c in candidates)
    prompt = f'Rank these documents by relevance to the claim. Call rank tool with top 10 ids.\n\nClaim: {query}\n\n{cand_text}'
    try:
        resp = gw.generate('deepseek-v4-flash', [{'role':'user','content':prompt}],
                          tools=RERANK_TOOL, role_tag='rerank', max_tokens=300)
        for tc in resp.tool_calls:
            if tc['function']['name'] == 'rank':
                try:
                    ids = json.loads(tc['function']['arguments']).get('ids',[])[:top_k]
                    return [str(i) for i in ids]
                except: pass
    except: pass
    return candidates[:top_k]

r_hyde_rerank = eval_recall(retrieve_hyde_rerank, 'HyDE + LLM 精排')


# ══════════════════════════════════════════════
# 强化 4: 全组合（HyDE×5 + BM25 + LLM 精排）
# ══════════════════════════════════════════════
print('\n=== 强化 4: 全组合（HyDE×5 + BM25 + LLM 精排）===', flush=True)
def retrieve_combo(query, top_k=10):
    # HyDE×5 + BM25 粗筛（同强化1）拿 top-20
    rewrites = hyde5_rewrite(query)
    fused = defaultdict(float)
    for rw in rewrites:
        for rank, h in enumerate(ret_orig.search(rw, top_k_recall=10, top_k_rerank=10, rerank=False)):
            cid = title_to_cid.get(h['title'],'')
            if cid: fused[cid] += 1.0/(60+rank+1)
        for rank, (cid, sc) in enumerate(bm25_search(rw, top_k=10)):
            fused[cid] += 1.0/(60+rank+1)
    candidates = [c for c,_ in sorted(fused.items(), key=lambda x:-x[1])][:20]
    if not candidates: return []
    # LLM 精排
    cand_text = '\n'.join(f'[{c}] {cid_to_title.get(c,"")[:80]}' for c in candidates)
    prompt = f'Rank by relevance. Call rank tool with top 10 ids.\n\nClaim: {query}\n\n{cand_text}'
    try:
        resp = gw.generate('deepseek-v4-flash', [{'role':'user','content':prompt}],
                          tools=RERANK_TOOL, role_tag='combo_rerank', max_tokens=300)
        for tc in resp.tool_calls:
            if tc['function']['name'] == 'rank':
                try: return [str(i) for i in json.loads(tc['function']['arguments']).get('ids',[])[:top_k]]
                except: pass
    except: pass
    return candidates[:top_k]

r_combo = eval_recall(retrieve_combo, '全组合 HyDE×5+BM25+精排')


# ══════════════════════════════════════════════
print(f'\n{"="*55}', flush=True)
print('索引创新深挖 recall@10 对比（n=50, 5183 篇）', flush=True)
print(f'{"="*55}', flush=True)
print(f'  baseline 原文 embedding:      81%', flush=True)
print(f'  HyDE×3 改写:                  87%', flush=True)
print(f'  强化1 HyDE×5+BM25:            {r_hyde5:.0%}  ({(r_hyde5-0.81)*100:+.0f}pt)', flush=True)
print(f'  强化2 关系图扩展:              {r_graph:.0%}  ({(r_graph-0.81)*100:+.0f}pt)', flush=True)
print(f'  强化3 HyDE+LLM精排:           {r_hyde_rerank:.0%}  ({(r_hyde_rerank-0.81)*100:+.0f}pt)', flush=True)
print(f'  强化4 全组合:                  {r_combo:.0%}  ({(r_combo-0.81)*100:+.0f}pt)', flush=True)
