"""全量 SciFact 粗索引（5183 篇）+ recall 验证 + BM25 baseline 对比。

用 deepseek-v4-flash + 10 并发，验证搜索引擎级索引在大规模上的可行性。
"""
import sys, time, json, concurrent.futures
sys.path.insert(0, 'src')
from pathlib import Path
from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.retriever import Retriever

art = _load_scifact_artifacts()
corpus = art['corpus']
samples = load_dataset_subsampled('scifact')[:50]  # 50 个 claim 测 recall

# 准备全量语料（5183 篇）
all_docs = [{'_id': cid, 'title': c['title'], 'text': c['text']}
            for cid, c in corpus.items()]
print(f'全量语料: {len(all_docs)} 篇, 评测 {len(samples)} claim\n', flush=True)

gw = ModelGateway()
COARSE_BATCH = 100  # 一次处理 100 篇
MAX_WORKERS = 8     # 并发数

# === 1. 全量粗索引（deepseek + 并发）===
print(f'=== 1. 全量粗索引（{len(all_docs)} 篇, batch={COARSE_BATCH}, 并发={MAX_WORKERS}）===', flush=True)

cache_file = Path('cache/full_coarse_index.json')
if cache_file.exists():
    coarse = json.loads(cache_file.read_text())
    print(f'命中缓存: {len(coarse)} 篇粗索引', flush=True)
else:
    coarse = {}
    # 切 batch
    batches = []
    for i in range(0, len(all_docs), COARSE_BATCH):
        batches.append(all_docs[i:i+COARSE_BATCH])
    print(f'{len(batches)} batch', flush=True)

    COARSE_TOOL = [{'type':'function','function':{'name':'save_coarse_index','parameters':{'type':'object','properties':{'entries':{'type':'array','items':{'type':'object','properties':{'doc_index':{'type':'integer'},'summary':{'type':'string'},'keywords':{'type':'array','items':{'type':'string'}}}}}}}}}]

    def process_batch(batch_idx, batch):
        """处理一个 batch，返回 {corpus_id: {summary, keywords}}。"""
        doc_strs = [f'[{i}] {d["title"][:60]}\n{d["text"][:600]}' for i, d in enumerate(batch)]
        prompt = f'For each document, generate one-sentence summary + 3-5 keywords. Call save_coarse_index.\n\n' + '\n\n'.join(doc_strs)
        try:
            resp = gw.generate('deepseek-v4-flash', [{'role':'user','content':prompt}],
                              tools=COARSE_TOOL, role_tag=f'coarse_{batch_idx}', max_tokens=8000)
            result = {}
            for tc in resp.tool_calls:
                if tc['function']['name'] == 'save_coarse_index':
                    try:
                        args = json.loads(tc['function']['arguments'])
                        for e in args.get('entries', []):
                            idx = e.get('doc_index', -1)
                            if 0 <= idx < len(batch):
                                cid = batch[idx]['_id']
                                result[cid] = {'summary': e.get('summary',''), 'keywords': e.get('keywords',[])}
                    except: pass
            return batch_idx, result
        except Exception as e:
            print(f'  batch {batch_idx} 失败: {str(e)[:80]}', flush=True)
            return batch_idx, {}

    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_batch, i, b): i for i, b in enumerate(batches)}
        done = 0
        for f in concurrent.futures.as_completed(futures):
            idx, result = f.result()
            coarse.update(result)
            done += 1
            if done % 10 == 0:
                elapsed = time.time() - t0
                print(f'  进度 {done}/{len(batches)} batch, {len(coarse)} 文档, {elapsed:.0f}s', flush=True)

    print(f'\n粗索引完成: {len(coarse)}/{len(all_docs)} 文档, 耗时 {time.time()-t0:.0f}s', flush=True)
    cache_file.write_text(json.dumps(coarse, ensure_ascii=False))

# === 2. BM25 baseline recall ===
print(f'\n=== 2. BM25 baseline recall（全文检索）===', flush=True)
ret = Retriever()
ret.build_index([{'title': d['title'], 'text': d['text']} for d in all_docs])

bm25_hits = 0
bm25_total = 0
for s in samples:
    claim = s['metadata']['claim']
    gold_cids = set(s['metadata']['gold_corpus_ids'])
    hits = ret.search(claim, top_k_recall=10, top_k_rerank=10, rerank=False)[:10]
    hit_titles = {h['title'] for h in hits}
    gold_titles = {corpus[gid]['title'] for gid in gold_cids if gid in corpus}
    n_hit = len(hit_titles & gold_titles)
    bm25_hits += n_hit
    bm25_total += len(gold_titles)
print(f'BM25 recall@10: {bm25_hits}/{bm25_total} = {bm25_hits/max(bm25_total,1):.0%}', flush=True)

# === 3. 粗索引 recall（LLM 判断）===
print(f'\n=== 3. 粗索引 LLM 判断 recall ===', flush=True)
# 构造粗索引文档列表（id + summary + keywords）
coarse_list = []
for cid, c in coarse.items():
    coarse_list.append({'_id': cid, 'summary': c['summary'], 'keywords': c['keywords']})

JUDGE_TOOL = [{'type':'function','function':{'name':'select','parameters':{'type':'object','properties':{'ids':{'type':'array','items':{'type':'string'}}}}}}]

coarse_hits = 0
coarse_total = 0
for s in samples:
    claim = s['metadata']['claim']
    gold_cids = set(s['metadata']['gold_corpus_ids'])
    # 给 LLM 看所有粗索引摘要（5183 太多，只给 claim 关键词相关的）
    # 先用关键词过滤
    claim_words = set(claim.lower().split())
    candidates = [c for c in coarse_list if any(w in claim.lower() or w in c['summary'].lower() for w in c['keywords'][:3])][:100]
    if not candidates:
        candidates = coarse_list[:100]
    doc_list = '\n'.join(f'[{c["_id"]}] {c["summary"][:80]}' for c in candidates)
    prompt = f'Select document ids relevant to: {claim}\n\n{doc_list}\n\nCall select tool.'
    try:
        resp = gw.generate('deepseek-v4-flash', [{'role':'user','content':prompt}],
                          tools=JUDGE_TOOL, role_tag='coarse_judge', max_tokens=500)
        sel_ids = set()
        for tc in resp.tool_calls:
            if tc['function']['name'] == 'select':
                try: sel_ids = set(json.loads(tc['function']['arguments']).get('ids',[]))
                except: pass
        n_hit = len(sel_ids & gold_cids)
        coarse_hits += n_hit
        coarse_total += len(gold_cids)
    except Exception as e:
        print(f'  judge 失败: {str(e)[:60]}', flush=True)
print(f'粗索引 LLM judge recall: {coarse_hits}/{coarse_total} = {coarse_hits/max(coarse_total,1):.0%}', flush=True)

print(f'\n=== 对比汇总 ===', flush=True)
print(f'BM25 (全文 embedding): {bm25_hits/max(bm25_total,1):.0%}', flush=True)
print(f'粗索引 (LLM judge):    {coarse_hits/max(coarse_total,1):.0%}', flush=True)
