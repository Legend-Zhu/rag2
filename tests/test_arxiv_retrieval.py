"""arXiv 2026 "模型不知道的语料" 检索管线。

466 篇 2026-07-21~27 最新论文，模型训练截止后发表。
建 embedding 索引 + grep 倒排索引，验证融合检索在新语料上的效果。

同时从摘要提取 claim（用于后续 A/B 准确率对照，需 API key）。
"""
import sys, time, json, re
sys.path.insert(0, 'src')
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from pathlib import Path
from collections import defaultdict
from rag2.methods.retriever import Retriever
from rag2.methods.fusion_retriever import FusionRetriever

# ── 加载 arXiv 2026 语料 ──
corpus = json.loads(Path('data/arxiv_2026_corpus.json').read_text())
print(f'arXiv 2026 语料: {len(corpus)} 篇', flush=True)

# ── 从摘要提取 claim ──
def extract_claim(abstract):
    """从摘要提取核心 claim（结论/发现句）。

    找含 "we show", "we demonstrate", "we achieve", "we find",
    "outperform", "state-of-the-art", "improve", "propose" 的句子。
    """
    sents = re.split(r'(?<=[.!?])\s+', abstract)
    # 优先找结论句
    indicators = ['we show', 'we demonstrate', 'we find', 'we achieve',
                  'we propose', 'outperform', 'state-of-the-art', 'sota',
                  'we introduce', 'we present', 'our approach', 'our method',
                  'we develop', 'we design', 'significantly improve',
                  'surpass', 'exceed', 'novel']
    for sent in sents:
        lower = sent.lower()
        if any(ind in lower for ind in indicators):
            # 清理
            sent = sent.strip()
            if 20 < len(sent) < 300:
                return sent
    # fallback: 最后一句（通常是结论）
    return sents[-1].strip() if sents else ''

# 为每篇论文提取 claim
claims = {}
for pid, doc in corpus.items():
    claim = extract_claim(doc['text'])
    if claim:
        claims[pid] = claim

print(f'提取 claim: {len(claims)}/{len(corpus)} 篇\n', flush=True)
print('前 5 个 claim 示例:', flush=True)
for pid, claim in list(claims.items())[:5]:
    print(f'  [{pid}] {claim[:100]}', flush=True)

# 保存 claims
Path('data/arxiv_2026_claims.json').write_text(json.dumps(claims, ensure_ascii=False, indent=2))

# ── 建融合检索器 ──
print(f'\n=== 建融合检索索引 ===', flush=True)
all_docs = [{'title': d['title'], 'text': d['text']} for d in corpus.values()]

ret = Retriever()
ret.build_index(all_docs)
print(f'  embedding 索引建好 ({len(all_docs)} 篇)', flush=True)

fr = FusionRetriever(retriever=ret, corpus=corpus)
_ = fr.inverted_index  # 触发倒排索引构建
print(f'  grep 倒排索引建好 ({len(fr.inverted_index)} 词)', flush=True)

# ── 检索验证：用 claim 检索自己的论文 ──
# 这是 recall 测试：claim 从论文摘要提取，检索应返回该论文
print(f'\n=== 检索验证（claim → 找回自己的论文）===', flush=True)

n = min(30, len(claims))
test_items = list(claims.items())[:n]

# 无 HyDE（没有 LLM，用原 claim 做 embedding）
# 只用 embedding + grep + CrossEncoder

def retrieve_emb_only(claim, top_k=10):
    """纯 embedding 检索（无 HyDE，因为需要 LLM 改写）"""
    results = ret.search(claim, top_k_recall=top_k, top_k_rerank=top_k, rerank=False)
    title_to_pid = {d['title']: pid for pid, d in corpus.items()}
    return [title_to_pid.get(r['title'], '') for r in results]

def retrieve_fusion(claim, top_k=10):
    """融合检索（embedding + grep + CrossEncoder，无 HyDE）"""
    # embedding top-20
    emb_results = ret.search(claim, top_k_recall=20, top_k_rerank=20, rerank=False)
    title_to_pid = {d['title']: pid for pid, d in corpus.items()}
    emb_cids = [title_to_pid.get(r['title'], '') for r in emb_results if title_to_pid.get(r['title'])]

    # grep MAX IDF top-10
    grep_cids = fr.grep_max_idf(claim, top_k=10)

    # 合并
    pool = list(dict.fromkeys(emb_cids + grep_cids))
    if not pool:
        return []

    # CrossEncoder 重排
    pairs = [[claim, f"{corpus[c]['title']}: {corpus[c]['text']}"] for c in pool if c in corpus]
    valid_pool = [c for c in pool if c in corpus]
    scores = fr.ce.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(valid_pool, [float(s) for s in scores]), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_k]]

# 跑 embedding only
t0 = time.time()
emb_hits = 0
for pid, claim in test_items:
    results = retrieve_emb_only(claim, top_k=10)
    if pid in results: emb_hits += 1
emb_recall = emb_hits / n
print(f'  embedding only recall@10: {emb_hits}/{n} = {emb_recall:.0%} ({time.time()-t0:.0f}s)', flush=True)

# 跑融合检索
t0 = time.time()
fusion_hits = 0
for pid, claim in test_items:
    results = retrieve_fusion(claim, top_k=10)
    if pid in results: fusion_hits += 1
fusion_recall = fusion_hits / n
print(f'  融合检索 recall@10: {fusion_hits}/{n} = {fusion_recall:.0%} ({time.time()-t0:.0f}s)', flush=True)

# miss 分析
print(f'\n  融合 miss case:', flush=True)
for pid, claim in test_items:
    results = retrieve_fusion(claim, top_k=10)
    if pid not in results:
        print(f'    [{pid}] {claim[:80]}', flush=True)

print(f'\n{"="*55}', flush=True)
print('arXiv 2026 融合检索 recall@10', flush=True)
print(f'{"="*55}', flush=True)
print(f'  embedding only:  {emb_recall:.0%}', flush=True)
print(f'  融合检索:        {fusion_recall:.0%}  ({(fusion_recall-emb_recall)*100:+.0f}pt)', flush=True)
print(f'\n  注意：此处无 HyDE 改写（需 LLM API），', flush=True)
print(f'  有 HyDE 后 recall 会进一步提升（SciFact 上 +6pt）', flush=True)
