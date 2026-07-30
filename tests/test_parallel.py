"""方向2: 并行化验证——测 token plan 的并发能力。"""
import sys, time, json
sys.path.insert(0, 'src')
import concurrent.futures
from rag2.gateway import ModelGateway
from rag2.data import _load_scifact_artifacts

art = _load_scifact_artifacts()
gw = ModelGateway()

# 准备 10 个独立的简单调用（不同文档的摘要）
docs = list(art['corpus'].values())[:10]
prompts = [
    [{"role": "user", "content": f"Summarize in one sentence: {d['title']}: {d['text'][:400]}"}]
    for d in docs
]

def call_one(idx):
    t0 = time.time()
    resp = gw.generate('qwen3.8', prompts[idx], role_tag=f'parallel_test_{idx}', max_tokens=200)
    return idx, time.time() - t0, resp.text[:50]

# 测不同并发数
for n_concurrent in [1, 3, 5, 10]:
    print(f'\n=== 并发 {n_concurrent} ===', flush=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as executor:
        futures = [executor.submit(call_one, i) for i in range(n_concurrent)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    total = time.time() - t0
    times = [r[1] for r in results]
    print(f'  总耗时: {total:.1f}s', flush=True)
    print(f'  单次平均: {sum(times)/len(times):.1f}s', flush=True)
    print(f'  加速比: {(sum(times))/total:.2f}x (vs 串行)', flush=True)
    # 看是否有错误/限流
    errors = [r for r in results if not r[2].strip()]
    if errors:
        print(f'  ⚠️ {len(errors)} 个空响应（可能限流）', flush=True)
