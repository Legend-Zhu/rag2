"""
分层索引验证：第一层粗索引（摘要+关键词），测一次调用能处理多少篇。

创新点：不全库深度标注（贵），分层——粗索引全库 + 精标注局部（查询时）。
"""
import sys, time, json
sys.path.insert(0, 'src')
from rag2.gateway import ModelGateway
from rag2.data import _load_scifact_artifacts

art = _load_scifact_artifacts()

# 粗索引的工具 schema（只生成 summary + keywords，轻量）
COARSE_TOOL = [{
    "type": "function",
    "function": {
        "name": "save_coarse_index",
        "description": "Save coarse index entries (summary + keywords only) for documents.",
        "parameters": {
            "type": "object",
            "properties": {
                "entries": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "doc_index": {"type": "integer", "description": "document index"},
                            "summary": {"type": "string", "description": "one concise sentence"},
                            "keywords": {"type": "array", "items": {"type": "string"}, "description": "3-5 key terms"},
                        },
                        "required": ["doc_index", "summary", "keywords"],
                    },
                },
            },
            "required": ["entries"],
        },
    },
}]

COARSE_PROMPT = """For each document below, generate a one-sentence summary and 3-5 keywords. Call save_coarse_index with all entries.

Documents:
{documents}"""

gw = ModelGateway()

# 测不同规模的粗索引：50 / 100 / 200 篇
for n_docs in [50, 100, 200]:
    docs = list(art['corpus'].values())[:n_docs]
    # 每篇截断到 600 字符（粗索引不需要全文）
    doc_strs = [f"[Doc {i}] {d['title'][:60]}\n{d['text'][:600]}" for i, d in enumerate(docs)]
    prompt = COARSE_PROMPT.format(documents='\n\n'.join(doc_strs))

    print(f'\n=== 粗索引 {n_docs} 篇 ===', flush=True)
    print(f'  prompt: {len(prompt)} 字符 ≈ {len(prompt)//4} tokens', flush=True)

    t0 = time.time()
    try:
        resp = gw.generate('qwen3.8', [{'role':'user','content':prompt}],
                          tools=COARSE_TOOL, role_tag=f'coarse_{n_docs}', max_tokens=16000)
        dt = time.time() - t0
        # 解析
        entries = []
        for tc in resp.tool_calls:
            if tc['function']['name'] == 'save_coarse_index':
                try:
                    args = json.loads(tc['function']['arguments'])
                    entries = args.get('entries', [])
                except: pass
        print(f'  耗时: {dt:.0f}s, finish={resp.finish_reason}, entries={len(entries)}/{n_docs}', flush=True)
        if entries:
            print(f'  示例: {entries[0]}', flush=True)
        elif resp.tool_calls:
            args_str = resp.tool_calls[0]['function']['arguments'][:200]
            print(f'  解析失败, args 前缀: {args_str}', flush=True)
    except Exception as e:
        print(f'  错误: {str(e)[:120]}', flush=True)
