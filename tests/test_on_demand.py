"""方向4: 按需局部索引——不预建全库，推理时边探查边建索引。

思路：模型自己决定查什么、读什么、对什么建索引。
类似 coding agent 在代码库里 grep/read，但每次读到的内容自动缓存为"局部索引"。

验证:
  1. 不预建索引，给模型 grep + read 工具
  2. 模型探查到的文档，自动精标注并缓存
  3. 下次类似查询，已标注的文档直接命中缓存
  4. 对比：按需局部 vs 预建全库 的 recall 和耗时
"""
import sys, time, json
sys.path.insert(0, 'src')
from rag2.gateway import ModelGateway
from rag2.data import load_dataset_subsampled, _load_scifact_artifacts
from rag2.methods.agentic_tools import DocumentCorpus, build_tool_schemas, dispatch_tool
from rag2.methods.fine_label_cache import FineLabelCache

art = _load_scifact_artifacts()
corpus = art['corpus']
samples = load_dataset_subsampled('scifact')[:5]

# 50 篇语料
gold_ids = set()
for s in samples:
    gold_ids.update(s['metadata']['gold_corpus_ids'])
docs = []
seen = set(gold_ids)
for gid in gold_ids:
    c = corpus.get(gid)
    if c: docs.append({'_id': gid, **c})
for cid, c in corpus.items():
    if cid in seen: continue
    docs.append({'_id': cid, **c})
    seen.add(cid)
    if len(docs) >= 50: break

gw = ModelGateway()
fine_cache = FineLabelCache()

# 把 docs 转成 DocumentCorpus 可用的格式（带 _id）
corpus_docs = [{'title': d['title'], 'text': d['text'], '_id': d['_id']} for d in docs]
dc = DocumentCorpus(docs=corpus_docs)

# 工具：grep + read + extract（提取即精标注，自动缓存）
ON_DEMAND_SYSTEM = """You explore a scientific corpus to verify claims. Use tools to find and read documents.

Strategy:
1. Use grep to find documents mentioning key terms from the claim.
2. Use read to examine promising documents.
3. When you read a document, call extract_entities to build a local index of it (this is cached for future queries).
4. Decide: does the evidence SUPPORT, REFUTE, or is there NOT_ENOUGH_INFO for the claim?

Be efficient: grep → read 2-3 most relevant → extract → verdict."""

def make_tools():
    """工具集：grep + read + extract（精标注并缓存）"""
    return [
        {"type":"function","function":{"name":"grep","parameters":{"type":"object","properties":{"pattern":{"type":"string"},"max_hits":{"type":"integer","default":5}},"required":["pattern"]}}},
        {"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"doc_id":{"type":"integer"},"max_chars":{"type":"integer","default":2000}},"required":["doc_id"]}}},
        {"type":"function","function":{"name":"extract_entities","parameters":{"type":"object","properties":{"doc_id":{"type":"integer"}},"required":["doc_id"]}}},
        {"type":"function","function":{"name":"submit_verdict","parameters":{"type":"object","properties":{"verdict":{"type":"string","enum":["SUPPORTED","REFUTED","NOT_ENOUGH_INFO"]},"explanation":{"type":"string"}},"required":["verdict"]}}},
    ]

def execute_tool(tool_name, args, dc, docs_list, gw, fine_cache):
    """执行工具，extract 时自动精标注并缓存。"""
    if tool_name == 'grep':
        return dc.grep(args.get('pattern',''), max_hits=args.get('max_hits',5))
    elif tool_name == 'read':
        return dc.read(args.get('doc_id',0), max_chars=args.get('max_chars',2000))
    elif tool_name == 'extract_entities':
        doc_id = args.get('doc_id',0)
        if doc_id not in dc._by_id:
            return f'doc {doc_id} not found'
        d = dc._by_id[doc_id]
        cid = d.get('_id', str(doc_id))
        # 查缓存
        cached = fine_cache.get(cid)
        if cached:
            return f'[CACHED] {json.dumps(cached, ensure_ascii=False)[:300]}'
        # miss → 标注
        prompt = f'Extract entities and key facts from:\n{d["title"]}\n{d["text"]}'
        resp = gw.generate('qwen3.8', [{'role':'user','content':prompt}], max_tokens=500, role_tag='ondemand_extract')
        label = {'summary': resp.text[:200], 'extracted': resp.text[:500]}
        fine_cache.put(cid, label)
        return f'[NEW] {label["extracted"][:300]}'
    return f'unknown tool {tool_name}'

# 跑 2 个 claim（第一个建索引，第二个部分命中）
print(f'=== 方向4: 按需局部索引（50 篇语料，2 claim）===\n', flush=True)

for idx, s in enumerate(samples[:2]):
    claim = s['metadata']['claim']
    gold_cids = set(s['metadata']['gold_corpus_ids'])
    print(f'--- claim {idx+1}: {claim[:60]} ---', flush=True)
    print(f'    gold: {gold_cids}', flush=True)

    messages = [
        {'role':'system','content':ON_DEMAND_SYSTEM},
        {'role':'user','content':f'Claim: {claim}'},
    ]
    tools = make_tools()

    t0 = time.time()
    n_steps = 0
    verdict = 'UNKNOWN'
    for step in range(8):  # max 8 steps
        resp = gw.generate('qwen3.8', messages, tools=tools, role_tag='ondemand', max_tokens=2000)
        messages.append({'role':'assistant','content':resp.text or '','tool_calls':resp.tool_calls})
        if not resp.tool_calls:
            break
        for tc in resp.tool_calls:
            fn = tc['function']
            name = fn['name']
            try: args = json.loads(fn['arguments'] or '{}')
            except: args = {}
            if name == 'submit_verdict':
                try: verdict = json.loads(fn['arguments']).get('verdict','UNKNOWN')
                except: pass
                break
            result = execute_tool(name, args, dc, docs, gw, fine_cache)
            messages.append({'role':'tool','tool_call_id':tc['id'],'content':result})
            n_steps += 1
            print(f'    step {n_steps}: {name}({args}) → {result[:60]}...', flush=True)
        else:
            continue
        break  # submit_verdict 触发退出

    print(f'    verdict: {verdict}, 步数: {n_steps}, 耗时: {time.time()-t0:.0f}s\n', flush=True)

print(f'=== 缓存统计 ===', flush=True)
print(f'  {fine_cache.stats()}', flush=True)
