"""
RAG² MLOps: FastAPI 服务。

端点：
  POST /search          - 检索（可选 strategy=vanilla|fusion，自动 A/B 路由）
  GET  /health          - 健康检查
  GET  /metrics         - 成本/延迟/缓存命中率摘要
  GET  /indices         - 列出所有索引版本
  POST /indices/{cid}   - 注册新索引
  PUT  /indices/{cid}/activate/{ver}  - 切换版本
  GET  /ab/results      - A/B 测试结果对比
  GET  /recent          - 最近查询

启动：
  python -m rag2.mlops.api_server --corpus scifact
  python -m rag2.mlops.api_server --corpus arxiv_2026 --corpus-file data/arxiv_2026_corpus.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag2.methods.retriever import Retriever
from rag2.methods.fusion_retriever import FusionRetriever
from rag2.mlops.metrics import MetricsCollector
from rag2.mlops.index_manager import IndexManager
from rag2.mlops.ab_router import ABRouter

# ── 请求/响应模型 ──────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3
    strategy: str | None = None  # None = A/B 自动路由, "vanilla" / "fusion"
    gold_cids: list[str] | None = None
    verdict: str = ""
    correct: bool | None = None

class SearchResponse(BaseModel):
    strategy: str
    results: list[dict]
    latency_s: float
    n_results: int

class IndexRegisterRequest(BaseModel):
    corpus_file: str  # JSON 文件路径
    corpus_id: str = ""

# ── 全局状态 ────────────────────────────────────────────

app = FastAPI(title="RAG² MLOps API", version="1.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state: dict = {}  # 运行时状态（在 startup 中初始化）


@app.on_event("startup")
def startup():
    """加载模型和索引（在启动时一次性完成）。"""
    corpus_id = _state.get("corpus_id", "default")
    corpus_file = _state.get("corpus_file", "")

    logging.info("启动 RAG² API: corpus=%s, file=%s", corpus_id, corpus_file)

    # 加载语料
    if corpus_file and Path(corpus_file).exists():
        corpus = json.loads(Path(corpus_file).read_text())
    else:
        # 默认用 SciFact
        from rag2.data import _load_scifact_artifacts
        art = _load_scifact_artifacts()
        corpus = {cid: {"title": c["title"], "text": c["text"]} for cid, c in art["corpus"].items()}
        corpus_id = "scifact"

    all_docs = [{"title": d["title"], "text": d["text"]} for d in corpus.values()]

    # 建检索器
    logging.info("建 embedding 索引 (%d 文档)...", len(all_docs))
    t0 = time.time()
    ret = Retriever()
    ret.build_index(all_docs)
    logging.info("embedding 索引建好 (%.1fs)", time.time() - t0)

    # HyDE 缓存
    hyde_path = f"cache/{corpus_id}_hyde_rewrites.json"
    if not Path(hyde_path).exists():
        hyde_path = "cache/hyde_rewrites.json"

    fr = FusionRetriever(retriever=ret, corpus=corpus, hyde_cache_path=hyde_path)
    _ = fr.inverted_index  # 预建 grep 索引
    logging.info("融合检索器就绪")

    # MLOps 组件
    mc = MetricsCollector()
    im = IndexManager()

    # 注册索引（如果还没注册）
    active = im.get_active(corpus_id)
    if not active:
        im.register(
            corpus_id=corpus_id,
            corpus=corpus,
            embed_dim=1024,
            grep_index="cache/grep_inverted_index.json",
            hyde_cache=hyde_path,
        )

    # A/B 路由器
    def vanilla_search(query, top_k=3):
        results = ret.search(query, top_k_recall=top_k, top_k_rerank=top_k, rerank=False)
        title_to_cid = {d["title"]: cid for cid, d in corpus.items()}
        return [{"cid": title_to_cid.get(r["title"], ""), "title": r["title"],
                 "text": r["text"], "score": r["score"], "source": "emb"}
                for r in results]

    def fusion_search(query, top_k=3):
        return fr.search(query, top_k=top_k, use_grep=True, use_rerank=True)

    router = ABRouter(
        strategies={"vanilla": vanilla_search, "fusion": fusion_search},
        metrics=mc, router_type="round_robin", corpus_id=corpus_id,
    )

    _state.update({
        "corpus": corpus, "corpus_id": corpus_id,
        "retriever": ret, "fusion_retriever": fr,
        "metrics": mc, "index_manager": im, "ab_router": router,
    })
    logging.info("RAG² API 启动完成")


# ── 端点 ────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "corpus": _state.get("corpus_id", "unknown"),
            "n_docs": len(_state.get("corpus", {}))}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    router = _state.get("ab_router")
    if not router:
        raise HTTPException(503, "服务未就绪")

    gold_cids = set(req.gold_cids) if req.gold_cids else None
    strat, results, latency = router.search(
        query=req.query, top_k=req.top_k,
        strategy=req.strategy, gold_cids=gold_cids,
        verdict=req.verdict, correct=req.correct,
    )
    return SearchResponse(
        strategy=strat, results=results[:req.top_k],
        latency_s=round(latency, 3), n_results=len(results),
    )


@app.get("/metrics")
def metrics():
    mc = _state.get("metrics")
    if not mc:
        raise HTTPException(503, "服务未就绪")
    return mc.get_metrics_summary()


@app.get("/indices")
def list_indices():
    im = _state.get("index_manager")
    if not im:
        raise HTTPException(503, "服务未就绪")
    return im.list_versions()


@app.post("/indices/{corpus_id}")
def register_index(corpus_id: str, req: IndexRegisterRequest):
    im = _state.get("index_manager")
    if not im:
        raise HTTPException(503, "服务未就绪")
    corpus = json.loads(Path(req.corpus_file).read_text())
    cid = req.corpus_id or corpus_id
    version = im.register(cid, corpus, embed_dim=1024)
    return {"corpus_id": cid, "version": version, "n_docs": len(corpus)}


@app.put("/indices/{corpus_id}/activate/{version_id}")
def activate_index(corpus_id: str, version_id: str):
    im = _state.get("index_manager")
    if not im:
        raise HTTPException(503, "服务未就绪")
    ok = im.activate(corpus_id, version_id)
    if not ok:
        raise HTTPException(404, f"版本 {version_id} 不存在")
    return {"corpus_id": corpus_id, "active_version": version_id}


@app.get("/ab/results")
def ab_results():
    mc = _state.get("metrics")
    if not mc:
        raise HTTPException(503, "服务未就绪")
    return mc.get_ab_results()


@app.get("/recent")
def recent_queries(limit: int = 20):
    mc = _state.get("metrics")
    if not mc:
        raise HTTPException(503, "服务未就绪")
    return mc.get_recent_queries(limit)


# ── HTML 仪表盘（零依赖，内嵌 FastAPI）─────────────────

@app.get("/dashboard")
def dashboard():
    """自包含 HTML 仪表盘，通过 JS 调用 /metrics /ab/results /indices /recent。"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(_DASHBOARD_HTML)


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG² MLOps Dashboard</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: -apple-system, sans-serif; background:#f0f2f5; color:#333; }
.header { background:#2563eb; color:#fff; padding:20px; text-align:center; }
.header h1 { font-size:1.5em; }
.container { max-width:1200px; margin:20px auto; padding:0 15px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:15px; margin-bottom:20px; }
.card { background:#fff; border-radius:8px; padding:20px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
.card h3 { color:#6b7280; font-size:0.85em; text-transform:uppercase; margin-bottom:8px; }
.card .value { font-size:1.8em; font-weight:bold; color:#2563eb; }
.card .sub { font-size:0.8em; color:#9ca3af; margin-top:4px; }
table { width:100%; border-collapse:collapse; font-size:0.85em; }
th { background:#f9fafb; text-align:left; padding:8px; border-bottom:2px solid #e5e7eb; }
td { padding:8px; border-bottom:1px solid #f3f4f6; }
.section { background:#fff; border-radius:8px; padding:20px; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.1); }
.section h2 { font-size:1.1em; margin-bottom:15px; color:#1f2937; }
.bar { height:24px; border-radius:4px; display:inline-block; min-width:2px; }
.badge { padding:2px 8px; border-radius:12px; font-size:0.75em; font-weight:bold; }
.badge-vanilla { background:#dbeafe; color:#2563eb; }
.badge-fusion { background:#dcfce7; color:#16a34a; }
.badge-active { background:#fef3c7; color:#d97706; }
.refresh { float:right; font-size:0.8em; color:#9ca3af; }
</style></head><body>
<div class="header"><h1>📊 RAG² MLOps Dashboard</h1></div>
<div class="container">

<div class="grid" id="metrics-grid">
  <div class="card"><h3>7天成本</h3><div class="value" id="cost">$0.00</div><div class="sub" id="cost-sub">0 次调用</div></div>
  <div class="card"><h3>检索延迟 P50</h3><div class="value" id="ret-lat">0s</div><div class="sub" id="ret-lat-sub">P95: 0s</div></div>
  <div class="card"><h3>LLM延迟 P50</h3><div class="value" id="llm-lat">0s</div><div class="sub" id="llm-lat-sub">P95: 0s</div></div>
  <div class="card"><h3>缓存命中率</h3><div class="value" id="cache">0%</div><div class="sub" id="cache-sub">7天</div></div>
</div>

<div class="section">
  <h2>🔬 A/B 测试结果 <span class="refresh" id="ab-refresh"></span></h2>
  <table><thead><tr><th>策略</th><th>样本数</th><th>准确率</th><th>平均延迟</th><th>准确率对比</th></tr></thead>
  <tbody id="ab-body"></tbody></table>
</div>

<div class="section">
  <h2>🗂️ 索引版本</h2>
  <div id="indices-content"></div>
</div>

<div class="section">
  <h2>🔍 最近查询 <span class="refresh">自动刷新 30s</span></h2>
  <table><thead><tr><th>时间</th><th>查询</th><th>策略</th><th>结果数</th><th>延迟</th></tr></thead>
  <tbody id="recent-body"></tbody></table>
</div>

</div>
<script>
async function fetchJSON(url) {
  try { const r = await fetch(url); return await r.json(); } catch(e) { return null; }
}
function fmt(s) { return s ? s.substring(11,19) : ''; }

async function refresh() {
  // metrics
  const m = await fetchJSON('/metrics');
  if (m) {
    document.getElementById('cost').textContent = '$' + (m.cost.cost_usd||0).toFixed(2);
    document.getElementById('cost-sub').textContent = (m.cost.n_calls||0) + ' 次调用';
    document.getElementById('ret-lat').textContent = (m.retrieval_latency.p50||0).toFixed(1) + 's';
    document.getElementById('ret-lat-sub').textContent = 'P95: ' + (m.retrieval_latency.p95||0).toFixed(1) + 's';
    document.getElementById('llm-lat').textContent = (m.llm_latency.p50||0).toFixed(1) + 's';
    document.getElementById('llm-lat-sub').textContent = 'P95: ' + (m.llm_latency.p95||0).toFixed(1) + 's';
    document.getElementById('cache').textContent = ((m.cache_hit_rate||0)*100).toFixed(0) + '%';
  }
  // A/B results
  const ab = await fetchJSON('/ab/results');
  if (ab && ab.length) {
    const maxAcc = Math.max(...ab.map(r => r.accuracy));
    document.getElementById('ab-body').innerHTML = ab.map(r => {
      const w = (r.accuracy / maxAcc * 200) || 1;
      const color = r.strategy === 'fusion' ? '#16a34a' : '#2563eb';
      return `<tr><td><span class="badge badge-${r.strategy}">${r.strategy}</span></td>
        <td>${r.n}</td><td>${(r.accuracy*100).toFixed(0)}%</td>
        <td>${(r.avg_latency||0).toFixed(1)}s</td>
        <td><div class="bar" style="width:${w}px;background:${color}"></div></td></tr>`;
    }).join('');
  } else {
    document.getElementById('ab-body').innerHTML = '<tr><td colspan="5" style="color:#9ca3af;text-align:center">暂无 A/B 数据</td></tr>';
  }
  // indices
  const idx = await fetchJSON('/indices');
  if (idx) {
    let html = '';
    for (const [cid, info] of Object.entries(idx)) {
      html += `<h3 style="margin:10px 0 5px">📁 ${cid}</h3>`;
      for (const [vid, v] of Object.entries(info.versions||{})) {
        const badge = v.active ? '<span class="badge badge-active">ACTIVE</span>' : '';
        html += `<div style="margin:5px 0;padding:5px 10px;background:#f9fafb;border-radius:4px">
          ${badge} <code>${vid}</code> | ${v.n_docs||'?'} 文档 | dim=${v.embed_dim||'?'} | ${v.created ? v.created.substring(0,19) : ''}</div>`;
      }
    }
    document.getElementById('indices-content').innerHTML = html || '<p style="color:#9ca3af">暂无索引</p>';
  }
  // recent
  const recent = await fetchJSON('/recent?limit=15');
  if (recent && recent.length) {
    document.getElementById('recent-body').innerHTML = recent.map(r =>
      `<tr><td>${fmt(r.timestamp)}</td><td style="max-width:300px;overflow:hidden;text-overflow:ellipsis">${r.query}</td>
       <td><span class="badge badge-${r.strategy}">${r.strategy}</span></td>
       <td>${r.n_results}</td><td>${(r.latency_s||0).toFixed(1)}s</td></tr>`
    ).join('');
  }
}
refresh();
setInterval(refresh, 30000);
</script>
</body></html>"""


# ── CLI ─────────────────────────────────────────────────

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="RAG² MLOps API Server")
    parser.add_argument("--corpus", default="scifact", help="语料标识")
    parser.add_argument("--corpus-file", default="", help="语料 JSON 文件路径")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _state["corpus_id"] = args.corpus
    _state["corpus_file"] = args.corpus_file

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
