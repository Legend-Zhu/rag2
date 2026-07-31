"""RAG² Web 平台前端：单页应用（SPA）。

4 个模块：
  1. 文档库管理：上传文件、查看索引状态、入库历史
  2. 检索测试：输入查询、dense/sparse/混合对比
  3. Agent 报告生成：输入主题、实时显示生成过程
  4. 监控仪表盘：成本、延迟、缓存、A/B

零前端构建依赖：纯 HTML + vanilla JS，内嵌 FastAPI。
"""
from __future__ import annotations

WEB_UI = r"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG² Platform</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,sans-serif; background:#f0f2f5; color:#1f2937; }
.navbar { background:#1e293b; padding:0 24px; display:flex; align-items:center; height:56px; }
.navbar h1 { color:#fff; font-size:1.1em; }
.tab-bar { display:flex; gap:4px; margin-left:32px; }
.tab { color:#94a3b8; padding:8px 16px; border-radius:6px; cursor:pointer; font-size:0.9em; }
.tab:hover { background:#334155; color:#e2e8f0; }
.tab.active { background:#3b82f6; color:#fff; }
.container { max-width:1100px; margin:0 auto; padding:24px 16px; }
.card { background:#fff; border-radius:8px; padding:20px; margin-bottom:16px; box-shadow:0 1px 3px rgba(0,0,0,0.08); }
.card h2 { font-size:1em; color:#6b7280; margin-bottom:12px; text-transform:uppercase; }
.btn { background:#3b82f6; color:#fff; border:none; padding:8px 20px; border-radius:6px; cursor:pointer; font-size:0.9em; }
.btn:hover { background:#2563eb; }
.btn:disabled { background:#9ca3af; cursor:not-allowed; }
.btn-sm { padding:4px 12px; font-size:0.8em; }
input[type=text], textarea, select { width:100%; padding:8px 12px; border:1px solid #d1d5db; border-radius:6px; font-size:0.9em; }
textarea { min-height:80px; font-family:inherit; }
label { font-size:0.85em; color:#6b7280; margin-bottom:4px; display:block; }
table { width:100%; border-collapse:collapse; font-size:0.85em; }
th { background:#f9fafb; text-align:left; padding:8px; border-bottom:2px solid #e5e7eb; }
td { padding:8px; border-bottom:1px solid #f3f4f6; }
.badge { padding:2px 8px; border-radius:12px; font-size:0.75em; font-weight:bold; }
.badge-green { background:#dcfce7; color:#16a34a; }
.badge-blue { background:#dbeafe; color:#2563eb; }
.badge-orange { background:#fef3c7; color:#d97706; }
.metric { display:flex; flex-direction:column; gap:4px; }
.metric .value { font-size:1.8em; font-weight:bold; }
.metric .label { font-size:0.8em; color:#9ca3af; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:16px; }
.result-item { padding:12px; border:1px solid #e5e7eb; border-radius:6px; margin-bottom:8px; }
.result-item .title { font-weight:600; font-size:0.9em; }
.result-item .meta { font-size:0.8em; color:#6b7280; margin-top:4px; }
.result-item .snippet { font-size:0.85em; color:#374151; margin-top:6px; line-height:1.5; }
.section-preview { padding:12px; border-left:3px solid #3b82f6; margin-bottom:12px; background:#f8fafc; }
.section-preview h3 { font-size:0.95em; margin-bottom:6px; }
.section-preview p { font-size:0.85em; line-height:1.6; }
.spinner { display:inline-block; width:16px; height:16px; border:2px solid #d1d5db; border-top:#3b82f6; border-radius:50%; animation:spin 0.8s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.status { font-size:0.85em; padding:8px 12px; border-radius:6px; margin:8px 0; }
.status-info { background:#eff6ff; color:#1e40af; }
.status-success { background:#f0fdf4; color:#166534; }
.status-error { background:#fef2f2; color:#991b1b; }
.hidden { display:none; }
</style></head><body>
<div class="navbar">
  <h1>RAG<sup>2</sup> Platform</h1>
  <div class="tab-bar">
    <div class="tab active" onclick="switchTab('library')">📚 文档库</div>
    <div class="tab" onclick="switchTab('search')">🔍 检索测试</div>
    <div class="tab" onclick="switchTab('report')">📝 报告生成</div>
    <div class="tab" onclick="switchTab('dashboard')">📊 监控</div>
  </div>
</div>

<div class="container">

<!-- Tab 1: 文档库管理 -->
<div id="tab-library" class="tab-content">
  <div class="card">
    <h2>上传文档</h2>
    <p style="margin-bottom:12px;color:#6b7280;font-size:0.85em">支持 PDF / DOCX / HTML / Markdown / TXT / PPTX / XLSX</p>
    <input type="file" id="file-input" multiple style="margin-bottom:12px">
    <input type="text" id="upload-corpus" placeholder="语料库名称（默认 upload）" style="margin-bottom:12px">
    <button class="btn" id="upload-btn" onclick="uploadFiles()">上传并入库</button>
    <div id="upload-status"></div>
  </div>

  <div class="card">
    <h2>目录入库</h2>
    <label>目录路径（服务器本地）</label>
    <input type="text" id="ingest-dir" placeholder="/path/to/documents" style="margin-bottom:12px">
    <label>语料库名称</label>
    <input type="text" id="ingest-corpus" placeholder="my_docs" style="margin-bottom:12px">
    <button class="btn" onclick="ingestDir()">开始入库</button>
    <div id="ingest-status"></div>
  </div>

  <div class="card">
    <h2>入库历史</h2>
    <button class="btn btn-sm" onclick="loadIngestHistory()">刷新</button>
    <table id="ingest-history-table"><thead><tr>
      <th>时间</th><th>语料</th><th>文件数</th><th>Chunks</th><th>总耗时</th>
    </tr></thead><tbody id="ingest-history"></tbody></table>
  </div>
</div>

<!-- Tab 2: 检索测试 -->
<div id="tab-search" class="tab-content hidden">
  <div class="card">
    <h2>检索查询</h2>
    <input type="text" id="search-query" placeholder="输入查询..." style="margin-bottom:12px"
      onkeydown="if(event.key==='Enter')doSearch()">
    <label>检索模式</label>
    <select id="search-mode" style="margin-bottom:12px">
      <option value="fusion">混合（dense + sparse + 重排）</option>
      <option value="vanilla">纯 dense</option>
    </select>
    <label>搜索语料</label>
    <select id="search-corpus" style="margin-bottom:12px">
      <option value="">arXiv 2026（主语料）</option>
    </select>
    <label>返回数量</label>
    <select id="search-topk" style="margin-bottom:12px">
      <option value="3">3</option><option value="5" selected>5</option><option value="10">10</option>
    </select>
    <button class="btn" onclick="doSearch()">搜索</button>
    <span id="search-time" style="margin-left:12px;font-size:0.85em;color:#6b7280"></span>
  </div>
  <div id="search-results"></div>
</div>

<!-- Tab 3: Agent 报告生成 -->
<div id="tab-report" class="tab-content hidden">
  <div class="card">
    <h2>生成研究报告</h2>
    <label>研究主题</label>
    <input type="text" id="report-topic" placeholder="例：LLM agent 安全研究进展" style="margin-bottom:12px">
    <button class="btn" id="report-btn" onclick="generateReport()">生成报告</button>
    <div id="report-status"></div>
  </div>
  <div id="report-output"></div>
</div>

<!-- Tab 4: 监控仪表盘 -->
<div id="tab-dashboard" class="tab-content hidden">
  <div class="grid" id="metrics-grid">
    <div class="card"><div class="metric"><div class="label">7天成本</div><div class="value" id="m-cost">$0.00</div><div class="label" id="m-cost-sub">0 次调用</div></div></div>
    <div class="card"><div class="metric"><div class="label">检索延迟 P50</div><div class="value" id="m-retlat">0s</div><div class="label" id="m-retlat-sub">P95: 0s</div></div></div>
    <div class="card"><div class="metric"><div class="label">LLM延迟 P50</div><div class="value" id="m-llmlat">0s</div><div class="label" id="m-llmlat-sub">P95: 0s</div></div></div>
    <div class="card"><div class="metric"><div class="label">缓存命中率</div><div class="value" id="m-cache">0%</div></div></div>
  </div>
  <div class="card">
    <h2>A/B 测试结果</h2>
    <table><thead><tr><th>策略</th><th>样本</th><th>准确率</th><th>平均延迟</th></tr></thead>
    <tbody id="ab-body"></tbody></table>
  </div>
  <div class="card">
    <h2>最近查询</h2>
    <table><thead><tr><th>时间</th><th>查询</th><th>策略</th><th>结果数</th><th>延迟</th></tr></thead>
    <tbody id="recent-body"></tbody></table>
  </div>
</div>

</div>

<script>
const API = '';
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById('tab-' + name).classList.remove('hidden');
  event.target.classList.add('active');
  if (name === 'dashboard') loadDashboard();
  if (name === 'library') loadIngestHistory();
}
async function api(url, opts = {}) {
  try {
    const r = await fetch(API + url, opts);
    return await r.json();
  } catch(e) { console.error(e); return null; }
}

// ── 文档库 ──
async function uploadFiles() {
  const input = document.getElementById('file-input');
  if (!input.files.length) return;
  document.getElementById('upload-status').innerHTML = '<div class="status status-info"><span class="spinner"></span> 上传中...</div>';
  const fd = new FormData();
  for (const f of input.files) fd.append('files', f);
  fd.append('corpus_id', document.getElementById('upload-corpus').value || 'upload');
  try {
    const r = await fetch(API + '/upload?corpus_id=' + (document.getElementById('upload-corpus').value || 'upload'), {
      method:'POST', body: fd
    });
    const d = await r.json();
    document.getElementById('upload-status').innerHTML = '<div class="status status-success">✓ 上传 ' + d.n_files + ' 文件, ' + d.n_chunks + ' chunks → 语料: ' + d.corpus_id + '</div>';
    refreshCorpusList();
  } catch(e) {
    document.getElementById('upload-status').innerHTML = '<div class="status status-error">上传失败: ' + e + '</div>';
  }
}
async function ingestDir() {
  const dir = document.getElementById('ingest-dir').value;
  const cid = document.getElementById('ingest-corpus').value || 'docs';
  if (!dir) return;
  document.getElementById('ingest-status').innerHTML = '<div class="status status-info"><span class="spinner"></span> 入库中（可能需要几分钟）...</div>';
  const d = await api('/ingest/dir', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({dir_path: dir, corpus_id: cid, build_index: true})
  });
  if (d && !d.error) {
    document.getElementById('ingest-status').innerHTML = '<div class="status status-success">✓ ' + d.n_files + ' 文件 → ' + d.n_chunks + ' chunks (' + d.total_time_s + 's)</div>';
    refreshCorpusList();
  } else {
    document.getElementById('ingest-status').innerHTML = '<div class="status status-error">入库失败: ' + (d?.error || '未知') + '</div>';
  }
}
async function loadIngestHistory() {
  const d = await api('/ingest/status');
  const tbody = document.getElementById('ingest-history');
  if (!d || !d.length) { tbody.innerHTML = '<tr><td colspan="5" style="color:#9ca3af;text-align:center">暂无记录</td></tr>'; return; }
  tbody.innerHTML = d.map(r => '<tr><td>' + (r.timestamp||'').substring(0,19).replace('T',' ') + '</td><td>' + r.corpus_id + '</td><td>' + r.n_files + '</td><td>' + r.n_chunks + '</td><td>' + (r.total_time_s||0).toFixed(1) + 's</td></tr>').join('');
  refreshCorpusList(d);
}
function refreshCorpusList(historyData) {
  const sel = document.getElementById('search-corpus');
  if (!sel) return;
  const current = sel.value;
  let opts = '<option value="">arXiv 2026（主语料）</option>';
  const data = historyData || [];
  const seen = new Set();
  for (const r of data) {
    if (r.corpus_id && !seen.has(r.corpus_id)) {
      seen.add(r.corpus_id);
      opts += '<option value="' + r.corpus_id + '">' + r.corpus_id + ' (' + r.n_chunks + ' chunks)</option>';
    }
  }
  sel.innerHTML = opts;
  if (current) sel.value = current;
}

// ── 检索 ──
async function doSearch() {
  const q = document.getElementById('search-query').value;
  if (!q) return;
  const mode = document.getElementById('search-mode').value;
  const topk = document.getElementById('search-topk').value;
  const corpus = document.getElementById('search-corpus').value;
  document.getElementById('search-results').innerHTML = '<div class="status status-info"><span class="spinner"></span> 搜索中...</div>';
  const t0 = performance.now();
  const body = {query: q, top_k: parseInt(topk), strategy: mode === 'vanilla' ? 'vanilla' : 'fusion'};
  if (corpus) body.corpus_id = corpus;
  const d = await api('/search', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)
  });
  const dt = ((performance.now() - t0) / 1000).toFixed(2);
  document.getElementById('search-time').textContent = dt + 's';
  if (!d) { document.getElementById('search-results').innerHTML = '<div class="status status-error">搜索失败</div>'; return; }
  let html = '<div class="card"><h2>结果（' + d.n_results + ' 条, 策略: ' + d.strategy + ', ' + dt + 's）</h2>';
  for (const r of (d.results || [])) {
    html += '<div class="result-item"><div class="title">' + (r.title || r.heading || '').substring(0,80) +
      ' <span class="badge ' + (r.source?.includes('sparse') ? 'badge-green' : 'badge-blue') + '">' + (r.source||'') + '</span></div>' +
      '<div class="meta">score: ' + (r.score||0).toFixed(4) + (r.rerank_score ? ' → rerank: ' + r.rerank_score.toFixed(4) : '') + '</div>' +
      '<div class="snippet">' + (r.text||'').substring(0,200) + '...</div></div>';
  }
  html += '</div>';
  document.getElementById('search-results').innerHTML = html;
}

// ── Agent 报告 ──
async function generateReport() {
  const topic = document.getElementById('report-topic').value;
  if (!topic) return;
  const btn = document.getElementById('report-btn');
  btn.disabled = true; btn.textContent = '生成中...';
  document.getElementById('report-status').innerHTML = '<div class="status status-info"><span class="spinner"></span> Agent 正在检索和写作（约 1-2 分钟）...</div>';
  document.getElementById('report-output').innerHTML = '';
  const d = await api('/report/generate', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({topic: topic, max_tool_calls: 25})
  });
  btn.disabled = false; btn.textContent = '生成报告';
  if (!d) { document.getElementById('report-status').innerHTML = '<div class="status status-error">生成失败</div>'; return; }
  document.getElementById('report-status').innerHTML = '<div class="status status-success">✓ ' + d.sections.length + ' 章节, ' + d.total_llm_calls + ' 次 LLM 调用, ' + d.elapsed_s + 's</div>';
  let html = '<div class="card"><h2>' + topic + '</h2>';
  if (d.summary) html += '<p style="margin:12px 0;color:#374151;line-height:1.6">' + d.summary + '</p>';
  html += '</div>';
  for (const s of d.sections) {
    html += '<div class="section-preview"><h3>' + s.title + '</h3><p>' + (s.content||'').replace(/\n/g,'<br>') + '</p>';
    if (s.cited_docs && s.cited_docs.length) html += '<div style="margin-top:8px;font-size:0.8em;color:#6b7280">引用: ' + s.cited_docs.join(', ') + '</div>';
    html += '</div>';
  }
  document.getElementById('report-output').innerHTML = html;
}

// ── 监控 ──
async function loadDashboard() {
  const m = await api('/metrics');
  if (!m) return;
  document.getElementById('m-cost').textContent = '$' + (m.cost.cost_usd||0).toFixed(2);
  document.getElementById('m-cost-sub').textContent = (m.cost.n_calls||0) + ' 次调用';
  document.getElementById('m-retlat').textContent = (m.retrieval_latency.p50||0).toFixed(1) + 's';
  document.getElementById('m-retlat-sub').textContent = 'P95: ' + (m.retrieval_latency.p95||0).toFixed(1) + 's';
  document.getElementById('m-llmlat').textContent = (m.llm_latency.p50||0).toFixed(1) + 's';
  document.getElementById('m-llmlat-sub').textContent = 'P95: ' + (m.llm_latency.p95||0).toFixed(1) + 's';
  document.getElementById('m-cache').textContent = ((m.cache_hit_rate||0)*100).toFixed(0) + '%';

  const ab = await api('/ab/results');
  const tbody = document.getElementById('ab-body');
  if (ab && ab.length) {
    tbody.innerHTML = ab.map(r => '<tr><td><span class="badge ' + (r.strategy==='fusion'?'badge-green':'badge-blue') + '">' + r.strategy + '</span></td><td>' + r.n + '</td><td>' + (r.accuracy*100).toFixed(0) + '%</td><td>' + (r.avg_latency||0).toFixed(1) + 's</td></tr>').join('');
  } else { tbody.innerHTML = '<tr><td colspan="4" style="color:#9ca3af;text-align:center">暂无数据</td></tr>'; }

  const recent = await api('/recent?limit=10');
  const rbody = document.getElementById('recent-body');
  if (recent && recent.length) {
    rbody.innerHTML = recent.map(r => '<tr><td>' + (r.timestamp||'').substring(11,19) + '</td><td>' + (r.query||'').substring(0,40) + '</td><td>' + r.strategy + '</td><td>' + r.n_results + '</td><td>' + (r.latency_s||0).toFixed(1) + 's</td></tr>').join('');
  } else { rbody.innerHTML = '<tr><td colspan="5" style="color:#9ca3af;text-align:center">暂无查询</td></tr>'; }
}
</script>
</body></html>"""
