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
  if (event && event.target) event.target.classList.add('active');
  if (name === 'dashboard') loadDashboard();
  if (name === 'library') loadIngestHistory();
}

// 统一 HTTP 请求（回调风格，无 async/await，无 fetch）
function httpGet(url, cb) {
  var xhr = new XMLHttpRequest();
  xhr.open('GET', API + url, true);
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { cb(null, JSON.parse(xhr.responseText)); }
        catch(e) { cb('JSON parse: ' + xhr.responseText.substring(0,100)); }
      } else { cb('HTTP ' + xhr.status); }
    }
  };
  xhr.send();
}
function httpPost(url, body, cb) {
  var xhr = new XMLHttpRequest();
  xhr.open('POST', API + url, true);
  xhr.setRequestHeader('Content-Type', 'application/json');
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      if (xhr.status >= 200 && xhr.status < 300) {
        try { cb(null, JSON.parse(xhr.responseText)); }
        catch(e) { cb('JSON parse: ' + xhr.responseText.substring(0,100)); }
      } else { cb('HTTP ' + xhr.status + ': ' + xhr.responseText.substring(0,100)); }
    }
  };
  xhr.send(JSON.stringify(body));
}

// ── 文档库 ──
function uploadFiles() {
  var input = document.getElementById('file-input');
  if (!input.files.length) return;
  document.getElementById('upload-status').innerHTML = '<div class="status status-info">上传中...</div>';
  var fd = new FormData();
  for (var i = 0; i < input.files.length; i++) fd.append('files', input.files[i]);
  var cid = document.getElementById('upload-corpus').value || 'upload';
  var xhr = new XMLHttpRequest();
  xhr.open('POST', API + '/upload?corpus_id=' + cid, true);
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          var d = JSON.parse(xhr.responseText);
          document.getElementById('upload-status').innerHTML =
            '<div class="status status-success">✓ 上传 ' + d.n_files + ' 文件, ' + d.n_chunks +
            ' chunks → 语料: ' + d.corpus_id + '</div>';
          addCorpusOption(d.corpus_id, d.n_chunks);
        } catch(e) {
          document.getElementById('upload-status').innerHTML = '<div class="status status-error">解析失败</div>';
        }
      } else {
        document.getElementById('upload-status').innerHTML = '<div class="status status-error">上传失败: HTTP ' + xhr.status + '</div>';
      }
    }
  };
  xhr.send(fd);
}
function ingestDir() {
  var dir = document.getElementById('ingest-dir').value;
  var cid = document.getElementById('ingest-corpus').value || 'docs';
  if (!dir) return;
  document.getElementById('ingest-status').innerHTML = '<div class="status status-info">入库中...</div>';
  httpPost('/ingest/dir', {dir_path: dir, corpus_id: cid, build_index: true}, function(err, d) {
    if (err) {
      document.getElementById('ingest-status').innerHTML = '<div class="status status-error">失败: ' + err + '</div>';
    } else if (d && d.n_files !== undefined) {
      document.getElementById('ingest-status').innerHTML =
        '<div class="status status-success">✓ ' + d.n_files + ' 文件 → ' + d.n_chunks + ' chunks (' + d.total_time_s + 's)</div>';
      loadIngestHistory();
    } else {
      document.getElementById('ingest-status').innerHTML = '<div class="status status-error">失败: ' + (d && d.error || '未知') + '</div>';
    }
  });
}
function loadIngestHistory() {
  httpGet('/ingest/status', function(err, d) {
    var tbody = document.getElementById('ingest-history');
    if (err || !d || !d.length) {
      tbody.innerHTML = '<tr><td colspan="5" style="color:#9ca3af;text-align:center">暂无记录</td></tr>';
      return;
    }
    var html = '';
    for (var i = 0; i < d.length; i++) {
      var r = d[i];
      html += '<tr><td>' + (r.timestamp||'').substring(0,19).replace('T',' ') + '</td>' +
        '<td>' + r.corpus_id + '</td><td>' + r.n_files + '</td><td>' + r.n_chunks + '</td>' +
        '<td>' + (r.total_time_s||0).toFixed(1) + 's</td></tr>';
    }
    tbody.innerHTML = html;
    refreshCorpusList(d);
  });
}
function refreshCorpusList(data) {
  var sel = document.getElementById('search-corpus');
  if (!sel) return;
  var current = sel.value;
  var opts = '<option value="">arXiv 2026（主语料）</option>';
  if (data) {
    var seen = {};
    for (var i = 0; i < data.length; i++) {
      var cid = data[i].corpus_id;
      if (cid && !seen[cid]) {
        seen[cid] = true;
        opts += '<option value="' + cid + '">' + cid + ' (' + data[i].n_chunks + ' chunks)</option>';
      }
    }
  }
  sel.innerHTML = opts;
  if (current) sel.value = current;
}
function addCorpusOption(cid, nChunks) {
  var sel = document.getElementById('search-corpus');
  if (!sel || !cid) return;
  for (var i = 0; i < sel.options.length; i++) {
    if (sel.options[i].value === cid) {
      sel.options[i].text = cid + ' (' + nChunks + ' chunks)';
      return;
    }
  }
  var opt = document.createElement('option');
  opt.value = cid;
  opt.text = cid + ' (' + nChunks + ' chunks)';
  sel.appendChild(opt);
}

// ── 检索 ──
function doSearch() {
  var q = document.getElementById('search-query').value;
  if (!q) return;
  var mode = document.getElementById('search-mode').value;
  var topk = parseInt(document.getElementById('search-topk').value);
  var corpus = document.getElementById('search-corpus').value;
  document.getElementById('search-results').innerHTML = '<div class="status status-info">搜索中...</div>';
  document.getElementById('search-time').textContent = '';
  var t0 = Date.now();
  var body = {query: q, top_k: topk, strategy: mode === 'vanilla' ? 'vanilla' : 'fusion'};
  if (corpus) body.corpus_id = corpus;
  httpPost('/search', body, function(err, d) {
    var dt = ((Date.now() - t0) / 1000).toFixed(2);
    document.getElementById('search-time').textContent = dt + 's';
    if (err) {
      document.getElementById('search-results').innerHTML = '<div class="status status-error">搜索失败: ' + err + '</div>';
      return;
    }
    if (!d || !d.results) {
      document.getElementById('search-results').innerHTML = '<div class="status status-error">无结果</div>';
      return;
    }
    var html = '<div class="card"><h2>结果（' + d.n_results + ' 条, 策略: ' + d.strategy + ', ' + dt + 's）</h2>';
    for (var i = 0; i < d.results.length; i++) {
      var r = d.results[i];
      var title = (r.title || r.heading || '').substring(0,80);
      var src = r.source || '';
      var badgeClass = src.indexOf('sparse') >= 0 ? 'badge-green' : 'badge-blue';
      var score = (r.score || 0).toFixed(4);
      var rerank = r.rerank_score ? ' → rerank: ' + r.rerank_score.toFixed(4) : '';
      var snippet = (r.text || '').substring(0,200);
      html += '<div class="result-item"><div class="title">' + title +
        ' <span class="badge ' + badgeClass + '">' + src + '</span></div>' +
        '<div class="meta">score: ' + score + rerank + '</div>' +
        '<div class="snippet">' + snippet + '...</div></div>';
    }
    html += '</div>';
    document.getElementById('search-results').innerHTML = html;
  });
}

// ── Agent 报告 ──
function generateReport() {
  var topic = document.getElementById('report-topic').value;
  if (!topic) return;
  var btn = document.getElementById('report-btn');
  btn.disabled = true; btn.textContent = '生成中...';
  document.getElementById('report-status').innerHTML = '<div class="status status-info">Agent 正在检索和写作（约 1-2 分钟）...</div>';
  document.getElementById('report-output').innerHTML = '';
  httpPost('/report/generate', {topic: topic, max_tool_calls: 25}, function(err, d) {
    btn.disabled = false; btn.textContent = '生成报告';
    if (err || !d) {
      document.getElementById('report-status').innerHTML = '<div class="status status-error">生成失败: ' + (err||'') + '</div>';
      return;
    }
    document.getElementById('report-status').innerHTML =
      '<div class="status status-success">✓ ' + d.sections.length + ' 章节, ' +
      d.total_llm_calls + ' 次 LLM 调用, ' + d.elapsed_s + 's</div>';
    var html = '<div class="card"><h2>' + topic + '</h2>';
    if (d.summary) html += '<p style="margin:12px 0;color:#374151;line-height:1.6">' + d.summary + '</p>';
    html += '</div>';
    for (var i = 0; i < d.sections.length; i++) {
      var s = d.sections[i];
      html += '<div class="section-preview"><h3>' + s.title + '</h3><p>' +
        (s.content||'').replace(/\n/g,'<br>') + '</p>';
      if (s.cited_docs && s.cited_docs.length) {
        html += '<div style="margin-top:8px;font-size:0.8em;color:#6b7280">引用: ' + s.cited_docs.join(', ') + '</div>';
      }
      html += '</div>';
    }
    document.getElementById('report-output').innerHTML = html;
  });
}

// ── 监控 ──
function loadDashboard() {
  httpGet('/metrics', function(err, m) {
    if (err || !m) return;
    document.getElementById('m-cost').textContent = '$' + ((m.cost && m.cost.cost_usd)||0).toFixed(2);
    document.getElementById('m-cost-sub').textContent = ((m.cost && m.cost.n_calls)||0) + ' 次调用';
    document.getElementById('m-retlat').textContent = ((m.retrieval_latency && m.retrieval_latency.p50)||0).toFixed(1) + 's';
    document.getElementById('m-retlat-sub').textContent = 'P95: ' + ((m.retrieval_latency && m.retrieval_latency.p95)||0).toFixed(1) + 's';
    document.getElementById('m-llmlat').textContent = ((m.llm_latency && m.llm_latency.p50)||0).toFixed(1) + 's';
    document.getElementById('m-llmlat-sub').textContent = 'P95: ' + ((m.llm_latency && m.llm_latency.p95)||0).toFixed(1) + 's';
    document.getElementById('m-cache').textContent = (((m.cache_hit_rate)||0)*100).toFixed(0) + '%';
  });
  httpGet('/ab/results', function(err, ab) {
    var tbody = document.getElementById('ab-body');
    if (err || !ab || !ab.length) {
      tbody.innerHTML = '<tr><td colspan="4" style="color:#9ca3af;text-align:center">暂无数据</td></tr>';
      return;
    }
    var html = '';
    for (var i = 0; i < ab.length; i++) {
      var r = ab[i];
      html += '<tr><td><span class="badge ' + (r.strategy==='fusion'?'badge-green':'badge-blue') + '">' + r.strategy + '</span></td><td>' + r.n + '</td><td>' + (r.accuracy*100).toFixed(0) + '%</td><td>' + (r.avg_latency||0).toFixed(1) + 's</td></tr>';
    }
    tbody.innerHTML = html;
  });
  httpGet('/recent?limit=10', function(err, recent) {
    var rbody = document.getElementById('recent-body');
    if (err || !recent || !recent.length) {
      rbody.innerHTML = '<tr><td colspan="5" style="color:#9ca3af;text-align:center">暂无查询</td></tr>';
      return;
    }
    var html = '';
    for (var i = 0; i < recent.length; i++) {
      var r = recent[i];
      html += '<tr><td>' + (r.timestamp||'').substring(11,19) + '</td><td>' + (r.query||'').substring(0,40) + '</td><td>' + r.strategy + '</td><td>' + r.n_results + '</td><td>' + (r.latency_s||0).toFixed(1) + 's</td></tr>';
    }
    rbody.innerHTML = html;
  });
}
</script>
</body></html>"""
