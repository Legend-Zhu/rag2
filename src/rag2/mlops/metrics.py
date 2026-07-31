"""
RAG² MLOps: 监控指标采集（SQLite 后端）。

零外部依赖（sqlite3 是 Python 内置）。三张表：
  - llm_calls: LLM 调用记录（模型、token、延迟、缓存命中）
  - retrieval_calls: 检索调用记录（查询、策略、结果数、延迟）
  - ab_outcomes: A/B 测试结果（策略、查询、判决、是否正确）

通过回调注入 ModelGateway 和 FusionRetriever，不改现有逻辑。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    model TEXT,
    role TEXT,
    role_tag TEXT,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    latency_s REAL DEFAULT 0,
    from_cache INTEGER DEFAULT 0,
    cost_estimate REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS retrieval_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    query TEXT,
    strategy TEXT,
    corpus_id TEXT,
    n_results INTEGER DEFAULT 0,
    latency_s REAL DEFAULT 0,
    top_cids TEXT,
    gold_found INTEGER
);

CREATE TABLE IF NOT EXISTS ab_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    strategy TEXT,
    query TEXT,
    verdict TEXT,
    correct INTEGER,
    latency_s REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ingestion_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    corpus_id TEXT,
    n_files INTEGER DEFAULT 0,
    n_chunks INTEGER DEFAULT 0,
    parse_time_s REAL DEFAULT 0,
    chunk_time_s REAL DEFAULT 0,
    index_time_s REAL DEFAULT 0,
    total_time_s REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_llm_ts ON llm_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_ret_ts ON retrieval_calls(timestamp);
CREATE INDEX IF NOT EXISTS idx_ab_ts ON ab_outcomes(timestamp);
CREATE INDEX IF NOT EXISTS idx_ing_ts ON ingestion_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_ing_corpus ON ingestion_events(corpus_id);
CREATE INDEX IF NOT EXISTS idx_ab_strategy ON ab_outcomes(strategy);
"""

# 粗略成本估算（美元 / 1K tokens），可按需更新
_COST_PER_1K = {
    "deepseek-v4-flash": {"prompt": 0.001, "completion": 0.002},
    "k3": {"prompt": 0.003, "completion": 0.012},
    "qwen3.8-max-preview": {"prompt": 0.002, "completion": 0.006},
}


class MetricsCollector:
    """SQLite 指标采集器。线程安全（每次操作独立连接）。"""

    def __init__(self, db_path: str = "logs/metrics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # ── LLM 调用记录 ──────────────────────────────────────

    def record_llm_call(
        self,
        model: str,
        role: str = "",
        role_tag: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_s: float = 0,
        from_cache: bool = False,
    ):
        """记录一次 LLM 调用。可通过回调注入 ModelGateway。"""
        cost = self._estimate_cost(model, prompt_tokens, completion_tokens)
        ts = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO llm_calls (timestamp, model, role, role_tag, "
                "prompt_tokens, completion_tokens, latency_s, from_cache, cost_estimate) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, model, role, role_tag, prompt_tokens, completion_tokens,
                 latency_s, int(from_cache), cost),
            )
            conn.commit()

    def _estimate_cost(self, model: str, pt: int, ct: int) -> float:
        rates = _COST_PER_1K.get(model, {"prompt": 0.002, "completion": 0.006})
        return pt / 1000 * rates["prompt"] + ct / 1000 * rates["completion"]

    # ── 检索调用记录 ──────────────────────────────────────

    def record_retrieval(
        self,
        query: str,
        strategy: str = "fusion",
        corpus_id: str = "",
        n_results: int = 0,
        latency_s: float = 0,
        top_cids: list[str] | None = None,
        gold_found: bool | None = None,
    ):
        ts = datetime.utcnow().isoformat()
        cids_json = json.dumps(top_cids[:10]) if top_cids else ""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO retrieval_calls (timestamp, query, strategy, corpus_id, "
                "n_results, latency_s, top_cids, gold_found) VALUES (?,?,?,?,?,?,?,?)",
                (ts, query[:500], strategy, corpus_id, n_results, latency_s,
                 cids_json, int(gold_found) if gold_found is not None else None),
            )
            conn.commit()

    # ── A/B 结果记录 ─────────────────────────────────────
    # ── 入库事件记录 ─────────────────────────────────────

    def record_ingestion(
        self,
        corpus_id: str,
        n_files: int = 0,
        n_chunks: int = 0,
        parse_time_s: float = 0,
        chunk_time_s: float = 0,
        index_time_s: float = 0,
        total_time_s: float = 0,
    ):
        """记录一次文档入库事件。"""
        ts = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO ingestion_events (timestamp, corpus_id, n_files, n_chunks, "
                "parse_time_s, chunk_time_s, index_time_s, total_time_s) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ts, corpus_id, n_files, n_chunks,
                 parse_time_s, chunk_time_s, index_time_s, total_time_s),
            )
            conn.commit()

    def get_ingestion_summary(self, corpus_id: str | None = None, limit: int = 20) -> list[dict]:
        """获取入库历史。"""
        with self._conn() as conn:
            if corpus_id:
                rows = conn.execute(
                    "SELECT * FROM ingestion_events WHERE corpus_id=? ORDER BY id DESC LIMIT ?",
                    (corpus_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM ingestion_events ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def record_ab_outcome(
        self,
        strategy: str,
        query: str,
        verdict: str = "",
        correct: bool | None = None,
        latency_s: float = 0,
    ):
        ts = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO ab_outcomes (timestamp, strategy, query, verdict, "
                "correct, latency_s) VALUES (?,?,?,?,?,?)",
                (ts, strategy, query[:500], verdict,
                 int(correct) if correct is not None else None, latency_s),
            )
            conn.commit()

    # ── 查询方法 ─────────────────────────────────────────

    def get_cost_summary(self, days: int = 7) -> dict:
        """返回最近 N 天的成本摘要。"""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as n_calls, SUM(prompt_tokens) as pt, "
                "SUM(completion_tokens) as ct, SUM(cost_estimate) as cost, "
                "SUM(from_cache) as cache_hits "
                "FROM llm_calls WHERE timestamp >= ?",
                (since,),
            ).fetchone()
            return {
                "days": days,
                "n_calls": row["n_calls"] or 0,
                "prompt_tokens": row["pt"] or 0,
                "completion_tokens": row["ct"] or 0,
                "cost_usd": round(row["cost"] or 0, 4),
                "cache_hit_rate": (row["cache_hits"] or 0) / max(row["n_calls"] or 1, 1),
            }

    def get_latency_stats(self, table: str = "retrieval_calls", days: int = 7) -> dict:
        """返回延迟统计（p50/p95/max）。"""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT latency_s FROM {table} WHERE timestamp >= ? AND latency_s > 0",
                (since,),
            ).fetchall()
            if not rows:
                return {"p50": 0, "p95": 0, "max": 0, "n": 0}
            vals = sorted(r["latency_s"] for r in rows)
            n = len(vals)
            return {
                "p50": round(vals[n // 2], 2),
                "p95": round(vals[int(n * 0.95)], 2),
                "max": round(vals[-1], 2),
                "n": n,
            }

    def get_cache_hit_rate(self, days: int = 7) -> float:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as n, SUM(from_cache) as hits "
                "FROM llm_calls WHERE timestamp >= ?",
                (since,),
            ).fetchone()
            return (row["hits"] or 0) / max(row["n"] or 1, 1)

    def get_ab_results(self) -> list[dict]:
        """返回 A/B 测试汇总（按策略分组）。"""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT strategy, COUNT(*) as n, "
                "SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as n_correct, "
                "AVG(latency_s) as avg_latency "
                "FROM ab_outcomes GROUP BY strategy"
            ).fetchall()
            return [
                {
                    "strategy": r["strategy"],
                    "n": r["n"],
                    "accuracy": round((r["n_correct"] or 0) / max(r["n"], 1), 3),
                    "avg_latency": round(r["avg_latency"] or 0, 2),
                }
                for r in rows
            ]

    def get_recent_queries(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT timestamp, query, strategy, n_results, latency_s "
                "FROM retrieval_calls ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_metrics_summary(self) -> dict:
        """一次性返回所有关键指标（供 /metrics 端点）。"""
        return {
            "cost": self.get_cost_summary(7),
            "retrieval_latency": self.get_latency_stats("retrieval_calls", 7),
            "llm_latency": self.get_latency_stats("llm_calls", 7),
            "cache_hit_rate": self.get_cache_hit_rate(7),
            "ab_results": self.get_ab_results(),
        }


# ── Gateway 回调适配器 ──────────────────────────────────

def make_gateway_callback(mc: MetricsCollector):
    """创建 Gateway 回调函数，用于注入 ModelGateway。

    用法：
        gw = ModelGateway()
        gw.on_call_complete = make_gateway_callback(mc)
    """
    def callback(model: str, role: str, role_tag: str,
                 prompt_tokens: int, completion_tokens: int,
                 latency_s: float, from_cache: bool):
        mc.record_llm_call(
            model=model, role=role, role_tag=role_tag,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            latency_s=latency_s, from_cache=from_cache,
        )
    return callback
