"""RAG² MLOps: 落地工程层。

组件：
  - metrics: SQLite 监控采集（LLM 调用 / 检索 / A/B 结果）
  - index_manager: 索引版本化 + 元数据管理
  - ab_router: A/B 测试路由
  - api_server: FastAPI 服务
  - dashboard: Streamlit 仪表盘
"""
from rag2.mlops.metrics import MetricsCollector
from rag2.mlops.index_manager import IndexManager
from rag2.mlops.ab_router import ABRouter

__all__ = ["MetricsCollector", "IndexManager", "ABRouter"]
