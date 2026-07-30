"""RAG² prompt 模板（统一来源，取代各实验脚本内联的 prompt）。"""
from rag2.prompts.verify import (  # noqa: F401
    VERIFY_SYSTEM,
    VERIFY_TOOL,
    VERIFY_SYSTEM_FORCED,
    VERIFY_TOOL_FORCED,
    build_user_message,
    parse_verdict,
    verify_claim,
)
