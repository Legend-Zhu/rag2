"""
统一验证 prompt（claim verification）。

背景：n=30 pilot（test_ab_accuracy.py）与 n=100 主实验（test_ab_scale100.py）此前用
两份不同的 verifier prompt（n=30 更严格），导致截断/agent 等衍生结果与主表不可比。
本模块为唯一来源，所有喂论文的实验（主表、截断扫描、agent、强制猜测基线）都引用此处。

默认三分类（SUPPORTED/REFUTED/NOT_ENOUGH_INFO）；forced=True 为强制猜测变体（P1-1，
禁止弃权以分离拒答效应）。
"""
from __future__ import annotations

import json
from typing import Any

# ─────────────────────────────────────────────────────────
# 默认：三分类验证
# ─────────────────────────────────────────────────────────
VERIFY_SYSTEM = """You are a scientific claim verifier. Given a scientific claim, determine if it is:
- SUPPORTED: the evidence clearly and directly supports the claim
- REFUTED: the evidence clearly contradicts the claim
- NOT_ENOUGH_INFO: insufficient evidence to determine

Call the verify tool with your verdict. Be rigorous: only return SUPPORTED if the evidence directly supports the claim; do not infer beyond what is stated."""

VERIFY_TOOL = [{'type': 'function', 'function': {'name': 'verify', 'parameters': {
    'type': 'object',
    'properties': {
        'verdict': {'type': 'string', 'enum': ['SUPPORTED', 'REFUTED', 'NOT_ENOUGH_INFO']},
        'reason': {'type': 'string'},
    },
    'required': ['verdict'],
}}}]

# ─────────────────────────────────────────────────────────
# 强制猜测变体（P1-1：禁止 NOT_ENOUGH_INFO，分离拒答效应）
# ─────────────────────────────────────────────────────────
VERIFY_SYSTEM_FORCED = """You are a scientific claim verifier. Given a scientific claim, you MUST decide between:
- SUPPORTED: the evidence leans toward supporting the claim
- REFUTED: the evidence leans toward contradicting the claim

You may NOT answer NOT_ENOUGH_INFO. Commit to SUPPORTED or REFUTED even if uncertain, choosing whichever the evidence more favors. Call the verify tool."""

VERIFY_TOOL_FORCED = [{'type': 'function', 'function': {'name': 'verify', 'parameters': {
    'type': 'object',
    'properties': {
        'verdict': {'type': 'string', 'enum': ['SUPPORTED', 'REFUTED']},
        'reason': {'type': 'string'},
    },
    'required': ['verdict'],
}}}]

# forced 模式下文本兜底解析的候选顺序（无 NEI）
_FORCED_TEXT_ORDER = ['REFUTED', 'SUPPORTED']
_DEFAULT_TEXT_ORDER = ['NOT_ENOUGH_INFO', 'SUPPORTED', 'REFUTED']


def build_user_message(claim: str, context: str | None, forced: bool = False) -> str:
    """构造验证 user message。context=None 凭记忆，否则基于证据。"""
    msg = f'Claim: {claim}'
    if context:
        msg += (f'\n\n--- Evidence ---\n{context}\n--- End Evidence ---\n\n'
                f'Based ONLY on the evidence above, verify the claim.')
    else:
        if forced:
            msg += ('\n\nBased on your knowledge, verify this claim. '
                    'You must commit to SUPPORTED or REFUTED.')
        else:
            msg += ('\n\nBased on your knowledge, verify this claim. '
                    'If you do not have sufficient information, answer NOT_ENOUGH_INFO.')
    return msg


def parse_verdict(resp: Any, forced: bool = False) -> str:
    """从 gateway 响应解析 verdict：优先 tool_call，其次文本兜底。"""
    order = _FORCED_TEXT_ORDER if forced else _DEFAULT_TEXT_ORDER
    for tc in getattr(resp, 'tool_calls', []) or []:
        if tc.get('function', {}).get('name') == 'verify':
            try:
                return json.loads(tc['function']['arguments']).get('verdict', 'UNKNOWN')
            except Exception:
                pass
    text = (getattr(resp, 'text', '') or '').upper()
    for v in order:
        if v in text:
            return v
    return 'UNKNOWN'


def verify_claim(gw, model: str, claim: str, context: str | None = None,
                 forced: bool = False, max_tokens: int = 500) -> str:
    """
    用 gateway 调模型验证 claim，返回 verdict 字符串。

    Args:
        gw: ModelGateway 实例
        model: 模型名
        claim: 待验证（通常为改写后的）claim
        context: 检索/oracle 证据；None 则凭记忆
        forced: True 走强制猜测变体（禁 NEI）
    """
    system = VERIFY_SYSTEM_FORCED if forced else VERIFY_SYSTEM
    tool = VERIFY_TOOL_FORCED if forced else VERIFY_TOOL
    role_tag = 'verify_forced' if forced else 'verify'
    user_msg = build_user_message(claim, context, forced=forced)
    resp = gw.generate(
        model,
        [{'role': 'system', 'content': system}, {'role': 'user', 'content': user_msg}],
        tools=tool, role_tag=role_tag, max_tokens=max_tokens,
    )
    return parse_verdict(resp, forced=forced)
