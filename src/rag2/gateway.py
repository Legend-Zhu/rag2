"""
RAG² ModelGateway — 统一模型接入层

职责：
  - 把 Kimi K3 与 Qwen3.8 统一成一个接口，屏蔽协议差异
  - 配置驱动切换（YAML），代码不硬编码模型/角色
  - 请求级缓存（API-only 下任何重复调用都是浪费）
  - 成本记账（token 用量 + 估算花费）

设计要点：
  - 两者均兼容 OpenAI 协议 → 主路径走 openai SDK
  - 保留 anthropic 协议备选（某模型在 openai 兼容层行为异常时切）
  - 角色解耦：同一模型可作 generator 或 verifier，由配置决定
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """单个模型的接入配置。"""
    name: str                  # 配置键名，如 "kimi-k3"
    protocol: str              # openai | anthropic
    base_url: str
    api_key_env: str
    model_name: str            # 实际传给 API 的模型名
    max_tokens: int = 8192     # 单次输出最大长度（非上下文窗口！）
    temperature: float = 0.0
    context_window: int = 1_000_000   # 上下文窗口（输入守卫 + 路由用）
    forced_temperature: float | None = None   # API 强制的温度（如 K3=1），None 则用 temperature


class ContextOverflowError(Exception):
    """输入超过模型上下文窗口。E0 长上下文实验的关键守卫——防静默截断。"""


@dataclass
class Response:
    """统一响应结构，屏蔽协议差异。"""
    text: str
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)   # {prompt_tokens, completion_tokens}
    raw: Any = None                             # 原始返回，便于 debug
    from_cache: bool = False
    finish_reason: str = ""                     # stop / length / tool_calls / content_filter


@dataclass
class CallRecord:
    """单次调用记账（落盘汇总成本）。"""
    model: str
    role: str                  # generator | verifier
    prompt_tokens: int
    completion_tokens: int
    elapsed_s: float
    from_cache: bool
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────
# ModelGateway
# ─────────────────────────────────────────────────────────

class ModelGateway:
    """统一模型接入层。上层只看 generate(...)，不关心底下走哪条协议。"""

    def __init__(self, config_path: str = "config/config.yaml"):
        self._load_dotenv()  # 先加载 .env（密钥不进 config，gitignore 保护）
        cfg = self._load_config(config_path)
        self.models: dict[str, ModelConfig] = {
            k: ModelConfig(name=k, **v) for k, v in cfg["models"].items()
        }
        self.roles: dict[str, str] = cfg["roles"]
        self.cache_cfg = cfg.get("cache", {})
        self.cache_enabled = self.cache_cfg.get("enabled", True)
        self.cache_dir = Path(self.cache_cfg.get("request_cache_dir", "cache/requests"))
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        # 成本记账
        self.cost_log_path = Path("logs/cost.jsonl")
        self.cost_log_path.parent.mkdir(parents=True, exist_ok=True)
        # 懒加载的 client 池
        self._clients: dict[str, Any] = {}

    @staticmethod
    def _load_config(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _load_dotenv(env_path: str = ".env") -> None:
        """从 .env 读 KEY=VALUE 注入环境变量（不覆盖已存在的）。

        密钥（如 DMX_API_KEY）放 .env（gitignore），config.yaml 只存环境变量名，
        避免密钥进版本控制。无 python-dotenv 依赖的精简实现。
        """
        p = Path(env_path)
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

    # ── 角色解析 ──────────────────────────────────────────

    def resolve(self, role_or_name: str) -> ModelConfig:
        """接受角色名（generator/verifier）或模型名，返回 ModelConfig。"""
        if role_or_name in self.roles:
            name = self.roles[role_or_name]
        elif role_or_name in self.models:
            name = role_or_name
        else:
            raise KeyError(f"未知角色或模型: {role_or_name}")
        return self.models[name]

    # ── client 懒加载 ────────────────────────────────────

    def _get_client(self, mc: ModelConfig):
        """按协议返回 client，缓存复用。"""
        if mc.name in self._clients:
            return self._clients[mc.name]
        api_key = os.environ.get(mc.api_key_env, "")
        if not api_key:
            logger.warning("环境变量 %s 未设置（%s），调用将失败", mc.api_key_env, mc.name)
        # 超时保护：避免 thinking + 工具调用组合下 API 挂起导致死等
        # thinking 模式可能很慢，给 180s（3 分钟）单次调用上限
        if mc.protocol == "openai":
            from openai import OpenAI
            client = OpenAI(
                base_url=mc.base_url or None, api_key=api_key or "dummy",
                timeout=180.0, max_retries=2,
            )
        elif mc.protocol == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(
                base_url=mc.base_url or None, api_key=api_key or "dummy",
                timeout=180.0, max_retries=2,
            )
        else:
            raise ValueError(f"不支持协议: {mc.protocol}")
        self._clients[mc.name] = client
        return client

    # ── 缓存 ─────────────────────────────────────────────

    def _cache_key(self, mc: ModelConfig, messages: list, tools: list | None, params: dict) -> str:
        """稳定哈希作 cache key。"""
        payload = json.dumps(
            {"model": mc.model_name, "messages": messages, "tools": tools, "params": params},
            sort_keys=True, ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cache_get(self, key: str) -> Response | None:
        if not self.cache_enabled:
            return None
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Response(**data, from_cache=True)
        return None

    def _cache_put(self, key: str, resp: Response) -> None:
        if not self.cache_enabled:
            return
        path = self.cache_dir / f"{key}.json"
        payload = {"text": resp.text, "tool_calls": resp.tool_calls, "usage": resp.usage}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 成本记账 ─────────────────────────────────────────

    def _log_cost(self, rec: CallRecord) -> None:
        with self.cost_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    # ── 主接口 ───────────────────────────────────────────

    def generate(
        self,
        role_or_name: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        role_tag: str = "generator",
        **params,
    ) -> Response:
        """
        统一生成入口。

        Args:
            role_or_name: 角色名(generator/verifier) 或模型名(kimi-k3/qwen3.8)
            messages: OpenAI 格式消息列表
            tools: 可选工具定义（agentic 检索用）
            role_tag: 记账用角色标签
            **params: 覆盖默认推理参数(temperature 等)
        Returns:
            Response
        """
        mc = self.resolve(role_or_name)
        # forced_temperature 优先（如 K3 API 强制温度=1），否则用默认/传入的 temperature
        # 注意：必须在算缓存键之前生效，否则温度变化会导致缓存键不一致
        if mc.forced_temperature is not None:
            params["temperature"] = mc.forced_temperature
        merged = {"max_tokens": mc.max_tokens, "temperature": mc.temperature, **params}

        # 输入长度守卫——防 E0 长上下文实验静默截断（最危险的失败模式）
        input_tokens = self._estimate_tokens(messages, tools)
        if input_tokens > mc.context_window:
            raise ContextOverflowError(
                f"输入 {input_tokens} tokens 超过 {mc.model_name} 上下文窗口 "
                f"{mc.context_window}（差 {input_tokens - mc.context_window}）。"
                f"这是显式守卫，避免 API 静默截断导致实验数据失真。"
            )

        # 1. 查缓存
        key = self._cache_key(mc, messages, tools, merged)
        cached = self._cache_get(key)
        if cached is not None:
            self._log_cost(CallRecord(
                model=mc.model_name, role=role_tag,
                prompt_tokens=cached.usage.get("prompt_tokens", 0),
                completion_tokens=cached.usage.get("completion_tokens", 0),
                elapsed_s=0.0, from_cache=True,
            ))
            return cached

        # 2. 实调（带指数退避重试，兜住 ServiceIsBusy/5xx/429 等临时过载）
        client = self._get_client(mc)
        t0 = time.time()
        resp = self._call_with_retry(client, mc, messages, tools, merged)
        elapsed = time.time() - t0

        # 3. 缓存 + 记账
        self._cache_put(key, resp)
        self._log_cost(CallRecord(
            model=mc.model_name, role=role_tag,
            prompt_tokens=resp.usage.get("prompt_tokens", 0),
            completion_tokens=resp.usage.get("completion_tokens", 0),
            elapsed_s=elapsed, from_cache=False,
        ))
        return resp

    # ── 截断续写 loop ────────────────────────────────────

    def generate_complete(
        self,
        role_or_name: str,
        messages: list[dict],
        max_continuations: int = 5,
        role_tag: str = "generator",
        **params,
    ) -> Response:
        """
        带截断自动续写的生成。

        当 finish_reason=length（输出被 max_tokens 截断）时，自动追加
        "continue" 消息让模型续写，拼接成完整输出。适合长 JSON/结构化输出。

        max_continuations: 最多续写次数（防失控）。每次续写用相同 max_tokens。
        """
        accumulated_text = ""
        accumulated_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        last_resp = None
        finish_reason = ""

        for cont_idx in range(max_continuations + 1):
            resp = self.generate(role_or_name, messages, role_tag=role_tag, **params)
            last_resp = resp
            accumulated_text += resp.text
            accumulated_usage["prompt_tokens"] += resp.usage.get("prompt_tokens", 0)
            accumulated_usage["completion_tokens"] += resp.usage.get("completion_tokens", 0)
            finish_reason = resp.finish_reason

            # 没被截断 → 完成
            if finish_reason != "length":
                break

            # 被截断 → 追加续写消息
            logger.info(
                "输出被截断（finish_reason=length），续写第 %d 次（已累积 %d 字符）",
                cont_idx + 1, len(accumulated_text),
            )
            # 把模型的截断输出作为 assistant 消息，再加 "continue" 引导续写
            messages = list(messages) + [
                {"role": "assistant", "content": resp.text},
                {"role": "user", "content": "Continue your previous output exactly from where it stopped. Do not repeat what you already wrote. Output ONLY the continuation."},
            ]

        # 合并最终 Response
        return Response(
            text=accumulated_text,
            tool_calls=last_resp.tool_calls if last_resp else [],
            usage=accumulated_usage,
            raw=last_resp.raw if last_resp else None,
            finish_reason=finish_reason,
            from_cache=last_resp.from_cache if last_resp else False,
        )

    # ── 输入长度估算 ──────────────────────────────────────

    @staticmethod
    def _estimate_tokens(messages: list[dict], tools: list[dict] | None) -> int:
        """
        粗估输入 token 数（用于守卫，非精确计数）。
        规则：英文 ~4 字符/token，中文 ~1.5 字符/token，取折中 3 字符/token 偏保守。
        每条 message 加固定开销 4 token（role 标记等）。
        """
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                # 多模态/工具结果可能为 list
                content = " ".join(str(x) for x in content)
            total += len(str(content)) / 3 + 4
        if tools:
            for t in tools:
                total += len(str(t)) / 3
        return int(total)

    # ── 临时错误重试 ─────────────────────────────────────

    @staticmethod
    def _is_retryable(e: Exception) -> bool:
        """判断是否可重试的临时错误（过载/服务端/连接/限流），认证与参数错误立即失败。"""
        msg = str(e).lower()
        name = type(e).__name__.lower()
        # 明确不可重试：认证 / 参数错误
        if any(k in msg for k in ("incorrect api key", "invalid api key", "authentication",
                                  "unauthorized", "invalid_api_key", "does not exist")):
            return False
        if any(k in name for k in ("timeout", "connection", "ratelimit")):
            return True
        return any(k in msg for k in ("busy", "overloaded", "503", "502", "500", "429",
                                      "rate limit", "service unavailable", "temporarily",
                                      "engine busy", "serviceisbusy"))

    def _call_with_retry(self, client, mc, messages, tools, merged, max_attempts: int = 6) -> Response:
        """对临时性 API 错误做指数退避重试（delay 5s→120s 封顶），避免长实验被瞬时过载打断。"""
        delay = 5.0
        for attempt in range(1, max_attempts + 1):
            try:
                if mc.protocol == "openai":
                    return self._call_openai(client, mc, messages, tools, merged)
                return self._call_anthropic(client, mc, messages, tools, merged)
            except Exception as e:
                if not self._is_retryable(e) or attempt == max_attempts:
                    raise
                logger.warning("API 临时错误(attempt %d/%d, %ss 后重试): %s",
                               attempt, max_attempts, delay, str(e)[:120])
                time.sleep(delay)
                delay = min(delay * 2, 120.0)

    def _call_openai(self, client, mc, messages, tools, params) -> Response:
        # ⚠️ 必须用流式：非流式请求服务端固定 300s 超时，thinking 模式会超时挂起
        # 文档：https://platform.qianwenai.com/docs/developer-guides/run-and-scale/streaming
        # 流式下：content/reasoning_content/tool_calls 都是增量 delta，需累积拼接
        kwargs = {
            "model": mc.model_name, "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            **params,
        }
        if tools:
            kwargs["tools"] = tools
        stream = client.chat.completions.create(**kwargs)

        accumulated_content = ""
        accumulated_reasoning = ""
        tool_calls_acc: dict[int, dict] = {}  # index → {id, function:{name, arguments}}
        finish_reason = ""
        usage = {}

        for chunk in stream:
            if chunk.usage:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                }
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            # 累积 content
            if delta.content:
                accumulated_content += delta.content
            # 累积 reasoning_content（thinking 模式）
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                accumulated_reasoning += delta.reasoning_content
            # 累积 tool_calls（分块返回，需按 index 拼接 arguments）
            if delta.tool_calls:
                for tc_chunk in delta.tool_calls:
                    idx = tc_chunk.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": tc_chunk.id or "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_chunk.id:
                        tool_calls_acc[idx]["id"] = tc_chunk.id
                    if tc_chunk.function:
                        if tc_chunk.function.name:
                            tool_calls_acc[idx]["function"]["name"] += tc_chunk.function.name
                        if tc_chunk.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += tc_chunk.function.arguments

        tool_calls = [tool_calls_acc[i] for i in sorted(tool_calls_acc.keys())]
        # 把 reasoning_content 附加到 content（便于下游使用，可选）
        # 注意：多轮对话传回 assistant 时需保留 reasoning_content（文档要求）
        return Response(
            text=accumulated_content,
            tool_calls=tool_calls, usage=usage,
            raw=accumulated_reasoning[:500] if accumulated_reasoning else "",
            finish_reason=finish_reason,
        )

    def _call_anthropic(self, client, mc, messages, tools, params) -> Response:
        # Anthropic 协议：分离 system 与 user/assistant
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [m for m in messages if m["role"] != "system"]
        r = client.messages.create(
            model=mc.model_name, system=system, messages=rest,
            max_tokens=params["max_tokens"], temperature=params["temperature"],
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        usage = {"prompt_tokens": r.usage.input_tokens, "completion_tokens": r.usage.output_tokens}
        return Response(text=text, tool_calls=[], usage=usage, raw=str(r))
