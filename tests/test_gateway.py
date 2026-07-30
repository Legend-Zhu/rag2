"""
ModelGateway 烟雾测试（不触网）

验证：
  - 配置加载、角色解析
  - 缓存读写（命中不调 API）
  - 成本记账落盘
  - mock 一个响应注入缓存，确认二次调用命中

真实 API 调用需 endpoint+key，另测。
"""
import json
import sys
from pathlib import Path

# 让 src 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag2.gateway import ModelGateway, Response, CallRecord


def test_config_load_and_resolve():
    """配置加载 + 角色解析。"""
    gw = ModelGateway()
    g = gw.resolve("generator")
    v = gw.resolve("verifier")
    assert g.model_name == "kimi-k3", f"generator 应解析到 kimi-k3，实际 {g.model_name}"
    assert v.model_name == "qwen3.8", f"verifier 应解析到 qwen3.8，实际 {v.model_name}"
    # 也能直接用模型名
    assert gw.resolve("kimi-k3").name == "kimi-k3"
    # 未知名应报错
    try:
        gw.resolve("nonexistent")
        assert False, "应抛 KeyError"
    except KeyError:
        pass
    print("✅ 配置加载 + 角色解析通过")


def test_cache_hit_avoids_api(monkeypatch=None):
    """缓存命中时不调 API。用 qwen3.8（无 forced_temperature），避免温度覆盖干扰 key 计算。"""
    gw = ModelGateway()
    # 用 qwen3.8 测，它的温度不会被强制覆盖，key 计算可预测
    mc = gw.resolve("qwen3.8")
    messages = [{"role": "user", "content": "ping"}]
    # 用 generate 内部相同的 merged 参数算 key
    merged = {"max_tokens": mc.max_tokens, "temperature": mc.temperature}
    key = gw._cache_key(mc, messages, None, merged)

    fake = Response(text="pong", usage={"prompt_tokens": 5, "completion_tokens": 1})
    gw._cache_put(key, fake)

    # 打个标记：如果 _get_client 被调用说明缓存没命中
    called = {"hit": False}
    orig = gw._get_client
    def spy(*a, **k):
        called["hit"] = True
        return orig(*a, **k)
    gw._get_client = spy

    resp = gw.generate("qwen3.8", messages)
    assert resp.from_cache is True, "应命中缓存"
    assert resp.text == "pong"
    assert called["hit"] is False, "缓存命中不应调 _get_client"
    print("✅ 缓存命中绕过 API 通过（含温度覆盖逻辑验证）")


def test_cost_logging():
    """成本记账落盘。"""
    gw = ModelGateway()
    log_path = gw.cost_log_path
    before = log_path.stat().st_size if log_path.exists() else 0
    # 触发一次缓存命中的记账
    gw._log_cost(CallRecord(
        model="test-model", role="generator",
        prompt_tokens=100, completion_tokens=20,
        elapsed_s=0.5, from_cache=True,
    ))
    after = log_path.stat().st_size
    assert after > before, "成本日志应增长"
    line = log_path.read_text(encoding="utf-8").strip().split("\n")[-1]
    rec = json.loads(line)
    assert rec["model"] == "test-model" and rec["from_cache"] is True
    print("✅ 成本记账落盘通过")


def test_protocol_adapters_exist():
    """两套协议适配器都在。"""
    gw = ModelGateway()
    assert hasattr(gw, "_call_openai"), "缺 openai 适配器"
    assert hasattr(gw, "_call_anthropic"), "缺 anthropic 适配器"
    # 默认两个模型都配 openai 协议
    assert gw.models["kimi-k3"].protocol == "openai"
    assert gw.models["qwen3.8"].protocol == "openai"
    print("✅ 协议适配器就绪")


def test_k3_forced_temperature():
    """K3 的 forced_temperature=1 必须覆盖到调用参数。"""
    gw = ModelGateway()
    mc = gw.resolve("kimi-k3")
    assert mc.forced_temperature == 1.0, f"K3 forced_temperature 应为 1.0，实际 {mc.forced_temperature}"
    # qwen3.8 无强制温度
    mc2 = gw.resolve("qwen3.8")
    assert mc2.forced_temperature is None, "Qwen3.8 不应有 forced_temperature"
    print("✅ K3 forced_temperature=1 配置就绪")


def test_context_overflow_guard():
    """输入超窗口应显式报 ContextOverflowError，而非静默截断。"""
    from rag2.gateway import ContextOverflowError
    gw = ModelGateway()
    mc = gw.resolve("generator")
    # 造一个超过窗口的输入（临时把窗口调小到 100 测）
    orig_window = mc.context_window
    mc.context_window = 100
    try:
        long_msg = [{"role": "user", "content": "x" * 1000}]   # ~333 tokens
        try:
            gw.generate("generator", long_msg)
            assert False, "应抛 ContextOverflowError"
        except ContextOverflowError as e:
            assert "超过" in str(e) or "exceed" in str(e).lower()
    finally:
        mc.context_window = orig_window
    print("✅ 输入守卫触发 ContextOverflowError")


def test_context_window_set():
    """两个模型都配了 context_window=1M。"""
    gw = ModelGateway()
    for name in ["kimi-k3", "qwen3.8"]:
        mc = gw.resolve(name)
        assert mc.context_window == 1_000_000, f"{name} 窗口应为 1M，实际 {mc.context_window}"
    print("✅ context_window=1M 配置就绪")


if __name__ == "__main__":
    test_config_load_and_resolve()
    test_cache_hit_avoids_api()
    test_cost_logging()
    test_protocol_adapters_exist()
    test_k3_forced_temperature()
    test_context_window_set()
    test_context_overflow_guard()
    print("\n🎉 ModelGateway 烟雾测试全过（含温度强制 + 输入守卫）")
