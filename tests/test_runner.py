"""
ExperimentRunner 端到端 dry-run 测试

用 MockGateway 替代真实 LLM（不触网），验证：
  - 配置驱动加载
  - TraditionalRAG 建索引 + 检索（真实 bge-m3 + FAISS）
  - 断点续跑（中断后再跑跳过已完成）
  - 评测 + bootstrap CI
  - 结果落盘

真实 LLM 调用需 endpoint+key，另测。
"""
import json
import sys
from pathlib import Path
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag2.gateway import ModelGateway, Response, ModelConfig, CallRecord
from rag2.runner import ExperimentRunner, ExperimentConfig


class MockGateway(ModelGateway):
    """模拟 gateway：不触网，从 supporting_docs 里抽答案片段作 mock 响应。"""

    def generate(self, role_or_name, messages, tools=None, role_tag="generator", **params):
        # 从最后一条 user message 里提取 question，造个 mock 答案
        last = messages[-1]["content"] if messages else ""
        # mock 策略：如果 prompt 里有 "Tracy" 相关，答 Tracy（模拟检索成功）
        # 否则答一个固定串（模拟生成）
        if "Mother" in last or "narrator" in last:
            text = "Tracy McConnell"
        else:
            text = "mock answer"
        return Response(
            text=text,
            usage={"prompt_tokens": 100, "completion_tokens": 5},
            raw='{"finish_reason": "stop"}',
        )


def test_dry_run_e2e():
    """端到端：真实数据 + mock LLM，跑 TraditionalRAG + LongContext。"""
    print("=== 端到端 dry-run（MockGateway，真实 MuSiQue 数据）===")
    gw = MockGateway()
    runner = ExperimentRunner(gw)

    # 用前 5 题快速验证（不跑满 1000）
    exp = ExperimentConfig(
        name="test_dry_run",
        dataset="musique",
        methods=["long_context", "traditional_rag"],
        model="kimi-k3",
        sample_n=5,
    )

    print(f"\n实验: {exp.name}")
    print(f"  dataset={exp.dataset}, methods={exp.methods}, n={exp.sample_n}")

    summary = runner.run(exp)

    # 验证结果结构
    assert "results" in summary, "缺 results"
    assert "long_context" in summary["results"], "缺 long_context 结果"
    assert "traditional_rag" in summary["results"], "缺 traditional_rag 结果"

    for method, metrics in summary["results"].items():
        print(f"\n  [{method}] 指标:")
        for m, v in metrics.items():
            if isinstance(v, dict) and "mean" in v:
                print(f"    {m}: {v['mean']:.2f} CI=[{v['ci_lo']:.2f}, {v['ci_hi']:.2f}]")

    # 验证落盘
    out_dir = Path("results") / exp.name
    assert (out_dir / "config.json").exists(), "config.json 未落盘"
    assert (out_dir / "summary.json").exists(), "summary.json 未落盘"
    assert (out_dir / "long_context_per_sample.jsonl").exists()
    assert (out_dir / "traditional_rag_per_sample.jsonl").exists()
    print("\n  ✅ 结果文件全部落盘")

    # 验证断点续跑
    print("\n=== 断点续跑测试（再跑一次应跳过已完成）===")
    summary2 = runner.run(exp)
    # per_sample 行数应不变（5 题）
    lc_file = out_dir / "long_context_per_sample.jsonl"
    lines = lc_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5, f"断点续跑后应为 5 行，实际 {len(lines)}"
    print(f"  ✅ 断点续跑正确（{len(lines)} 行，未重复）")

    print("\n🎉 ExperimentRunner 端到端 dry-run 全过")


def test_results_reproducible():
    """两次跑同样配置，结果应一致（可复现）。"""
    print("\n=== 可复现性测试 ===")
    gw = MockGateway()
    runner = ExperimentRunner(gw)
    exp = ExperimentConfig(
        name="test_repro", dataset="musique",
        methods=["long_context"], model="kimi-k3", sample_n=3,
    )
    # 清掉旧结果
    out_dir = Path("results") / exp.name
    if out_dir.exists():
        for f in out_dir.glob("*"):
            f.unlink()
    s1 = runner.run(exp)
    em1 = s1["results"]["long_context"]["em"]["mean"]
    # 清缓存再跑
    for f in out_dir.glob("*"):
        f.unlink()
    s2 = runner.run(exp)
    em2 = s2["results"]["long_context"]["em"]["mean"]
    assert em1 == em2, f"两次 EM 应相同: {em1} vs {em2}"
    print(f"  ✅ 可复现（EM={em1} 两次一致）")


if __name__ == "__main__":
    test_dry_run_e2e()
    test_results_reproducible()
    print("\n🎉🎉 ExperimentRunner 全部测试通过")
