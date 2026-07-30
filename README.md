# RAG² — Temporal Evaluation（论文复现说明）

> 论文投稿中，暂未公开；本仓库为配套代码与数据。
> 核心主张：RAG 的准确率在"模型不知道的语料"（训练截止后发表）上才显现；公开基准测的是记忆。

---

## 目录结构

```
├── src/            # 主程序代码（rag2 包：检索、融合、评测、prompts）
├── config/         # 模型与管线配置（config.yaml，不含密钥）
├── run_e0.sh       # 主实验启动脚本
├── tests/          # 实验代码（主实验 test_ab_scale100.py 及各组件实验）
├── scripts/        # 实验脚本（数据下载、补充实验、统计）
├── data/           # 实验数据（arXiv 2026 语料、claims、人工标注）
├── data_raw/       # 原始基准数据（scifact / sciq / lfrqa / subsamples）
├── results/        # 实验结果（JSON + 统计 CSV）
├── docs/           # 技术文档（开发手册、架构决策、共享层方案）
└── cache/          # LLM 请求/索引缓存（本地生成，不入库）
```

## 环境

```bash
pip install -r requirements.txt
# 密钥放 .env（gitignore），gateway 启动时自动加载（src/rag2/gateway.py::_load_dotenv）
echo 'DMX_API_KEY=<你的 dmxapi key>' > .env
# 模型接入配置在 config/config.yaml（只存环境变量名，不含密钥）
```

主模型：`deepseek-v4-flash`（dmxapi.cn 代理，generator + verifier 同模型）。本地检索模型 `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3`（HF 缓存后离线）。

## 数据管线

| 步骤 | 脚本 | 产出 |
|---|---|---|
| 1. 下载 arXiv 语料 | `scripts/download_arxiv_corpus.py` | `data/arxiv_2026_corpus.json`（466 篇 cs.CL+cs.AI，2026-07 一周，模型训练截止后） |
| 2. 抽取 claim | `tests/test_arxiv_retrieval.py` | `data/arxiv_2026_claims.json`（每篇 1 条，启发式指标词） |
| 3. 改写 claim | 主跑脚本内（`reformulate_claim`） | `cache/arxiv_reformulated_claims.json`（LLM 换词，使检索非平凡） |

语料快照：原始 arXiv API XML 响应存于 `data/arxiv_*.xml`。

## 实验

```bash
# 主实验：A/B1/B2/B3/C 五条件（n=全部 466）
python tests/test_ab_scale100.py            # -> results/ab_scale466.json

# 补充实验（复用主跑缓存；可 --n 调规模）
python scripts/exp_extra.py forced          # P1-1 强制猜测 -> results/exp_forced_guess.json
python scripts/exp_extra.py b3decouple      # P1-3 B3 解耦 2x2 -> results/exp_b3_decouple.json
python scripts/exp_extra.py truncation --n 200  # P2-1 截断扫描 -> results/exp_truncation.json

# 统计：从结果 JSON 生成论文 CSV（accuracy + paired + McNemar，seed 固定）
python scripts/compute_stats.py results/ab_scale466.json
```

**条件定义**（`tests/test_ab_scale100.py`）：A 无检索 / B1 embedding top-3 / B2 融合(HyDE+grep+CrossEncoder) top-3 / B3 embedding top-20 不重排 / C 单文档 oracle（直接给金文档）。
**指标**：Accuracy = 判为 SUPPORTED 的比例（claim 摘自来源摘要，gold=SUPPORTED）；配对比较 + McNemar 精确检验（`src/rag2/eval/metrics.py`）；bootstrap 95% CI（10000 次，seed=42）。

**统一验证 prompt**：`src/rag2/prompts/verify.py`（n=30/n=100 旧双版本已合并；`forced=True` 为强制猜测变体）。所有喂论文的实验均引用此唯一来源。

## 人工标注（ground truth）

```bash
python scripts/build_annotation_set.py      # -> data/annotation/claims_to_annotate.csv
```

人工填 `human_label`/`confidence`/`notes`（见 `data/annotation/README.md`），用于报告 LLM-judge vs 人工一致率 + 争议样本敏感性分析。

## 关键缓存

- `cache/requests/` — 请求级 LLM 缓存（prompt-hash 键，改 prompt 即失效，保复现）
- `cache/arxiv_reformulated_claims.json` / `arxiv_hyde_rewrites.json` — 改写/HyDE 缓存
- `cache/indices/` + `cache/index_manifest.json` — embedding/FAISS 索引（corpus-hash 键，466 篇已建）
- `cache/grep_inverted_index.json` — grep 倒排索引（6.3MB，0.2s 建）

## 复现注意

- 改 `src/rag2/prompts/verify.py` 会使全部 verify 缓存失效 → 需重跑主实验（请求缓存自动断点续跑）。
- 统计数字一律由 `scripts/compute_stats.py` 从结果 JSON 生成，**不要手填 CSV**（历史上 cleaned 列就是这么算错的）。
