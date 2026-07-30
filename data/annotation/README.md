# 人工标注说明（P0-2 人工 ground truth）

## 目的
论文假定"所有从摘要抽取的 claim 真值均为 SUPPORTED"，但无人工验证。本标注提供独立人工 ground truth，用于：
1. 报告 LLM-judge（条件 C oracle）与人工判断的一致率（agreement + Cohen's κ）。
2. 裁决 3 条争议样本（oracle ≠ SUPPORTED），做敏感性分析。

## 文件
- `claims_to_annotate.csv` — 标注表（**你要填的文件**），争议样本已置顶。
- `all_claims_verdicts.csv` — 全量 verdict 参考（n=100 当前 run 的系统判断）。

## 标注任务
对每一行，阅读 `reformulated_claim`（模型实际验证的改写 claim）与 `abstract`（来源论文摘要），判断：

> 该 claim 是否被其来源摘要的证据所支持？

填三列：
| 列 | 取值 | 说明 |
|---|---|---|
| `human_label` | `SUPPORTED` / `REFUTED` / `NOT_ENOUGH_INFO` | 摘要证据明确支持 / 明确反驳 / 证据不足 |
| `confidence` | `high` / `med` / `low` | 你对该判断的把握 |
| `notes` | 自由文本 | 简述理由（尤其 REFUTED/NEI） |

## 判定标准
- **SUPPORTED**：摘要中有直接证据支持该 claim（即使措辞不同）。
- **REFUTED**：摘要证据与 claim 矛盾（如 claim 说"提升"，摘要说"下降"）。
- **NOT_ENOUGH_INFO**：claim 过于模糊、或摘要未涉及 claim 所断言的具体点，无法判定。
- 仅依据**该论文自身摘要**判定（与 oracle 条件 C 一致：单文档来源）。不要借助外部知识或检索。

## 优先级
1. 先标 `is_disputed=True` 的 3 条（idx 3 / 9 / 15）——这是审稿人最关心的。
2. 再标其余。当前 n=100 全部待标（`to_annotate=True`）。

## 备注
- `verdict_A/B1/B2/B3/C` 列是系统当前判断（n=100 run，统一 prompt 重跑后会刷新），**仅供参照，不要让其影响你的独立判断**。
- `gold_recall_B1/B2/B3` 表示该条件是否检索到了金文档。
- 标注完成后交回，我会算一致率并对 `human_label ≠ SUPPORTED` 的 claim 修订全表 + 写敏感性分析（tex+md 同步）。
