# 五候选明确更新：措辞与候选顺序对照

状态更新：60 次真实调用已完成并独立复算，更新/状态 60/60，顺序配对 30/30、措辞组 20/20、任务组 10/10 均正确，
没有调用失败、非法输出或干扰项误选。下一步[代表性导航验证](language_navigation_representatives.md)已完成 CPU 准备，15 个新 episode 待运行。
本页真实推理与评分命令已执行，已有输出不覆盖。

入口：`python -m scripts.language_navigation_robustness`。
这是基于已验收 HSSD 五候选任务的**离线任务更新实验**，不创建新的导航 episode。
固定 Qwen、系统提示、候选描述/别名、旧策略状态、生成参数和状态应用规则，只改变当前消息与候选排列。
不继续 history/state 对比，不加入歧义、拒绝或视觉定位实验。

## 1. 实验矩阵与语言来源

来源 `outputs/hssd_language_candidates5_001` 的 10 个固定基础任务。五候选导航批次已经独立验收：
30 个全新 episode 完整、更新和导航闭环均为 10/10、无干扰目标误选，Oracle/Ignore 与两候选的轨迹和观测匹配。
新实验全部重新调用模型，不复用此前预测。

每个任务生成以下六个请求：

| 因素 | 条件 |
| --- | --- |
| 当前消息 | `original`、`paraphrase_1`、`paraphrase_2` |
| 候选顺序 | `ascending`（按 ID 正序）、`reversed`（逆序） |
| 上下文 | 同一个来源任务的事件前策略状态，固定 `state` 模式 |

共 **10 × 3 × 2＝60 条请求、60 次计划模型调用**。
它们是 10 个任务的相关变体，不是 60 个独立任务，不测试连续状态传递。

原句直接取自任务；两种英文改写由
`configs/hssd_language_wordings_v1.json` 声明，分别改变句式与目标名词短语。
例如沙发使用 “the dark-colored couch” 和 “the couch that is dark in color”，仍明确取消旧目标。
候选中原有的 “the dark sofa” 及别名保持不变，改写不会作为新别名提供给模型。
这些是编写的开发措辞，不是独立收集并由多名标注者审核的自然语言数据；正式发布前仍需语言质检。

正逆序不改变中间候选的位置，相关配对会标记 `target_position_changed=false`。
本次不能据此声称覆盖了所有排列或消除了全部位置偏差。

## 2. 已完成的 CPU 准备

准备需使用 `habitat` 环境，对来源五候选任务做 CPU 网格/图像独立审计，不构造 Simulator 或调用模型。
以下命令已执行，现有 `_001` 无需再次准备：

```bash
python -m scripts.language_navigation_robustness prepare \
  --source-dir outputs/hssd_language_candidates5_001 \
  --wordings configs/hssd_language_wordings_v1.json \
  --output-dir outputs/hssd_language_robustness_dev_001
```

| 目录/文件 | 内容与权限 |
| --- | --- |
| `public/` | 请求、请求哈希、原系统说明和固定生成参数；推理只需要此目录 |
| `private/records.json` | 任务分组、独立参考更新与预期状态，仅评分使用 |
| `private/wording_review.csv` | 当前消息与期望目标，供人工检查，不能交给模型 |
| `manifest.json` | 来源原始证据、代码、措辞和公开/私有计划哈希 |
| `prompt_plan/` | 60 条实际 HTTP 请求体预览，没有预测文件 |
| `cpu_check.json` | 合约与参考状态重放检查，不是模型性能 |

请求 ID 是不透明哈希，不编码操作名、条件标签或任务答案。
公开请求不含参考更新、目标坐标、场景、实例元数据或任务分组；初始符号目标属于本轨道已知上下文。
准备只接受完整可信的来源证据，不要求来源模型必须预测成功，避免按正确性筛选任务。
源配置、原始证据或措辞变化会使评分校验失败，不静默改变实验。

## 3. 下一步：运行真实模型

推理与评分均只需标准库，可在仓库根目录使用 `mivln` 环境。
维持原 Qwen3.5-9B 服务与权重；模型别名、地址及参数从公开清单继承，运行时不提供覆盖选项：
`qwen3.5-9b`、`http://127.0.0.1:8000/v1`、非思考、temperature=0、max_tokens=512、timeout=120。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate mivln

python -m scripts.language_navigation_robustness predict \
  --requests-dir outputs/hssd_language_robustness_dev_001/public \
  --output-dir outputs/hssd_language_robustness_qwen_001

python -m scripts.language_navigation_robustness score \
  --prepared-dir outputs/hssd_language_robustness_dev_001 \
  --run-dir outputs/hssd_language_robustness_qwen_001 \
  --output-dir outputs/hssd_language_robustness_eval_001
```

预测复用原客户端，逐请求使用新会话，不重试、不修复 JSON、不回填答案。
调用失败仍继续后续请求；受控中断会为剩余槽位写入 null。每条试验都独立使用原旧状态，错误不会传给下一个试验。
预测结束保存原始请求/响应、计数和 `sealed_run.json`，评分核验全部文件并重建实际请求体。
应使用本入口生成封存记录；直接运行通用客户端产生的未封存结果不属于本协议。
完整保留的失败或受控中断可评分；硬终止导致缺少封存或日志时，先排查证据，不覆盖原目录。

预测返回 1 表示含调用失败；评分返回 1 表示不是全部正确，两者都不表示应删除失败结果。
每个阶段使用新目录，评分不覆盖旧结果，也不修改来源导航数据。

## 4. 产物、指标与验收

评分目录保存：

- `summary.json`：60 条请求的正确数/率、按措辞/顺序/目标位置分组、调用与格式失败。
- `cases.csv/jsonl`：实际原文、解析/应用状态、旧目标/新目标是否正确、干扰项误选和结果状态。
- `order_pairs.csv/jsonl`：固定 **30 组**，同任务同措辞下两个顺序的预测/状态一致性及两者都正确情况。
- `wording_groups.csv/jsonl`：固定 **20 组**，同任务同顺序下三种措辞的一致性及三者都正确情况。
- `task_groups.csv/jsonl`：固定 **10 组**，每个基础任务的六种条件是否全部正确。

`correct` 同时要求合法应用、预测更新完全正确和结果状态正确。
`consistent` 比较规范化预测及应用后的策略状态，忽略 JSON 空白；相同错误也可一致，因此必须同时看 `all_correct_count`。
空回答/非法 JSON 不会因为同为 null 获得一致性分数。所有失败保留在固定分母中。
共享执行器保证未修改字段保留，这不是额外的模型记忆成绩。

验收先看 `integrity_passed=true`、`statistics.planned=60`，确认实际请求、响应与固定控制一致。
再看正确率、30/20/10 组的全部正确数、顺序翻转是否改变正确性，以及错误目标/非法输出/调用失败。
真实错误是研究结果，不要求模型满分才认为数据有效。

本轮只验收离线更新。[代表性措辞的实际导航入口](language_navigation_representatives.md)现已准备五个固定目的地，
使用 `paraphrase_2/reversed` 新跑十五个三条件 episode，GPU 待执行；不能将这 60 个离线预测表述为 60 条已完成的导航轨迹。

## 5. 验证与提交

新增 18 项 CPU 测试，覆盖矩阵、公开信息隔离、模型参数固定、全分母、错误目标真实应用、
一致但错误、独立试验状态、调用失败/中断、实际请求与原始响应篡改、防覆盖和私有参考重建。
Habitat 全量 **254 项通过**；核心环境发现 254 项并跳过 70 项原生依赖测试。
真实来源在禁止构造 Simulator/真实调用的条件下验证了 60/30/20/10 计划与合成传输评分；
临时合成预测已随测试目录清理，正式准备目录没有模型预测或真实评分。

本功能的配置、入口、测试和指南可独立提交：`feat: add wording and order robustness evaluation`。
`outputs/`、数据和权重不提交；上一阶段五候选功能是单独的逻辑提交。
