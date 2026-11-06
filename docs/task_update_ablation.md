# 任务更新提示消融：历史呈现与拒绝判断

更新日期：2026-11-05。入口已实现，完成 CPU 预览与测试；两个新条件的真实模型调用待执行。
目标是分别检查历史组织和拒绝说明的作用，保持模型、公共候选及顺序、样例、状态构建与评分不变。

## 1. 三个独立条件

| `--prompt-variant` | 改动 | dev 输入变化 |
| --- | --- | --- |
| `baseline` | 保留原系统提示和用户 JSON，包括原有空格与换行 | 0/216 |
| `history_timeline` | 仅 history 改为初始指令、编号历史、末尾当前消息；明确历史变更已生效 | 72/216 |
| `refusal_check` | 三种模式统一增加候选唯一性、未知对象/区域和上下文不足检查；用户 JSON 不变 | 216/216 |

提示实现位于 `mivln/task_update_prompts.py`。历史条件不计算当前目标，不加入参考答案、几何或结构化状态；
“任务变更已生效”表示已接收此前介入，不表示已完成导航。拒绝条件仍允许明确更新，不采用一律拒绝策略。
本轮不叠加两个条件，以便分别解释变化来源。latest/history/state 是信息模式，表中的三个条件是提示变体。

## 2. 已完成的 CPU 输入冻结

以下命令可复现预览，目录必须是新目录；本次已生成 `_001`，重跑时更换编号：

```bash
python -m scripts.task_update_ablation prepare \
  --requests-dir outputs/task_update_pairs_dev_requests_002 \
  --model qwen3.5-9b \
  --thinking-mode off --temperature 0 --max-tokens 512 --timeout 120 \
  --output-dir outputs/task_update_ablation_dev_preview_001
```

每个条件保存 `system_prompt.txt`、`model_inputs.jsonl` 和 `run.json`，总目录另有 `plan.json`。
预览不加载模型、不发 HTTP 请求，也不生成预测或分数，不能当作已完成的实验传给比较命令。
每条输入记录公共请求哈希、完整 HTTP 请求体及其规范 JSON 哈希，实际推理复用同一请求体构造函数。
新推理同样保存输入日志，以便检查实际呈现的提示，而不仅检查原始 JSON。

本次原设置系统提示 SHA256 为
`be471f29075d3c406ad9b5929fa5a62ffc0851a7a783aa8e4f3a2bbf9b6d9aff`，
216 条预览均与旧 dev 运行重建的请求体一致。旧基线没有当时的请求体日志；比较报告会明确标记为重建证据。
本地聊天模板与分词器 CPU 检查：原设置输入为 692–748 token，历史条件为 692–804，拒绝条件为 892–948；
最长输入加 512 token 输出预算为 1460，小于当前 4096 上限。检查没有加载模型权重。

## 3. 下一步：用户执行两个 dev 批次

前提：同一个 Qwen3.5-9B 服务仍以 `qwen3.5-9b` 在本机 8000 端口提供服务，
保持权重、聊天模板、服务启动参数和 GPU 设置一致。启动说明见 [本地部署](qwen35_local.md)。
客户端只需要标准库，可以在 `mivln` 环境运行；以下命令已实现。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate mivln

python -m scripts.predict_task_updates \
  --requests-dir outputs/task_update_pairs_dev_requests_002 \
  --base-url http://127.0.0.1:8000/v1 --model qwen3.5-9b \
  --thinking-mode off --temperature 0 --max-tokens 512 --timeout 120 \
  --prompt-variant history_timeline \
  --output-dir outputs/task_update_ablation_history_dev_001

python -m scripts.predict_task_updates \
  --requests-dir outputs/task_update_pairs_dev_requests_002 \
  --base-url http://127.0.0.1:8000/v1 --model qwen3.5-9b \
  --thinking-mode off --temperature 0 --max-tokens 512 --timeout 120 \
  --prompt-variant refusal_check \
  --output-dir outputs/task_update_ablation_refusal_dev_001
```

每个批次预期 216 个预测槽位，三种信息模式各 72 条；成功调用时 `received=216, failed=0`。
输出包含原始 `predictions.jsonl`、调用元数据 `responses.jsonl`、完整输入日志及 `run.json`。
不修复 JSON，不根据答案重试；调用失败或中断后的未调用槽位保留 `null`，仍参与评分。
目录已有时更换编号，同时更新下方比较路径。不要覆盖旧预测或重新导出候选集合。

## 4. 同一评分器审计与比较

两个批次结束后执行 CPU 比较，复用已完成的原设置 dev 运行：

```bash
python -m scripts.task_update_ablation compare \
  --requests-dir outputs/task_update_pairs_dev_requests_002 \
  --baseline-run outputs/task_update_pairs_qwen35_dev_001 \
  --history-run outputs/task_update_ablation_history_dev_001 \
  --refusal-run outputs/task_update_ablation_refusal_dev_001 \
  --output-dir outputs/task_update_ablation_dev_compare_001
```

比较先检查冻结请求、系统提示、输入日志、预测哈希、完整请求名单及生成参数，再用当前同一评分器重新评分三组。
不修改旧评分目录；新报告记录评分代码的 SHA256。模型别名和客户端参数一致不能单独证明服务权重一致。

- `comparison.json`：三组按模式/类别的决策分数、上下文/候选顺序配对指标、错误计数及相对基线的增减。
- `comparison_cases.csv`：同一请求在三个条件下的决策正确与否。
- `baseline_score/`、`history_timeline_score/`、`refusal_check_score/`：固定协议完整评分与逐条前后状态。

`errors_by_mode` 的诊断定义：

- `reference_goal_wrong_count`：明确可执行请求中，契约合法且操作正确，但 `old_goal`/`resume_goal` 错误；
  分母为 `reference_goal_required`。其中误填最初目标计入 `initial_goal_confusion_count`。
- `refusal_execution_count`：应拒绝的请求仍输出合法 `status=ok`；`refusal_applied_count` 进一步要求更新可应用。
  分母为 `refusal_required`，涵盖信息缺失、歧义和未知候选。
- `false_refusal_count`：明确可执行请求被拒绝，分母为 `actionable_required`。
- `invalid_or_missing_count`：输出契约无效或缺失，单列统计；这些记录始终保留在完整计划分母中。

`vs_baseline` 给出同条请求新增正确/丢失正确的数量与各模式错误计数变化。
历史条件的 latest/state 共 144 条输入应完全相同；报告同时统计其原始回答变化，作为服务稳定性对照。
温度为 0 也不保证逐次输出一致；若这些对照变化明显，先以新目录重跑 `--prompt-variant baseline` 检查稳定性。

验收先看原 dev 基线是否复算为 **34/72、21/72、52/72**，再看历史模式的目标混淆是否减少，
拒绝条件的误执行是否减少，同时检查明确更新是否因过度拒绝退化。不预设两个提示一定有效，不要求 100% 正确。
比较退出码 0 表示没有调用失败；格式/适配错误另记各条件的 `score_return_code`，保留错误记录和完整报告。

## 5. 后续范围与提交

先用 dev 选择提示；已观察的 test 只作回归检查。新的泛化结论仍需要另建冻结验证组，该工作尚未实现。
连续自生成任务状态和接入导航闭环属于后续阶段，本次不改动。

可提交提示模块、客户端扩展、比较入口、专项测试及说明，建议：
`feat: add controlled prompt ablations for task updates`。
验证命令为 `python -m unittest discover -s tests -q`：Habitat 环境 166 项通过，
核心环境发现 166 项并跳过 62 项；其中消融专项 11 项覆盖实际 HTTP 请求体、变体边界、旧基线兼容、
输入篡改拒绝、预览隔离、错误分母和目标混淆/拒绝诊断，全部使用本地桩。
CPU 检查不代表真实模型效果验证完成；`outputs/` 保持 Git 忽略，不自动暂存、提交或推送。
