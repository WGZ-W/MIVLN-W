# 明确更新实验：连续状态传递

更新日期：2026-11-06。入口、CPU 检查及真实 Qwen 的 288 次调用已完成，因果重放和评分复算一致。
state 更新及状态正确 144/144、完整对话 24/24；history 为 124/144、19/24，五次操作错误造成后续状态残留，
见 [真实模型结果与原因分析](qwen35_rollout_dev_results_2026-11-06.md)。
本阶段优先研究明确更新中的连续状态，暂不扩展拒绝判断。入口：`python -m scripts.task_update_rollout`。

## 1. 与前一阶段的区别

前一阶段每条请求的旧状态由固定规则从公开历史重新构建。本实验只在对话开始时用同一公开语言解析器确定初始目标；
此后两个分支分别持有自己的实际策略状态。模型本轮预测的更新经过共享执行器，结果直接传入后续轮次，不重置为参考状态。

- **state**：读取本分支实际的 `previous_state`，包含模型先前有效更新及错误的累积结果。
- **history**：读取初始指令，以及本分支累计的用户消息和执行反馈，默认按时间线呈现。
  反馈记录实际尝试的符号更新和是否应用，不包含完整状态、几何或参考答案。失败请求不会被描述为已生效。
- **独立评价**：从人工声明的参考更新构建单独的正确轨迹，仅在评分时比较，不修复策略状态。

模型仍预测三类增量更新，不是每轮重新生成整个状态 JSON；未修改字段保留和挂起栈转换仍由执行器保证。
history 新增了真实执行反馈，本轮提示和数据也与旧单步实验不同，不能直接把新旧分数当作同一个实验比较。

## 2. 对话设计与完成通知

配置：`configs/task_update_rollouts_v1.json`。两个序列模板 × 中英文 × 三种初始目标 × 两种候选顺序，共 24 条变体对话。
每条对话有 6 次明确更新和 1 次外部合成完成通知。两种模式共 **288 次模型调用、48 次完成事件**；
每模式 144 次更新、24 次完成事件、24 条对话，每轮更新固定分母为 24。

1. 替换目标 → 新增禁区 → 插入子目标 → 完成并恢复 → 再次替换 → 新增另一禁区 → 显式替换。
2. 插入子目标 → 完成并恢复 → 替换目标 → 新增禁区 → 显式替换 → 新增另一禁区 → 再次替换。

例如原目标 A 改为 B，再临时插入 C；完成 C 后应恢复 B。若模型第一步合法地误改为 D，后续输入和恢复就真实继承 D，
不会由评价器偷偷改回 B。若更新格式非法、调用失败或状态应用失败，策略状态保持原样，后续轮次继续保留。

完成通知是预先固定、对两模式相同的**文本状态机夹具**，不需要模型调用。仅当本分支确实具有匹配的活动子目标与挂起目标时才执行恢复，
否则记录失败并保留状态。它没有 RGB、距离、朝向或实测完成证据，不能表述为 HSSD 导航成功。
本轮在正确参考轨迹上均为明确消息；错误分支后续可能出现旧目标不一致，这正是要保留的错误传播。
24 个对话变体只有两个序列模板，不是 24 个独立语义类型或未见泛化基准。

## 3. 已完成的 CPU 检查

以下产物已生成，重复运行这些命令需要更换输出编号：

```bash
python -m scripts.task_update_rollout export \
  --output-dir outputs/task_update_rollout_dev_requests_001

python -m scripts.task_update_rollout run \
  --requests-dir outputs/task_update_rollout_dev_requests_001 \
  --output-dir outputs/task_update_rollout_rules_001

python -m scripts.task_update_rollout score \
  --requests-dir outputs/task_update_rollout_dev_requests_001 \
  --run-dir outputs/task_update_rollout_rules_001 \
  --output-dir outputs/task_update_rollout_rules_eval_001
```

规则基线两模式均为更新/状态 144/144、完整对话 24/24、完成事件 24/24，累计错误为 0。
这是针对已知语法的 CPU 流程检查，不是真实模型结果或方法优势证据。
错误注入测试确认：错误但合法的目标真正传入下一轮；失败与拒绝保持旧状态；错误完成通知不修复分支；
中断后的剩余槽位全部填 null；重写文件哈希也无法掩盖状态重置或动态提示篡改。

全项目 `python -m unittest discover -s tests -q`：Habitat 环境 **183 项通过**，核心环境发现 183 项并跳过 62 项。
连续更新专项 17 项通过，均使用 CPU 与本地 HTTP 桩，没有访问真实模型服务。
规则轨迹的本地分词器检查：history 输入 880–1343 token，state 为 877–907，最长加 512 输出预算为 1855，低于 4096；
实际模型反馈和状态可能不同，真实 token 用量仍以新批次记录为准。

## 4. 已完成的真实连续更新实验

前提：使用原来的 Qwen3.5-9B 服务与权重，模型别名为 `qwen3.5-9b`，端口为 8000；保持 GPU 服务参数不变。
输入目录已冻结，以下两个模型/评分输出目录均已存在。命令保留用于复现，重复运行需要新目录。客户端只需标准库，可使用 `mivln` 环境。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate mivln

python -m scripts.task_update_rollout predict \
  --requests-dir outputs/task_update_rollout_dev_requests_001 \
  --base-url http://127.0.0.1:8000/v1 --model qwen3.5-9b \
  --thinking-mode off --temperature 0 --max-tokens 512 --timeout 120 \
  --history-format timeline \
  --output-dir outputs/task_update_rollout_qwen35_dev_001

python -m scripts.task_update_rollout score \
  --requests-dir outputs/task_update_rollout_dev_requests_001 \
  --run-dir outputs/task_update_rollout_qwen35_dev_001 \
  --output-dir outputs/task_update_rollout_qwen35_dev_eval_001
```

`predict` 在同一批次运行两个模式，固定模型、系统提示、生成参数、候选及外部事件。每次 HTTP 调用都是新会话，
连续性来自显式状态或历史输入，不依赖服务隐式会话。默认时间线标签针对执行反馈重新设计，未复用旧提示中“所有历史更新已生效”的表述。
`--history-format json` 是以后可选的呈现对照；本次先固定 timeline，不混用运行结果。

流程验收预期 `planned_updates=attempted=received=288, failed=0`。模型语义失败仍保留完整轨迹和固定分母，
不预设准确率等于 100% 或 state 一定更好。预测命令返回 1 表示存在调用失败；评分返回 1 表示存在调用、契约、状态应用或完成事件错误，
仍保存报告。合法但语义错误的更新也会降低分数，即使退出码为 0。

## 5. 产物、重放与指标

公共导出目录保存 `dialogues.jsonl`、`instructions.txt`、`executor_regions.json` 和哈希清单。
`executor_regions.json` 只有合成执行器几何，不能发给模型；模型只接收按白名单构造的当轮请求。
生成时不打开源数据集或读取标签；评分时才核验源文件哈希并载入独立参考。
后续状态依赖真实预测，因此不能提前导出所有轮次的结构化请求，也不能使用旧的静态预测命令替代本入口。

预测目录包含：

- `predictions.jsonl`、`responses.jsonl`：288 个实际原始响应槽位及调用元数据，不修复 JSON、不重试。
- `model_inputs.jsonl`：每轮动态请求哈希、完整 HTTP 请求体及哈希。
- `trace.jsonl`：336 条事件记录，包括实际前后状态、反馈、完成事件和失败原因。
- `run.json`、`system_prompt.txt`：固定参数、调用计数、文件哈希和实际系统提示。

评分先按原始预测因果重放两个分支，重新构造每条请求体及反馈，核对完整名单、日志和哈希，再创建新评分目录。
评分产物为 `summary.json`、`cases.jsonl/csv`、`trajectories.jsonl`，记录评分源码哈希。

重点检查：

- `by_mode.*.by_turn`：六轮分别报告固定分母下的更新及状态正确率，观察随轮次变化。
- `all_steps_correct_count`、`final_state_exact_count`：全对话和最终状态是否正确；`paired_dialogues` 给出同对话两模式的配对得失。
- `carried_error_count`、`new_error_count`、`continued_error_count`、`recovered_count`：错误进入后续轮次、首次产生、持续和恢复的次数。
- `active_goal_correct_count` 与 `field_correct_count`：区分当前目标、挂起目标、约束及历史记录。取消目标记录可能持续错误，
  所以完整状态未恢复不等于当前目标仍然错误，错误残留次数也不能单独证明每轮产生了更多新错误。
- `completion_restoration_correct_count`：完成后目标、挂起栈和已完成子目标是否正确；`completion_correct_count` 进一步要求完整状态一致。
- `operation_correct`、`target_argument_correct`、`reference_argument_correct`：分别定位操作、新目标/区域和旧/恢复目标错误。

## 6. 提交与后续范围

连续更新核心、独立评分、配置、CLI、共享 HTTP 传输提取、17 项测试及本文可作为功能提交：
`feat: add sequential task update rollouts with causal state replay`。
前轮尚未提交的消融结果文档建议先单独提交，避免与新功能混在一起。
CPU 验收与真实模型批次审计已完成；结果报告及相关状态说明可单独提交为：
`docs: record sequential task update results and error propagation`。
此前提出的新措辞与禁区位置对照见 [结果中的方案](qwen35_rollout_dev_results_2026-11-06.md)，当前进一步 history/state 验证暂缓。
当前 [HSSD 模型更新回调及三条件单例](language_navigation.md)的真实 GPU 验收已完成，Oracle/model 成功，Ignore 按预期失败，
见 [闭环结果](hssd_language_results_2026-11-09.md)。下一步扩展其他固定任务。
其他介入类型、批量导航及拒绝相关诊断仍列后续。所有 `outputs/`、模型权重与数据继续忽略，不自动暂存、提交或推送。
