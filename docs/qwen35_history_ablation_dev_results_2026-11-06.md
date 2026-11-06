# Qwen3.5-9B 历史呈现消融：dev 结果

检查日期：2026-11-06。历史条件的 216 次真实调用和原基线逐条复算通过。
history 决策正确从 21/72 提升至 39/72，但明确替换退化、歧义拒绝未解决，不能据总分认定该提示全面更好。
拒绝说明条件尚未运行，本次是两个条件的阶段比较，不是完整三条件消融结论。
后续拒绝条件与三组比较现已完成，见 [三条件结果](qwen35_prompt_ablation_dev_results_2026-11-06.md)；本页保留历史条件的阶段记录。

## 1. 输入、控制与证据

- 公共输入：`outputs/task_update_pairs_dev_requests_002/`，六个语义组、72 个变体样例、216 条模式请求。
- 原基线：`outputs/task_update_pairs_qwen35_dev_001/`；原评分：`outputs/task_update_pairs_qwen35_dev_eval_001/`。
- 新预测：`outputs/task_update_ablation_history_dev_001/`，`prompt_variant=history_timeline`。
- 新评分：`outputs/task_update_ablation_history_dev_eval_001/`；审计：同目录 `ablation_audit.json`。
- 客户端固定 `qwen3.5-9b`、同一 API 地址、非思考模式、temperature=0、max_tokens=512、timeout=120。

planned/attempted/received 均为 216，failed=0，所有响应均为 `finish_reason=stop`。
调用日志记录于 11 月 5 日北京时间 19:56:37—19:59:19；本次核查未调用模型或启动 GPU。
源配置、请求、索引、系统提示、实际请求体及预测哈希核验通过，216 个 ID 完整且顺序一致。

只有 72 条 history 请求体发生变化；latest/state 的 144 条输入与原基线一致，原始回答也逐条一致。
系统提示没有变化，历史条件只调整公开历史的呈现，不增加正确目标、结构化状态、坐标或参考答案。
新批次的请求体日志与当前条件重建完全一致；旧基线当时未保存请求体，仅能按原提示和参数重建，不能表述为旧 HTTP 日志证据。
模型别名与客户端参数相同，也不能独立证明服务端权重和 GPU 配置完全相同。

新预测 SHA256：

```text
00b42e1ad0299ba97c099d14c033ee962faaf62cee2defd80629fd074ffb8451
```

两组共 432 条评分记录及各项汇总均从原始预测重新计算一致；原基线和旧输出未修改。
新评分返回码为 1，来自契约或状态应用错误，不表示调用缺失；评分文件已完整保存。

## 2. 主结果与逐条得失

| 有依据的决策正确数 | 原基线 | 历史呈现条件 |
| --- | ---: | ---: |
| latest | 34/72 | 34/72 |
| history | 21/72（29.2%） | 39/72（54.2%） |
| state | 52/72 | 52/72 |

history 净增 18 条、提高 25 个百分点：新增正确 25 条，丢失正确 7 条，14 条保持正确，26 条仍然错误。
总分增加并不表示所有请求类型都改善：

| history 语义组 | 原基线 | 历史呈现条件 |
| --- | ---: | ---: |
| 隐式目标替换 `replace_dark` | 1/12 | 8/12 |
| 隐式子目标恢复 `resume_dark` | 0/12 | 5/12 |
| 明确新增禁区 `avoid_red` | 12/12 | 12/12 |
| 显式目标替换 `explicit_replace` | 8/12 | 2/12 |
| 沙发歧义拒绝 `ambiguous_sofa` | 0/12 | 0/12 |
| 未知书架拒绝 `unknown_goal` | 0/12 | 12/12 |

明确更新合计从 21/48 升至 27/48；其余新增正确主要来自未知物体拒绝。
两种候选顺序均正确的组数从 9/36 升至 15/36，但仅要求预测一致的组数从 21/36 降至 17/36。
三个不同当前目标均决策正确的上下文组从 0/8 升至 1/8，仍未形成稳定的历史跟踪。

## 3. 改善与新增问题

### 目标混淆有所减少，但伴随操作错误增加

对 24 条隐式更新的原始 JSON 检查，`old_goal` 或 `resume_goal` 填入最初目标的次数从 23 降至 9。
该诊断也计入操作类型错误的输出；入口中较严格的诊断只统计契约合法、操作正确的明确请求，对应计数为 22→7。
两种口径不同，不能直接混用。

48 条明确更新中的原始操作类型错误从 5 增至 14，新的错误均为将目标替换写成 `insert_subgoal`。
其中显式替换组有 10/12 输出了错误操作。七条从正确变错误的配对记录全部来自该组，
例如“Instead of the dining table, go to the light sofa.”被解释成临时插入，而不是替换原目标。
这表明历史呈现影响了当前消息解释；具体是哪条说明或位置造成影响，仍需后续独立验证。

### 未知引用改善，歧义误执行仍未解决

未知书架的 12 条消息全部正确拒绝；沙发歧义的 12 条原始回答却仍全部输出 `status=ok`。
其中两条新输出不符合契约，不能把它们当成正确拒绝。
应拒绝请求中的合法执行尝试由 22/24 降至 10/24，但能通过适配器应用的错误执行反而由 4/24 升至 9/24。
前者包含未知引用拒绝的改善，后者暴露了歧义猜测的风险，不能只报告尝试次数下降。

history 契约合法数从 69/72 降至 63/72；新批次九条契约失败均为子目标与恢复目标相同。
没有出现明确请求被合法拒绝的情况。状态一致性保护能阻止部分非法更新，无法阻止所有语义错误。

## 4. 下一步与提交

保留本次历史条件及原基线，暂不改提示或候选。下一步先运行独立 `refusal_check` 条件，保持当前历史基线 JSON 输入，
只统一增加拒绝说明；不要同时叠加历史时间线，以便区分两个因素。

```bash
python -m scripts.predict_task_updates \
  --requests-dir outputs/task_update_pairs_dev_requests_002 \
  --base-url http://127.0.0.1:8000/v1 --model qwen3.5-9b \
  --thinking-mode off --temperature 0 --max-tokens 512 --timeout 120 \
  --prompt-variant refusal_check \
  --output-dir outputs/task_update_ablation_refusal_dev_001

python -m scripts.task_update_ablation compare \
  --requests-dir outputs/task_update_pairs_dev_requests_002 \
  --baseline-run outputs/task_update_pairs_qwen35_dev_001 \
  --history-run outputs/task_update_ablation_history_dev_001 \
  --refusal-run outputs/task_update_ablation_refusal_dev_001 \
  --output-dir outputs/task_update_ablation_dev_compare_001
```

前提是同一模型服务可用、参数保持一致、两个输出目录尚不存在。预期拒绝条件保留 216 个预测槽位，received=216、failed=0，
比较生成 `comparison.json` 和逐条 CSV。验收同时检查歧义/缺失上下文拒绝、可应用误执行、明确更新及契约错误，不能只看总分。
这些命令已实现，本次未执行新的 GPU 请求；旧 test 仅作回归，新泛化结论仍需另建冻结验证组。

本结果与状态说明可作为文档提交：`docs: record history prompt ablation dev results`。
本轮无代码改动，检查使用 CPU 输入审计和逐条评分复算；`outputs/` 不提交，不自动暂存、提交或推送。
