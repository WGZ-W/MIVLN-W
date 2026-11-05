# 独立语言任务更新：契约、规则基线与离线评测

更新日期：2026-11-05。入口 `python -m scripts.task_update` 仅使用标准库，不需要 Habitat、GPU、模型权重或 API。
这一步落实“语言理解与增量更新”接口，支持三类操作、旧状态保留检查及外部模型预测导入。

## 当前交付与范围

- `mivln/task_update.py`：公共输入、严格预测校验及原 `TaskState` 的事务式适配器。
- `mivln/rule_task_updater.py`：有限中英文短语与候选别名匹配的**规则基线**，不是语言模型。
- `mivln/task_update_eval.py`：独立参考状态、固定上下文构建与逐字段评价。
- `configs/task_update_cases_v1.json`：18 条合成开发样例，包括 15 条三类更新及 3 条应拒绝猜测的消息。
- `scripts/task_update.py`：运行规则、导出无答案请求、导入外部预测评分。

当前是**给定固定历史上下文的一次任务更新评测**。共同的规则构建器从初始指令和历史消息独立建立旧策略状态，
不是把参考正确状态直接提供给更新器。初始或历史语言无法解析时保留失败，不补入正确答案。
外部模型只预测当前消息，先前状态并非该模型在连续对话中自己产生的结果。
视觉落地、完整对话记忆、任务版本、主动求助及 GPU 导航接入仍待实现。

## 1. 公共输入与三种上下文

每个请求只接受以下字段，禁止完整 episode 或参考答案透传：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 当前为整数 `1` |
| `request_id` | 样例与模式的唯一标识 |
| `mode` | `latest`、`history` 或 `state` |
| `message` | 当前待解释的消息 |
| `candidate_goals` | `goal_id`、文本描述和别名，不含场景坐标 |
| `candidate_regions` | `region_id`、文本描述和别名，不含区域几何 |
| `context` | 当前模式允许的上下文 |

- `latest`：`context: {}`，只有当前消息和公共候选描述。
- `history`：`initial_instruction` 和此前收到的 `message_history`。
- `state`：`previous_state`，包含有效目标、挂起目标、约束引用、已完成/已取消目标及状态。
  `active_constraints` 是区域标识列表，不是坐标字典。

三模式共用解析器、当前消息、候选集合及执行器，只改变上下文表示。
旧状态通过公开语言构建，不读取 `reference.initial_goal` 或 `reference.context_updates`。
区域几何只在执行适配器内部按预测标识查询，不交给预测函数。

## 2. 输出与状态应用

三类成功输出：

```json
{"status":"ok","task_update":{"operation":"replace_goal","old_goal":"island","new_goal":"sofa_dark"}}
```

```json
{"status":"ok","task_update":{"operation":"add_avoid_region","region_id":"region_red"}}
```

```json
{"status":"ok","task_update":{"operation":"insert_subgoal","subgoal":"sofa_dark","resume_goal":"island"}}
```

不明确时允许拒绝猜测：

```json
{"status":"abstain","reason":"ambiguous_goal"}
```

拒绝原因限定为 `missing_context`、`unknown_goal`、`ambiguous_goal`、`unknown_region`、
`ambiguous_region`、`unsupported_message`、`invalid_context`。
例如有深色、浅色两张沙发时，“改去沙发”应拒绝猜测；有两个候选区域时，“避开标记区域”不足以确定其一。

未知标识、额外字段、重复 JSON 键、非有限数字及 Markdown 围栏不自动修复。
旧目标或恢复目标不匹配、重复禁区等更新失败时，输入状态不被修改。
有效更新经同一个执行器应用：替换目标保留约束，增加禁区保留目标，插入子目标保留恢复要求。
这些保留行为是**共享状态机保证**，不是模型已经学会记忆的证据。

本契约要求输出 `old_goal` 和 `resume_goal` 作为一致性保护，因此 latest 的部分失败来自缺少填写这些字段的上下文。
不能据此声称仅读最新消息的导航器无法前往一句话中明说的新目标。

## 3. 本轮已经执行的开发检查

```bash
python -m scripts.task_update run --mode all --output-dir outputs/task_update_rules_001
python -m scripts.task_update export --mode all --output-dir outputs/task_update_requests_001
python -m scripts.task_update score \
  --requests-dir outputs/task_update_requests_001 \
  --predictions outputs/task_update_rules_001/predictions.jsonl \
  --updater-name imported_bilingual_rules_v1 --output-dir outputs/task_update_rescore_001
```

三个模式共 54 个固定记录，无非法或缺失输出；导出后重新导入相同规则预测，评分完全一致。

| 模式 | 预测及完整新状态均正确 |
| --- | --- |
| latest | 12/18 |
| history | 18/18 |
| state | 18/18 |

样例针对已知规则编写，只用于开发检查，不是留出语言集、HSSD episode 或模型泛化结论。
区域坐标为合成状态机夹具，未绑定真实场景。history 与 state 同分，没有结构化记忆优于历史拼接的证据。

## 4. 可执行命令与模型导入

本轮 `_001` 目录已存在，重跑时选择新目录。核心环境即可：

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate mivln

python -m scripts.task_update run \
  --mode all --output-dir outputs/task_update_rules_002

python -m scripts.task_update export \
  --mode all --output-dir outputs/task_update_requests_002
```

模型读取 `requests.jsonl` 和 `instructions.txt`；响应包装时用 `request_index.json` 对齐标识与哈希。
`manifest.json` 冻结源数据、模式、请求顺序与文件哈希，导出目录不含答案或区域几何。
规则运行目录的 `cases.jsonl` 含参考评分内容，仅用于人工检查。

模型预测 JSONL 每行格式如下，其中标识和哈希从实际请求复制：

```json
{"request_id":"样例标识__模式","request_sha256":"对应请求哈希","prediction":{"status":"abstain","reason":"missing_context"}}
```

`prediction` 也可保存模型原始 JSON 字符串。真实预测文件生成后，评分命令已经可用：

```bash
python -m scripts.task_update score \
  --requests-dir outputs/task_update_requests_002 \
  --predictions outputs/model_task_update_predictions.jsonl \
  --updater-name model_under_test --output-dir outputs/task_update_model_eval_001
```

上述模型预测文件尚不存在。本次未接入外部 API 或本地模型推理。
下一项实现是确定模型与运行后端，生成同格式预测；要求所有请求均可对齐，并保留错误和未回答的分母。

## 5. 指标与失败处理

- `summary.json`：固定分母、按模式/操作分组、状态字段准确率。
- `cases.csv`：每个样例/模式一行，保留上下文、输出或缺失错误。
- `cases.jsonl`：逐条预测、独立参考、前后状态与错误。
- `predictions.jsonl`：规则运行时的实际输出，用于回环检查。

分别统计上下文正确、输出契约合法、更新可应用、操作/参数正确、完整状态正确和未受影响字段保留。
拒绝理由也按参考检查；只保留旧状态但漏掉更新，不能获得完整正确。
缺失、无效或哈希不匹配的响应保留原计划槽位；重复/未知请求标识及输入篡改在创建输出前拒绝。
输出目录禁止重复使用。命令返回 0 表示没有输入/输出异常，**不要求准确率等于 100%**；
有效但错误的更新会降低指标，非法或缺失输出会保留评分文件并返回 1。

未来语言改写按 `group_id` 与基础任务划分，避免同一含义的中英改写跨开发/测试泄漏。
正式评测还需留出语言、明确区域候选依据，并保持信息边界后再接入导航。

## 6. 测试与提交

```bash
python -m unittest tests.test_task_update -v
python -m unittest discover -s tests -v
```

专项测试覆盖答案隔离、别名歧义、中英文三类更新、旧约束和恢复目标保留、事务失败、严格 JSON、
上下文失败分母、请求冻结、外部响应缺失/错位、导出导入一致性和拒绝覆盖。
本轮 20 项新增专项测试通过；全项目在 `habitat` 环境 136 项通过，核心环境通过并跳过 62 项依赖测试。
测试使用 CPU 假模拟器，没有启动真实 GPU 导航。
本轮核心接口、规则、评价、脚本、样例、测试及相关说明可以提交为：
`feat: add isolated language task update contracts and offline evaluation`。
图稿改动单独处理；运行输出保持 Git 忽略，不自动暂存、提交或推送。
