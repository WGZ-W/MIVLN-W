# 独立语言任务更新：契约、规则基线与离线评测

更新日期：2026-11-05。入口 `python -m scripts.task_update` 仅使用标准库，不需要 Habitat、GPU、模型权重或 API。
预测客户端 `python -m scripts.predict_task_updates` 同样只用标准库，但真实推理需要用户配置模型服务。
这一步落实“语言理解与增量更新”接口，支持三类操作、旧状态保留检查及外部模型预测导入。

## 当前交付与范围

- `mivln/task_update.py`：公共输入、严格预测校验及原 `TaskState` 的事务式适配器。
- `mivln/rule_task_updater.py`：有限中英文短语与候选别名匹配的**规则基线**，不是语言模型。
- `mivln/task_update_eval.py`：独立参考状态、固定上下文构建与逐字段评价。
- `configs/task_update_cases_v1.json`：18 条合成开发样例，包括 15 条三类更新及 3 条应拒绝猜测的消息。
- `scripts/task_update.py`：运行规则、导出无答案请求、导入外部预测评分。
- `scripts/predict_task_updates.py`：通过 Chat Completions 兼容服务逐条生成原始预测，不启动模型服务。

当前是**给定固定历史上下文的一次任务更新评测**。共同的规则构建器从初始指令和历史消息独立建立旧策略状态，
不是把参考正确状态直接提供给更新器。初始或历史语言无法解析时保留失败，不补入正确答案。
外部模型只预测当前消息，先前状态并非该模型在连续对话中自己产生的结果。
预测客户端已用模拟服务测试，并完成首次真实 Qwen3.5-9B 开发样例运行；
latest/history/state 完整正确分别为 15/18、15/18、16/18，详见 [本轮结果与限制](qwen35_task_update_results_2026-11-05.md)。
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

上述根目录预测文件尚不存在。可以使用下面的客户端直接生成子目录中的预测文件，再评分。

### 4.1 自动生成真实模型预测

前提：已运行一个支持 `POST /chat/completions` 的服务，已知 API 基址和实际模型标识。
例如 [vLLM 的兼容服务](https://docs.vllm.ai/en/stable/serving/openai_compatible_server/) 使用 `/v1/chat/completions`。
基址应包含服务所需的 `/v1`，客户端会追加 `/chat/completions`。
本入口不会下载权重、安装推理框架或启动 GPU 服务，不保证支持所有供应商特有的接口与参数。

`outputs/task_update_requests_003` 已导出 54 条有效公共请求，无需再次导出。
以下地址仅为本地服务示例，模型名必须替换为该服务真实暴露的名称。
若没有模型服务，应先确定本地部署或远程 API；此时还不能运行真实预测。

```bash
conda activate mivln
export TASK_UPDATE_BASE_URL="http://127.0.0.1:8000/v1"
export TASK_UPDATE_MODEL="替换为服务的实际模型名"

python -m scripts.predict_task_updates \
  --requests-dir outputs/task_update_requests_003 \
  --base-url "$TASK_UPDATE_BASE_URL" \
  --model "$TASK_UPDATE_MODEL" \
  --output-dir outputs/task_update_model_001

python -m scripts.task_update score \
  --requests-dir outputs/task_update_requests_003 \
  --predictions outputs/task_update_model_001/predictions.jsonl \
  --updater-name "$TASK_UPDATE_MODEL" \
  --output-dir outputs/task_update_model_eval_001
```

如果服务需要认证，在当前终端设置 `TASK_UPDATE_API_KEY`；客户端从环境变量读取，不要求将密钥写入代码。
可用 `--api-key-env` 指定其他变量名。所有输出目录必须为新目录。
每条请求单独使用系统说明和本条公共请求，不共享对话，也不读取源数据或参考答案。
系统提示在导出说明末尾明确要求只预测内部对象，外层 ID 与哈希由客户端填写。

生成目录包含：

- `predictions.jsonl`：54 个计划槽位；`prediction` 保存模型原始文本。调用失败用 `null` 占位。
- `responses.jsonl`：请求错误、服务报告的模型名、结束原因及用量（服务提供时）。
- `system_prompt.txt`：实际使用的系统提示。
- `run.json`：输入和提示哈希、服务地址、模型名、参数、调用计数及时间；不保存 API 密钥。

默认参数为 `--temperature 0 --max-tokens 512 --timeout 60`；温度为零不保证服务完全确定。
`--thinking-mode default|off|on` 控制兼容服务的 `chat_template_kwargs.enable_thinking`：
默认不发送该扩展，`off` 发送布尔 `false`，`on` 发送布尔 `true`，配置保存到 `run.json`。
此开关适用于 vLLM/Qwen 聊天模板；供应商的其他参数形式仍需单独适配。
本地已下载的 Qwen3.5-9B 路径、环境安装、非思考服务和首次 54 条预测命令见
[本地部署说明](qwen35_local.md)。
不自动重试、不移除围栏、不修复 JSON。调用失败继续保留后续槽位；调用期间中断时，尚未尝试的槽位填 `null`。
`received` 只表示收到文本，格式与语义是否正确由后续评分决定；截断响应也保留原文和结束原因。
生成命令遇到调用失败返回 1，仍可对已保存文件评分。
首次本地 Qwen3.5-9B 真实预测已完成并逐条复算，记录完整性通过；八条语义错误与四条缺少上下文依据的命中已单独分析。
平衡原目标与恢复目标的 [成对上下文评测](task_update_pairs.md) 已实现，
dev/test 各 216 条公共请求已冻结，单独评价缺失上下文、猜中答案和歧义误执行。
真实 dev 的 216 条记录已核对，决策正确数 latest/history/state 分别为 34/72、21/72、52/72，
详见 [dev 结果](qwen35_pairs_dev_results_2026-11-05.md)。同设置 test 也已完成，分别为 36/72、25/72、46/72；
详见 [test 阶段结论](qwen35_pairs_test_results_2026-11-05.md)。
[历史组织与拒绝判断消融入口](task_update_ablation.md) 已实现并完成 CPU 预览；下一步执行两个 dev 模型批次并统一比较。
连续模型状态与导航接入仍待实现。

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
python -m unittest tests.test_predict_task_updates -v
python -m unittest tests.test_task_update_ablation -v
python -m unittest discover -s tests -v
```

专项测试覆盖答案隔离、别名歧义、中英文三类更新、旧约束和恢复目标保留、事务失败、严格 JSON、
上下文失败分母、请求冻结、外部响应缺失/错位、导出导入一致性和拒绝覆盖。
任务更新专项 20 项、预测客户端专项 8 项、成对评测专项 11 项、提示消融专项 11 项通过；
全项目在 `habitat` 环境 166 项通过，核心环境通过并跳过 62 项依赖测试。
测试使用 CPU 假模拟器，没有启动真实 GPU 导航。
本轮核心接口、规则、评价、脚本、样例、测试及相关说明可以提交为：
`feat: add isolated language task update contracts and offline evaluation`。
图稿改动单独处理；运行输出保持 Git 忽略，不自动暂存、提交或推送。
预测客户端作为后续独立提交时，可用 `feat: add model prediction generation for task update evaluation`。
