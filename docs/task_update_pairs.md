# 成对上下文任务更新评测

更新日期：2026-11-05。目标是区分“利用已有上下文正确更新任务”“信息不足时合理拒绝”和“猜中隐藏目标”。
本阶段只做固定上下文的文本评测，不启动 Habitat，不涉及模型训练或连续模型记忆。

## 1. 实现与实验规模

- 配置：`configs/task_update_context_pairs_v1.json`；标签按作者配置独立构建，不从规则输出生成答案。
- 数据与评价：`mivln/task_update_pairs.py`；命令：`python -m scripts.task_update_pairs`。
- 真实预测继续使用已有 `scripts.predict_task_updates`，无需另装客户端依赖。

12 个语义组 × 中英文 2 种表达 × 3 种原目标 × 2 种候选顺序 = 144 个样例，
再分别提供 latest/history/state，合计 **432 条请求**。
dev/test 各 6 个语义组、72 个样例、216 条请求，每种模式各 72 条。
同一语义组的上下文、语言和顺序变体始终属于同一分区。

| 类型 | 设计 |
| --- | --- |
| 隐式恢复与目标替换 | 当前消息保持不变，原目标分别为 island、bed、table |
| 历史更新 | 初始目标与当前目标不同；先替换目标，再增加一条禁区约束 |
| 明确更新对照 | 明确命名旧/恢复目标，或明确指出新增禁区颜色，latest 也能解释 |
| 歧义拒绝 | 两张沙发均符合泛称；区域颜色明确未指定，避免指回旧禁区的标签歧义 |
| 未知引用 | 书架、紫色区域均不在公共候选集合中 |

每组使用循环移位和反转两种候选顺序；同一上下文对照中候选顺序完全相同。
每个分区的 216 条请求中，六种目标各占首位 36 次，两种区域各占首位 108 次。
这里只覆盖每组两种顺序，并不代表测试了所有排列。

公开请求使用不含任务含义的哈希标识；语义组、分区、原目标配置、期望拒绝和配对关系留在评价侧。
坐标仅为合成执行器夹具，未绑定 HSSD。旧策略状态仍由公共语言规则构建，无参考答案回填。

## 2. 两条评分轴

原有 `summary.json` 顶层和 `cases.csv` 保留统一任务参考答案的评分，用于观察是否真正完成更新。
新指标在 `summary.json["paired"]`、`decision_cases.csv` 及扩充后的 `cases.jsonl` 中。

- **明确更新**：操作、参数和新状态都正确，才获得 `decision_correct`。
- **缺少上下文**：隐式目标替换/恢复的 latest 请求应输出 `abstain/missing_context` 并保留原状态。
- **歧义与未知**：应输出对应拒绝理由；语法合法却随意选择候选，仍然判错。
- **猜中答案**：latest 在缺少原目标时预测更新，即使吻合隐藏参考，也只计入原始任务匹配，
  同时标记 `unsupported_guess` 和 `unsupported_guess_hit`，不计入决策正确。

例如“先去深色沙发，再继续原任务”在三个原目标下，history/state 应给出三个不同的恢复目标；
latest 不知道“原任务”具体是什么，合理行为是拒绝猜测。
因此，latest 的正确拒绝与 state 的完成任务不是相同执行效果，必须同时报告两条评分轴。
替换操作需要 `old_goal` 的契约保护，也影响 latest 的可回答性，不能据此推断它无法识别一句话中的新目的地。

主要字段：

- `paired.by_split_mode`：按 dev/test 和模式报告决策结果。
- `actionable_*`、`missing_context_*`、`ambiguity_*`、`unknown_*`：各类固定分母与正确率。
- `unsupported_guess_rate`：以缺少上下文的请求为分母；`ambiguity_execution_rate`：以歧义请求为分母。
- `context_groups_by_mode`：同消息、同顺序、三种旧目标全部正确的组数，以及始终填同一旧/恢复目标的组数。
  latest 的全部正确表示组内都合理拒绝，不表示理解了隐藏目标。
- `order_groups_by_mode`：两种候选顺序下的预测一致组数、两条均正确组数。
  同样答错也可能一致，不能单看一致率。

缺失、非法、哈希不匹配或上下文构建失败均保留原请求及配对组分母。
旧版 18 样例及 54 条 Qwen 结果的参考答案、评分语义与原始文件保持不变。

## 3. 本轮已完成的 CPU 检查

正式冻结前，将可能指回历史区域的“那片区域”改为“颜色暂不明确”的措辞。
早期 `_001` 目录为开发中间产物；后续使用与当前配置哈希一致的 **`_002`** 目录。

```bash
python -m scripts.task_update_pairs run \
  --split all --output-dir outputs/task_update_pairs_rules_002
python -m scripts.task_update_pairs export \
  --split dev --output-dir outputs/task_update_pairs_dev_requests_002
python -m scripts.task_update_pairs export \
  --split test --output-dir outputs/task_update_pairs_test_requests_002
```

以上命令已执行；重复运行需另选新目录。dev/test 均成功导出 216 条公共请求。
本地 Qwen 分词器与非思考聊天模板检查：dev 输入 692–748 token，test 输入 694–755 token；
最长输入加 512 token 输出预算为 1267，适配现有 4096 上下文配置。
规则开发基线的决策结果：

| 分区 | latest | history | state |
| --- | ---: | ---: | ---: |
| dev | 72/72 | 72/72 | 72/72 |
| test | 36/72 | 36/72 | 36/72 |
| all | 108/144 | 108/144 | 108/144 |

test 中三个语义组采用现有短语规则未覆盖的表达，规则会返回 `unsupported_message`。
这反映规则基线的语法范围，不是模型实验结果，也不代表状态表示无效。
标签不因规则是否答对而调整。此表仅报告 CPU 规则基线；真实 Qwen dev 结果见下一节。
原始任务匹配分数则为 latest 72/144、history/state 各 108/144；
latest 在隐式目标任务上的安全拒绝不会被计为完成更新。

11 项新增测试覆盖分区隔离、控制变量、候选平衡、公开字段边界、猜中与拒绝区分、
常量目标猜测、顺序扰动、失败分母、请求冻结和导出导入。
全项目在 Habitat 环境 166 项通过，核心环境发现 166 项并跳过 62 项依赖测试，均未启动真实导航。

## 4. dev 的 216 条真实预测（已完成）

用户已完成 dev，本次核对 216 次调用全部返回，有依据的决策正确为 latest 34/72、history 21/72、state 52/72。
明确更新 state 为 48/48，但歧义拒绝仍为 0/12；详见 [dev 实测报告](qwen35_pairs_dev_results_2026-11-05.md)。
以下 `_001` 模型与评分目录现已存在，保留命令用于复现，重复运行需选择新目录。

前提：本地 `qwen3.5-9b` 服务已启动，按 [部署说明](qwen35_local.md) 配置。
使用与首批相同的非思考模式、温度和输出预算；所有后续输出仍要求使用新目录。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
conda activate mivln
curl --fail --silent --show-error http://127.0.0.1:8000/v1/models

python -m scripts.predict_task_updates \
  --requests-dir outputs/task_update_pairs_dev_requests_002 \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.5-9b \
  --thinking-mode off \
  --temperature 0 --max-tokens 512 --timeout 120 \
  --output-dir outputs/task_update_pairs_qwen35_dev_001

python -m scripts.task_update_pairs score \
  --requests-dir outputs/task_update_pairs_dev_requests_002 \
  --predictions outputs/task_update_pairs_qwen35_dev_001/predictions.jsonl \
  --updater-name qwen3.5-9b-nonthinking \
  --output-dir outputs/task_update_pairs_qwen35_dev_eval_001
```

本轨道必须使用 `task_update_pairs score`，其会从冻结的成对配置重建完整请求；
通用 `task_update score` 面向旧版显式样例配置，不支持本配置展开格式。
验收先检查 216 条槽位与请求哈希完整对齐，每种模式分母 72，再分析决策分类和配对组。
不要求模型 100% 正确；更新应用失败会使命令返回 1，但完整评分文件仍保留。

## 5. 随后运行 test

test 已完成，216 条记录逐条复算一致；提示哈希和生成参数与 dev 相同。
有依据决策正确 latest/history/state 分别为 36/72、25/72、46/72，
state 明确更新 46/48，但 24 条应拒绝消息均未正确拒绝，见 [test 结果与阶段结论](qwen35_pairs_test_results_2026-11-05.md)。
以下为已执行命令，`_001` 输出目录已经存在，保留用于复现；重跑时需使用新目录。

```bash
python -m scripts.predict_task_updates \
  --requests-dir outputs/task_update_pairs_test_requests_002 \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.5-9b \
  --thinking-mode off \
  --temperature 0 --max-tokens 512 --timeout 120 \
  --output-dir outputs/task_update_pairs_qwen35_test_001

python -m scripts.task_update_pairs score \
  --requests-dir outputs/task_update_pairs_test_requests_002 \
  --predictions outputs/task_update_pairs_qwen35_test_001/predictions.jsonl \
  --updater-name qwen3.5-9b-nonthinking \
  --output-dir outputs/task_update_pairs_qwen35_test_eval_001
```

dev/test 是本轮预先指定的语义组分区，作者已了解此前同类任务和物体；
部分 dev 表达沿用旧样例。test 是新增的受控合成检查，不构成独立采集的自然语言泛化基准。
432 个请求也不是 432 个独立语义任务，统计时应考虑语义组及配对关系。
完整连续对话状态与 HSSD 模型导航仍是后续阶段。
历史信息组织与拒绝判断的独立变体、实际输入日志和统一比较入口已实现，
见 [提示消融命令](task_update_ablation.md)。历史 dev 变体已审计：history 从 21/72 升至 39/72，
但显式替换退化、歧义拒绝未解决，见 [历史消融结果](qwen35_history_ablation_dev_results_2026-11-06.md)。
拒绝变体和三组比较已完成，latest/history/state 为 35/72、27/72、53/72，见 [三条件分析](qwen35_prompt_ablation_dev_results_2026-11-06.md)。
当前优先开展明确更新，[连续状态传递入口](task_update_rollout.md)的 288 次真实调用已审计，state/history 更新及状态正确为 144/144、124/144，
见 [连续实验结果](qwen35_rollout_dev_results_2026-11-06.md)。当前进一步 history/state 验证暂缓，
已完成 [HSSD 模型更新回调与三条件目标替换入口](language_navigation.md)的真实 GPU 单例验收，
见 [结果与视频一致性](hssd_language_results_2026-11-09.md)。下一步扩展其他固定任务。
独立的可回答性与拒绝诊断仍未实现，暂列后续。
后续调参使用 dev；已观察的 test 作为回归集，新增效果结论需要新的冻结验证组。

## 提交建议

源配置、成对生成/评价模块、命令入口、测试、通用评分器的小幅扩展及相关说明可以合并提交为：
`feat: add controlled context pairs for task update evaluation`。
CPU 检查与真实 dev/test 记录核对完成；`outputs/` 保持 Git 忽略，不自动提交。
