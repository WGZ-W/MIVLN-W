# MI-VLN 项目周报 / Weekly Report

日期 / Date：2026-11-05

统计周期 / Reporting Period：2026-10-30—2026-11-05

## 中文

### 子目标插入与恢复实验

完成 HSSD 场景中“先完成子目标 B，再恢复原目标 A”的批量验证。在同一场景的 10 个固定任务上，完成单时机 30 个 episode 和早、中、晚三时机 90 个 episode。

Follow 在早、中、晚介入下的平均路程分别为 20.37、22.60、25.72 米，在这组固定任务中，较晚插入带来的额外行程更大。

### 语言任务更新接口与真实模型评测

实现独立于模拟器的语言更新接口，支持目标替换、禁区新增和临时子目标插入，统一输出校验、事务式状态更新与离线评分。接入本地 Qwen3.5-9B，比较仅最新消息（latest）、初始指令加历史（history）和结构化任务状态（state）三种输入。首次 18 个合成样例展开为 54 次调用，完整更新及状态正确数分别为 15/18、15/18、16/18，主要用于验证真实模型链路。

进一步构建 12 个语义组，通过中英文表达、三种当前目标和两种候选顺序形成 144 个变体样例，展开为 dev/test 各 216 条请求。评价区分“匹配隐藏任务答案”和“依据当前输入作出合理决策”，避免将上下文不足时的猜中计为理解正确。

首批与成对实验共 486 次模型调用，均返回结果，无调用失败或长度截断标记。state 在 dev/test 的明确更新中分别正确 48/48、46/48；history 在两批各 24 条隐式更新中均有 22 条误用初始目标。

### 当前边界

当前证据来自单场景 Oracle 导航与固定合成语言样例。Oracle 使用坐标和结构化更新；语言模型未接入导航，先前任务状态由固定语言规则构建，尚未验证模型连续维护状态、视觉落地或跨场景泛化。历史输入的呈现方式也可能影响结果，现阶段不将差异归结为结构化记忆的普遍优势。

### 下周计划

1. 基于已实现的入口完成消融实验准备，设置原设置、仅改历史呈现、仅加强拒绝说明三个独立条件，核验请求体、哈希记录和统一评分。保持模型、候选和生成参数固定，运行两个 dev 变体，每批 216 条请求。复用原基线比较目标混淆、误执行和误拒绝，同时检查明确更新是否退化。
2. 在 dev 上选择提示，将已观察的 test 作为回归集；另行设计并冻结新的验证语义组，避免依据旧 test 调参后继续将其视为未见评测。
3. 补充子目标代表视频的人工播放记录；梳理模型连续生成任务状态的评测方案，再逐步对接固定 HSSD 导航后端，区分语言更新错误与导航执行失败。

## English

### Subgoal Insertion and Resumption

Batch validation of visiting subgoal B before resuming the original goal A was completed in HSSD. Ten fixed tasks in one scene were evaluated in a 30-episode single-timing batch and a 90-episode early/middle/late batch.

Mean Follow path lengths were 20.37, 22.60, and 25.72 m for early, middle, and late interventions. Later insertion required more additional travel on this fixed task set.

### Language Task Updates and Real-Model Evaluation

A simulator-independent language-update interface was implemented for goal replacement, avoid-region addition, and temporary-subgoal insertion, with shared output validation, transactional state updates, and offline scoring. Local Qwen3.5-9B inference was integrated to compare the latest message alone (latest), the initial instruction plus history (history), and structured task state (state). An initial check expanded 18 synthetic examples into 54 calls. Exact update-and-state scores were 15/18, 15/18, and 16/18, respectively, establishing that the real-model pipeline operates.

A controlled evaluation then combined 12 semantic families with two languages, three current goals, and two candidate orders, producing 144 variant examples and 216 requests per dev/test partition. Scoring separates hidden task-answer matches from decisions supported by the available input, preventing lucky guesses without context from counting as successful understanding.

All 486 calls across the initial and paired experiments returned responses, without call failures or length-truncation flags. State correctly handled 48/48 clear dev updates and 46/48 clear test updates. History incorrectly reused the initial goal in 22 of 24 implicit updates in each partition.

### Current Scope

Current evidence covers Oracle navigation in one scene and fixed synthetic language examples. The Oracle uses coordinates and structured updates. The language model has not been connected to navigation, and prior task states are constructed by fixed language rules. Continuous model-maintained state, visual grounding, and cross-scene generalization remain untested. History presentation may also affect the observed differences, which do not establish a general advantage of structured memory.

### Plan for Next Week

1. Complete ablation preparation using the implemented entry points, defining three separate conditions: baseline, history presentation only, and refusal instructions only. Verify request bodies, hash records, and shared scoring. Keep the model, candidates, and generation settings fixed, and run two dev variants with 216 requests each. Compare against the baseline for goal confusion, erroneous execution, and false refusal, while checking for degradation on clear updates.
2. Select prompts on dev and retain the observed test partition for regression checks. Design and freeze new validation families before making claims about unseen performance after tuning.
3. Record manual playback reviews of representative subgoal videos. Specify evaluation of continuously model-generated task state, followed by gradual integration with the fixed HSSD navigation backend, separating language-update errors from navigation failures.

## 实验记录 / Experiment Records

- [HSSD 子目标验收 / Subgoal acceptance](hssd_subgoal_results_2026-11-03.md)
- [首次真实模型评测 / Initial real-model evaluation](qwen35_task_update_results_2026-11-05.md)
- [成对 dev 结果 / Paired dev results](qwen35_pairs_dev_results_2026-11-05.md)
- [成对 test 结果 / Paired test results](qwen35_pairs_test_results_2026-11-05.md)
- [消融入口与执行命令 / Ablation entry points and commands](task_update_ablation.md)
