# MI-VLN 项目周报 / Weekly Report

日期 / Date：2026-11-13

统计周期 / Reporting Period：2026-11-07—2026-11-13

## 中文

### 语言更新导航与稳健性验证

完成 Qwen3.5-9B 语言更新驱动的 HSSD 导航批量验证。两候选与五候选批次均覆盖 10 个固定任务、30 个 episode，模型更新及导航闭环均为 10/10。进一步完成 60 条措辞改写与候选逆序请求，更新及状态正确数为 60/60；选取五个代表任务完成 15 个新导航 episode，在线更新、导航闭环及在线/离线一致性均为 5/5。上述结果验证的是语言更新与固定特权导航后端的衔接，不是端到端视觉导航能力。

### Benchmark 协议与数据流程

明确以可复现 Benchmark 建设为优先目标，整理任务协议、输入权限、统一清单与独立评价。对既有替换、禁区、插入恢复共 220 个历史 episode 完成只读重审，全部符合各自验收协议，保留原始结果及失败恢复证据。当前主采集范围调整为目标替换、子目标插入与恢复、取消任务并立即停止；禁区因边界及视觉可观察性问题后置。

实现语言标注、修订、审核及版本化导出流程。基于 10 个基础任务整理并完成人工文本/RGB 确认，共 40 条原模板语言：初始指令、替换、插入恢复和取消各 10 条，导出 30 个离线案例。审核与原始模板来源分别保留；修改初始指令会使相关介入审核失效，未获批记录不能通过导出门槛。

### 三类 V 导航接口与真实 HSSD 接线

实现逐步 V 输入与动作闭环，向普通 Agent 提供初始语言、当前/历史 RGB、实际动作及已投递消息，隔离坐标、正确更新和实例诊断。生成三类、三时机共 90 个 HSSD 准备配置；完成五条件共 450 个 CPU 合成 episode，全部按预期验收，其中 150 个任务成功、300 个负条件按预期失败。延迟一次动作不必导致替换或插入失败，但取消必须当步 STOP。

完成真实 Habitat backend，并在 `task_001 + middle` 上运行三类 Oracle/Ignore 共 6 个真实 episode：三个 Oracle 均成功，三个 Ignore 均按预期失败。介入在动作 22 之前投递；插入任务有效完成沙发 B 后恢复岛台 A，取消任务 STOP 延迟为 0、取消后路程为 0。最新只读重审通过，6/6 分支验收、配对错误 0，运行源文件前后不变。全量 392 项自动测试在 Habitat 环境通过，核心环境按依赖跳过 83 项。

### 当前边界

当前仍是单场景、10 个基础任务的开发原型。450 个 CPU episode 使用合成几何和图像，不能计为真实 HSSD 实验；新的三类 V 真实导航目前仅验证一个基础任务。Oracle 直接使用正确更新、坐标和网格，V 链路尚未接入真实视觉策略。语言仍为经审核的模板表达，多场景划分、完整运行隔离及代表视频人工播放审核尚待完成，尚未达到正式发布条件。

### 下周计划

1. 实现三类真实 HSSD middle 批量验收，将范围扩展至 10 个基础任务、60 个计划 Oracle/Ignore 分支；满足配置和版本绑定条件时复用已有结果，保留全部失败与缺失槽位。
2. 导出并人工复核替换、插入恢复、取消的代表视频，重点检查介入时序、B 的实例完成、A 的恢复以及取消后的立即 STOP。
3. middle 批次通过后，再扩展 early/late，按类型和时机分别统计任务成功与错误原因，保持输入权限和评分一致。
4. 准备第二个场景及更丰富的初始/介入表达，明确 scene-disjoint 分组与数据质检要求，推进可复现发布所需的数据说明和评价边界。

## English

### Language-Update Navigation and Robustness

Batch validation of Qwen3.5-9B language updates driving HSSD navigation was completed. Both the two-candidate and five-candidate batches covered ten fixed tasks and 30 episodes, achieving 10/10 correct updates and successful navigation pipelines. A further 60 requests tested paraphrasing and reversed candidate order, with 60/60 correct updates and states. Fifteen new navigation episodes on five representative tasks achieved 5/5 for online updates, navigation success, and online/offline agreement. These results validate language updates coupled to a fixed privileged navigation backend, rather than end-to-end visual navigation.

### Benchmark Protocol and Data Workflow

Reproducible Benchmark construction became the primary project objective, covering task protocols, input permissions, unified inventories, and independent evaluation. A read-only audit verified all 220 historical episodes across replacement, avoidance, and subgoal insertion/resumption under their respective acceptance protocols, preserving original results and recovery evidence. Current collection focuses on goal replacement, subgoal insertion/resumption, and task cancellation with immediate stopping. Avoid-region collection is deferred because region boundaries and visual observability remain unresolved.

A versioned workflow for language annotation, revision, review, and export was implemented. Human confirmation of text and RGB evidence covered 40 original template annotations on ten base tasks: ten initial instructions and ten updates for each intervention type. Thirty offline cases were exported. Review declarations and template provenance remain distinct. Revising an initial instruction invalidates dependent intervention reviews, and unapproved records block export.

### Three-Task V Interface and Real HSSD Integration

A stepwise V input/action loop was implemented. Ordinary agents receive initial language, current/historical RGB, executed actions, and delivered messages, while coordinates, correct updates, and instance diagnostics remain private. Preparation produced 90 HSSD configurations across three task types and three timings. All 450 synthetic CPU episodes across five conditions met their expected outcomes: 150 task successes and 300 expected negative outcomes. A one-action delay can still permit replacement or subgoal success, whereas cancellation requires STOP in the delivery slot.

The real Habitat backend was integrated and exercised in six Oracle/Ignore episodes for the three task types on `task_001 + middle`. All three Oracle episodes succeeded, and all three Ignore episodes failed as expected. Interventions arrived before action 22. The subgoal task completed sofa B before resuming kitchen island A; cancellation achieved zero STOP delay and zero post-cancellation travel. The latest read-only audit accepted all six branches, found no pairing errors, and confirmed unchanged source files. All 392 automated tests passed in the Habitat environment; the core environment skipped 83 tests requiring native dependencies.

### Current Scope

The project remains a single-scene development prototype with ten base tasks. The 450 CPU episodes use synthetic geometry and images and are not real HSSD runs. The new three-task V pipeline has been validated on only one base task in the real scene. Oracle policies access correct updates, coordinates, and the navigation mesh; a real visual policy has not yet been connected to V. Language remains reviewed template text. Multi-scene splits, full runtime isolation, and manual playback review of representative videos remain pending, so the Benchmark is not release-ready.

### Plan for Next Week

1. Implement batch acceptance for real three-task HSSD navigation at middle timing, expanding to ten base tasks and 60 planned Oracle/Ignore branches. Reuse existing results only when configuration and version bindings match, retaining all failed and missing slots.
2. Export and manually review representative videos for replacement, insertion/resumption, and cancellation, checking delivery timing, instance-level completion at B, restoration of A, and immediate STOP after cancellation.
3. After middle-batch acceptance, extend to early/late timings and report success and failure causes separately by task type and timing, with consistent input permissions and scoring.
4. Prepare a second scene and more varied initial/intervention expressions. Specify scene-disjoint grouping and data-quality checks, and continue documenting data and evaluation boundaries for reproducible release.

## 实验记录 / Experiment Records

- [两候选导航批次 / Two-candidate navigation](language_navigation_dataset.md)
- [五候选扩展 / Five-candidate expansion](language_navigation_candidates.md)
- [措辞及候选顺序 / Wording and candidate order](language_navigation_robustness.md)
- [代表任务导航 / Representative navigation](language_navigation_representatives.md)
- [历史资产只读审计 / Historical asset audit](benchmark_audit.md)
- [语言审核与导出 / Language review and export](language_annotations.md)
- [三类 CPU 导航 / Three-task CPU navigation](benchmark_navigation.md)
- [真实 HSSD middle 入口 / Real HSSD middle entry](hssd_visual_navigation.md)
- 最新真实运行：`outputs/hssd_v_middle_run_001/`。
- 本次周报只读核验：`outputs/hssd_v_middle_weekly_audit_2026-11-13/audit.json`，`passed=true`、`source_files_unchanged=true`。
