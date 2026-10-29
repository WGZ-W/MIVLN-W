# MI-VLN 项目周报 / Weekly Report

日期 / Date：2026-10-29

统计周期 / Reporting Period：2026-10-23—2026-10-29

## 中文

### HSSD 场景接入与语义任务构建

完成 HSSD 家庭场景接入，核验场景配置、134 个家具物体及包含静态家具的导航网格。通过目标近景预览与显式审核，将床、沙发、浴缸、厨房岛台和餐桌五类实例的语言描述、稳定实例标识与可达停止点绑定，构建了 10 个固定任务。配置、场景、导航网格和审核证据均记录哈希，保证重复实验中的任务与环境一致。

### 动态任务实验与评价完善

在 HSSD 中完成早、中、晚目标替换及 Ignore 负对照验证。针对原批次中三个“到达停止点但未朝向目标”的案例，增加终点转向与指定实例可见性诊断，并在相同 10 个任务上完成新变体的 100 个 episode：70 条参考、控制和 Follow 条件成功，30 条 Ignore 按预期失败，三个终帧问题均已解决。新批次记录 6448 组观测、0 个碰撞标记，并导出 70 段视频。

实现新增禁区介入，保持原目标不变，通过受约束 Oracle 规划完成绕行，并以事件生效后的实际运动线段判定违规，避免仅检查离散端点造成漏检。完成单例及 10 任务、30 个 episode 的批量 GPU 验收：Control 与 Follow 均为 10/10 成功；Ignore 均到达原目标，但因违规被正确判失败，共记录 51 个违规动作。Follow 平均路程由 Control 的 8.659 米增加至 9.414 米，配对平均增加 0.755 米；碰撞标记与规划错误均为 0，30 段视频已导出。

实现临时子目标插入与恢复单例入口，支持“先去沙发 B，再继续前往岛台 A”。新增计步但不移动的 `COMPLETE_SUBGOAL` 逻辑动作，评价器根据实测距离、朝向和实例可见性确认 B 完成后才恢复 A，策略声明与评价真值保持分离。无渲染 CPU 原生动作检查完成了 B→A 序列，共 83 个动作；99 项自动测试全部通过，GPU 三条件验收尚待完成。

### 当前问题与验证边界

当前结果主要验证单个 HSSD 场景中的任务更新、连续执行和独立评价流程。Oracle 直接使用目标坐标和结构化任务更新，尚未实现自然语言理解或跨场景泛化评测。视频已导出，禁区批次的视频帧数已核对，完整人工播放审核仍需补充；子任务恢复的 CPU 导航检查不能替代实际 GPU 中间目标与最终目标的可见性验收。

### 下周计划

1. 执行子任务插入恢复的 GPU 单例，验证 Control 到 A、Follow 先 B 后 A、Ignore 漏掉 B 的三种条件，检查中间完成证据、位姿连续性和最终 STOP。
2. 完成代表性视频复核；单例通过后，扩展固定任务与“只完成 B、不恢复 A”等负例，保留失败分母，分析顺序错误、恢复失败及诊断异常。
3. 整理三类任务更新的统一评测接口，设计最新指令、历史拼接与结构化任务状态三种语言更新基线，明确模型输入与评价真值之间的信息边界。

## English

### HSSD Integration and Semantic Task Construction

HSSD household-scene integration was completed, including checks of the scene configuration, 134 furniture objects, and a navigation mesh incorporating static furniture. Close-up previews and explicit review linked five object types—beds, sofas, bathtubs, kitchen islands, and dining tables—to language descriptions, stable instance identifiers, and navigable stopping points. Ten fixed tasks were constructed. Hashes of configurations, scene assets, navigation meshes, and review evidence were recorded to preserve task and environment consistency across runs.

### Dynamic-Task Experiments and Evaluation

Early, middle, and late goal-replacement experiments and Ignore negative controls were verified in HSSD. Three original cases reached their stopping points without facing the target. Terminal turning and a visibility diagnostic for the specific target instance were added, followed by 100 episodes on the same ten tasks. All 70 reference, control, and Follow episodes succeeded, while 30 Ignore episodes failed as expected. The three terminal-view issues were resolved. The new batch produced 6,448 observation sets, zero collision flags, and 70 videos.

An avoid-region intervention was implemented while preserving the original destination. A constrained Oracle planner produced detours, and violations were evaluated using actual motion segments after event delivery rather than discrete endpoints alone. A single-case experiment and a ten-task batch of 30 episodes passed GPU acceptance checks. Control and Follow each achieved 10/10 successes. All ten Ignore episodes reached the original destination but correctly failed because of constraint violations, totaling 51 violating actions. Mean path length increased from 8.659 m for Control to 9.414 m for Follow, with a paired mean increase of 0.755 m. Collision flags and planner errors were zero, and 30 videos were exported.

A temporary-subgoal insertion and resumption entry was implemented for visiting sofa B before returning to the original kitchen-island goal A. The counted logical action `COMPLETE_SUBGOAL` causes no movement. Evaluator restoration to A requires measured distance, heading, and instance visibility at B, independently of the policy's completion claim. A CPU native-action check without rendering completed the B→A sequence in 83 actions. All 99 automated tests passed; GPU validation of the three conditions remains pending.

### Open Issues and Validation Scope

Current results primarily validate task updates, continuous execution, and independent evaluation in one HSSD scene. The Oracle directly accesses goal coordinates and structured task updates; natural-language understanding and cross-scene generalization have not been evaluated. Videos have been exported, and frame counts were checked for the avoid-region batch. Full manual playback review remains outstanding. CPU navigation checks for subgoal resumption do not establish intermediate or final target visibility in an actual GPU episode.

### Plan for Next Week

1. Run the GPU subgoal-insertion experiment: Control reaches A, Follow completes B then A, and Ignore omits B. Verify intermediate completion evidence, pose continuity, and final STOP.
2. Review representative videos. After single-case acceptance, expand the fixed tasks and negative cases, including completion of B without resuming A. Retain failures in the denominators and analyze ordering errors, resumption failures, and diagnostic errors.
3. Consolidate evaluation interfaces for the three update types. Design language-update baselines using the latest instruction, concatenated history, and structured task state, with explicit separation between model inputs and evaluator ground truth.
