# MI-VLN 项目周报 / Weekly Report

日期 / Date：2026-10-22

## 中文

### 环境配置和仿真实验

完成服务器环境配置，核心逻辑和仿真脚本可在环境中共同运行。Habitat-Sim 的 RGB-D 渲染、动作执行、最短路径计算及视频导出均已跑通。固定目标导航在 49 个动作后成功停止，SPL 为 1.0，终点误差为 0.268 米。

实现途中目标替换闭环，接入任务状态管理、事件投递和从当前位置重新规划的流程。介入事件在动作选择前生效，执行过程保留机器人位姿与历史轨迹；成功判定使用实际位置到有效目标的导航距离，并要求执行 STOP。

构建 10 个不同起点、朝向与 A/B 目标组合，分别测试早、中、晚介入。共完成 70 个 episode，30 组配对实验全部通过自动验收，生成 40 段视频。介入前轨迹与 RGB-D 观测一致，所有 episode 的终点误差均不超过 0.3 米。

完成“忽略介入”负对照，将策略目标与评价器真值分离。事件在第 29 步正常投递后，策略继续前往 A，评价器保持 B 为有效目标；机器人最终停在 A，被正确判定为失败，响应延迟记录为 null。该负例的物理轨迹与无介入对照一致。

### 当前问题与验证边界

批量实验中出现两次运动过滤器碰撞标记，分别位于 task_004 的晚期介入和 task_009 的早期介入，需要结合视频进一步复核。当前结果验证了测试场景中的 Oracle 导航与评测流程，语言理解和家庭场景语义目标尚待接入。

### 下周计划

1. 核验 HSSD 数据完整性并记录版本，补充数据集配置加载支持，完成一个家庭场景的 RGB-D、导航网格和语义标注检查。
2. 手工选择两个语义明确的物体目标及可达停止点，建立自然语言描述、对象 ID 和目标区域的对应关系。
3. 在同一语义任务上复用无介入、正确处理介入和忽略介入三种对照，跑通一个完整的目标替换单例，保存配置、轨迹、指标与视频，并完成重点片段的人工复核。

## English

### Environment Setup and Simulation Experiments

The server environment was configured, allowing core task logic and simulator scripts to run together. Habitat-Sim RGB-D rendering, action execution, shortest-path computation, and video export were verified. The fixed-goal run stopped successfully after 49 actions, achieving an SPL of 1.0 and a final navigation error of 0.268 m.

Goal replacement during navigation was implemented through task-state management, event delivery, and replanning from the current pose. Events take effect before action selection while preserving robot pose and trajectory history. Success requires a STOP action within the permitted geodesic distance of the active goal.

Ten distinct combinations of starting position, orientation, and A/B targets were constructed, each evaluated with early, middle, and late interventions. All 30 matched pairs passed automatic validation across 70 episodes, and 40 videos were generated. Pre-intervention trajectories and RGB-D observations matched their controls, and every episode finished within the 0.3 m success radius.

An Ignore Intervention negative control was completed with separate policy state and evaluator ground truth. Following event delivery at step 29, the policy continued toward A while B remained the valid evaluation target. Stopping at A was correctly classified as failure, with response latency recorded as null. Physical execution matched the no-intervention control.

### Open Issues and Validation Scope

Two motion-filter collision flags occurred: task_004 with late intervention and task_009 with early intervention. These segments require further video review. Current results validate Oracle navigation and evaluation in the test scene. Language understanding and semantic targets in household scenes remain future work.

### Plan for Next Week

1. Verify HSSD data completeness and record the dataset revision. Add dataset-configuration loading support and inspect RGB-D rendering, navigation meshes, and available semantic annotations in one household scene.
2. Select two clearly identifiable object targets and navigable stopping locations. Establish mappings among natural-language descriptions, object IDs, and goal regions.
3. Reuse no-intervention, intervention-following, and intervention-ignoring controls on the same semantic task. Complete one goal-replacement episode, preserve its configuration, trajectory, metrics, and videos, and manually review key segments.
