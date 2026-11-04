# HSSD 临时子目标插入与恢复：单例实验

## 目的与当前状态

初始前往岛台 A，途中收到“先去沙发 B，再继续去 A”，从当前位姿完成 B→A。
协议标识 `insert_subgoal_v1`，入口 `python -m scripts.subgoal`，与历史目标替换、禁区实验分开评测。

代码、99 项 CPU/模拟器替身测试及真实 HSSD 网格上的原生动作检查已完成。
`outputs/hssd_subgoal_prepare_001/config.json` 已在 CPU 上准备好；
2026-10-30 已对 `outputs/hssd_subgoal_001/` 的 GPU 三分支结果完成独立离线验收，全部符合预期。
本协议只支持一次插入、一个临时目标；嵌套插入、多次介入与语言解析未实现。
2026-11-03 补齐固定任务与早/中/晚批量入口，30/90 episode GPU 已运行且通过修正后的独立验收，
详见 [子目标批量与阶段验收](subgoal_dataset.md) 和 [实测与验收器修复](hssd_subgoal_results_2026-11-03.md)。

## 任务与中间完成语义

```json
{
  "operation": "insert_subgoal",
  "subgoal": "target_054",
  "resume_goal": "target_051"
}
```

事件在动作 k 前投递一次：原目标 A 进入 `suspended_goals`，B 成为 `active_goal`；
A 不进入 `cancelled_goals`。完成 B 后将 B 加入 `completed_subgoals`，弹出并恢复 A。
任务状态保留既有约束与历史；本轮不混合新增禁区和插入事件。

**本协议显式增加逻辑动作 `COMPLETE_SUBGOAL`**，与最终 `STOP` 区分：

- 到达 B 停止点后，Oracle 用原有左右转动作面向 B；随后发出 `COMPLETE_SUBGOAL`。
- 该逻辑动作计入原有总动作预算，不调用 `sim.step`，不移动、不推进物理、不重置或结束 episode。
- 最终在 A 发出 `STOP` 才结束任务；中间 STOP 不能被解释为完成 B 后继续。
- 这是新实验协议的动作扩展，旧 v0.1 动作、STOP 语义和既有实验输出不追溯修改。

策略状态的完成声明与评价真值分开。评价器必须独立确认 B：

1. 实测 geodesic 距离 ≤ 0.3 m；
2. 面向 B 实例中心的水平角差绝对值 ≤ 7.5°；
3. 指定 B 实例可见像素 ≥ 64；不是同类别所有实例的像素之和；
4. 临时实例标签和位姿都已恢复/保持，诊断没有失败。

评价器只在上述条件成立时恢复 A。策略自己的状态会按其完成意图恢复 A，
因此诊断失败或 B 不可见时，评价器仍要求 B，策略可能继续前往 A，联合任务仍失败。
诊断观测不给策略；策略用特权坐标与结构化更新规划，当前不代表语言或视觉理解能力。

## 1. CPU 准备：本轮已执行

```bash
python -m scripts.subgoal prepare \
  --source-config outputs/hssd_dataset_facing_001/configs/task_001/base.json \
  --reference-dir outputs/hssd_dataset_facing_001/tasks/task_001/reference \
  --output-dir outputs/hssd_subgoal_prepare_001
```

不要重复创建同一准备目录。继承 source 的起点、朝向、目标停止点、实例、种子、传感器和 500 步预算；
不采样新目标、不自动审批实例。A/B 均须与已有显式审核目录匹配。
存在多个备选目标时需用 `--subgoal` 明确指定，不猜测语言指代。

准备冻结源配置、参考 episode/轨迹/指标/终点诊断的哈希，以及审核实例中心。
触发时机为参考实际行程约 25% 推导的第 14 步，动作前生效；
B 是 `target_054` 沙发，恢复目标 A 是 `target_051` 厨房岛台。
触发位姿→B geodesic 约 4.377 m，B→A 约 4.870 m，停止区域互不重叠。

无渲染 CPU 原生动作检查记录在 `cpu_navigation_check.json`：
第 45 步可完成 B（误差约 0.151 m、角差 3.089°），第 82 步可在 A STOP
（误差约 0.278 m、角差 −2.661°），合计 83 个动作，包含计步的中间完成动作。
该检查没有创建 `habitat_sim.Simulator`、没有渲染，也没有证明 B/A 在 GPU 中可见。

## 2. GPU 三条件：本轮已执行

在仓库根目录激活 Habitat 环境，使用分配的 GPU：

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python -m scripts.subgoal run \
  --config outputs/hssd_subgoal_prepare_001/config.json \
  --gpu-id 0 --output-dir outputs/hssd_subgoal_001

# CPU 离线重审，需 Habitat 的 PathFinder，但不创建模拟器或使用 GPU。
python -m scripts.subgoal summarize \
  --output-dir outputs/hssd_subgoal_001
```

输出目录必须是新的。默认保存逐帧数据，不编码视频。
依次运行 Control、Follow（`intervention/`）、Ignore；失败保留输出，不阻止后续分支，分母固定为 3。
不支持覆盖、续跑、自动重试或更换失败任务。

## 3. 验收结果与解释

实测 `summary.json`：`passed: true`、`expected_episodes: 3`，三条件验收均通过。
本次检查另外调用只读 `check_experiment` 重算，未改写原始输出，也未启动 GPU 模拟器。

| 条件 | 根目标 A 到达＋STOP | 完成 B 并恢复 A | 联合任务成功 |
| --- | --- | --- | --- |
| Control | true | 无插入要求 | true |
| Follow | true | true，且严格先 B 后 A | true |
| Ignore | true | false | false，符合负对照预期 |

| 条件 | 动作数 | 实走路程 | 最终目标 A 点误差 | 联合任务成功 |
| --- | --- | --- | --- | --- |
| Control | 43 | 7.000003 m | 0.296612 m | true |
| Follow | 83 | 10.753810 m | 0.277641 m | true |
| Ignore | 43 | 7.000003 m | 0.296612 m | false |

事件在第 14 步动作前投递；Follow 在第 45 步以 `COMPLETE_SUBGOAL` 完成沙发 B，
实测误差 0.151145 m、水平角差 3.089447°、实例像素 7482。
完成动作前后位姿一致，随后恢复岛台 A，第 82 步发出最终 STOP，共 83 个计步动作。
A 终点实例像素 11939、水平角差 −2.660878°；全组三条件碰撞和规划错误均为 0。
Follow 比配对 Control 多走约 3.753808 m；该差值不是两阶段最优路径指标。

独立核查通过配置/场景/参考哈希、动作时序、状态重放、中间及最终几何与掩码证据、
Follow 介入前观测一致，以及 Ignore 与 Control 全程物理执行和实际观测一致。
人工查看了 Follow 的 B 完成图和 A 终点图，分别可见沙发和岛台。
单例三段视频后续已导出，完整人工播放审核仍待记录；结果仅支持单场景特权 Oracle 协议验收。

- `root_goal_success` 单独衡量最终是否在最初目标 A STOP；三条件都应为 true。
- `point_goal_success` 继续针对**评价器当前 active_goal**，Ignore 尚欠 B，所以该值为 false，
  `final_goal` 为 B，`policy_goal` 仍为 A；这不表示 Ignore 没有到达 A。
- Follow 的 `subgoal_completed`、`resumed_original_goal`、`sequence_success` 都应为 true，
  最终任务状态为空挂起栈、`completed_subgoals: [B]`、无被取消的 A。
- Ignore 的 `subgoal_events.json` 和 `replans.json` 为空，响应延迟为 null，
  终止原因是 `subgoal_sequence_incomplete`，不能由超时、规划失败或没有 STOP 代替。
- 只完成 B、在 B STOP、漏掉 B、预算耗尽、诊断失败或仅声称完成，都不能成功。
- `post_intervention_spl` 为 null；旧的介入后 geodesic 字段只到当时的 B，不等于 B→A 两阶段最优路径。

Follow 的 `subgoal_events.json` 保存动作步号、测量距离/角差、真假状态、证据路径和哈希。
`subgoals/<step>/` 保存 B 完成时的独立实例掩码、RGB 和 `terminal_diagnostics.json`。
`replans.json` 记录插入 B、恢复 A 两次重规划；恢复不设置机器人位姿。

独立汇总重新加载固定网格，重算 B 完成与最终 A 的距离、从四元数重算朝向，
核查原始掩码像素、证据哈希和 RGB；再从事件与实测完成证据重放评价/策略状态。
策略记录的状态、完成标记和最终 success 字段都不能替代这些检查。
另查事件前真实观测前缀、Control 对原参考、Ignore 对 Control 的完整物理执行/观测匹配。

## 4. 单例视频审核与下一步批量实验

单例三段视频已存在，先人工审核，不重复运行以下历史导出命令（会拒绝覆盖）。
历史命令仅编码已有帧，不重新运行 GPU 导航：

```bash
python -m scripts.export_video --episode-dir outputs/hssd_subgoal_001/control
python -m scripts.export_video --episode-dir outputs/hssd_subgoal_001/intervention
python -m scripts.export_video --episode-dir outputs/hssd_subgoal_001/ignore
```

核查中间 B 的证据图和 Follow 视频，确认完成动作不移动、B 后恢复 A、最后才 STOP。
预期生成各条件目录下的 `trajectory.mp4`；三段视频分别对应 44、84、44 组观测（含初始化帧）。
人工审核需记录审核者、日期、检查范围和异常，确认没有介入传送、没有中间 STOP，
Follow 顺序为 B→A，Control/Ignore 沿原路线到 A。视频抽查不等同于全部帧人工审核。
失败时保留所有文件，优先查看 `summary.json` 条件错误、`subgoal_events.json` 和两阶段终点图；
不要覆盖重跑、删除完成证据或放宽阈值。
多任务与多时机入口已实现并完成 30/90 episode 自动验收，下一步补齐视频人工审核记录与版本提交。
“只做 B 不恢复 A”等额外负例已在 CPU 替身测试中覆盖，不声称已执行额外 GPU 负例分支。

## 测试与提交

```bash
python -m unittest discover -s tests -v
```

新增测试覆盖挂起与恢复、约束/历史保留、一次投递、测量失败与不可见不推进真值、
三条件连续循环、B 完成即超时、缺失/篡改证据、中间 STOP 伪装完成，以及输入哈希预检。
单例初版在 Habitat 环境共 99 项测试通过；2026-11-03 补齐批量、负例与投影验收回归后，全项目 116 项测试全部通过，
核心环境因缺少原生依赖跳过 62 项。测试不创建真实 GPU Simulator。
`git diff --check` 和历史单例的独立只读重审通过；GPU 单例与批量均已验收，完整视频人工播放仍待记录。
此前禁区功能已提交；当前剩余插入恢复的导航改动、辅助模块、入口、测试及本文档可作为一个完整功能提交：
`feat: add measured subgoal insertion and resumption experiment`。
视频人工审核仍待完成；该状态应保留在提交说明中。批量入口与后续验收见 [批量说明](subgoal_dataset.md)。
`data/`、`outputs/` 不提交，不自动暂存、提交或推送。
