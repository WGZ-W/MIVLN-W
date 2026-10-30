# HSSD 新增避让区域：单例三条件实验

## 目的与当前状态

机器人继续前往原目标 A，在第 k 个动作之前收到一个新增地面禁区，从当前位置绕行。
入口是 `python -m scripts.avoid_region`，提供 CPU `prepare`、GPU `run` 和 CPU `summarize`。
这是一条独立的实验协议 `add_avoid_region_v1`；原 v0.1 目标替换结果和输出保持不变。

当前已完成代码、89 项自动测试和真实 HSSD 网格上的无渲染原生动作检查。
单例 `outputs/hssd_avoid_001/` 三分支 GPU 验收已通过：Control/Follow 分别行走约 7.00/8.25 m，
Ignore 行走约 7.00 m、发生 5 个违规动作，三者都到达原目标且目标可见，碰撞标记均为 0。
三个单例视频已导出；后续 [10 任务批量](avoid_dataset.md) 的 30 个 episode 也已完成 GPU 验收，
详见 [批量验收与下一步](hssd_avoid_results_2026-10-29.md)。
仅支持单场景、单次新增矩形禁区，没有多区域、撤销约束或子任务恢复。

## 区域和事件语义

结构化更新格式：

```json
{
  "operation": "add_avoid_region",
  "region": {
    "region_id": "avoid_001",
    "type": "rectangle_xz",
    "bounds_xz": [[-7.637286, -0.664476], [-7.137286, -0.164476]],
    "floor_y": 0.153721,
    "floor_tolerance_m": 0.5
  }
}
```

坐标为世界坐标、单位米；`bounds_xz` 是 `[[min_x, min_z], [max_x, max_z]]`。
Y 用于区分楼层，以机器人脚点为准，范围是 `floor_y ± floor_tolerance_m`。
评价时矩形在 XZ 两方向各外扩 `agent_radius`，形成保守方形足迹；不是精确圆形 Minkowski 和。
区域是闭集，接触外扩边界也算违规；几何数值容差为 1e-6 m。

事件 k 恰好投递一次，在动作 k 前同时更新评价真值和 Follow 策略状态。
目标、历史与位姿保持不变，Ignore 只更新真值。事件前的动作不追溯处罚。
每条已执行动作的前后位姿构成一条线段，检查其与外扩区域是否相交；
两个端点都在外面仍可能违规。原地转向和 STOP 也检查所在位置。
首轮准备拒绝触发位姿或目标过近的区域，不处理“生效时已经在禁区内”的逃离策略。

禁区是逻辑约束，不修改原始 HSSD 网格或添加物理障碍；当前 RGB 中不会绘制标记。
“marked region”的几何由结构化输入直接提供给特权 Oracle，尚无自然语言指代解析或视觉定位。

## 规划方式与边界

`scripts/avoid_planner.py` 在禁区周围确定性采样三圈角点和侧面中点，吸附到导航网格。
用 Habitat 的 CPU geodesic 查询连接候选节点，排除穿过禁区及规划缓冲带的边，
用 Dijkstra 找到该有限图中的最短路径，然后用原有 Greedy follower 依次跟随途经点。
中间途经点不会成为任务子目标，也不会发出 STOP；原动作空间和总步数预算不变。
默认规划净空 0.35 m，途经点容差 0.2 m。

每次前进前还用 `PathFinder.try_step` 预测运动过滤后的线段；不安全时记录规划失败，
不会退回穿越禁区的无约束路线。独立评价器再次检查实际执行的线段。
有限图可能漏掉真实可行路线，失败只表示本规划器未找到路线，不证明场景全局不可绕行。
规划路线长度也不是连续空间最优值，禁区实验的 `post_intervention_spl` 因此设为 null；
旧的 `post_intervention_shortest_path_length_m` 仍是无约束 geodesic，不应当作受约束最优距离。

## 1. 准备固定单例：CPU

从已验收的无事件 reference 继承起点、朝向、目标实例、种子、传感器和动作预算，
在参考行程约 25% 处确定触发步，在后续路线中选择区域。记录被拒候选和原因，
要求原路线在事件后违规、触发位姿在区域外、目标保留安全空间，并找到合法绕行路线。
源配置、轨迹、指标及场景/网格均用哈希核查。

```bash
python -m scripts.avoid_region prepare \
  --reference-dir outputs/hssd_dataset_facing_001/tasks/task_001/reference \
  --output-dir outputs/hssd_avoid_prepare_002
```

**本轮已经执行上述准备命令，不要重复创建同一目录。**
`outputs/hssd_avoid_prepare_002/config.json` 是本轮使用的最终准备配置；
`_001` 是开发期间的 CPU 检查记录，使用 `_002` 的准备路线与当前规划器一致。
目标为 `target_051` 厨房岛台，第 14 步生效；原路线第 15 步首次违规，共 5 个违规动作。
新有限图参考路线长约 6.354 m，准备报告保存在 `preparation.json`。
真实网格＋原生动作控制的 CPU 检查完成介入后 47 个导航动作，最终点误差约 0.173 m，
全过程没有创建 `habitat_sim.Simulator` 或进行 GPU 渲染；这不是完整 GPU episode 验收。

## 2. 执行三条件：GPU，由使用者运行

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python -m scripts.avoid_region run \
  --config outputs/hssd_avoid_prepare_002/config.json \
  --gpu-id 0 --output-dir outputs/hssd_avoid_001

# 离线复核：不创建模拟器，会更新本批次 summary.json。
python -m scripts.avoid_region summarize \
  --output-dir outputs/hssd_avoid_001
```

使用分配的 GPU 和新的运行目录；入口默认只保存逐帧数据，不自动编码视频。
依次运行 `control`、`intervention`（Follow）、`ignore`，失败分支保留输出，另外的分支继续执行，
汇总分母固定为 3。不支持续跑、覆盖旧目录或自动重试。

预期 `summary.json` 的 `passed: true`，`expected_episodes: 3`，三个条件的验收 `passed` 均为 true：

| 条件 | 点目标到达＋STOP | 约束违规动作数 | dynamic_success |
| --- | --- | --- | --- |
| Control | true | 0（无新增约束） | true |
| Follow | true | 0 | true |
| Ignore | true | 大于 0 | false |

三者目标始终为 A。所有终点仍须满足朝向和指定实例可见性；
因此 Ignore 的 `terminal_semantic_success` 可以为 true，但 `constrained_semantic_success` 必须为 false。
`goal_update_compliance` 仍只表示最终目标匹配，不能用它代替 `constraint_compliance`。
Ignore 的响应延迟为 null，终止原因应为 `constraint_violation`，不能以超时或规划失败代替负例。

独立汇总重算事件前共享轨迹和观测、Ignore 与 Control 完整物理执行及观测、
任务/策略约束时间线、线段违规、终点实例证据及配置哈希；Control 也须复现准备时的原 reference。
每分支增加 `constraint_violations.json`；Follow 的 `replans.json` 保存途经点、合法路线和方法名。
若 CLI 返回非零，保留 `run_status.json`、`summary.json`、分支指标和轨迹，不降低约束追求通过。

## 3. 导出和后续验收

三条件通过后，用现有 CPU 视频入口：

```bash
python -m scripts.export_video --episode-dir outputs/hssd_avoid_001/control
python -m scripts.export_video --episode-dir outputs/hssd_avoid_001/intervention
python -m scripts.export_video --episode-dir outputs/hssd_avoid_001/ignore
```

视频不会画出禁区，人工核查须结合准备配置的坐标、`replans.json` 和违规步号。
确认 Follow 实际绕行、无传送，Ignore 保持原路线且在到达 A 后仍判失败。
单例通过后已新增 [多个固定任务的批量入口](avoid_dataset.md)，先运行同一时机的 10 个任务、30 个 episode，
再扩展到至少 20 组和多时机比较。当前尚无多时机禁区批量入口，不直接复用目标替换批次运行器。

## 测试与提交

```bash
python -m unittest discover -s tests -v
```

新增测试覆盖边界与楼层、半径外扩、端点在外但线段穿越、动作前一次投递、
到达但违规的失败、缺少约束测量时拒绝评价、三条件状态隔离、失败分母、
无合法图路线时禁止回退、离散动作安全检查和输入哈希篡改。
Habitat 环境 89 项全部通过，核心环境 89 项中 44 项因缺少原生依赖跳过。
可以将核心、入口、测试和文档作为一个功能提交：
`feat: add matched avoid-region intervention experiment`；注明单例和批量 GPU 都已通过。
输出和数据继续不入 Git，本步骤不自动暂存、提交或推送。
