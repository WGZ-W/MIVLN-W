# Ignore Intervention 真实负对照

本实验复用已完成任务的介入配置和无介入控制轨迹，只运行一次忽略介入的导航。
机器人仍使用 Oracle 坐标导航，但接到替换目标事件后继续追求 A。
评价器独立更新任务真值为 B，并根据最终实际位置判定任务失败。

## 状态与验收逻辑

| 状态 | 事件前 | 事件后 |
|---|---|---|
| 评价器 `active_goal` | A | B |
| 策略 `policy_goal` | A | A |
| 被取消的目标 | 无 | A |

`EventEngine` 始终更新评价器的 `TaskState`；导航策略持有另一个独立的 `TaskState`。
默认 `follow` 模式将结构化更新同步给策略；`ignore` 模式保留原策略目标和规划器。
两种模式都完整记录事件和任务状态，成功条件始终使用评价器的最终有效目标。

忽略模式每步记录到有效目标 B 与策略目标 A 的距离，两者不能混用。
`response_latency_steps` 和 `first_compliant_action_step` 为 null，表示没有以新目标规划动作。

## 运行一个已验证任务的负例

```bash
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat
cd /mnt/sdc/weiguanzhao/MIVLN-W

python -m scripts.run_ignore_intervention \
  --config outputs/debug_dataset_001/tasks/task_001/timing/configs/middle.json \
  --reference-dir outputs/debug_dataset_001/tasks/task_001/timing/middle/control \
  --output-dir outputs/ignore_intervention_001 \
  --gpu-id 0 \
  --no-video

cat outputs/ignore_intervention_001/negative_check.json
```

使用分配给你的 GPU 编号。此命令复用已有控制结果，仅新增一个 GPU episode。
起点、朝向、动作参数、目标坐标和触发时机均来自保存的配置。
事件照常在指定动作前投递，不会重置机器人位姿或清空轨迹。

验收目标：

```json
{
  "negative_check_passed": true,
  "expected_dynamic_success": false,
  "observed_dynamic_success": false,
  "reference_checked": true
}
```

报告中的 `metrics` 应包含：

```json
{
  "intervention_received": true,
  "ignored_event_count": 1,
  "policy_goal": "point_a",
  "final_goal": "point_b",
  "stopped_goal": "point_a",
  "dynamic_success": false,
  "goal_update_compliance": false,
  "response_latency_steps": null,
  "termination": "stopped_at_cancelled_goal"
}
```

负对照脚本在上述预期失败得到正确验证时返回退出码 0。
规划错误、超时、未投递事件、错误成功标记、参考轨迹不匹配等情况均不能通过验收。
直接使用 `scripts.run_navigation --intervention-policy ignore` 时，退出码仍按任务成功与否返回；
这里使用专门入口来区分“任务按预期失败”和“负对照实验异常”。

## 输出与复核

输出包含原有的 RGB-D、轨迹、事件和几何指标，以及：

- `negative_check.json`：所有负对照检查与最终指标。
- `final_task_state.json`：评价器的最终真值，B 有效、A 已取消。
- `final_policy_state.json`：策略的最终状态，仍追求 A。
- `trajectory.json`：逐步记录 `active_goal`、`policy_goal` 和两个状态快照。
- `events.json`：记录 `ignored_by_policy` 和事件前后的两种状态。
- `replans.json`：忽略模式为空，表明策略没有为 B 重建规划器。

参考轨迹比较只使用动作、位姿、碰撞和单步距离等物理执行字段，
可以复用旧版输出，不要求其包含新增的策略状态字段。
整个物理执行轨迹应与无介入控制一致，而评价器真值在事件后有所不同。

导出视频检查机器人继续到 A：

```bash
python -m scripts.export_video --episode-dir outputs/ignore_intervention_001
```

已有输出不会覆盖；重跑时更换为 `outputs/ignore_intervention_002`。
单例通过后，可选择其他已保存配置做相同负对照；系统性比较时再扩展到全部 30 个配置。
