# HSSD 终点朝向与独立实例可见性诊断

## 基线验收与本次目的

`outputs/hssd_dataset_001/` 已完成 10 个固定任务、30 组 control/follow 配对、30 个 Ignore 负对照，
共 100 个 episode、6147 组观测；停止点协议全部通过，碰撞标记为 0。
其中 70 条正常条件成功，30 条 Ignore 按预期失败；不是 100 次导航任务都成功。

但以下三条 Follow 的最终视野没有沙发：`task_001` early/middle、`task_004` late。
点目标误差仍满足 0.3 m，问题是最终朝向。旧 `success` 定义不追溯修改，旧文件不覆盖。

本次增加可选的终点转向策略和独立诊断，以同一批固定任务进行新实验。
代码和 CPU 测试已完成；新策略的实际 GPU 效果尚待运行，不预先宣称解决了全部问题。

## 控制与诊断严格分离

- 导航器首次建议 STOP 时，Oracle 根据策略当前目标物体的世界坐标中心计算水平角差。
- 若偏差超过 7.5°，执行原有 `turn_left`/`turn_right`（15°），直至满足容差再 STOP。
- 不调用位姿设置、不传送、不改变相机俯仰，保持旧动作空间；转向计入原有 500 步预算。
- 每个策略目标最多允许 24 次终点转向；不足以对齐时失败，不伪造 STOP。
- 事件仍在动作前生效，即使转向途中收到介入也先更新任务；Ignore 始终朝向旧目标 A，而不是泄漏真值 B。

该控制器使用特权物体中心，不代表语言/视觉识别能力。仅做水平对齐不能保证解决遮挡、低矮物体或不合理停止点；
这些问题由独立诊断报告，不通过临时俯视相机或修改目标来隐藏。

## 为什么不是直接数类别像素

HSSD 中多个实例可能共用类别 ID，例如床 ID 27。诊断在轨迹全部保存后：

1. 从 source index 和唯一模板绑定具体 runtime object。
2. 保存当前位姿、原始类别标签及观测副本。
3. 只给该实例临时设置一个未使用的独立语义标签，重新渲染其可见掩码。
4. 在 `finally` 恢复原始标签，并验证类别渲染和位姿均已恢复。

没有动作或物理推进，诊断观测不提供给策略，原有 RGB-D/类别语义轨迹不改写。
默认“可见”阈值为至少 64 个实例像素，固定在本实验配置中；不等同于整件物体完全入镜或模型识别正确。

## 1. 派生同一批任务：CPU

```bash
python -m scripts.hssd_dataset derive-facing \
  --source-dir outputs/hssd_dataset_001 \
  --output-dir outputs/hssd_dataset_facing_001
```

本轮已执行此命令，不要重复覆盖现有派生目录。
10 个任务的起点、朝向、A/B 停止点、实例、语言、种子及动作预算保持不变，新增终点设置并重新固定配置哈希。
新 manifest 保存父数据集路径、manifest 哈希及各父配置哈希；不会重新随机采样。
`smoke_config.json` 保留首任务实际验证的 early 触发点；完整批次则会重新执行参考并推导时机。

## 2. 先做两分支 smoke test：GPU，由使用者执行

在仓库根目录、`habitat` 环境中执行；使用分配的 GPU 和新的输出目录：

```bash
python -m scripts.run_intervention \
  --config outputs/hssd_dataset_facing_001/smoke_config.json \
  --gpu-id 0 --output-dir outputs/hssd_facing_smoke_001 --no-video
```

`pair_summary.json` 的 `passed` 仍然检查点导航、任务更新及共享前缀，不能独自作为终点可见性的验收。
追加离线检查：

```bash
python - <<'PY'
from pathlib import Path
from scripts.intervention_timing import load_json
from scripts.hssd_dataset import terminal_record

config = load_json(Path('outputs/hssd_dataset_facing_001/smoke_config.json'))
for condition in ('control', 'intervention'):
    folder = Path('outputs/hssd_facing_smoke_001') / condition
    metrics = load_json(folder / 'metrics.json')
    trajectory = load_json(folder / 'trajectory.json')
    result = terminal_record(config, metrics, trajectory, folder, condition)
    print(condition, 'point:', metrics['point_goal_success'],
          'semantic:', metrics['terminal_semantic_success'],
          'turns:', metrics['terminal_alignment_turns'], 'diagnostic:', result)
    assert result['passed'], result
PY
```

预期两分支 `point_goal_success` 与 `terminal_semantic_success` 均为 true，
`terminal_alignment_complete: true`、绝对 yaw error ≤ 7.5°，实例像素数 ≥ 64。
检查新 `terminal_rgb.png` 和 `terminal_instance_mask.png`，确认控制到原物体、Follow 到沙发且朝向合理。
若诊断失败，保留文件和错误，不把未知当作“不可见”，也不降低像素阈值追求通过。

可继续验证相同配置的 Ignore：

```bash
python -m scripts.run_ignore_intervention \
  --config outputs/hssd_dataset_facing_001/smoke_config.json \
  --reference-dir outputs/hssd_facing_smoke_001/control \
  --gpu-id 0 --output-dir outputs/hssd_facing_ignore_smoke_001 --no-video
```

预期负对照验收通过、point/semantic success 都为 false；策略对齐旧物体 A，评价器测量新目标 B。

## 3. Smoke 通过后再执行完整新批次

```bash
python -m scripts.hssd_dataset run --output-dir outputs/hssd_dataset_facing_001 --gpu-id 0
python -m scripts.hssd_dataset summarize --output-dir outputs/hssd_dataset_facing_001
```

预期原来的 10/30/30 点协议验收继续通过，同时：

- `terminal_protocol_passed: true`；`terminal_checks.expected/evaluated/passed_episodes` 均为 100。
- 70 条 reference/control/follow 的 `terminal_semantic_success` 为 true。
- 30 条 Ignore 的该指标为 false，负对照仍符合预期。
- `terminal_checks.missing` 为 0，`failed_episodes` 为空。

新增证据是 `terminal_diagnostics.json`、原始实例掩码 NPY、掩码 PNG 和终点 RGB；
日志保存模板/源索引、运行时 ID、像素数、标签恢复检查、位姿和证据哈希。
`cases.csv` 与按时机统计增加终点指标。若点协议通过但终点协议失败，两个结果分别保留，CLI 返回非零提醒检查。

## 指标兼容性与后续

`success`/`dynamic_success` 和新别名 `point_goal_success` 仍表示旧的 STOP＋点距离协议。
`terminal_semantic_success = point_goal_success AND 指定实例可见`，单独报告，不取代旧指标。
诊断失败时该值为 null，而不是 false；完整新协议不能因此通过。
新配置的对照与参考必须同时开启相同设置，不能混用旧参考文件做逐行前缀匹配。

本批基线仍保存在 `outputs/hssd_dataset_001/`；不要覆盖或修改其结果。
代码提交应区分 CPU/模拟器替身验证与实际 GPU 效果；原始资产和运行输出继续不入 Git。
