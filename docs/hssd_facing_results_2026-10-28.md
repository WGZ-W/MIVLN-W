# HSSD 终点朝向实验验收与下一步

检查日期：2026-10-28。实验目录：`outputs/hssd_dataset_facing_001/`。

## 1. 验收结论

本轮 10 个固定任务、100 个 episode 的点协议和终点实例可见性协议均通过。
这是单场景、五类已审核实例的 Oracle 实验，尚不代表语言理解或跨场景泛化。

| 项目 | 实际结果 |
| --- | --- |
| episode 构成 | 10 reference + 30 control + 30 follow + 30 Ignore |
| 任务 / 配对 / 负对照验收 | 10/10、30/30、30/30 |
| 独立终点诊断 | 100/100 有效，缺失与失败均为 0 |
| 点成功且指定实例可见 | 70/70 正常条件；30 条 Ignore 均按预期失败 |
| 原始观测 | 6448 组 RGB、depth、semantic |
| 正常条件最大停止点误差 | 0.296612 m，阈值 0.3 m |
| 正常条件最少实例像素 | 4787，阈值 64 |
| 全部 episode 最大水平朝向误差 | 7.404526°，阈值 7.5° |
| 碰撞标记 / 规划错误 | 0 / 0 |
| 子进程阶段 | 80/80 返回码为 0 |

除读取汇总外，本次逐条重新检查了冻结配置及场景/网格哈希、实例证据文件哈希、
事件动作前投递、任务真值与策略状态、配对共享前缀及实际观测内容、Ignore 与 control 的物理执行匹配。
另外，直接读取 NPY/PNG 重新计算掩码像素，核对终点 RGB 与最后一帧一致，
使用 CPU PathFinder 重算全部终点距离并检查全部轨迹点可导航，检查 depth/semantic 数组形状与有限值，
从最终四元数独立计算策略目标朝向。没有启动 GPU 模拟器或改写原批次结果。
当前 73 项自动测试也全部通过。

### 原有三个终帧问题

| Follow 案例 | 新实例像素 | 新朝向误差 | 终点转向动作 |
| --- | --- | --- | --- |
| task_001 early | 7482 | 3.089448° | 6 |
| task_001 middle | 7415 | 0.246882° | 6 |
| task_004 late | 7853 | 3.347580° | 5 |

三张终点 RGB 已查看，均能看到沙发。可见性指标表示指定实例达到像素阈值，
不表示物体完整入镜或模型识别正确。终点控制仍使用特权实例中心。
完整视频人工审核尚未完成；检查时该批次有 0 段 MP4。

## 2. 立即执行：导出视频并做人工复核

无需重新运行 100 个 episode，也无需再次派生任务。在仓库根目录执行：

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# 先导出包含原先三个问题的两个任务，共 14 段视频。
# CPU 编码；FFmpeg 已存在于 habitat 环境中，不需要启动 GPU。
python -m scripts.hssd_dataset videos \
  --output-dir outputs/hssd_dataset_facing_001 --task-id task_001
python -m scripts.hssd_dataset videos \
  --output-dir outputs/hssd_dataset_facing_001 --task-id task_004

# 再补齐其余任务；已有视频会跳过，最终共 70 段。
python -m scripts.hssd_dataset videos \
  --output-dir outputs/hssd_dataset_facing_001

# 检查最终导出状态；这里只读取结果，不重写 summary。
python - <<'PY'
import json
from pathlib import Path

root = Path('outputs/hssd_dataset_facing_001')
summary = json.loads((root / 'summary.json').read_text())
status = json.loads((root / 'video_status.json').read_text())
videos = list(root.glob('tasks/**/trajectory.mp4'))
assert summary['terminal_protocol_passed'] is True
assert summary['terminal_checks']['passed_episodes'] == 100
assert not status['failed_exports'], status
assert len(videos) == 70, len(videos)
assert all(video.stat().st_size > 0 for video in videos)
print('100 episodes accepted; 70 nonempty videos exported.')
PY
```

预期产物是各 episode 内的 `trajectory.mp4` 以及根目录 `video_status.json`。
每任务导出一条 reference、三条 follow、三条 Ignore；相同控制轨迹不重复导出。
重点先看以下 Follow 视频：

- `tasks/task_001/timing/early/intervention/trajectory.mp4`
- `tasks/task_001/timing/middle/intervention/trajectory.mp4`
- `tasks/task_004/timing/late/intervention/trajectory.mp4`

人工验收检查介入前后无传送、转向自然、最终沙发可见、STOP 前没有异常移动；
再从其余四类目标各抽一例，并核对相同时机的 Ignore 确实停在旧目标。
将审核者、案例路径、通过/异常及简短原因写入新的审核记录；视频生成完成不等于人工审核完成。
发现异常时保留日志和输出，不降低阈值、不删除失败样例、不重采样替代。

## 3. 下一阶段目标：P3 新增避让区域

完成当前证据整理后，下一项实现目标是**单次介入增加禁区，同时保留原目标**。
先完成一个有合法绕行路线的 HSSD 单例，以及 Ignore 违反约束的负例，再考虑批量扩展。
不要直接重跑目标替换批次作为新任务类型实验。

以下功能尚未实现，当前没有可执行的禁区实验入口：

1. 定义区域坐标、单位、边界、事件生效时刻，以及区域内触发如何处理；
   建议从世界 XZ 平面的矩形和已确认的同楼层路线开始。
2. 新增约束更新语义：事件前动作不追溯处罚，事件后目标保持 A；真值更新，Ignore 策略保持原约束。
3. 实现受约束规划与连续轨迹评价：检查每条动作线段是否穿越区域，不能只检查离散端点。
   原 Habitat 最短路接口不会自动遵守新逻辑禁区，需要新的规划处理。
4. CPU 测试覆盖恰好一次投递、无位姿重置、边界穿越、不可绕行情形与 STOP 距离要求。
5. 通过后再提供 GPU smoke 配置和命令：同起点/朝向/目标，control 不加约束，
   follow 绕行且完成目标，Ignore 保持旧策略并因违反已生效约束被判失败。

采样前必须确认事件位姿在区域外、原后续路线会穿越区域、存在合法替代路线。
先通过一个正例和一个负例，再扩展到至少 20 条自动仿真样例；不得用修改旧批次替代新实验。

## 4. 提交建议

终点控制代码已在 `5625fce` 提交，本步只更新实验结论与路线文档，可以单独提交：
`docs: record HSSD terminal-facing acceptance and next steps`。
数据、轨迹、图片和视频继续留在 Git 忽略目录中。未执行暂存、提交或推送。
