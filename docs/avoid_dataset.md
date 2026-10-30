# HSSD 禁区批量实验：10 个固定任务、30 个 episode

## 目的与已完成检查

复用 `outputs/hssd_dataset_facing_001/` 的全部 10 个任务和无事件参考轨迹，
每任务新增一个有效禁区，运行 Control、Follow、Ignore 三种条件。
研究不同起点、目标和区域下的任务生成与评价可靠性；这是单场景特权 Oracle 实验。
本轮固定使用参考实际行程约 25% 的介入时机，尚无早/中/晚批量比较。

脚本：`scripts/avoid_dataset.py`。当前状态：

- CPU 已准备 10/10 个任务，计划 30 个 episode；没有替换或重新采样原始任务。
- 最终批次目录为 `outputs/hssd_avoid_batch_001/`，不要再次执行同目录准备。
- 真实 HSSD 网格与原生动作的无渲染 CPU 检查通过 10/10，记录在 `cpu_native_checks.json`。
  该检查只覆盖介入后的导航，不覆盖 GPU 观测、终点朝向或实例可见性。
- 89 项测试在 Habitat 环境全部通过；核心环境因缺少原生依赖跳过 44 项。
- GPU 批量已完成并逐条验收：10/10 个任务、30/30 个 episode，20 条正常条件成功、10 条 Ignore 按预期失败。
- 30 段视频已导出且编码帧数符合轨迹；碰撞标记与规划错误为 0，平均配对增加路程约 0.755 m。
  详细核查和下一步见 [本轮验收记录](hssd_avoid_results_2026-10-29.md)。

## 1. CPU 准备：已执行

```bash
python -m scripts.avoid_dataset prepare \
  --source-dir outputs/hssd_dataset_facing_001 \
  --output-dir outputs/hssd_avoid_batch_001
```

从源 manifest 读取完整任务列表，先核查所有源配置、审核目录、参考轨迹和终点实例证据，
再逐任务调用现有单例准备器。保留起点、朝向、语言、目标实例、导航设置和步数预算，
为每个任务分配独立 episode ID。
要求事件后原路线违规、触发位姿在禁区外、目标周围保留空间，并找到有限图中的合法绕行路线。
区域、单位、足迹外扩及边界语义沿用 [单例禁区协议](avoid_region.md)。

`manifest.json` 固定源 manifest 哈希、完整任务顺序、参考文件哈希、准备配置/报告哈希及失败原因。
`preparations/<task_id>/` 保存单例配置和候选筛选证据；`preparation_logs/` 保存每任务准备日志。
准备失败时记录原因和已拒绝候选，不删除任务、不找替代起点；退出码为 1，但仍可运行已准备的任务。
分母始终是源任务数乘 3，准备失败的任务也有三条计划记录。

## 2. GPU 运行：本轮已完成，以下为命令记录

当前目录已经运行，不要重复启动。复现时用新的准备与运行目录，保留本批次结果。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python -m scripts.avoid_dataset run \
  --output-dir outputs/hssd_avoid_batch_001 --gpu-id 0

# CPU 离线复核；重新生成本批次 summary.json 与 cases.csv。
python -m scripts.avoid_dataset summarize \
  --output-dir outputs/hssd_avoid_batch_001
```

使用分配的 GPU。每任务启动一个独立子进程，内部依次运行三个条件；一次只运行一个子进程。
启动首个 GPU 子进程前检查所有任务的配置和源文件。
禁止覆盖、续跑或自动重试；已出现 `run_status.json`、`tasks/` 或 `run_logs/` 时拒绝重新启动。
子进程非零退出不阻止后续任务，运行失败与部分输出完整保留，不扩大成功率分母之外的任务集合。
`run_logs/<task_id>.log` 保存子进程日志，`run_status.json` 保存每任务返回码。

本批次不再生成新的参考 episode；沿用此前已冻结的 10 条参考，所以计划新增运行数是 30。
各条件产物位于 `tasks/<task_id>/{control,intervention,ignore}/`。

## 3. 验收与统计

本轮 `summary.json` 的实测结果，也作为后续复现的验收标准：

```text
passed: true
prepared_tasks: 10
passed_tasks: 10
expected_episodes: 30
observed_episode_metrics: 30
missing_episode_metrics: 0
preparation_failed_tasks: []
input_errors: []
```

| 条件 | 点目标到达 | 事件后违规 | dynamic_success | 验收通过数 |
| --- | --- | --- | --- | --- |
| Control | 10/10 | 无新增约束 | 10/10 | 10/10 |
| Follow | 10/10 | 0 | 10/10 | 10/10 |
| Ignore | 10/10 | 每条至少一次 | 0/10 | 10/10 |

所有条件保持原目标、合法 STOP、动作预算和终点实例可见性。
Ignore 到达原目标但联合任务失败是正常负例，不允许超时或规划失败替代此结果。

汇总调用单例的只读 `check_experiment` 重新核查原始配置、事件、位姿、任务/策略状态、
连续线段违规和实例证据，以及 Control 对原参考、Follow 的事件前前缀、Ignore 对 Control 的实际观测匹配。
不信任已有子任务 `summary.json`，也不覆盖这些子任务汇总。

`cases.csv` 固定有 30 行数据，包括缺失和准备失败的条件。
每种条件同时报告原始指标宣称的 `reported_*_successes` 和通过独立验证的成功数；
`dynamic_success_rate` 使用验证后的成功数除以全部计划任务数。
路径均值仅使用有效验收数据，并报告样本数。
`paired_overheads` 是完整验收任务的 Follow 总路程减 Control 总路程，另报配对样本数和均值；
它衡量增加的实走路程，不是受约束最优 SPL。

## 4. 视频与失败处理

验收完成后导出 30 段视频：

```bash
python -m scripts.avoid_dataset videos \
  --output-dir outputs/hssd_avoid_batch_001
```

已有非空视频会跳过；缺失分支和导出失败记录到 `video_status.json`，不会因为任务准备失败而忽略。
当前视频不绘制禁区，应结合区域坐标、`replans.json` 和 `constraint_violations.json` 复核。
发现失败后保留整个批次，先按任务查看 `run_logs`、汇总错误与轨迹；不要覆盖重跑或降低约束阈值。

本批次没有验收失败，下一项实现优先补齐子任务插入与恢复，见 [下一步目标](hssd_avoid_results_2026-10-29.md)。
禁区至少 20 组和早/中/晚比较保留为扩展计划；多时机批量入口尚未实现，不将重复同一配置当作新时机实验。

## Git 与验证

```bash
python -m unittest discover -s tests -v
```

新增测试包含完整批量替身运行、准备失败保留分母、子进程失败、缓存汇总失效、原始轨迹篡改，
以及在首个子进程启动前拒绝源文件/后续任务配置变化和拒绝覆盖旧批次。
CPU 测试中的 Simulator 均为替身，未执行真实 GPU 实验。
代码、测试和文档可以提交；单例和批量 GPU 都已通过。`outputs/` 和 `data/` 不提交。
