# HSSD 子目标插入与恢复：批量、多时机与阶段验收

更新日期：2026-11-03。单例协议见 [subgoal.md](subgoal.md)。本阶段继续限定为一次介入、一个临时 B、恢复原目标 A；
不包含嵌套插入、多次介入、禁区与插入混合或语言解析。当前策略仍为特权 Oracle。

## 已完成与待完成

- 新增 `scripts.subgoal_dataset` 的 `prepare / cpu-check / run / summarize / videos`。
- 复用 `hssd_dataset_facing_001` 的全部 10 个源任务，保留原起点、朝向、A/B 实例、停止点、种子与预算，不重新抽样或自动审核。
- 早期批次 `outputs/hssd_subgoal_batch_001/`：10/10 配置与 30/30 GPU episode 已独立验收。
- 三时机批次 `outputs/hssd_subgoal_timing_001/`：30/30 配置与 90/90 GPU episode 已在修正检查器后独立验收。
- 早期批次原生动作范围 73–205，三时机批次范围 73–243，均包含中间逻辑完成和最终 STOP，预算为 500。
- 两批共 120 段视频已导出，帧数与轨迹一致。30 张代表性 B/A 证据图已查看，完整人工视频播放仍待记录。
- 原 `task_004_middle` 的动作位移拒绝已由原生重放证明为检查器误判；原始失败和返回码保留。
  最新只读重审证据在 `outputs/hssd_subgoal_acceptance_2026-11-03/`，详见 [实际验收记录](hssd_subgoal_results_2026-11-03.md)。

CPU 检查仅使用 SceneGraph、无传感器 Agent 和固定 PathFinder，不创建 `habitat_sim.Simulator`，没有渲染或物理世界。
它检查事件前位姿前缀、动作预算、可导航位姿、B→A 几何与朝向；不能证明目标在图像中可见。

## 1. 已完成：10 任务、单时机、30 episode

准备、GPU 导航和视频均已完成，不重复 `prepare / run / videos`。
需要刷新顶层汇总时，激活 Habitat 环境运行 CPU 离线检查：

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# 以下为 CPU 离线重审，读取真实网格和原始证据，不创建模拟器。
python -m scripts.subgoal_dataset summarize \
  --output-dir outputs/hssd_subgoal_batch_001
```

三条件的 30 段视频已编码，不需重新导出。

预期文件：`manifest.json`、`cpu_navigation_check.json`、`run_status.json`、`summary.json`、`cases.csv`、
`run_logs/<case_id>.log` 和 `tasks/<task_id>/<timing>/<condition>/` 下的逐帧与中间/最终证据。
CSV 应保留 30 行数据；视频导出后应有 30 段，`video_status.json` 中 `failed_exports` 为空。

该批次在 `prepare` 后允许首次 `run` 写入，但只要出现运行状态、任务输出或运行日志，就拒绝覆盖、续跑和自动重试。
失败时保留目录和日志，先诊断，不更换失败任务或修改阈值。

## 2. 已完成：10 任务、三时机、90 episode

该目录的 GPU 运行与视频已完成。原始汇总受旧检查器误判影响仍为 29/30，
独立证据目录已保留旧副本与修正后的 30/30 结果；需要刷新原顶层汇总时只运行 CPU 命令：

```bash
python -m scripts.subgoal_dataset summarize \
  --output-dir outputs/hssd_subgoal_timing_001
```

三时机按同一无介入参考的**累计实际行走距离**约 25%/50%/75% 取动作前步号。
这不是动作数百分比，也不是原目标剩余 geodesic 的减少比例；与历史目标替换时机定义分别报告。
若短参考导致不同请求比例落在同一动作，后一个配置记准备失败，不静默重复或重新采样。

三时机批次应有 30 个任务/时机配置、90 行 CSV 和 90 段视频。
它包含与先行批次同任务同协议的 early 条件；两批结果分别汇总，不能拼成 40 个独立任务。
时机分析仍只有 10 个基础任务，统计不确定性须按基础任务分组。

## 3. 全阶段验收标准

| 项目 | 先行批次 | 三时机批次 |
| --- | --- | --- |
| 固定基础任务数 | 10 | 10 |
| 准备并通过的任务/时机配置 | 10/10 | 30/30 |
| 已生成并独立验收的 episode | 30/30 | 90/90 |
| Control：A STOP，联合成功 | 10/10 | 30/30 |
| Follow：B 完成→恢复 A→A STOP | 10/10 | 30/30 |
| Ignore：到 A、漏 B，联合失败且负对照通过 | 10/10 | 30/30 |
| 缺失 episode、输入错误与准备失败 | 0 | 0 |

两批 `summary.json` 都应 `passed: true`，对应 `passed_cases` 为 10、30，`expected_episodes` 为 30、90。
各条件 `validation_passed` 应达到对应配置数；`dynamic_success_count` 仅 Control/Follow 达到该数，Ignore 为 0。
`by_timing` 单独报告各时机。不能把所有验收通过解释成 Ignore 也完成了任务。

Follow 的 B 完成和最终 A 必须有实测距离 ≤ 0.3 m、水平角差 ≤ 7.5°、指定实例像素 ≥ 64 及完好的独立诊断证据。
中间 `COMPLETE_SUBGOAL` 计步、不移动；最终 STOP 必须发生在 A，事件和 B 完成不能重置位姿或历史。
Ignore 的 `root_goal_success` 应为 true，但其评价真值仍要求 B，`point_goal_success`、`sequence_success` 和 `dynamic_success` 均为 false。
Ignore 终止原因须为 `subgoal_sequence_incomplete`，不能靠超时、规划异常或缺少 STOP 充当负例。

测试中的额外负例必须继续被拒绝：只完成 B 随后停止、不返回 A；B 完成后立即耗尽预算；
B 不可见或诊断失败；中间 STOP 冒充完成；伪造距离、任务状态或掩码证据。
这些是 CPU 自动化负例，不能写成已经执行的额外 GPU 分支。

## 4. 独立核查与诚实统计

- 每个 GPU 子进程前先核查**整个批次**：源 manifest、全部源配置/参考、审核目录、场景/网格和各准备文件哈希。
- 仅接收原任务 × 已声明时机的完整有序列表；准备错误保留三条件槽位和原因，不换目标、不丢分母。
- 每个子进程运行真实单例三条件入口，失败后继续其他配置，保存返回码和日志。
- 汇总调用只读 `subgoal.check_experiment` 重算 B/A 距离、四元数朝向、原始实例掩码与证据哈希，重放真假任务状态。
- 核查同任务介入前实际观测相同、Ignore 与 Control 全程物理执行/观测相同、Control 与冻结参考相同。
- 子输出还须绑定本批次对应的准备文件与配置哈希；别的合法任务结果复制过来同样失败。
- 不读取子 `summary.json` 的成功标记，不改写子汇总；只有通过原始证据核验的数据计入成功分子及平均值。
- 超出历史动作位移上限时，必须由固定网格与原生 Agent 的相同动作重放证明终点一致；不以放宽阈值代替证据。
- 原子运行均完成、原始返回码为 1 且新原始证据验收和原生投影证明均通过时，可记 `recovered_audit_cases`；返回码和旧子汇总继续保留。
- 原始声称成功计入 `reported_*` 诊断字段；缺失、失败和准备拒绝仍进入计划分母。均值报告有效样本数，null 不替换成 0。
- 增加路程为同任务、同时机 Follow 减 Control 的实际路程，不是两阶段最优 SPL。

只读检查接口：

```python
from pathlib import Path
from scripts.subgoal_dataset import check_batch

report, cases = check_batch(Path("outputs/hssd_subgoal_batch_001"))
assert report["passed"], report["failed_cases"]
```

`summarize` 仅重写本批次的汇总和 CSV。`cpu-check` 仅重写 CPU 几何检查报告，不改变准备输入或导航结果。

## 5. 视频人工审核

每个时机至少覆盖五类根目标和不同 B，优先最长路线、单例岛台→沙发，以及所有异常案例。
若抽看而非逐条播放，应记录范围，不能宣称全批完整人工审核。

记录格式：审核者 / 日期 / task_id / timing / condition / 视频及证据图路径 / 通过或异常 / 原因。
关注介入无传送、完成 B 后继续、最后在 A STOP；Ignore 应保持原路线且明确漏 B。
将步号与 `events.json`、`subgoal_events.json`、`replans.json` 对照；仅凭视频不能代替独立实例诊断。

## 6. 新批次复现与测试

只有重做新批次时才使用新目录，避免覆盖本轮已准备的两个目录：

```bash
python -m scripts.subgoal_dataset prepare \
  --source-dir outputs/hssd_dataset_facing_001 \
  --timings early middle late --output-dir outputs/hssd_subgoal_timing_002

python -m scripts.subgoal_dataset cpu-check \
  --output-dir outputs/hssd_subgoal_timing_002

python -m unittest discover -s tests -v
```

默认 `prepare` 选择全部三个时机；显式 `--timings early` 得到先行单时机批次。
模块导入保持标准库兼容，CPU 资产/网格与替身集成测试在 Habitat 环境运行。
2026-11-03 在 Habitat 环境运行全项目 116 项测试全部通过；核心环境同样通过，因缺少原生依赖跳过 62 项。
其中子目标专项共 27 项，覆盖单例、负例、固定批次、三时机与原生投影验收；另在真实 HSSD 网格上检查了全部 10/30 个准备配置。
Python 3.9 语法检查、旧 GPU 单例的只读重审和 `git diff --check` 均通过。
本阶段的实现与 GPU 批量自动验收已完成；完整视频人工播放仍待记录，不能声称已完成全部人工审核。

实现、测试和文档可以提交：`feat: add fixed-task multi-timing subgoal evaluation`。
单例功能此前仍未提交时，可与本次批量扩展合为一个子目标功能提交；研究框架图与汇报文档另行提交。
`data/`、`outputs/` 保持忽略，不自动暂存、提交或推送。
