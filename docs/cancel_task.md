# 取消任务并停止：cancel_task_v1

日期：2026-11-13。此类型加入目标替换、子目标插入与恢复，成为当前第三类主任务；禁区仍后置。
已交付独立状态/评分扩展、V 消息投递、CPU 三条件样例、只读审计与 HSSD 参考驱动的准备入口。
真实 Habitat 取消任务运行器和真实模型验收尚待实现/完成；[语言标注审核工具](language_annotations.md)已实现，人工审核尚未完成。

## 1. 语义

初始指令要求前往 A，途中收到“取消任务，现在停下”。评价器私有更新为：

```json
{"operation": "cancel_task"}
```

取消发生在动作 k 选择之前，必须恰好投递一次。所有尚未完成的导航目标被取消，当前目标为 null、状态为 cancelled。
历史、已完成子目标及预存约束保留，取消不会自动移动机器人或产生 STOP，也不表示机器人已停下。
首版采集限定单个初始目标、一次取消，不混合禁区或插入事件；状态类对已有挂起义务的取消支持不等于发布了多轮任务。

模型仍读取初始语言、当前/历史 RGB、实际动作和已投递语言，并自行输出动作。
正确取消 opcode、null 目标及评价状态不发送给模型。原目标处停下不能代替服从取消。
这与目标替换不同：没有新的物体目的地，也没有恢复原目标的义务；未来恢复/重启需要新协议。

## 2. 首版成功条件

**动作 k 必须是 STOP。** 因而 `stop_delay_actions = stop_step − k = 0`，取消后的计步动作只有这一个 STOP。
STOP 计入原预算，不推进物理，实际位姿必须保持。位置容差为 1e−5 m，四元数误差容差为 1e−6，允许 q/−q 表示同一朝向。
这些是数值容差，不是允许继续行走的安全距离。取消后累计实走路程不得超过位置容差。

不要求到达 A、不测物体朝向或实例可见性；取消任务的 STOP 用投递、时序和位姿证据评分。
目标到达型任务原有“STOP＋目标距离/朝向/实例”条件继续保留，不能把本类型的例外套用到它们。

| 情况 | 结果 |
| --- | --- |
| 正确投递后，立即在当前位姿 STOP | 成功，即使远离 A |
| 先移动、原地转向或发 COMPLETE_SUBGOAL，再 STOP | 失败 |
| 忽略取消，继续到 A 再 STOP | 失败 |
| 介入前提前 STOP，没收到取消 | 失败，保留计划槽位 |
| 收到取消但不 STOP、超出预算、调用/动作失败 | 失败 |
| STOP 时传送/转动、事件处重置位姿、重复/错误投递、日志不完整 | 证据无效且不授予成功，另列原因 |

## 3. 评分与兼容实现

`mivln/cancel_task.py` 提供：

- `CancellationTaskState`：继承旧状态，显式支持取消和 null 当前目标；旧 `TaskState` 仍拒绝未知取消操作。
- `CancellationVisualSession`：复用 V RGB/动作 schema，仅扩展私有投递检查；不将取消强制变为模型的 STOP。
- `evaluate_cancellation`：从原配置、实际位姿/动作及投递日志独立重放。既有指标缓存和模型意图不作为成功证据。

旧状态机、旧评价器、旧 V 类和旧 U 更新预测契约保持原样，以保留冻结代码哈希。
旧 `run_navigation`、旧 U 预测入口不能直接接收本类型。新取消状态不要传给旧 U `previous_state` 验证器。
V 的动作仍是已有 STOP，不增加动作标签或给模型正确任务类型。

主要新增字段：

| 字段 | 定义 |
| --- | --- |
| `cancellation_success` | 投递/预算/立即 STOP/位姿及证据均通过 |
| `evidence_valid` | 配置、连续轨迹、事件、距离及 STOP 不移动等证据可核验 |
| `stop_delay_actions` | STOP 动作下标减投递步；立即为 0，未收到或没有 STOP 为 null |
| `post_cancel_action_count` | 从动作 k 到执行结束的动作数，成功时为 1 |
| `post_cancel_path_length_m` | 从投递前位姿起累计实际位移；出发又返回仍计入 |
| `post_cancel_net_displacement_m` | 投递位姿到最终位姿的距离，不能代替累计路程 |
| `failure_reason` / `errors` | 区分未投递、未立即停、无 STOP、超预算及证据问题 |

轨迹初始行为 INITIALIZE，动作 k 的结果位于第 k+1 行；事件 before/after 位姿与第 k 行一致。
停止延迟以动作计，0 不代表零秒模型推理；真实模型接入时应另记回调秒数，本版不设置壁钟截止时间。
`inspect_trajectory` 检查实测位移、连续步号及非运动动作，无 native mesh/物理动作重放声明。
本类型成功率分母为其全部计划运行，缺失和失败不删除。现有 `dynamic_success` 不自动扩展为取消成功。

## 4. 可执行命令

普通 Python 环境即可；输出目录必须全新。下面编号供重复执行，已完成的首轮产物见第 5 节。

```bash
# 从已验收参考准备，不执行新导航，不覆盖旧配置。
python -B -m scripts.cancel_task prepare \
  --source-dir outputs/hssd_dataset_facing_001 \
  --output-dir outputs/hssd_cancel_prepare_002

# 合成一维走廊、脚本策略、V 实际 PNG 输入；不是 HSSD/模型表现。
python -B -m scripts.cancel_task demo --output-dir outputs/cancel_task_cpu_003

# 新目录存重审，原样例文件不回写。
python -B -m scripts.cancel_task audit \
  --input-dir outputs/cancel_task_cpu_003 \
  --output-dir outputs/cancel_task_audit_003

python -B -m unittest tests.test_cancel_task -v
```

prepare 为每个基础任务保留 early/middle/late 三个计划槽位，按参考累计实际行程 25%/50%/75% 推导动作前触发步。
继承 A、起点、朝向、种子和预算，移除原替换事件及配对目标 B；绑定来源配置/参考证据，拒绝输入改变。
单个参考不合格则保留其三个未准备槽位与原因，不重采样替换。
新的停止消息是模板，`language_review_status=not_recorded`，不能当作已完成人工语言审核。
配置均为评价器私有数据。不得把 prepare 目录或完整原配置传给视觉模型。

demo 固定 Control/Follow/Ignore 三条件：Control 无事件，Follow 根据消息立即停，Ignore 完整复现 Control 的物理轨迹和 RGB。
Control 仅在合成无障碍走廊用欧氏距离作为等价 geodesic 判点，不能把其检查器当成真实 HSSD 物体终点评价器。
只读重审检查原始响应、实际公共消息/RGB、执行动作、位姿及真值重放，不信任 summary/metrics 缓存。
失败的样例仍保留轨迹、输入、原始响应和三条件计划分母。

## 5. 当前验收

- `outputs/hssd_cancel_prepare_001/`：10 基础任务、30/30 配置准备完成，90 个计划 episode，**尚未执行导航**；62 份绑定来源 JSON 哈希前后一致。
- `outputs/cancel_task_cpu_002/`、`outputs/cancel_task_audit_002/`：CPU 三条件和只读重审通过，重审源文件前后不变。
- Follow：STOP 延迟 0、取消后路程 0，在距原目标 0.5 m 处取消成功；Ignore：到达 A，但延迟 2 动作、多走 0.5 m，取消失败。
- 17 项新回归通过；全量 340 项在 Habitat 环境全部通过，核心环境按原生依赖跳过 74 项。

`passed=true` 表示 Control/Follow 成功、Ignore 按预期失败以及证据/配对通过；不是三条取消任务全成功。
没有运行真实模型或 GPU。真实场景/视觉模型效果仍待验收。

## 6. 下一完整步骤

新增第三类已纳入[语言标注、审核与版本化导出](language_annotations.md)，既有 10 个基础任务已生成 40 条待审语言。下一步结合 RGB 人工改写与独立复核，未复核不自动批准；新审核不回写本取消准备包或旧配置。
并行接入新版本 V 导航：仅支持明确的三类主任务，取消后仍由模型决定动作，独立评价器检查立即 STOP，保存实际请求/动作/位姿。
先用 CPU 替身覆盖正确、忽略、延迟与错误输出，再安排单任务 HSSD 验证。真实 Habitat 取消任务 run 命令尚未实现，不能把 prepare 配置交给旧运行器。
