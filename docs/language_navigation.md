# HSSD 语言更新驱动导航：目标替换闭环

更新日期：2026-11-09。模型回调、三条件单例、真实 GPU 运行及独立验收已完成，见 [单例结果与视频一致性](hssd_language_results_2026-11-09.md)。
Qwen 正确更新至深色沙发，Oracle/model 均成功，Ignore 按预期失败。Oracle/model 共享后端且更新相同，轨迹、帧文件及视频哈希均一致。
反向沙发→岛台任务也已通过真实验收，Oracle/model 均为 57 动作、8.549 m、终点误差 0.114 m，见 [反向结果](hssd_language_reverse_results_2026-11-09.md)。
入口：`python -m scripts.language_navigation`。本阶段固定结构化状态输入，不继续比较 history/state，也不扩展拒绝实验。

## 1. 当前实现与信息边界

导航循环在事件对应动作之前调用模型。模型读取当前介入消息、公开候选描述和本分支实际 `previous_state`，
输出增量更新；执行器将合法目标替换应用到策略状态，再按该状态重建导航规划器。

- `mivln/navigation_updates.py`：公开请求和事务式策略更新，不读取评价参考。
- `scripts/run_navigation.py`：新增 `intervention_policy="model"` 和 `task_updater(request)` 回调。
  回调返回 `(raw_text, metadata, call_error)`；默认 follow/ignore 入口继续可用。
- `scripts/language_navigation.py`：复用已审核任务的 prepare/run/summarize/videos 入口。

评价器仍独立接收预设参考更新。模型不会收到完整配置、事件参考 `task_update`、坐标、网格、位姿、物体中心、实例诊断或评价状态。
请求 ID 为公开信息生成的不透明哈希，不直接使用含操作名的事件 ID。
候选 ID 绑定已有审核描述、稳定实例及停止点，仅向模型暴露 ID、描述和别名。

本阶段只执行一次 `replace_goal`。不合法输出、调用失败、拒绝或其他操作均被记录，策略保留原状态；不会用正确答案补救，也不会修复 JSON 或重试。
合法但语义错误的目标会实际进入策略状态；规划路径和终点转向始终跟随实际策略目标，不借用参考目标的路线。
导航仍使用已知坐标、网格和物体中心，属于**语言更新加 Oracle 导航后端**，不是图像目标定位或学习型端到端导航。

## 2. 三个匹配条件

| 目录 | 策略更新来源 | 预期行为 |
| --- | --- | --- |
| `oracle/` | 正确结构化参考 | 从当前位姿切换到新目标 B |
| `model/` | Qwen 对语言和实际状态的预测 | 由真实预测决定后续目标，错误保留 |
| `ignore/` | 不应用策略更新 | 仍到 A，评价器按新任务 B 判失败 |

三者复用场景、网格、起点、朝向、目标、随机种子、动作预算、事件时机和终点协议。
独立验收要求事件前物理轨迹与 RGB/depth/semantic 文件内容一致，并核验一次投递、动作前应用及事件处位姿连续。
模型调用同步阻塞在动作之前，等待期间模拟器不执行动作；`response_latency_steps=0` 不表示零秒推理，
另用 `model_callback_seconds` 记录完整回调墙钟耗时，含 HTTP 和日志开销。

## 3. 已完成的 CPU 准备

已执行以下命令；`_001` 已存在，重复准备需使用新编号：

```bash
python -m scripts.language_navigation prepare \
  --source-dir outputs/hssd_dataset_facing_001 \
  --task-id task_001 --timing middle \
  --output-dir outputs/hssd_language_prepare_001
```

准备从原任务、已成功参考轨迹和原时机配置核验当前消息、参考更新与真实进度触发点，不重新采样、重审或修改旧任务。
`task_001/middle` 在第 **22 个动作前**投递，候选为已审核的带抽屉木质岛台和深色沙发，ID 为 `target_051`、`target_054`。
事件位姿到两个停止点的 CPU 网格距离分别为约 3.538 m、3.454 m，均可达。
本地 Qwen 分词器检查：输入 520 token，加 512 输出预算共 1032，小于当前 4096 上下文上限。

准备目录包含 `config.json`、来源/配置哈希清单 `manifest.json`，另保存了
`model_request_preview.json` 和 `cpu_check.json`。后两者是 CPU 检查产物，不是模型预测或 GPU 验收。
准备和检查均没有创建 Simulator 或调用模型；源场景、网格、审核及参考文件保持原样。

## 4. 已完成的真实三条件单例

前提：原 Qwen3.5-9B 服务保持可用，在本机 8000 端口暴露 `qwen3.5-9b`；模型部署见 [本地部署](qwen35_local.md)。
导航使用 `habitat` 环境，模型服务在其独立环境运行。将 GPU 0 替换为分配给本次 Habitat 的 GPU 编号。
以下输出目录已存在，命令保留用于复现；重复实验必须使用新目录。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python -m scripts.language_navigation run \
  --prepared-dir outputs/hssd_language_prepare_001 \
  --base-url http://127.0.0.1:8000/v1 --model qwen3.5-9b \
  --thinking-mode off --temperature 0 --max-tokens 512 --timeout 120 \
  --gpu-id 0 \
  --output-dir outputs/hssd_language_navigation_001
```

该命令顺序运行 3 个 episode，只在 model 分支事件送达时调用一次模型。模型更新错误属于待报告结果，
不会自动替换目标或重新发请求。某分支异常也保留计划位置，并继续记录其余条件；中断时未运行条件仍留在清单中。
`run` 结束会自动进行 CPU 轨迹、网格距离与终点证据验收并保存 `summary.json`，不自动编码视频。

完成后可分别复核和导出视频：

```bash
python -m scripts.language_navigation summarize \
  --output-dir outputs/hssd_language_navigation_001

python -m scripts.language_navigation videos \
  --output-dir outputs/hssd_language_navigation_001 --fps 10
```

`summarize` 需要 Habitat 的 CPU PathFinder、NumPy 和 Pillow，不创建 GPU Simulator、不调用模型。
`videos` 使用 FFmpeg，不重跑导航；已有视频不覆盖。

## 5. 产物与验收标准

三个分支保存已有导航产物：实际配置、逐步轨迹、事件、重规划、真实/策略最终状态、指标及 RGB/depth/semantic。
`model/policy_updates.json` 额外记录实际请求、原始预测、解析/应用结果、前后策略状态及回调秒数，并在执行下一动作前落盘。

总目录保存：

- `model_inputs.jsonl`、`model_responses.jsonl`：实际 HTTP 请求体、哈希、原文、结束原因、token 用量及失败记录。
- `model_system_prompt.txt`、`run.json`：实际系统提示、固定生成参数、GPU 编号、代码及日志哈希。
- `run_status.json`：固定 oracle/model/ignore 三个位置及是否完成。
- `summary.json`：独立验收、更新正确性、导航结果、失败阶段及两者组成的闭环结果。

验收关注：

1. `integrity_passed=true`：来源哈希、完整清单、事件前轨迹/观测、位姿连续和真值/策略状态独立重放一致。
2. 模型恰好一次请求与一次响应，实际请求体与公开边界重建一致；原始响应可重放出日志中的策略变化。
3. 重新查询最终各目标网格距离，核验实际 STOP、目标实例掩码像素、诊断身份/文件哈希及策略目标朝向。
   异常长的前进位移复用原生网格动作终点重放检查，不直接放宽位移阈值。
4. Oracle 到 B 并满足距离、朝向和实例可见；Ignore 到 A 且按 B 正确判失败。
5. model 的 `model_update_correct` 与导航 `joint_success` 分开报告，二者成立才得到 `model_pipeline_success=true`。

`passed=true` 表示三个条件完整且符合上述单例预期。模型任务失败也可具有 `integrity_passed=true`，不可把完整日志等同于任务成功。
`model_failure_stage` 区分证据不完整、模型调用、更新和导航问题。命令返回 1 时先检查 summary 与状态清单，完整记录不会被删除。

## 6. 验证与提交

新增 19 项测试覆盖白名单隔离、错误目标实际规划、事务失败、无 Oracle 回填、动作前投递、位姿连续、
真实请求体记录、接口失败不重试、固定三条件分母及提示/实例掩码篡改检查。全部使用 CPU 假模拟器与 HTTP 桩。
`python -m unittest discover -s tests -q`：Habitat 环境 **202 项通过**；核心环境发现 202 项并跳过 70 项原生依赖测试。
真实 GPU 单例现已完成并独立复算通过，模型回调约 1.099 秒；两成功分支均为 51 动作、约 7 m，距沙发停止点约 0.232 m。
此结果验证一个已知候选任务的闭环，不代表多任务或视觉落地成绩。

建议将回调模块、导航循环修改、实验入口、测试及相关文档作为一个功能提交：
`feat: add model update callbacks and HSSD navigation experiments`。
`outputs/`、场景、权重与凭据不提交，不自动暂存、提交或推送。

本单例与三段视频的编码/帧数已核对，用户反馈 Oracle/model 画面相近，原始帧和哈希核查证实完全相同；未补写完整人工播放审核记录。
正反两个任务已通过；反向三段视频尚未导出，命令及下一步批量方案见 [反向结果](hssd_language_reverse_results_2026-11-09.md)。
10 任务/30 episode 的统一批量运行与汇总入口尚未实现；
禁区和子目标的模型闭环也属于后续阶段，当前仅目标替换。
