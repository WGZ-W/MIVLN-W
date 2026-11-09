# HSSD 反向语言目标替换：沙发到岛台

检查日期：2026-11-09。反向任务 `task_002/middle` 的三个真实 episode 及独立 CPU 验收通过。
模型更新和导航均正确；与正向任务合计，两个方向的语言更新及闭环成功均为 2/2。

## 1. 批次与原始证据

- 准备：`outputs/hssd_language_task002_prepare_001/`。
- 预测与轨迹：`outputs/hssd_language_task002_001/`。
- 独立验收：实验目录中的 `summary.json`。
- 本轮审计：同目录 `reverse_review_2026-11-09.json`。
- Habitat GPU：2；模型别名 `qwen3.5-9b`，非思考、temperature=0、max_tokens=512、timeout=120。

日志时间为 11 月 9 日北京时间 16:47:50—16:48:03，包括三个导航分支和 CPU 验收。
三个分支全部执行完成，只有 model 调用一次接口，响应 `finish_reason=stop`，输入 521 token、输出 32 token，无调用错误。
回调墙钟耗时 **0.922 秒**，包含 HTTP 与日志开销。

本次禁止构造 Simulator，重新调用 CPU 独立验收，与原 summary 完全一致，没有重跑模型或 GPU。
来源、代码、请求体、原始输出重放、位姿连续、事件前观测、网格距离、终点朝向和实例掩码检查均通过。
`integrity_passed`、`model_update_correct`、`model_pipeline_success` 和 `passed` 均为 true。

## 2. 模型如何执行反向更新

机器人起初前往深色沙发 `target_054`，第 **37 个动作前**收到：

> Cancel the previous destination. Go to the wooden kitchen island with drawers instead.

模型读取实际旧状态与公开候选，原始输出为：

```json
{"status":"ok","task_update":{"operation":"replace_goal","old_goal":"target_054","new_goal":"target_051"}}
```

执行后活动目标变为岛台 `target_051`，沙发进入已取消目标列表。模型未收到参考更新、坐标或评价状态，
预测被直接应用到策略状态，评价器仍按独立参考更新判定任务。事件只投递一次，发生在动作之前，没有重置位姿。

## 3. 三条件结果

| 指标 | Oracle | model | Ignore |
| --- | ---: | ---: | ---: |
| 最终策略目标 | 岛台 | 岛台 | 深色沙发 |
| 动作数 | 57 | 57 | 62 |
| 路程 | 8.549 m | 8.549 m | 10.799 m |
| 到评价目标岛台停止点的距离 | 0.114 m | 0.114 m | 4.734 m |
| 岛台实例像素 | 10750 | 10750 | 0 |
| 策略目标终点角差 | −1.052° | −1.052° | −6.151° |
| 碰撞标记 / 规划错误 | 0 / 0 | 0 / 0 | 0 / 0 |
| 联合导航成功 | true | true | false，负对照符合预期 |

Oracle/model 在 0.3 m 半径内 STOP，满足朝向及实例可见性。Ignore 到达旧目标沙发，虽然完成自己的旧策略，
仍因未满足新任务岛台而正确失败。不同条件的路线对应不同终点，不能用其长度直接评价相同任务下的规划效率。

Oracle/model 的完整轨迹 JSON 一致；58 组 RGB、depth、semantic 文件哈希逐一一致。
当前三个目录尚无 `trajectory.mp4`，因此本次只核对原始帧，没有报告反向 MP4 哈希。
两者原始画面相同仍是正确更新与相同导航后端的预期结果，视频中无需出现不同路线才能证明模型生效。

## 4. 与正向结果合并解释

| 模型闭环指标 | 正向 task_001 | 反向 task_002 |
| --- | --- | --- |
| 更新方向 | 岛台→沙发 | 沙发→岛台 |
| 更新与闭环成功 | true / true | true / true |
| 动作数 / 路程 | 51 / 7.000 m | 57 / 8.549 m |
| 到新目标停止点误差 | 0.232 m | 0.114 m |
| 回调墙钟耗时 | 1.099 s | 0.922 s |

正向记录见 [首个单例](hssd_language_results_2026-11-09.md)。两个任务的公开候选及顺序相同，模型输出的新目标随任务方向改变，
在这两例中没有始终固定选择沙发或固定候选位置。
两个任务有不同起点、朝向与种子，反向任务是独立固定任务，不是把正向轨迹倒放。
Habitat GPU 也分别为 0、2；两次回调耗时仅为单次观测，不据此推断推理性能提升。

合计六个 episode 均通过协议验收：Oracle 2/2、model 2/2 导航成功，Ignore 0/2 导航成功但 2/2 按预期失败。
仍只有一个场景、两个方向、两个候选；替换时只有一个不同于当前目标的候选，不能仅凭这两例排除“总选另一个候选”的简单策略。
它们验证的是闭环接通及两个方向的执行一致性，尚不证明复杂多候选指代、图像目标定位或学习型导航能力。
导航仍依赖已有审核目标坐标和 Oracle 规划器。

## 5. 下一步、命令与提交

先导出并人工查看反向三段视频：

```bash
conda activate habitat
python -m scripts.language_navigation videos \
  --output-dir outputs/hssd_language_task002_001
```

预期生成 oracle/model/ignore 各自的 `trajectory.mp4`；前两者各 58 帧，Ignore 为 63 帧。该命令只编码现有帧，不重跑模型或导航。
查看任务切换前后是否连续、终点实例是否符合岛台/沙发条件，并记录审核范围。

随后应补齐 10 个固定任务的统一批量运行与汇总入口。现有两个已审计任务可在验证配置、提示、模型参数和代码一致后复用；
余下八个任务计划 24 个 episode、八次模型请求，保持 middle 时机和当前评分，失败仍计入完整分母。
验收输出应含固定计划清单、逐条件状态、更新正确率、联合导航成功率、调用耗时和错误阶段。
**批量入口尚未实现**，现有单例 prepare/run/summarize 可复用；若要研究语言目标选择，还应另行设计至少三个审核候选的对照配置。
本次没有执行新 GPU 条件或导出视频。

反向结果与相关状态说明可作为文档提交：`docs: record reverse HSSD language navigation results`。
本轮只进行 CPU 结果核查，原始预测、日志与评分未修改，`outputs/` 不提交，不自动暂存、提交或推送。
