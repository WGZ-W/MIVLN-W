# HSSD 语言目标替换单例：真实结果与视频一致性

检查日期：2026-11-09。`task_001/middle` 的 Oracle/model/Ignore 三条件实际运行和独立验收通过。
Oracle 与 model 视频完全相同：模型生成了正确的任务更新，随后两者使用相同导航后端执行。

## 1. 原始证据与验收

- 准备：`outputs/hssd_language_prepare_001/`。
- 实验：`outputs/hssd_language_navigation_001/`，GPU 0，场景 `102344280`。
- 独立验收：实验根目录 `summary.json`。
- 视频核查：同目录 `video_equivalence_audit_2026-11-09.json`。

三个 episode 都完成，模型仅调用一次，原始响应 `finish_reason=stop`，输入 520 token、输出 32 token，无调用错误。
记录时间为 11 月 9 日北京时间 16:34:41—16:34:52，包含三个分支和 CPU 验收，不是纯模型推理耗时。
本次在禁止构造 Simulator 的条件下重新执行 CPU 验收，与保存的 summary 完全一致；没有重跑 GPU 或模型。
核验覆盖来源/代码/日志哈希、实际请求体、原始预测重放、三分支事件前轨迹与观测、位姿连续、最终网格距离、朝向和实例掩码。

`integrity_passed`、`model_update_correct`、`model_pipeline_success` 和 `passed` 均为 true。
回调只收到当前消息、两个审核候选与策略旧状态；请求中没有参考更新、坐标、位姿或场景实例诊断。

## 2. 实测结果

| 指标 | Oracle | model | Ignore |
| --- | ---: | ---: | ---: |
| 最终策略目标 | 深色沙发 `target_054` | 深色沙发 `target_054` | 岛台 `target_051` |
| 动作数 | 51 | 51 | 43 |
| 路程 | 7.000 m | 7.000 m | 7.000 m |
| 到评价目标深色沙发的距离 | 0.232 m | 0.232 m | 4.619 m |
| 深色沙发实例像素 | 7415 | 7415 | 0 |
| 策略目标终点角差 | −0.247° | −0.247° | 0.176° |
| 联合导航成功 | true | true | false，符合负对照预期 |

Oracle/model 在 0.3 m 半径内 STOP，满足目标朝向和实例可见性；Ignore 仍前往岛台，被评价器正确判失败。
三分支路程碰巧都约 7 m，不表示路线或任务相同，Ignore 的终点和画面不同。

## 3. 为什么 Oracle 和 model 的视频相同

第 22 个动作前收到消息：

> Cancel the previous destination. Go to the dark sofa instead.

Qwen 原始输出为：

```json
{"status":"ok","task_update":{"operation":"replace_goal","old_goal":"target_051","new_goal":"target_054"}}
```

Oracle 直接使用预设结构化更新；model 通过 HTTP 根据语言和实际旧状态预测，记录的原文恰好与该更新一致。
二者因此得到相同策略状态：活动目标为深色沙发，岛台进入已取消目标列表。
导航动作由共享的 Oracle 路径规划与终点转向控制器决定；同一起点、姿态、网格、种子、触发时机和目标产生了相同动作与观测。
这里模型负责任务更新，移动动作由固定后端执行。

核查结果：

- Oracle/model 的完整 `trajectory.json` 内容一致，各包含 INITIALIZE 加 51 个动作，共 52 条记录。
- 全部 52 组 RGB、depth、semantic 文件哈希逐一相同。
- 两个 MP4 均为 256×256、10 fps、52 帧、5.2 秒，文件 SHA256 相同。
- Ignore 为 44 帧、4.4 秒，视频哈希不同。

Oracle/model MP4 SHA256：

```text
31dd5f9e767a18b263d0e6b311fd275af2c905c4913ad3c3fe95569ea420ce35
```

模型回调墙钟耗时约 **1.099 秒**，包含 HTTP 与日志开销。模拟器在等待期间没有执行动作。
视频导出只按连续动作帧以固定 10 fps 编码，没有为 HTTP 等待插入重复帧，所以视频也看不到这段等待。
`response_latency_steps=0` 表示更新在下一动作前生效，不表示模型零秒完成调用。

因此，视频一致符合本单例预期，当前没有发现参考答案回填或模型分支绕过调用的证据。
展示两分支区别时，应同时展示输入语言、模型原始 JSON、调用秒数和更新正确性，画面说明的是导航执行结果。

## 4. 结论与下一步

本单例验证了“真实语言预测 → 实际策略状态更新 → 当前位姿继续导航 → 独立实测成功”的链路。
只有一个消息、两个已知候选和一个场景，尚不能据此评价复杂指代、多场景泛化、图像目标定位或学习型导航。
当前导航后端仍使用特权坐标和网格；Oracle 与 model 的区别在更新来源。

下一步扩展已有固定任务，保持模型、候选审核、middle 时机和评分不变，统计模型更新正确率、联合导航成功率和调用耗时。
当前 prepare/run/summarize 单例入口可直接复用；10 任务/30 episode 的统一批量运行与汇总入口尚未实现。
可先运行已存在的反向任务 `task_002`（深色沙发→岛台）检查反向语言更新，以下是已有可执行命令，输出目录当前未创建：

```bash
conda activate habitat

python -m scripts.language_navigation prepare \
  --source-dir outputs/hssd_dataset_facing_001 \
  --task-id task_002 --timing middle \
  --output-dir outputs/hssd_language_task002_prepare_001

python -m scripts.language_navigation run \
  --prepared-dir outputs/hssd_language_task002_prepare_001 \
  --base-url http://127.0.0.1:8000/v1 --model qwen3.5-9b \
  --thinking-mode off --temperature 0 --max-tokens 512 --timeout 120 \
  --gpu-id 0 --output-dir outputs/hssd_language_task002_001
```

前提是同一模型服务和已分配 GPU 可用。预期保留三个条件、一次模型请求，分别报告更新、导航和负对照结果；
不将任务失败删除或改用 Oracle 预测。本次未准备或执行新 GPU 条件。

本轮结果、视频一致性分析及状态文档可作为一次文档提交：
`docs: record HSSD model navigation smoke results`。
只做了 CPU 结果审计与 FFprobe 元数据检查，原始日志、预测、评分和视频未改；`outputs/` 不提交，不自动暂存、提交或推送。
