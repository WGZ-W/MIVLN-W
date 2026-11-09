# 五候选目标替换：受控扩展实验

状态更新：五候选真实批次现已完成独立验收，30/30 episode 完整、更新和闭环均为 10/10、无干扰项误选；
Oracle/Ignore 的物理轨迹与观测均与原两候选匹配，30 段视频编码与帧数通过检查，完整人工播放仍待记录。
本页运行命令已执行，已有目录不要重跑或重导出。下一步为[措辞与顺序离线对照](language_navigation_robustness.md)，60 条新请求已准备。

入口：`python -m scripts.language_navigation_candidates`。
本阶段将每个任务的候选从两个扩展到五个，检验目标选择是否依赖“总选另一个目标”的捷径。
保留原模型、系统提示、生成参数、当前策略状态表示、初始位姿、随机种子、语言消息、事件触发点及评分。
候选顺序统一按 ID 排序，本阶段不同时研究语言改写、顺序、history/state 或拒绝判断。

## 1. 已有证据与新实验边界

原批次 `outputs/hssd_language_batch_001` 已完成 10 任务、30 episode：模型更新和闭环均为 10/10，
Oracle 10/10 成功，Ignore 10/10 按预期失败。30 段视频编码/帧数已核对，完整人工播放审核待记录。
它是本实验的两候选对照，准备与汇总只读原始证据，不覆盖旧结果。

新候选来自同一场景 `102344280`、同一审核目录中的五个实例：

| ID | 公开描述对应物体 |
| --- | --- |
| `target_020` | 金属框床 |
| `target_047` | 独立浴缸 |
| `target_051` | 带抽屉的厨房岛台 |
| `target_054` | 深色沙发 |
| `target_112` | 放有水果的长方形餐桌 |

各任务的原目标和正确新目标保持原样，额外三个实例作为干扰项。公开输入仍只有候选 ID、审核描述、别名、
当前消息和实际策略状态；坐标、实例诊断和参考更新不交给模型。
执行器增加这些实例的停止点与元数据，使合法但错误的预测也能实际进入规划，不用正确答案回填。
导航仍使用特权网格、坐标和物体中心，本实验不验证视觉落地或学习型导航。

## 2. 已完成的准备与 CPU 检查

使用 `habitat` 环境。准备会独立审核旧批次全部任务，冻结模型参数和原始证据哈希，然后生成新批次。
以下命令已执行，现有 `_001` 无需再次准备；重复准备必须更换编号。

```bash
python -m scripts.language_navigation_candidates prepare \
  --baseline-dir outputs/hssd_language_batch_001 \
  --output-dir outputs/hssd_language_candidates5_001

python -m scripts.language_navigation_candidates check \
  --output-dir outputs/hssd_language_candidates5_001
```

新批次为 **10 个任务 × Oracle/model/Ignore＝30 个全新 episode，计划 10 次全新模型调用**。
不将旧结果算作五候选表现。原参考的三个 JSON 文件按字节保留，仅用于核验同一触发步号，
不复制参考观测，也不将它们表述为新执行的五候选参考 episode。
`collection/` 是派生准备来源，实际运行产物位于 `batch/tasks/`。

CPU 检查核验五个停止点与审核实例的对应关系，并查询每个任务起点及事件前位置到全部候选的可达性。
无法到达的候选会阻止运行，不静默删除候选或更换任务。检查不创建 Simulator、不调用模型。
`request_previews.jsonl` 是准备阶段的公开请求预览，实际模型输入以运行后的日志为准。

## 3. 五候选 GPU 批次命令（已完成）

前提：维持原 Qwen3.5-9B 服务与部署配置，别名 `qwen3.5-9b`，端口 8000。
客户端继承原批次的非思考、temperature=0、max_tokens=512、timeout=120；运行时不提供参数覆盖。
将 GPU 2 替换为分配给本次 Habitat 的编号。此时不需要重新准备或重跑原批次。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python -m scripts.language_navigation_candidates run \
  --output-dir outputs/hssd_language_candidates5_001 --gpu-id 2
```

入口先重验来源与可达性，然后复用原有批量执行器顺序运行 10 个新任务。
每个任务三个条件；单任务非零返回不会删除结果或跳过剩余任务。
批次结束自动保存新实验评分和与旧批次的对照，完整失败仍计入固定分母。
输出目录不覆盖；中断后保留状态，当前新入口不支持续跑或导入旧两候选结果。

## 4. 复核与视频

```bash
python -m scripts.language_navigation_candidates summarize \
  --output-dir outputs/hssd_language_candidates5_001

python -m scripts.language_navigation_candidates videos \
  --output-dir outputs/hssd_language_candidates5_001 --fps 10
```

汇总只使用 CPU PathFinder、NumPy/Pillow，重新读取两批原始证据；视频使用 FFmpeg，
写入 `batch/videos/task_001/oracle.mp4` 等目录，不修改旧视频。可加 `--task-id task_003` 只导出一个任务。

| 产物 | 作用 |
| --- | --- |
| 根目录 `manifest.json` | 代码、原批次输入、模型参数、固定候选和计划分母 |
| `collection/`、`batch/preparations/` | 候选扩展来源与十个实际运行配置 |
| `cpu_check.json` | 每任务、每候选的起点/介入点网格距离，单位米 |
| `batch/summary.json` | 新批次固定 30 个 episode 的独立评分与失败阶段 |
| `batch/run_status.json`、`batch/run_logs/` | 每任务执行状态、返回码和日志 |
| 根目录 `summary.json` | 新旧模型结果、干扰项误选、Oracle/Ignore 行为对照 |
| `comparison.csv`、`comparison.jsonl` | 固定十行的新旧任务配对表 |

验收首先要求新批次 `audited_tasks=10`、`audited_episodes=30`、`results_provisional=false`；
根汇总的 `comparison_complete=true` 表示原始证据完整且 Oracle/Ignore 的全程物理轨迹与原批次一致。
跨批次 RGB/depth/semantic 的哈希是否一致单独记录为 `*_observations_matched`，GPU 不同时需据此复核渲染差异。
同一新任务三个条件的事件前观测一致性仍由原有独立审计严格检查。

分别报告 `expanded_update_correct_count/10` 和 `expanded_pipeline_success_count/10`。
`selected_distractor_count` 记录模型选择额外三个目标的次数；旧目标误选和其他失败可查看逐行预测/失败阶段。
完整真实错误可以有 `comparison_complete=true`、`passed=false`；返回 1 时先分析结果，不覆盖重跑。
运行前的 0/10 是待运行进度，不能视为模型准确率为零。

## 5. 验证与提交

新增 17 项 CPU 回归测试，覆盖审核来源、五候选公开边界、只改变候选、原参考保留、防覆盖、
证据篡改、未运行分母、真实原始响应重放、错误干扰目标实际应用，以及跨批次控制检查。
Habitat 全量 **236 项通过**；核心环境发现 236 项、跳过 70 项原生依赖测试。
真实来源的十个任务已在禁止构造 Simulator 和调用模型的条件下完成 CPU 准备与五候选可达检查；
五候选模型/GPU 表现已独立验收通过，审核记录见输出目录中的 `result_review_2026-11-09.json` 与 `video_review_2026-11-09.json`。

建议代码、测试和相关文档一次提交：`feat: add five-candidate HSSD navigation experiment`。
旧执行/评分代码保持原样，因此既有两候选结果仍可按原哈希复算。`outputs/`、数据、权重不提交。
