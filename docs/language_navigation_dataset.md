# HSSD 语言更新导航：统一批量入口

状态更新：两候选批次已完成 10 任务/30 episode，独立验收全部通过；模型更新和闭环均为 10/10，
Ignore 10/10 按预期失败。30 段视频已核对编码和帧数，人工播放审核待记录。
下述运行命令保留用于理解流程，已有批次勿重跑或重导出视频。
下一实验为[五候选受控扩展](language_navigation_candidates.md)，已完成 CPU 准备，新的 30 个 episode 待运行。

入口：`python -m scripts.language_navigation_dataset`。本阶段将原有单任务协议扩展至固定集合，
不改变模型提示、候选、策略更新或评价规则。模型预测语言更新，导航后端使用特权坐标、网格和物体中心；
实验用于验证更新能否可靠驱动导航闭环，暂不声称视觉落地或跨场景泛化。

## 1. 任务与固定分母

- 来源：`outputs/hssd_dataset_facing_001`，HSSD 场景 `102344280`，全部 10 个既定任务。
- 时机：统一 `middle`，每个任务沿用其原参考轨迹确定的动作前触发点。
- 每个任务运行 `oracle/model/ignore`，共 **30 个计划 episode、10 次计划模型调用**。
- 不允许按结果筛选任务或缩小分母。模型调用失败、错误更新、导航失败、缺失证据和未运行位置均保留。
- Oracle 使用正确结构化更新；model 只读取公开描述、当前消息和实际策略状态；Ignore 保持原目标。
  具体事件与终点评价语义见 [单任务指南](language_navigation.md)。

## 2. 准备与旧结果复用

使用 `habitat` 环境。没有复用时准备仅需标准库；复用包含 CPU PathFinder、NumPy/Pillow 的独立验收。
准备与汇总均不创建 GPU Simulator、不调用模型。

以下准备命令已在本地执行，**无需再次执行**；重新准备需更换输出编号：

```bash
python -m scripts.language_navigation_dataset prepare \
  --source-dir outputs/hssd_dataset_facing_001 --timing middle \
  --model qwen3.5-9b --base-url http://127.0.0.1:8000/v1 \
  --thinking-mode off --temperature 0 --max-tokens 512 --timeout 120 \
  --reuse-run outputs/hssd_language_navigation_001 \
  --reuse-run outputs/hssd_language_task002_001 \
  --output-dir outputs/hssd_language_batch_001
```

复用按任务 ID 匹配，核验完整来源、任务/时机、生成参数、系统提示、执行代码和三条件原始证据。
仅要求证据完整可信，不要求模型预测成功；完整的失败结果也可复用。重复任务、不同参数或证据异常会中止准备，
不会静默改为重跑。旧目录只读，不复制视频或覆盖旧汇总；新清单记录原始文件哈希及独立审核快照。
后续添加可选视频不影响已冻结的原始证据。

`task_001` 和 `task_002` 的 6 个 episode 已核验复用；
本批次当时新增 `task_003`～`task_010`：8 个任务、24 个 episode、8 次模型调用，现已全部验收。
旧任务 Habitat GPU 分别为 0、2；跨任务允许 GPU 不同并逐项记录，同一任务的三个条件保持一致。
客户端参数相同不保证服务端权重或部署配置自动相同，继续使用原 Qwen3.5-9B 服务及其启动配置。

## 3. GPU 实验命令（本批次已完成）

前提：原 Qwen 服务在 `http://127.0.0.1:8000/v1` 提供 `qwen3.5-9b`，使用同一权重和服务配置；
场景、网格、审核目录、旧运行目录仍位于原路径。模型服务保持在其独立环境运行。
将下面 GPU 2 改为分配给 Habitat 的编号；不要修改已冻结的模型参数或代码。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python -m scripts.language_navigation_dataset run \
  --output-dir outputs/hssd_language_batch_001 --gpu-id 2
```

入口顺序启动 8 个独立单任务子进程，各进程顺序执行三个条件。日志实时写入
`run_logs/task_003.log` 等文件，批次终端只显示任务进度，模型输出保存在各任务日志/原始响应中。
运行前会核验全体输入和复用证据，运行后自动汇总。

子进程返回非零也继续后续任务，保留真实失败。最终 `run`/`summarize` 返回 0 表示全体任务完整且符合实验预期；
返回 1 可能仅表示真实模型错误，先看汇总和日志，**不要直接重跑覆盖**。
中断后保留已完成/待运行位置；当前入口不支持原目录续跑。需要续跑时，显式准备新批次，
用重复的 `--reuse-run` 导入每个完整任务结果；不完整任务需重新运行，且先保存/报告原批次的失败。

## 4. 复核、视频与验收

```bash
python -m scripts.language_navigation_dataset summarize \
  --output-dir outputs/hssd_language_batch_001

python -m scripts.language_navigation_dataset videos \
  --output-dir outputs/hssd_language_batch_001 --fps 10
```

视频导出需 FFmpeg，写入新目录 `videos/task_001/oracle.mp4` 等，包括复用任务的视频；
不会写回旧运行目录。只导出一个任务可加 `--task-id task_003`，已有视频不覆盖。
视频选择不会改变评分分母，`video_status.json` 记录导出失败。

主要产物：

| 文件/目录 | 内容 |
| --- | --- |
| `manifest.json`、`preparations/` | 固定任务、参数、哈希、复用映射与全部准备配置 |
| `reuse_audits/` | 准备时 CPU 审核快照；评分仍重新读取原始证据 |
| `batch_run.json`、`run_status.json` | 批次来源、时间、GPU、逐任务状态/返回码 |
| `tasks/task_003/` 等 | 新任务完整三条件运行及模型请求/原始响应 |
| `summary.json` | 固定分母的成功数/率、完整性、失败阶段、回调耗时及样本数 |
| `tasks.jsonl`、`tasks.csv` | 每任务种子、目标、触发点、GPU、更新/闭环结果及模型相对 Oracle 的动作/路程差 |
| `cases.jsonl`、`cases.csv` | 始终 30 行，按任务和条件记录导航证据/指标或错误 |
| `model_calls.csv` | 始终 10 行，含未运行请求位置与调用/更新/导航失败阶段 |

完成后检查 `audited_tasks=10`、`audited_episodes=30`、`results_provisional=false`。
Oracle 的 `joint_success_count` 和 Ignore 的 `ignore_expected_failure_count` 预期均为 10；
模型分别报告 `model_update_correct_count/10` 和 `model_pipeline_success_count/10`，任何真实错误都保留分析，
不以模型必须满分作为日志有效性的条件。`passed=true` 是全条件均符合预期的更强结论。
耗时是含 HTTP/日志的回调墙钟时间，分新/复用来源给出样本数，不能当作相同硬件上的纯模型速度。

## 5. 本地检查与提交边界

新增 17 项 CPU 测试覆盖复用绑定、防覆盖、原始证据篡改、失败/中断和固定分母。
Habitat 环境全量 **219 项通过**；核心环境发现 219 项，跳过 70 项原生依赖测试。
真实两任务在禁止创建 Simulator 和调用模型的条件下重新验收通过，旧原始证据哈希未改变。
准备后快照为 2/10 个已审核任务、6/30 个 episode、8 个待运行任务，`results_provisional=true`；
此时模型正确数显示 2/10 只是固定分母的进度，不能解读为最终 20% 准确率。
后续 GPU 实验已完成，顶层汇总为 10/10 任务、30/30 episode、`results_provisional=false`；
完整视频人工播放审核待记录。五候选实验见[新入口指南](language_navigation_candidates.md)。

代码、测试和指南可作为一次功能提交：`feat: add batch HSSD language navigation with verified result reuse`。
`outputs/`、场景、权重与凭据不提交。
