# 三类真实 HSSD middle：统一批量入口

入口：`python -B -m scripts.hssd_visual_dataset`。新增协调层复用已有单任务执行/评价器，不改其代码哈希、旧配置或运行结果。
范围固定为来源清单中的全部基础任务、middle、替换/插入恢复/取消三类、Oracle/Ignore 两条件。
当前为 **10 基础任务、30 个介入案例、60 个计划 episode**；时机/条件不是新的独立基础任务。
状态更新（2026-11-16）：真实运行与只读审计已完成，60/60 条件验收、30 个 Oracle 成功、30 个 Ignore 按预期失败，见[批量实测结果](hssd_v_middle_batch_results_2026-11-16.md)。

## 1. 本次已经准备什么

- 来源：`outputs/benchmark_v_navigation_prepare_002/`；语言仍绑定到已批准的 `benchmark_language_annotations_002`。
- 批量准备包：`outputs/hssd_v_middle_batch_prepare_001/`，每个任务包含原格式的三类 middle 配置和六分支计划。
- 复用 `outputs/hssd_v_middle_run_001/` 的 `task_001` 六分支，重新从原始动作/RGB/实例证据审计通过，未重跑或复制原轨迹。
- 后续 `task_002`—`task_010` 的 **54 个新分支已运行并验收**，加上六个复用分支，全批有效证据和验收均为 60/60。
- 原 `preparation_summary.json` 保留执行前 valid/accepted 6/60、provisional 状态，不回写。最新状态来自运行及正式审计：`results_provisional=false`、`passed=true`。

所有任务保留原初始语言、实例、几何、起始姿态、种子、预算和参考累计路程触发槽；不接新模型，也不改变 Oracle 特权。
每个任务的三类/两条件共享该任务自己的 middle 步，不把全部任务强制设为 task_001 的步 22。

## 2. 已执行的命令与复现

以下 run + audit 已执行，`_001` 目录已有有效结果，不覆盖重跑。复现时使用配置好的 Habitat 环境和全新输出目录，并选择分配的 GPU；本次使用 0：

```bash
conda activate habitat

python -B -m scripts.hssd_visual_dataset run \
  --input-dir outputs/hssd_v_middle_batch_prepare_001 \
  --gpu-id 0 \
  --output-dir outputs/hssd_v_middle_batch_run_001

python -B -m scripts.hssd_visual_dataset audit \
  --input-dir outputs/hssd_v_middle_batch_run_001 \
  --output-dir outputs/hssd_v_middle_batch_audit_001
```

run 只启动剩余九个任务的单任务子进程，各顺序运行六个分支，共 54 次新 GPU episode。
每个任务沿用原 Oracle/Ignore 六分支协议，父级日志为 `logs/task_XXX.log`，分支日志在 `tasks/task_XXX/run/logs/`。
任务失败后继续后续任务。中断保留 interrupted/pending，不在旧目录按成功与否选择性重试；重新运行需新输出目录。
不同任务可以使用不同 GPU；复用任务保留其原 GPU，新任务的六个分支统一使用本次指定 GPU。

如果只想重新生成汇总，也可运行下列无渲染命令；它仍重放原始证据，不复制缓存分数：

```bash
python -B -m scripts.hssd_visual_dataset summarize \
  --input-dir outputs/hssd_v_middle_batch_run_001 \
  --output-dir outputs/hssd_v_middle_batch_summary_001
```

audit 进一步比较批次运行、批次准备、来源准备、外部复用运行/准备的完整文件快照。两种检查都不会创建 Simulator 或调用模型，但需要 Habitat 原生网格重放及 NumPy/Pillow。
未完成或未通过时仍输出完整汇总并返回 1；不能仅依据返回码判断是基础设施故障还是有效任务失败，应检查逐项记录。

## 3. 复用门槛与来源保护

准备命令支持多个 `--reuse-run`，但同一基础任务只能声明一次，且必须在完整来源 roster 内。
任务/来源、三类配置、时机、初始位姿、种子、预算、语言版本、几何和执行/评价代码必须一致。
GPU 可跨任务不同，不能用不同输入或不同评分结果替代本批任务。

复用不读取缓存成功标志来筛选任务。六分支原始证据完整、有效且配对正确的失败结果也可复用，并继续作为失败计入分母；缺失/无效证据不能伪装成可复用结果。
绑定整个旧运行目录的文件集合与哈希、其准备版本以及本次独立审计快照。运行前会再次预检全部任务及全部复用，任何变化在启动首个 GPU 子进程前报错。
旧运行目录及旧准备目录不能作为新报告/运行的父目录或子目录。复用源新增文件也会改变绑定，视频等后续导出应使用独立目录。

如需另建准备包，可使用：

```bash
python -B -m scripts.hssd_visual_dataset prepare \
  --source-dir outputs/benchmark_v_navigation_prepare_002 \
  --reuse-run outputs/hssd_v_middle_run_001 \
  --output-dir outputs/hssd_v_middle_batch_prepare_002
```

prepare 不启动 GPU。指定真实复用时需要 Habitat 环境进行 CPU 独立重审；不指定复用时准备只需标准库。
若某个来源任务无法准备，其六个槽位仍保留为 preparation_failed，prepare 返回 1，不悄悄缩小 roster。

## 4. 汇总与验收怎样看

| 文件 | 内容 |
| --- | --- |
| 准备 `manifest.json` | 完整任务/条件计划、代码与来源哈希、逐任务准备及外部复用绑定 |
| 准备 `reuse_audits/` | 复用时从原始证据重新计算的六分支审核结果 |
| 运行 `manifest.json`、`execution_status.json` | 本次 GPU、全部任务的 reused/pending/completed/failed/interrupted 状态及原返回码 |
| `summary.json` | 完整分母、证据覆盖、按类型/条件的任务成功和条件验收、基础任务统计及执行错误 |
| `cases.csv/jsonl` | 始终 60 个计划分支及其实际来源、有效性、成功/失败和逐项指标 |
| `tasks.csv/jsonl` | 10 个基础任务各自的证据、Oracle 成功和 Ignore 预期失败数量 |
| 审计 `audit.json`、`source_bindings.json` | 无渲染重算、配对检查，以及所有受保护源目录前后快照 |

最终正常预期：

- `planned_episodes=60`、`valid_evidence_episodes=60`、`accepted_episode_slots=60`、`results_provisional=false`、`passed=true`。
- 每类 Oracle 10/10 成功，三类共 30 个成功任务；每类 Ignore 10/10 按预期失败。
- Oracle 宏平均成功率为三类各自成功率的均值；Ignore 分别报告任务失败和负对照验收，不混入方法成功率。
- 每类/条件分母始终为 10。未运行、缺失或证据无效的槽位不进入成功分子；完成之前的分数标为 provisional，不作为最终性能结论。
- 完整可信的任务失败仍有有效证据，最终 `passed=false` 并保留失败；父任务非零返回码也原样报告，不用汇总成功掩盖执行异常。

独立重审复用原单任务 `check_run`，连续重放实际动作并核对 V 历史、PNG、网格位姿/距离/碰撞/朝向和实例掩码。
不信任单任务或批次的 metrics/summary 缓存。每个任务仍检查三条 Ignore 全程一致、Oracle/Ignore 的介入前物理/RGB 前缀一致。
该流程不独立重新渲染图像；实际像素与场景的对应依赖可信采集器，完整进程隔离和公开隐藏评测仍待完善。

## 5. 当前验证及下一完整步骤

新增 14 项回归以隔离合成来源和假子进程/渲染器验证完整分母、只读复用、失败继续、中断、目录边界、提前全量预检及逐项评分。
全量 406 项在 Habitat 环境通过；核心环境同套件通过，按原生依赖跳过 89 项。
真实准备和后续运行现已完成：10 任务/60 分支，6 复用、54 新运行，全部按预期验收，正式审计源文件前后不变。
已验收单任务及旧输入/执行代码保持不变，来源说明见[六分支实测](hssd_v_middle_results_2026-11-13.md)。

最新实测见[批量结果](hssd_v_middle_batch_results_2026-11-16.md)。[V 视频入口](visual_video_review.md)已在独立目录导出 task_001 六分支及 task_007 插入恢复两分支，共八段，编码验证通过但全部待审。
立即下一步是人工播放并导入审核记录；质检后推进第二场景/scene-disjoint 与 early/late。早/晚真实运行入口仍需适配，不直接把旧视频脚本或中期参数用于这些任务。
不新增模型预测或时机机制作为当前批次前置条件。数据与运行产物在 `outputs/`，不纳入 Git。
