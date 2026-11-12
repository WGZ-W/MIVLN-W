# 三类任务统一清单与只读审计

## 目的和范围

B0 将现有目标替换、禁区、子目标插入与恢复资产关联到共同基础任务，保留历史结果，重新检查原始执行证据。
这是**评价器私有开发清单**，包含精确坐标等特权输入；不能直接作为普通 Agent 的公开输入或正式排行榜数据。
清单不要求参与方法内部采用结构化 `TaskState`，也不依赖模型预测更新。

默认读取：

```text
outputs/hssd_dataset_facing_001/
outputs/hssd_avoid_batch_001/
outputs/hssd_subgoal_timing_001/
outputs/hssd_subgoal_acceptance_2026-11-03/
```

| 层级 | 当前计划规模 | 计数方式 |
| --- | --- | --- |
| 场景 / 基础任务 | 1 / 10 | 以目标替换的冻结 roster 为共同来源 |
| 目标替换配置 / episode | 30 / 100 | 三时机 × 三条件，另有十个共享参考 |
| 禁区配置 / episode | 10 / 30 | 一个时机 × 三条件 |
| 子目标配置 / episode | 30 / 90 | 三时机 × 三条件 |
| 合计配置 / episode | 70 / 220 | 配置、条件、时机不是新的独立基础任务 |

## 执行命令

从仓库根目录执行。完整审计需要已配置的 `habitat` 环境，使用 CPU 的 PathFinder、无传感器原生 Agent 和实例图像检查，
**不会创建 Simulator、运行导航、调用模型或启动 GPU 实验**。目录必须全新，重复执行请更换编号。

```bash
conda activate habitat
python -B -m scripts.benchmark_audit \
  --output-dir outputs/benchmark_audit_002
```

只生成清单时，普通 Python 核心环境即可；此模式保持 `audit_status=not_audited`，不声称原始证据通过验收。

```bash
python -B -m scripts.benchmark_audit --index-only \
  --output-dir outputs/benchmark_index_001
python -B -m unittest tests.test_benchmark_audit -v
```

四个输入可分别通过 `--replacement-dir`、`--avoid-dir`、`--subgoal-dir`、`--reaudit-dir` 指定。
首版适配要求规范的完整三时机子目标集合。缺少派生 manifest、未准备或失败的配置仍保留规范计划槽位并报错；
无法读取或唯一识别共同基础 roster 时直接终止，不凭空生成基础任务。

## 输出和字段

- `manifest.json`：输入 manifest 绑定、审计代码 SHA256、历史 summary 声明和开发协议版本。
- `base_tasks.{jsonl,csv}`：场景身份、共同基础 ID、原配置和共享参考；当前十行。
- `task_index.{jsonl,csv}`：介入配置、触发语义、协议版本、参考关联、配置哈希、运行/审核/划分状态；当前七十行。
- `episode_index.{jsonl,csv}`：全部计划执行槽位、实际分支目录、原始运行状态、原报告成功标志、逐项审计及证据；当前二百二十行。
- `audits/*.json`：三类新审计结果；子目标包含新计算的原生投影端点证明。
- `summary.json`：固定分母覆盖、输入校验、条件验收、任务成功和 Ignore 预期失败分别统计。
- `issues.{jsonl,csv}`：错误与待发布问题。失败样本继续进入固定分母。
- `source_snapshot.json`：四个源目录全部文件（含帧和视频）的前后 SHA256 清单；必须 `unchanged=true`。

`base_task_id` 由场景 ID、场景文件 SHA256 和原 task ID 组成；同源三类任务共享此 ID。
配置 ID 追加类型和时机，episode ID 再追加条件。`reference_id` 指向共享参考。
`scene_group` 用于后续场景分组，`leakage_group` 关联基础任务；当前 `split=null`，未生成 train/test。
原 `intervention` 分支在统一清单中标为 `follow`，同时保留 `source_condition`。
Control 按原始任务评分；Follow/Ignore 按更新后的任务评分，不能直接混合解释成功率。

## 如何解释验收

完整审计退出 0 表示当前资产符合**各自已有验收协议**，不表示语言理解有效或已经达到正式发布条件。
Ignore 的 `accepted` 应伴随 `reported_dynamic_success=false`；它是按预期失败的负对照。
未通过证据审查的原报告成功标志不会计入成功分子。Oracle 可行性按通过审查且成功的 Follow 配置单独统计。
输入完整性、条件验收、Oracle 成功、Ignore 预期失败、人工审核是不同字段；条件验收本身包含行为预期，不能当作纯文件完整性。

目标替换复用只读时序、STOP、配对帧哈希与终点诊断检查；禁区重放连续轨迹违规及状态更新；
子目标重新测量完成、恢复、最终 STOP、实例证据和必要的原生投影端点。前者不是新增的完整原生距离重放。
旧 `summary.json` 和子项缓存均不用于新评分，禁止调用会回写源目录的旧 `summarize` 入口。

历史 `task_004_middle` 的父运行返回码 1 必须保留。只有新完整审计通过、原子分支执行完整且有原生端点证明，
才标记 `audit_recovered=true`；其他非零退出不能仅凭旧 acceptance 的成功声明恢复。
历史重审包的原文件/检查器哈希另外绑定；缺失或陈旧会警告，不能替代新审计。

源目录拒绝符号链接，输出禁止与输入重叠。完整 SHA 比对只覆盖上述四棵源目录；外部场景、网格、目标目录依赖由现有输入验证器检查绑定。
审计期间应保持源文件静止；若前后文件内容或文件集合不同，本次审计失败。索引模式成功退出只表示清单构建成功，其 summary 的 `passed` 仍为 false。

## 首次验收记录（2026-11-11）

已在 Habitat 环境执行完整 CPU 审计，产物位于 `outputs/benchmark_audit_001/`：

- 10 基础任务、70 配置、220 episode；三类输入检查通过，220/220 episode 验收通过。
- Follow 共 70/70 成功，Ignore 共 70/70 按预期失败；另有 70 个 Control 和 10 个参考成功。
- 四个源目录共 67,961 个文件，前后文件集合及 SHA256 全部一致；没有重跑导航或改写旧报告。
- 子目标旧 summary 仍为失败，`task_004_middle` 原返回码仍为 1；新审计保留 step 64 的原生端点证明，端点误差为 0 m。
- 错误 0 项，发布缺口警告 5 项；`release_ready=false`。
- 新增 12 项回归通过；全量 283 项在 Habitat 环境全部通过，核心环境通过并按依赖条件跳过 74 项。

## 后续步骤与边界

当前仍有输入权限未冻结、禁区可观察性未定义、触发进度口径不同、缺少跨场景划分、人工语言/视频审核未完整登记、
禁区/子目标共同效率指标未定义等问题。报告明确 `release_ready=false`，不把代表帧浏览算作全量人工通过。

下一完整步骤是路线中的 **B1：第一版协议与输入权限表**：

1. 以前述清单及 `issues` 为依据，分别冻结三类更新、动作/完成条件、触发、预算和 STOP 定义。
2. 明确公开语言/观测、Oracle 特权和私有评价真值，先解决禁区如何被普通 Agent 获知。
3. 增加契约回归，核对公开输入不含未授权坐标/结构化答案，旧 Oracle 资产仍可用于可行性检查。

这些 B1 产物与公开导出入口尚未实现，当前没有对应执行命令；本次清单不自动启动新采集、模型或仿真。
