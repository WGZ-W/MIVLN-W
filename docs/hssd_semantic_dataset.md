# HSSD 多任务语义目标替换实验

## 目标与当前边界

在同一个 HSSD 场景中构建 10 个固定基础任务，改变起点、朝向和具体物体配对。
每个任务包括 1 条无事件参考、早/中/晚三组 control/follow/ignore，共 10 个 episode；整批计划 100 个。
这是带特权坐标、网格与结构化更新的 Oracle 调试集，不是语言理解或跨场景泛化测试。

新入口分为 `scripts.hssd_targets`（候选、预览、审核）和 `scripts.hssd_dataset`（任务采样、运行、汇总、视频）。
不会随机采样 A/B 坐标后仍沿用旧物体标签，也不会在 GPU 失败后换目标或重新采样任务。

## 1. 准备目标目录：CPU

使用已有 HSSD smoke 输出和导航配置：

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python -m scripts.hssd_targets prepare --output-dir outputs/hssd_catalog_001
```

本轮实现期间已执行此 CPU 命令，生成了下面四个**待审核**候选；现有目录不要重复覆盖：

| 目标 ID | 源实例索引 | metadata 名称 |
| --- | ---: | --- |
| `target_018` | 18 | Dawson Full Panel Bed, White |
| `target_020` | 20 | Brushed Gold Full Metal Bed |
| `target_054` | 54 | Magnolia Home Sinclair Luxe Sofa By Joanna Gaines, Midnight |
| `target_114` | 114 | Haskell California King Panel Bed |

目录使用模板与源实例索引标识具体物体，不依赖不稳定的运行时 ID。
类别冲突和无法唯一映射的实例写入 `rejected`；停止点从世界 AABB 外侧采样，检查网格、楼层和可达性。
这些仅是几何候选：接近 AABB 不等于隔墙可见，也不等于真实物体表面距离。

## 2. 生成四个目标的预览：GPU，由使用者执行

```bash
python -m scripts.hssd_targets preview \
  --catalog outputs/hssd_catalog_001/catalog.json \
  --output-dir outputs/hssd_catalog_preview_001 \
  --gpu-id 0
```

使用分配的 GPU。此步骤有意将机器人直接放到候选点，只用于离线检查，不是导航实验。
输出各目标子目录中的 `rgb.png`、`depth.npy`、`semantic.npy`、`category_mask.png`，以及 `preview.json`。
相机保持零滚转并朝向物体；报告保存相机姿态和文件哈希。程序不会自动标记审核通过。

人工检查每个 RGB 是否确实显示指定物体、是否被遮挡、停止点是否合理，
并选择能从视觉证据识别的唯一语言描述。不要直接以商品名称或类别像素数代替语义审核。
床类别 ID 27 被多个物体共享，类别掩码不是实例真值。

## 3. 显式审核：CPU，必须先看图

下面只演示一个已观察目标的审核命令，**不是预先批准所有候选**。description 应依据本次实际预览填写：

```bash
python -m scripts.hssd_targets review \
  --catalog outputs/hssd_catalog_001/catalog.json \
  --previews outputs/hssd_catalog_preview_001/preview.json \
  --target-id target_020 --decision approve \
  --description "the gold metal bed" \
  --reviewer "YOUR_REVIEWER_NAME" --note "Describe the actual visual checks here" \
  --output outputs/hssd_catalog_001/reviewed_1.json
```

继续审核时，输入上一次生成的 `reviewed_1.json`，输出新的 `reviewed_2.json`，以此类推。
不合格目标使用 `--decision reject` 并说明原因；已经做出的决定和原目录不覆盖。
审核结论保存在新 catalog 的 `review` 中；原始 `preview.json` 的 `manually_verified: false` 不会被改写。
至少 3 个不同实例、不同且明确的描述获批后，才允许生成正式任务；建议四个目标都完成审核。
不能仅修改原始 `targets.json` 的一个布尔值绕过证据检查。

例如最终版本叫 `reviewed_4.json`，检查：

```bash
python -m scripts.hssd_targets validate --catalog outputs/hssd_catalog_001/reviewed_4.json
```

预期输出审核通过的目标数；原始输入、停止点或预览文件改变都会使校验失败。
若目标被拒绝或描述不唯一，先修订候选并重新预览/审核，不降低门槛强行凑数。

## 4. 冻结 10 个任务：CPU，审核完成后才能执行

```bash
python -m scripts.hssd_dataset prepare \
  --catalog outputs/hssd_catalog_001/reviewed_4.json \
  --count 10 --seed 20261026 \
  --output-dir outputs/hssd_dataset_001
```

将 catalog 路径替换为真实的最终审核版本。任务按不同无序物体配对及其反向顺序均衡采样；
10 个任务可以覆盖最多 5 种配对的双向变化，至少需要 3 个目标。
`--count` 必须是至少 6 的偶数，以保证至少三种配对和反向覆盖。

筛选规则：起点到 A 为 4～15 m，到 B 为 2～20 m，A/B 间距超过 2 m 且不超过 20 m；
B 到参考最短路径折线至少 1 m，不同起点至少相距 1 m，同层高度差不超过 0.5 m。
任务清单只有在全部任务采样成功后才写出；某个配对不可行时报告失败，不悄悄替换。

输出 `manifest.json`、`configs/task_*/base.json` 和 `reference.json`。
base 中的事件步号 1 是占位符，不应直接用于正式实验；运行器会先跑参考，再根据真实距离进度生成三档事件。
清单保存目标审核快照、配置哈希、采样种子和几何统计。冻结之后不要手动改配置。

## 5. 批量执行：GPU，由使用者执行

```bash
python -m scripts.hssd_dataset run --output-dir outputs/hssd_dataset_001 --gpu-id 0
```

每个任务先执行参考导航，然后用现有 timing 流程与 `--keep-target` 生成约 25%/50%/75% 的事件。
各时机分别跑 control/follow 配对，再跑 Ignore。所有分支复用相同场景、网格、物体与停止点。

参考失败则保留该任务并继续下一个；某一时机失败不替换任务，也不阻止其他时机运行。
若 follow 失败但 control 成功，仍可检查 Ignore；control 不成功时，Ignore 明确标记未运行。
规划错误、未触发、超时、缺失结果均保留在计划分母中。

运行日志位于 `tasks/task_*/`，三条件数据位于 `timing/<early|middle|late>/<control|intervention|ignore>/`。
已有日志、部分运行结果或 `run_status.json` 时拒绝完整重跑；需要重跑时用同一 seed prepare 到新目录。
当前未实现断点续跑，不要删除失败任务来改变分母。

## 6. 汇总与视频：CPU

```bash
python -m scripts.hssd_dataset summarize --output-dir outputs/hssd_dataset_001
python -m scripts.hssd_dataset videos --output-dir outputs/hssd_dataset_001 --task-id task_001
```

完整自动验收目标：

- `passed_tasks: 10 / total_tasks: 10`；`passed_pairs: 30 / total_pairs: 30`。
- `passed_negative_controls: 30 / total_negative_controls: 30`。
- `observed_episode_metrics: 100`；成功 control/follow 停在正确目标，Ignore 按预期失败。
- 原始 RGB-D/语义观测前缀匹配；Ignore 与 control 的完整物理轨迹及观测匹配。
- 任务配置、目标实例、事件含义和参考推导的触发步号不变，STOP 和轨迹记录正确。

`summary.json` 包含逐任务失败、分条件统计和 `by_timing`；`cases.csv` 为每任务每时机每条件一行，
10 个任务固定 90 行数据。指标均值报告有效样本数；缺失结果不从成功率分母删除。
Ignore 的 `dynamic_success_rate: 0` 是正常预期，不应把它与负对照验收失败混为一谈。

视频命令每任务导出 1 条参考、3 条 follow、3 条 Ignore（需要 FFmpeg）；不重复导出相同控制轨迹。
自动验收不替代人工视频复核，尤其要检查新目标、介入附近的转向、碰撞标记和最终 STOP。

## 验证与 Git

```bash
python -m unittest discover -s tests -v
```

测试使用临时合成审核材料与 CPU fake simulator，不创建 GPU 模拟器。
最新扩展目录有 12 个候选，5 个不同类别实例经显式审核后，已完成真实 10 任务、100 episode 的点协议验证。
代码/测试与文档可以作为独立提交；自动通过不代表完整人工视频审核或终点可见性全部通过。
`data/` 与 `outputs/` 继续忽略；正式发布前再将审核后的必要配置和结果摘要整理到版本化目录。

本批真实 10 任务的点协议现已通过；3 条 Follow 的终帧没有目标，不能据此宣称所有目标在 STOP 时可见。
保留旧结果，后续独立派生终点朝向实验，见 [终点朝向与实例可见性](hssd_terminal_facing.md)。
