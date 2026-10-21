# 10～20 个目标替换调试任务

本阶段在已安装的官方测试场景内改变起点、初始朝向、原目标 A 和替换目标 B，
检验采样、导航、事件投递和评测是否稳定。控制器仍直接读取结构化目标，
结果只代表 Oracle 调试任务的验收，不表示语言理解或跨场景泛化能力。

## 1. 生成固定任务

```bash
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat
cd /mnt/sdc/weiguanzhao/MIVLN-W

python -m scripts.debug_dataset prepare \
  --count 10 \
  --seed 20261021 \
  --output-dir outputs/debug_dataset_001
```

此步骤只使用 CPU 导航网格查询。默认筛选规则：

- 起点到 A 的导航距离为 6～12 米；起点到 B、A 到 B 为 2～20 米。
- A、B 与起点的高度差不超过 0.5 米。
- B 到起点至 A 的参考路径折线至少相距 2 米，使用连续线段上的欧氏距离。
- 新任务的三个点与已接受任务的点至少相距 1 米。
- 每个任务使用固定采样种子和初始朝向；坐标与采样次数全部保存。

几何筛选完成后，才写入 `manifest.json` 和 `configs/task_001/`～`configs/task_010/`。
`base.json` 中的事件步号是待参考轨迹替换的占位值，不应直接拿来执行正式介入实验。
`reference.json` 没有事件，用于首次参考导航。
增加 `--count 20` 可以生成 20 个任务，需使用新的输出目录。

## 2. 批量运行

```bash
python -m scripts.debug_dataset run \
  --output-dir outputs/debug_dataset_001 \
  --gpu-id 0
```

将 GPU 编号改为实际分配的设备。每个任务依次执行：

1. 运行无介入参考导航。
2. 按到 A 的剩余导航距离减少 25%、50%、75% 选择三个介入时机。
3. 保留预先采样的 B，检查从三个实际介入位置到 B 的可达性与距离。
4. 执行早、中、晚三组配对实验，每组包含控制和介入两个分支。

10 个任务最多运行 70 个 episode：10 条参考轨迹，加上 30 组配对实验。
运行保存 RGB-D、事件、位姿、目标状态和指标，视频稍后单独导出。
各任务的 `reference.log`、`timing_prepare.log`、`timing_run.log` 保存完整子进程日志。

参考导航失败时，该任务停止后续阶段并记录失败，然后继续下一个任务。
固定 B 未通过实际参考轨迹筛选或介入规划失败时，同样保留失败记录。
运行过程中不会因为失败而替换任务或重新采样目标。

## 3. 汇总与验收

```bash
python -m scripts.debug_dataset summarize \
  --output-dir outputs/debug_dataset_001
```

终端显示通过任务数、配对数和失败任务 ID。完整结果在：

- `summary.json`：逐任务状态、失败阶段、三个时机的指标。
- `cases.csv`：每个任务、每个时机一行，共 30 行数据。
- `run_status.json`：各执行阶段的退出码与日志路径。
- `tasks/task_001/timing/early/pair_summary.json`：单组配对检查。

完整通过时，`passed` 为 true、`passed_tasks` 为 10、`total_tasks` 为 10、
`passed_pairs` 为 30、`total_pairs` 为 30，且 `failed_tasks` 为空。
还会检查每个任务三组控制轨迹与参考轨迹一致，所有分支保留同一起点、朝向、目标和动作参数。
失败或缺失任务仍占据总数，不会从分母中删除。

可达性不保证离散动作 Oracle 总能成功。出现失败时，应依据日志、轨迹与视频区分
几何筛选、初始化、规划、超时和评测问题，再决定如何修正规则。

## 4. 导出并人工复核视频

```bash
python -m scripts.debug_dataset videos \
  --output-dir outputs/debug_dataset_001
```

每个任务导出一段参考导航和三段介入视频，10 个任务最多 40 段。
检查起点和目标是否合理、介入前连续性、转向和碰撞、最终 STOP 位置。
也可以先查看一个任务：

```bash
python -m scripts.debug_dataset videos \
  --output-dir outputs/debug_dataset_001 \
  --task-id task_001
```

已有视频会跳过，导出错误写入 `video_status.json`。

## 目录与重跑

```text
outputs/debug_dataset_001/
├── manifest.json
├── configs/task_001/{base,reference}.json
├── run_status.json
├── summary.json
├── cases.csv
└── tasks/task_001/
    ├── reference.log
    ├── reference/
    ├── timing_prepare.log
    ├── timing_run.log
    └── timing/
        ├── manifest.json
        ├── configs/{early,middle,late}.json
        ├── summary.json
        └── {early,middle,late}/{control,intervention}/
```

准备和运行均拒绝覆盖已有任务输出。需要完整重跑时，使用同一 seed，
重新 prepare 到 `outputs/debug_dataset_002`，然后对该目录执行 run。
排查单个时机时，复用保存的配置并选择新的输出目录，例如：

```bash
python -m scripts.run_intervention \
  --config outputs/debug_dataset_001/tasks/task_003/timing/configs/middle.json \
  --gpu-id 0 \
  --output-dir outputs/debug_task003_middle_repeat \
  --no-video
```

单组复现命令要求该任务已完成配置生成；若参考导航失败，应先复现其
`configs/task_003/reference.json`，使用 `scripts.run_navigation` 并指定新的输出目录。
