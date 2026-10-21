# 独立新目标与早、中、晚介入

这一阶段使用同一个起点、原目标 A 和独立新目标 B，分别在早期、中期、晚期替换目标。
每个时机都运行无介入和介入两个分支，一共 6 个 episode。
仍使用 Oracle 控制器验证导航和任务语义，不评价语言理解。

## 1. 准备三个配置

```bash
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat
cd /mnt/sdc/weiguanzhao/MIVLN-W

python -m scripts.intervention_timing prepare \
  --reference-dir outputs/intervention_001/control \
  --output-dir outputs/intervention_timing_001
```

准备过程只加载导航网格，不启动渲染或占用 GPU。
它使用固定采样种子 123，生成一个 B 点，并把坐标、筛选距离、采样次数、
参考轨迹和导航网格的 SHA256、软件版本写入 `manifest.json`。

B 点满足以下过滤条件：

- 高度与起点相差不超过 0.5 米。
- 到起点、A 和三个介入位置的导航距离均在 2～20 米之间。
- 到原轨迹已记录位姿的最小三维欧氏距离至少为 1 米。
- 三组使用同一 B 坐标。

最后一项几何过滤针对已保存的离散位姿，不代表整条连续路径的精确最小距离。
可达性也不保证离散动作 Oracle 一定成功；实际规划失败会保留为失败结果。

介入时机根据成功的无介入参考轨迹选择：

```text
progress = 1 - 当前到 A 的导航距离 / 初始到 A 的导航距离
```

第一次达到 25%、50%、75% 的状态分别对应 early、middle、late。
事件在该状态之后的下一个动作选择前生效。原地转向不会增加此进度。
准备脚本拒绝已 STOP、步号不连续或无法提供三个独立时机的参考轨迹。
当前已验证参考轨迹对应动作 15、26、37 前介入，实际进度约为 26.7%、51.1%、75.5%。
它们是相对目标距离的进度，不能解释为固定的动作数百分比或所有场景中的临近到达状态。

生成文件：

```text
outputs/intervention_timing_001/
├── manifest.json
└── configs/
    ├── early.json
    ├── middle.json
    └── late.json
```

配置完整保存起点、朝向、目标坐标和事件，可以单独复现。

## 2. 批量运行

```bash
python -m scripts.intervention_timing run \
  --output-dir outputs/intervention_timing_001 \
  --gpu-id 0 \
  --no-video
```

将 GPU 编号改为实际分配的设备。脚本依次调用现有配对实验入口，保存每组的
`pair_summary.json`、事件、轨迹、RGB-D、位姿和指标；某组返回失败时会继续运行其余组。
随后重新从保存的轨迹核对配对条件，汇总到 `summary.json` 和 `summary.csv`。
子进程错误记录在 `run_status.json` 中。

预期 `summary.json` 满足：

```json
{
  "passed": true,
  "passed_cases": 3,
  "total_cases": 3,
  "controls_identical": true,
  "failed_processes": []
}
```

每个介入分支应在 B 点附近执行 STOP，导航误差不超过 0.3 米；三组无介入轨迹应完全一致。
汇总保留各组的触发步号、实际触发进度、动作数、路径长度、介入后 SPL、误差和碰撞数。
3 组只是单任务的时机检查，不能作为总体成功率或性能提升的证据。

查看结果，或者不占 GPU 地重新汇总：

```bash
cat outputs/intervention_timing_001/summary.json

python -m scripts.intervention_timing summarize \
  --output-dir outputs/intervention_timing_001
```

## 3. 导出视频

```bash
for timing in early middle late; do
  python -m scripts.export_video \
    --episode-dir "outputs/intervention_timing_001/$timing/intervention"
done

python -m scripts.export_video \
  --episode-dir outputs/intervention_timing_001/early/control
```

三个控制组轨迹通过一致性验收后，看一段控制组视频即可；三个介入视频都需要人工检查。
如果运行时去掉 `--no-video`，则自动导出全部 6 段视频。

## 重跑与单组复现

准备和批量运行均拒绝覆盖已有实验结果。需要重新运行时，重新 prepare 到一个新目录，
例如 `outputs/intervention_timing_002`，再对该目录执行 run。
仅复现某组时，可以使用已保存配置，输出到一个新的目录：

```bash
python -m scripts.run_intervention \
  --config outputs/intervention_timing_001/configs/middle.json \
  --gpu-id 0 \
  --output-dir outputs/intervention_middle_repeat \
  --no-video
```

修改采样种子可以产生其他候选 B 点，但生成后应人工检查轨迹和画面。
三组时机均验证通过后，再扩展到少量不同起点和目标组合。
