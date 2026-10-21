# 单次目标替换与无介入对照实验

本实验复用已验证的 `skokloster-castle` 官方测试场景、起点、朝向和导航参数。
控制器是读取目标坐标和结构化更新的 Oracle，验证事件时序、连续导航和几何评测。
自然语言指令会保存，但本实验没有实现语言理解，也没有把测试点解释成厨房或书房。

## 实验定义

- 目标 A：上一阶段固定导航的终点。
- 目标 B：初始出发位置，含义是“取消前往 A，返回起点”。
- 无介入分支 `control`：一直前往 A，最后执行 STOP。
- 介入分支 `intervention`：执行动作 0～19，随后在动作 20 前替换目标，返回 B 后 STOP。
- 两个分支分别创建一次模拟器，起点、朝向、场景、seed 和动作参数一致。
- 介入分支内部始终保留同一个模拟器和机器人；切换目标只更新任务状态与规划器。

配置位于 `configs/simulator/intervention_001.json`，不需要下载新场景。
B 与 A 的可达性以及成功区域不重叠会在运行时验证。

## 执行命令

在这台新服务器上执行：

```bash
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat
cd /mnt/sdc/weiguanzhao/MIVLN-W

python -m unittest discover -s tests -v

python -m scripts.run_intervention \
  --gpu-id 0 \
  --output-dir outputs/intervention_001 \
  --no-video

cat outputs/intervention_001/pair_summary.json
```

GPU 参数使用实际分配的编号。当前代码支持 Python >=3.9，可以直接在已配置的
Python 3.9 Habitat 环境中同时导入 MIVLN 和 Habitat；从项目根目录运行无需 pip 安装。
依赖齐全的 Habitat 环境中应全部测试通过，仿真循环测试使用替身模拟器，不启动 GPU 渲染。
没有 Habitat 的核心环境会跳过依赖仿真器导入的测试，其余测试仍可运行。

`pair_summary.json` 中 `passed: true` 表示自动验收通过。
即使某个分支规划失败或超时，脚本也会保存该分支的轨迹和指标，并继续产生配对报告；
无效配置、场景初始化错误等异常则会直接中止。
验收失败时脚本返回非零退出码。

再导出两段视频：

```bash
python -m scripts.export_video --episode-dir outputs/intervention_001/control
python -m scripts.export_video --episode-dir outputs/intervention_001/intervention
```

人工检查两段视频：开始的 20 个动作一致，介入组随后改变朝向并返回起点。
默认视频为 10 FPS；介入发生在第 20 个动作之后，约 2 秒处，回放速度不是物理时间。
不带 `--no-video` 运行时会自动生成两段视频。

已有输出不会覆盖。重跑时更换 `--output-dir`，例如 `outputs/intervention_002`；
配置中的 episode ID 不会因输出目录改名而改变。

## 输出

```text
outputs/intervention_001/
├── pair_summary.json
├── control/
└── intervention/
    ├── episode.json           # 场景、输入指令、结构化目标、事件、软件版本
    ├── oracle_path.json       # 起点到 A 的参考路径
    ├── replans.json           # 介入时从当前位置到 B 的参考路径
    ├── events.json            # 投递步号、事件前后位姿、任务状态变化
    ├── final_task_state.json
    ├── trajectory.json        # 每步动作、实际位姿、有效目标、任务状态及帧路径
    ├── metrics.json
    ├── trajectory.mp4         # 导出视频后生成
    ├── rgb/
    ├── depth/
    └── depth_preview/
```

`control/` 的文件布局相同，但事件和重规划列表为空。
轨迹首项是 `INITIALIZE`，其 `step` 为 null；下标 1 对应动作 0。
因此在第 20 步动作前介入时，两个分支的前 21 条轨迹记录应完全一致。

## 自动验收与指标

配对检查覆盖：两组成功停止、事件在指定步投递一次、介入前完整轨迹记录一致、
事件前后位姿一致且等于上一帧、任务状态从 A 更新到 B 并取消 A、
更新在动作选择前生效，以及控制组没有收到事件。

几何评测先计算最终位姿到每个目标的导航网格最短距离，再确定 STOP 所处的目标。
到最终有效目标的距离必须不超过 0.3 米；仅记录 `active_goal = point_b` 不算成功。
停在 A、未 STOP、超时或新目标不可达都不能使介入组通过验收。

无介入组使用普通 `spl`。介入组的 `spl` 为 null，因为原始 A 路径不再是有效的
完整任务参考；另报 `post_intervention_spl`，使用介入位置到 B 的最短路径和
介入后的实际路径长度计算，仅评价切换后的导航效率。

`response_latency_steps` 表示 Oracle 第一次以新目标规划动作的延迟，预期为 0；
这个数值不表示机器人已经发生位移，也不衡量语言理解能力。

通过后，按照 [intervention_timing.md](intervention_timing.md) 生成独立 B 点，
比较早、中、晚三个介入时机，再扩展为少量调试样例。
具有房间或物体含义的任务需要后续接入带语义信息的场景。
