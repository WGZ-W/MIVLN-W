# 单个无介入导航任务

本阶段验证固定目标下的导航、停止和评测。使用 Habitat-Sim 0.3.3 自带的
`GreedyGeodesicFollower`，读取完整导航网格和目标坐标，作为 Oracle 控制器。
任务不依赖语言理解，也不需要安装 Habitat-Lab。

## 运行

```bash
conda activate habitat
cd /remote-home/guanzhao/MIVLN-W
python -m scripts.run_navigation --gpu-id 7
cat outputs/navigation_001/metrics.json
```

起点、目标和朝向由 `configs/simulator/navigation_001.json` 固定。坐标来自已经
通过的 `habitat_smoke_test`，位于 `skokloster-castle` 官方测试场景，参考路径约
10.18 米。前进 0.25 米，转向 15 度，成功半径 0.3 米，最多 500 个动作。

每一步根据当前位置重新查询下一个动作，通过 `sim.step` 实际执行移动和转向。
跟随器返回 `STOP` 时，由 episode 循环结束任务，不调用模拟器的移动接口。
`STOP` 计入动作预算；到达成功半径但没有发出 `STOP` 仍然失败。

## 输出与验收

```text
outputs/navigation_001/
├── episode.json         # 配置、场景、控制器和依赖版本
├── oracle_path.json     # 初始最短路径和长度
├── trajectory.json      # 初始状态、动作、位姿、碰撞、目标距离及帧文件路径
├── metrics.json
├── trajectory.mp4       # RGB 视频，默认 10 帧/秒
├── rgb/                 # 初始帧及每个动作之后的 RGB
├── depth/               # 对应的 float 深度 NPY，单位米
└── depth_preview/       # 固定 0～10 米映射的深度预览 PNG
```

- `success`、`stopped` 均为 `true`；
- `termination` 为 `stopped`；
- `navigation_error_m <= success_radius_m`；
- 最后一个动作是 `STOP`，动作数不超过 `max_steps`；
- 查看 RGB 帧，确认画面正常。

`trajectory.json` 的初始状态没有动作步号（`step: null`），动作从第 0 步开始。
第 `k` 步后的帧编号为 `k + 1`。`STOP` 帧保留同一物理状态。

路径长度是相邻实际位姿间的三维欧氏距离之和，原地转向和 `STOP` 不增加路程。
导航误差使用最终位置到目标的导航网格最短距离，失去可达路径时记为 `null`
并判定失败。

单 episode 的 `success_rate` 为 0 或 1。
`SPL = success × L / max(L, P)`，其中 `L` 为初始最短路径长度，`P` 为实际路程。
规划器错误和超时均保留已执行轨迹与失败指标，进程返回非零状态。
报告的碰撞来自运动过滤器，不等同于验证了真实机器人外形的碰撞行为。

同一路径中的历史输出不会被覆盖。复现时使用新的输出目录：

```bash
python -m scripts.run_navigation --gpu-id 7 --output-dir outputs/navigation_001_repeat
```

比较两次的 `trajectory.json` 和 `metrics.json`，在相同软件、配置和场景下应一致。

## 视频导出

导航脚本默认在运行结束后调用 `ffmpeg`，把保存的 RGB 帧合成为 H.264 MP4。
当前服务器已有 `/usr/bin/ffmpeg`。在其他环境中，需要先确保 `ffmpeg` 位于 PATH 中。
编码在模拟器关闭后进行，规划器错误或超时产生的轨迹也会导出，便于检查失败原因。

已有结果可以直接生成视频，无需启动 Habitat 或占用 GPU：

```bash
python -m scripts.export_video --episode-dir outputs/navigation_001
```

默认保存为 `outputs/navigation_001/trajectory.mp4`，10 帧/秒。
每个动作对应一帧，另含初始帧，50 帧约为 5 秒；这是动作序列的回放速度，
不代表仿真或实际机器人执行时间。深度数据仍保存在逐帧 NPY 和 PNG 中。

新导航任务可通过 `--video-fps 5` 调整回放速度，或用 `--no-video` 仅保存原始帧。
单独导出时可指定新文件名，已有视频不会覆盖：

```bash
python -m scripts.export_video --episode-dir outputs/navigation_001 \
  --fps 5 --output outputs/navigation_001/trajectory_5fps.mp4
```

## 后续边界

当前 MIVLN 包和仿真脚本均支持 Python >=3.9，可以共用现有 `habitat` 环境。
单次目标替换与无介入配对实验见 [intervention_001.md](intervention_001.md)。
固定坐标测试通过后，再为 HSSD 场景设置有明确语义的房间或物体目标。
