# MIVLN 环境安装、运行与迁移命令

整理日期：2026-10-21。适用目标：Linux x86_64、NVIDIA GPU、Bash、Miniconda。

后续代码已验证支持 Python >=3.9，任务逻辑与仿真可以共用 Python 3.9 的
`habitat` 环境。单次目标替换与对照实验见 [intervention_001.md](intervention_001.md)。
本文中的双环境安装过程保留为原机器复现记录。

本文根据本次操作记录、Conda 安装历史、现有脚本和实际运行结果整理。
包含已执行的主要安装和运行命令，以及迁移时补充的源码传输、FFmpeg 安装和环境归档命令。
它是一份按执行顺序排列的操作手册，不是完整的终端历史，也不需要整篇一次性执行。
“可选”“排查”“补充”小节只在对应情况下使用。

## 1. 已验证的环境与结果

| 项目 | 原机器实际配置 |
|---|---|
| 系统 | Ubuntu 20.04.6 LTS，x86_64 |
| GPU | NVIDIA RTX A6000，48 GB |
| NVIDIA 驱动 | 550.54.14 |
| `nvidia-smi` 显示的 CUDA 兼容版本 | 12.4，不代表另行安装了 CUDA Toolkit |
| Conda | 26.9.3 |
| `mivln` 环境 | Python 3.10.22；用于单元测试、任务状态和文本 demo |
| `habitat` 环境 | Python 3.9.23；用于独立仿真脚本 |
| Habitat-Sim | 0.3.3，headless + Bullet |
| Habitat-Sim Conda 构建 | `py3.9_headless_bullet_linux_acbe6f4922e68145e401e55c30f9dfea460a3f24` |
| NumPy（habitat 环境） | 1.26.4 |
| FFmpeg | 系统 `/usr/bin/ffmpeg`，4.2.7，支持 libx264 |
| 原机器项目路径 | `/remote-home/guanzhao/MIVLN-W` |
| 原机器使用的 GPU | 第 8 张卡，编号 7 |

已确认的结果：7 项单元测试通过；目标替换 demo 成功；官方场景 RGB-D、动作和最短路径验证成功；
无介入导航在 49 个动作内停止，成功率 1.0、SPL 1.0、导航误差约 0.2683 米。
视频编码已用临时图像验证；是否已有正式导航视频，以对应输出目录为准。

另一台机器不必使用相同型号的显卡或相同用户名，但应具有可用的 NVIDIA 驱动和 EGL 渲染环境。
当前预编译 Habitat 环境针对 Linux x86_64；其他系统或架构需要单独适配。

## 2. 检查目标机器

```bash
cat /etc/os-release
uname -m
nvidia-smi
```

如果已经安装 Conda：

```bash
conda --version
conda env list
python --version
```

GPU 编号从 0 开始。新机器只有一张 GPU 时使用 0；原机器的 `--gpu-id 7` 不能直接照搬。

## 3. 安装和启用 Miniconda（迁移补充，已安装则跳过安装）

原机器在本次任务开始前已安装 Miniconda，下面是新机器的安装方法：

```bash
cd /tmp
curl -fLO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda --version
```

若安装到了其他目录，修改 `source` 的路径。以后每次打开新终端，也可以执行这个 `source` 命令。
可选地初始化 Bash，使新终端自动识别 Conda：

```bash
conda init bash
```

执行后重新打开终端。不要在已有的 Miniconda 目录上重复运行安装命令。

## 4. 迁移项目源码

### 4.1 当前工作区打包（迁移补充，适用于尚未提交的改动）

整理本文时，新增的仿真脚本、配置和部分文档尚未提交到远端。
只克隆远端仓库可能缺少这些文件，因此当前可以打包工作区源码。

在原机器执行，先把路径改为原项目位置：

```bash
MIVLN_SOURCE_DIR=/remote-home/guanzhao/MIVLN-W
tar --exclude='./.git' --exclude='./data' --exclude='./outputs' \
  --exclude='./.venv' --exclude='./venv' --exclude='__pycache__' \
  --exclude='./.aws' --exclude='./.codex' --exclude='./.agents' \
  -czf /tmp/MIVLN-W-source.tar.gz -C "$MIVLN_SOURCE_DIR" .
```

通过你使用的文件传输工具，把 `/tmp/MIVLN-W-source.tar.gz` 放到新机器的 `/tmp/`。
归档包含本地未提交的源文件，不包含场景资产、实验输出或 Conda 环境。

在新机器执行，项目位置可以自行修改：

```bash
MIVLN_PROJECT_DIR="$HOME/projects/MIVLN-W"
mkdir -p "$MIVLN_PROJECT_DIR"
tar -xzf /tmp/MIVLN-W-source.tar.gz -C "$MIVLN_PROJECT_DIR"
cd "$MIVLN_PROJECT_DIR"
```

### 4.2 从 Git 获取（可选，仅当所需文件已推送到远端）

与上面的源码归档方式二选一，使用一个新的目标目录：

```bash
MIVLN_PROJECT_DIR="$HOME/projects/MIVLN-W"
mkdir -p "$HOME/projects"
git clone https://github.com/WGZ-W/MIVLN-W.git "$MIVLN_PROJECT_DIR"
cd "$MIVLN_PROJECT_DIR"
```

### 4.3 检查必要文件并选择 GPU

```bash
ls scripts/run_demo.py scripts/check_habitat.py \
  scripts/run_navigation.py scripts/export_video.py
ls configs/demo_episode.json configs/task_cases_v0.json \
  configs/simulator/minimal.json configs/simulator/navigation_001.json

MIVLN_GPU_ID=0
nvidia-smi -i "$MIVLN_GPU_ID"
```

将 `MIVLN_GPU_ID` 改为分配给你的空闲 GPU 编号。原机器用的是 7。
后面的命令均从项目根目录运行。每次打开新终端，重新进入项目并设置这个变量。
此处通过脚本参数选卡，不额外设置 `CUDA_VISIBLE_DEVICES`，避免设备编号发生重映射。

## 5. 创建 mivln 环境并验证核心逻辑（已执行）

原安装命令是 `conda create -n mivln python=3.10`。
迁移时可以锁定原机器实际使用的补丁版本：

```bash
conda create -n mivln python=3.10.22 -y
conda activate mivln
python --version

python -m unittest discover -s tests -v
python -m scripts.run_demo
cat outputs/demo_001/metrics.json
```

当前核心代码没有第三方依赖，从项目根目录执行不需要额外安装项目包。
预期单元测试显示 `Ran 7 tests` 和 `OK`，demo 输出满足：

```json
{
  "dynamic_success": true,
  "intervention_received": true,
  "goal_update_compliance": true,
  "final_goal": "bedroom",
  "stopped_goal": "bedroom"
}
```

demo 还会输出响应延迟、路径长度和步数，并写入：

```text
outputs/demo_001/
├── episode.json
├── events.json
├── trajectory.json
└── metrics.json
```

已手工定义的 5 条任务保存在 `configs/task_cases_v0.json`。
下面的命令可补充检查 JSON 格式；当前 demo 不会批量运行这 5 条任务：

```bash
python -m json.tool configs/task_cases_v0.json > /dev/null
```

## 6. 创建 Habitat 环境（已执行）

原机器执行过的命令如下，按顺序运行：

```bash
conda create -n habitat python=3.9 pip numpy=1.26.4 \
  -c conda-forge --override-channels -y
conda activate habitat

conda install habitat-sim=0.3.3 headless withbullet numpy=1.26.4 \
  -c aihabitat -c conda-forge --override-channels -y
```

若需要同时锁定 Python 补丁版本，在创建命令中把 `python=3.9` 改为 `python=3.9.23`。
`headless` 使用无显示器渲染，`withbullet` 启用物理功能。
这里安装的是 aihabitat 的预编译包，不是通过 `pip install habitat-sim` 安装。

验证导入与版本：

```bash
python --version
python -c "import habitat_sim; print('Habitat-Sim:', habitat_sim.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
conda list habitat-sim
```

预期 Habitat-Sim 为 0.3.3，NumPy 为 1.26.4。
当前仅验证独立的 Habitat-Sim 脚本，没有安装 Habitat-Lab、HSSD 或训练框架。
MIVLN 包的 Python 要求现已调整为 >=3.9，并通过核心与仿真循环测试，可直接接入现有环境。

## 7. 下载官方测试场景（已执行）

确保位于项目根目录并已激活 `habitat`：

```bash
conda activate habitat
python -m habitat_sim.utils.datasets_download \
  --uids habitat_test_scenes \
  --data-path data/

ls -l data/scene_datasets/habitat-test-scenes
ls data/scene_datasets/habitat-test-scenes/skokloster-castle.glb
ls data/scene_datasets/habitat-test-scenes/skokloster-castle.navmesh
```

`data/` 被 Git 忽略，新机器需要重新下载。
原机器下载器创建的 `data/scene_datasets/habitat-test-scenes` 是指向原机器绝对路径的符号链接；
直接复制这个链接到新机器可能失效。重新下载会在新机器生成对应的数据布局。

这些测试场景没有语义标注。加载时出现 `.scn`、`info_semantic.json` 缺失及
`does not contain semantic annotations` 信息，不影响本阶段 RGB-D 与导航网格验证。

## 8. 检查 FFmpeg（视频导出需要）

原机器使用已有的系统 FFmpeg，未在 Habitat 环境中额外安装 `imageio-ffmpeg`：

```bash
command -v ffmpeg
ffmpeg -version
ffmpeg -hide_banner -encoders | grep libx264
```

若新机器没有 FFmpeg，下面是补充的 Ubuntu 安装命令，需要 sudo 权限：

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends ffmpeg
```

安装后重新执行上面的检查。没有 FFmpeg 时也能先通过 `--no-video` 运行导航。

## 9. 仿真最小验证（已执行）

```bash
conda activate habitat
python -m scripts.check_habitat --gpu-id "$MIVLN_GPU_ID"
cat outputs/habitat_smoke_test/metrics.json
```

配置文件：`configs/simulator/minimal.json`。
应看到 NVIDIA 渲染器、`success: true`、`rotation_changed: true`、有效 RGB-D 形状和非零移动距离。
本次实际移动约 2 米，RGB 为 `[256, 256, 4]`，Depth 为 `[256, 256]`。

输出目录：

```text
outputs/habitat_smoke_test/
├── rgb.png
├── depth.png
├── depth.npy
├── rgb_final.png
├── depth_final.png
├── depth_final.npy
├── config.json
├── trajectory.json
├── shortest_path.json
└── metrics.json
```

## 10. 无介入 Oracle 导航（已执行，当前脚本默认增加视频）

```bash
conda activate habitat
python -m scripts.run_navigation --gpu-id "$MIVLN_GPU_ID"
cat outputs/navigation_001/metrics.json
```

配置文件：`configs/simulator/navigation_001.json`。
固定场景、起点、朝向和目标；前进 0.25 米、转向 15 度、成功半径 0.3 米，最多 500 个动作。
使用导航网格的最短路径跟随器，不涉及语言理解。

原机器成功结果：

```json
{
  "success": true,
  "success_rate": 1.0,
  "spl": 1.0,
  "stopped": true,
  "termination": "stopped",
  "steps": 49,
  "path_length_m": 10.002098888158798,
  "shortest_path_length_m": 10.180318832397461,
  "navigation_error_m": 0.268307089805603,
  "success_radius_m": 0.3,
  "collision_count": 0,
  "planner_error": null
}
```

新机器以成功停止、导航误差不超过 0.3 米、没有规划错误为主要验收条件。
不同机器或底层库版本可能造成浮点差异，不要求跨机器 JSON 文本完全相同。

输出目录：

```text
outputs/navigation_001/
├── episode.json
├── oracle_path.json
├── trajectory.json
├── metrics.json
├── trajectory.mp4
├── rgb/
├── depth/
└── depth_preview/
```

导航不会覆盖非空的输出目录。再次运行时换一个输出目录：

```bash
python -m scripts.run_navigation --gpu-id "$MIVLN_GPU_ID" \
  --output-dir outputs/navigation_002
```

不生成视频，或调整回放帧率：

```bash
python -m scripts.run_navigation --gpu-id "$MIVLN_GPU_ID" \
  --output-dir outputs/navigation_no_video --no-video

python -m scripts.run_navigation --gpu-id "$MIVLN_GPU_ID" \
  --output-dir outputs/navigation_slow --video-fps 5
```

## 11. 相同机器上的复现比较（已给出的检查命令）

```bash
python -m scripts.run_navigation --gpu-id "$MIVLN_GPU_ID" \
  --output-dir outputs/navigation_001_repeat

diff outputs/navigation_001/metrics.json \
     outputs/navigation_001_repeat/metrics.json

diff outputs/navigation_001/trajectory.json \
     outputs/navigation_001_repeat/trajectory.json
```

在相同软件、配置和场景下，通常应一致；`diff` 无输出表示文件一致。
此比较不包含视频、硬件版本和场景绝对路径信息。

## 12. 从已有 RGB 帧导出视频（已实现）

这个步骤不加载场景，不启动仿真，也不占用 GPU。
如果自动生成的 `trajectory.mp4` 已存在，选择下面的其他输出文件名。

```bash
conda activate habitat
python -m scripts.export_video --episode-dir outputs/navigation_001
```

默认文件为 `outputs/navigation_001/trajectory.mp4`，H.264、10 帧/秒。
慢速回放并另存为新文件：

```bash
python -m scripts.export_video --episode-dir outputs/navigation_001 \
  --fps 5 --output outputs/navigation_001/trajectory_5fps.mp4
```

每帧对应一个动作之后的图像，另含初始帧，STOP 保留相同位置。
50 帧在 10 FPS 下约为 5 秒，这是动作序列的回放速度，不表示机器人的真实运动速度。
原始深度保存在以米为单位的 `.npy` 中，视频只展示 RGB。

## 13. 已遇到的问题及排查命令（按需执行）

### APT 代理连接被拒绝

原机器曾尝试下载 `nvtop` 及旧版 NVIDIA 库，但连接 `103.237.248.3:17890` 被拒绝。
原始 APT 安装命令未记录，不能把那次失败当作必需安装步骤。
当前 GPU 驱动已可用，无需为安装监控工具去安装这些旧库。

GPU 监控可以直接使用：

```bash
nvidia-smi -l 2
```

按 Ctrl+C 退出。若确实需要使用 APT，且服务器允许直接联网，可临时绕过代理更新索引：

```bash
sudo apt-get \
  -o Acquire::http::Proxy=DIRECT \
  -o Acquire::https::Proxy=DIRECT \
  update
```

这个配置只作用于本次命令。后续 APT 命令如仍需直连，也要带上对应选项。
若服务器必须经过代理联网，需要恢复有效代理；不要沿用原机器已经失效的地址。

### 新机器无法找到场景

```bash
ls -l data/scene_datasets/habitat-test-scenes
ls data/versioned_data/habitat_test_scenes/
```

检查是否复制了指向旧机器绝对路径的符号链接，并按第 7 节重新下载。

### GPU 渲染失败

```bash
nvidia-smi
nvidia-smi -i "$MIVLN_GPU_ID"
```

检查驱动是否可用，以及所选 GPU 是否存在；保留完整的 EGL/OpenGL 报错再排查。

## 14. 保存更精确的环境快照（可选的迁移补充）

上面的安装命令固定主要版本，但间接依赖仍可能随软件源变化。
如需同为 Linux x86_64 的机器重建相同 Conda 包构建，可在原机器另外导出：

```bash
mkdir -p outputs/environment_snapshot
conda list -n mivln --explicit > outputs/environment_snapshot/mivln-linux-64.explicit.txt
conda list -n habitat --explicit > outputs/environment_snapshot/habitat-linux-64.explicit.txt
conda run -n mivln python --version
conda run -n habitat python --version
ffmpeg -version > outputs/environment_snapshot/ffmpeg-version.txt
```

单独把 `outputs/environment_snapshot/` 传到新机器，因为第 4 节的源码归档不包含 `outputs/`。
使用空闲环境名恢复时，可以用下面的命令替代第 5、6 节的创建和安装命令：

```bash
conda create -n mivln --file outputs/environment_snapshot/mivln-linux-64.explicit.txt -y
conda create -n habitat --file outputs/environment_snapshot/habitat-linux-64.explicit.txt -y
```

显式清单不包含系统驱动、FFmpeg、场景资产、源码，以及通过 pip 额外安装的包。
需要分别按本文准备；也仍然需要软件源中的包可下载。
不要直接复制 Conda 环境目录到另一个路径，包内部可能包含原机器的绝对路径。

## 15. 官方参考

- [Miniconda 安装入口](https://www.anaconda.com/docs/getting-started/installation)
- [Habitat-Sim v0.3.3 安装与测试场景说明](https://github.com/facebookresearch/habitat-sim/tree/v0.3.3#installation)
- [Conda 环境管理与显式清单](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html)
- [APT 代理配置与 DIRECT](https://manpages.ubuntu.com/manpages/focal/man1/apt-transport-http.1.html)
