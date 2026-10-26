# HSSD 单场景检查

## 目的与当前状态

下载完成后，先确认一个完整的带家具场景能够加载，再准备语义导航目标。
检查脚本使用 HSSD 的 dataset config 和 scene instance，不把单独的房屋 GLB 当作完整场景。
这一步只验证资产、渲染、导航网格与语义输出，不是目标替换实验，也不证明语言理解能力。

当前数据路径为 `data/scene_datasets/hssd-hab/`，默认场景 `102344280`。
2026-10-23 的 CPU 检查中，数据版本为 `4369cb9876214c7fbebcf552eb532380e4d287e4`，
该场景的 290 个依赖文件检查通过，包含 134 个物体实例、406 个词表类别、15 条区域标注。
这是所选场景的依赖检查，不代表全数据集逐文件验证。

原始标注存在需要注意的问题：23 个实例的模板没有 `semantic_id`，
25 个实例的模板 ID 对应词表标签与 metadata 类别不同。例如，一个花瓶的模板 ID 被词表解释为 bed。
这些是标注诊断，不能直接判断为下载失败，也不能据此直接建立可靠的语义目标。
脚本不修改原始数据；metadata 类别仅作为候选标签，仍需 RGB 人工核验。

## 执行命令

在服务器中运行；GPU 实验由使用者执行。将 `--gpu-id 0` 换成分配的 GPU 编号。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

# 可选：仅 CPU 资产检查，不启动模拟器。
python -m scripts.check_hssd --assets-only --output-dir outputs/hssd_assets_001

# 实际加载完整场景，并采集固定位置的 24 个朝向。
python -m scripts.check_hssd --scene-id 102344280 --gpu-id 0 --output-dir outputs/hssd_smoke_001

cat outputs/hssd_smoke_001/metrics.json

# 可选：导出 RGB 环视视频；需要 FFmpeg，不需要再次启动 Habitat。
python -m scripts.export_video --episode-dir outputs/hssd_smoke_001 --fps 2
```

每次执行选择新的输出目录；脚本拒绝覆盖非空目录。若数据位于其他位置，使用
`--dataset-config /absolute/path/hssd-hab.scene_dataset_config.json`。

## 输出与验收

| 文件或目录 | 内容 |
| --- | --- |
| `assets.json` | 所选场景依赖、数据版本、输入哈希、缺失文件和语义诊断 |
| `config.json` | 本次场景、传感器、智能体尺寸与随机种子 |
| `source_objects.json` | 源实例索引、模板 ID、metadata 类别及原始词表类别 |
| `objects.json` | 运行时物体实例 ID、模板、位姿和变换后的世界坐标 AABB |
| `regions.json` | 数据集原始区域标注，不等同于运行时已加载的 semantic regions |
| `semantic_classes.json` | 原始类别 ID 到词表标签的映射 |
| `scene.navmesh` | 重新计算的导航网格，包含静态家具；默认高度 1.5 m、半径 0.1 m |
| `shortest_path.json` | 最大连通区域上采样的起终点及可达参考路径 |
| `rgb/`, `depth/`, `semantic/` | RGB 帧、米制深度 NPY、原始语义 ID NPY |
| `depth_preview/`, `semantic_preview/` | 可视化预览；语义颜色不是经过验证的实例或类别对应关系 |
| `trajectory.json`, `metrics.json` | 原地转向轨迹与检查结果 |

通过标准为 `assets.json` 的 `passed: true`，以及 `metrics.json` 的 `success: true`。
后者要求物体模板及实例数量匹配、导航网格有效、参考路径可达、RGB-D 有效且渲染到正值语义 ID。
这不是完整语义正确性检查：当前数据的 `semantic_labels_consistent` 预计为 `false`，
应与场景加载结果分别解读。运行时 semantic objects/regions 为空也不单独判失败，原始区域标注另行保存。

人工查看 RGB，确认房屋和家具正常显示，没有大片空白或明显缺失。
下一步结合 `objects.json` 与 RGB，选择两个类别清楚的候选物体，
在它们附近采样导航网格上的可达停止点，再复用目标替换和 Ignore Intervention 对照。
物体原点或 AABB 中心可能位于家具内部，不能直接作为导航终点；
场景、导航网格参数和目标配置也必须同时接入导航实验，不能只替换原来的场景 GLB 路径。
入口现已适配，床 A 的首次实际导航配置与运行命令见 [HSSD 首个实际导航](hssd_navigation_001.md)。

## 无 GPU 回归测试

```bash
python -m unittest discover -s tests -v
```

资产测试仅使用标准库与临时数据；Habitat 配置/几何测试不创建模拟器，缺少 Habitat 时自动跳过。
