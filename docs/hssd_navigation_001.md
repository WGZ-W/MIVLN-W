# HSSD 首个实际导航：到床 A 停止

## 目的与范围

在已检查的 HSSD 场景 `102344280` 中，从原始起点实际移动到床 A 附近的可达停止点。
这一步验证完整家具场景下的运动、碰撞过滤、Oracle 跟随与 STOP，不再使用直接放置到目标的近景替代导航。

配置为 [`configs/simulator/hssd_navigation_001.json`](../configs/simulator/hssd_navigation_001.json)。
控制器仍读取真实目标坐标和导航网格，不执行语言理解；成功只表示距配置的目标点不超过 0.3 m 并执行 STOP，
不是正式的物体可见性/朝向成功指标。最终 RGB 和语义帧应另行复核，近景时使用的向下看相机姿态不会自动套到导航中。

## 已核验输入

- 场景 dataset config 和 scene instance 都来自原始 HSSD；不是单独的房屋 GLB。
- 使用 `outputs/hssd_smoke_001/scene.navmesh`，高度 1.5 m、半径 0.1 m，包含静态家具。
- 网格 SHA256 固定为 `0c32b35b070d1f48971698a82cd608d691bc4f5a8ee62778d9c48212c8653215`。
- 起点到床 A 的几何导航距离约 7.036 m；沙发 B 约 2.309 m，两点间约 9.341 m。
- 床 A 对应源实例索引 20、模板 `7c3c32c496d7eb37286ff8ad3ddf5384ba30b2ae`；
  沙发 B 对应源实例索引 54、模板 `ed4ed55a08fac410b09e2d1f351d84893eaa7136`。
- 两处 RGB 近景已检查，物体清楚可见。原始 `targets.json` 的复核标志未改写，配置保留其来源与哈希；
  `goal_source` 是来源说明，不是运行时自动认证的可见性条件。

配置同时保留两点，当前 `events` 为空，只执行到 `bed_a` 的普通导航。
`dataset_revision` 来自此前资产检查记录，不是对全部物体资产的逐文件运行时验证；
入口会实际校验配置中声明的场景 JSON、dataset config 和网格哈希。

## 运行命令

由使用者在分配的 GPU 上执行：

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python -m scripts.run_navigation \
  --config configs/simulator/hssd_navigation_001.json \
  --gpu-id 0 \
  --output-dir outputs/hssd_navigation_001 \
  --no-video

cat outputs/hssd_navigation_001/metrics.json

# 可选：完成后导出视频，不重新启动模拟器。
python -m scripts.export_video --episode-dir outputs/hssd_navigation_001 --fps 10
```

输出目录必须为空或不存在；重跑换新目录。不要通过覆盖旧网格或随意改哈希绕过输入检查。
在其他服务器复现时，先获取相同 HSSD 资产并复制已核验网格，或重新检查新网格及所有目标，再更新配置。

## 输出与验收

沿用既有 `episode.json`、`trajectory.json`、`events.json`、`oracle_path.json`、
`final_task_state.json`、`final_policy_state.json`、`metrics.json` 和 RGB-D 文件；
新增 `semantic/*.npy` 原始语义帧，逐帧路径记录在 trajectory 的 `frames.semantic` 中。

`episode.json` 保存解析后的输入路径、实际文件哈希、网格尺寸与家具加载数量；
`metrics.json` 同时保存 `scene_provenance`、`navmesh_settings` 和 `loaded_rigid_objects`。
这些是带特权信息的诊断日志，不应直接作为未来语言策略的输入。

通过条件：

1. `success` 和 `dynamic_success` 为 `true`，`stopped` 为 `true`。
2. `termination` 为 `stopped`，`final_goal` 与 `stopped_goal` 均为 `bed_a`。
3. `navigation_error_m <= 0.3`，最后一个轨迹动作为 STOP，未超出 500 动作预算。
4. `loaded_rigid_objects` 为 134，网格包含静态物体，尺寸与配置一致。
5. `delivered_event_count` 为 0，`planner_error` 为 null；核查碰撞标记及最终 RGB。

床类别 ID 27 在此场景中对应多个物体，不能以最终语义图存在像素 27 单独证明到达指定床。
图像不理想时应区分“目标点到达成功”和“目标视角不理想”，不要修改评价真值以隐藏失败。

## 入口适配与兼容性

旧 GLB 配置不需要修改。HSSD 新增字段：

- `scene`：`scenes/<scene_id>.scene_instance.json` 的路径，用于来源校验。
- `scene_dataset_config`、`scene_id`：完整场景加载入口。
- `navmesh`、`navmesh_sha256`、`agent_height`、`agent_radius`：HSSD 必填，拒绝网格回退、尺寸不匹配或未包含静态家具的网格。
- `scene_sha256`、`scene_dataset_config_sha256`：可选输入哈希；本例均已固定。
- `semantic_sensor: true`：可选，保存原始语义帧；旧配置默认仍为 RGB-D。

`run_intervention` 和 `run_ignore_intervention` 复用同一个导航循环，后续可使用同样的 HSSD 字段。
负对照和 timing 的参考轨迹检查也会比较场景、网格、尺寸及声明的哈希，避免误用旧控制组。
HSSD timing 准备必须使用 `--keep-target`，禁止随机新坐标继承原语义目标标签；
当前 `debug_dataset` 的随机坐标采样流程不作为语义数据生成器使用。

当前只提供无介入导航配置。实际参考轨迹通过后，再选择动作前的有效触发位置，生成 A→B 的事件配置，
运行匹配的 control/follow/ignore 三条件；不要直接使用未验证的固定步号宣称三条件已通过。
