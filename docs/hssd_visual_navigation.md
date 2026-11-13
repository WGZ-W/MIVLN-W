# Middle：真实 HSSD 三类 V 导航接线

范围：`task_001 + middle`，目标替换、插入恢复、取消三类 × `oracle`/`ignore` 两条件，共 **6 个计划 episode**。
入口：`python -B -m scripts.hssd_visual_navigation`；真实 GPU 六分支已验收，见[实测结果](hssd_v_middle_results_2026-11-13.md)。
这一步检查已审语言、真实 RGB/动作、介入和独立评价的接线；Oracle 使用正确私有更新及几何，不代表语言或视觉理解。

## 1. 固定输入和中期时机

来源为 `outputs/benchmark_language_annotations_002/` 的 40 条已批准原模板，以及
`outputs/benchmark_v_navigation_prepare_002/` 中的真实几何准备配置。
新准备包仅选择一个基础任务、三类 middle，保留原始语言、几何、位姿、种子、500 动作预算和实例审核绑定。
不修改已冻结的旧导航代码、V 循环、历史指标或语言审核。

本批场景 `102344280`，A=`target_051` 岛台，B=`target_054` 深色沙发。
middle 来自冻结无介入 A 参考累计实走路程 50% 的外生动作槽，**在动作 22 之前投递**，三类/两条件共用该步。
不是当前方法自己走过 50%，也不重置预算、位姿、轨迹或策略记忆。

## 2. 实际接入哪些内容

| 模块 | 实现 |
| --- | --- |
| `scripts/hssd_visual_backend.py` | 真实 Habitat 加载、RGB、物理动作/碰撞、网格距离、物体朝向与实例诊断；另提供无渲染 NativeReplay |
| `scripts/hssd_visual_navigation.py` | 固定 middle 准备、原生前缀预演、六分支运行和只读重审 |
| 原 `mivln/benchmark_navigation.py` | 原样复用逐步 V 会话、真值更新、测量完成和三类评价 |

每个分支初始化一次真实 Agent，物理动作由 `sim.step` 执行；STOP/COMPLETE_SUBGOAL 计步但不推进物理。
相机标定读取实际 RGB sensor 的分辨率及 HFOV，RGBA 去除 alpha 后采集为规范 PNG。
普通 V 正文仍只含初始语言、当前/历史 RGB、实际动作/碰撞和已投递消息，不含候选、坐标、真值状态、实例掩码或未来消息。

评价器在距停止点达到成功半径时，使用已有孤立实例重标记诊断保存 RGB/掩码和身份；离开半径不伪造可见性完成证据。
诊断恢复原语义标签并核对位姿不变，诊断图像/掩码不进入 Agent 输入。网格距离单位 m，四元数 XYZW，朝向误差单位度。
Oracle 根据路径及目标实例中心转向，在 B 提交中间完成或在最终目标 STOP；评价器再根据实测证据决定是否有效。
Ignore 收到同样的介入语言，策略仍前往 A；评价真值照常改变，不能将到 A 算作服从介入。

## 3. 已完成的 CPU 检查

- `outputs/hssd_v_middle_prepare_001/`：原准备包保留执行前状态；对应的六个真实 RGB 分支现已全部运行并验收。
- `outputs/hssd_v_middle_geometry_001/geometry.json`：在真实冻结网格上用无 sensor 原生 Agent 重放参考前 22 个动作，位姿和碰撞反馈一致，参考文件前后不变。
- 投递前到 A/B 的网格距离分别为 3.538 m / 3.454 m。此预演只证明参考前缀的网格/动作接线，不证明 B→A 完整路径或实例可见性。
- 新增 14 项回归使用隔离的合成数据、假渲染器和假重放器，检查完整六分支、一次初始化、不重置、标签恢复、原始输入、独立掩码审计及失败分母。
- 全量 392 项在 Habitat 环境通过；核心环境同套件通过，因原生依赖跳过 83 项。源代码哈希及旧 V 准备包保持有效。

后续真实 GPU 运行 `outputs/hssd_v_middle_run_001/` 和正式审计 `outputs/hssd_v_middle_audit_001/` 均通过：3 个 Oracle 成功，3 个 Ignore 按预期失败，配对错误 0。尚未调用视觉模型。

## 4. 已执行的命令与复现

以下真实运行及审计已完成，`_001` 目录已有有效结果，不覆盖重跑。复现时使用配置好的 `habitat` 环境和全新输出目录，并选择已分配 GPU；本次使用 0：

```bash
conda activate habitat

python -B -m scripts.hssd_visual_navigation run \
  --input-dir outputs/hssd_v_middle_prepare_001 \
  --gpu-id 0 \
  --output-dir outputs/hssd_v_middle_run_001

python -B -m scripts.hssd_visual_navigation audit \
  --input-dir outputs/hssd_v_middle_run_001 \
  --output-dir outputs/hssd_v_middle_audit_001
```

`run` 顺序启动六个子进程，每个分支单独创建/关闭 Simulator，日志保存在 `logs/`。
返回码 0 需要六个分支均按预期验收；子进程失败仍继续其余分支，缺失/失败不得从分母删除。
中断时保留 interrupted/pending 状态；不在旧目录静默重试，改用新目录。
进程崩溃且没有完整轨迹时，仅保留已有日志，不能补造请求或授予成功。

如需另开准备或复查原生前缀，现有命令是：

```bash
python -B -m scripts.hssd_visual_navigation prepare \
  --input-dir outputs/benchmark_v_navigation_prepare_002 \
  --base-task task_001 \
  --output-dir outputs/hssd_v_middle_prepare_002

python -B -m scripts.hssd_visual_navigation geometry \
  --input-dir outputs/hssd_v_middle_prepare_002 \
  --output-dir outputs/hssd_v_middle_geometry_002
```

prepare 不依赖 Habitat；geometry/audit 需要原生 Habitat/NumPy/Pillow，但不创建 renderer 或 Simulator。
准备包绑定代码与源文件；代码更新后应新建准备版本，不回写旧包或旧验收。

## 5. 应看到什么结果

| 类型 | Oracle 预期 | Ignore 预期 |
| --- | --- | --- |
| 替换 | 在 B 满足距离/朝向/实例条件后 STOP | 在 A STOP，替换失败 |
| 插入恢复 | B 有效 COMPLETE_SUBGOAL，恢复 A，最后在 A STOP | 在 A STOP，未完成 B，顺序失败 |
| 取消 | 动作 22 为 STOP，延迟 0、取消后路程 0 | 继续到 A，取消失败 |

整体 `passed=true` 代表 **3 个 Oracle 成功 + 3 个 Ignore 按预期失败**、证据有效及配对一致，不能解释为六个任务全成功。
三条 Ignore 应有一致的完整物理/RGB 轨迹；每类 Oracle/Ignore 在动作 22 之前共享物理前缀及投递时的 RGB。

`private/` 保存实际配置、backend 来源、轨迹/测量、事件、原始响应、诊断及评价状态。
`public/` 保存实际请求、动作封装及所发送 PNG；全包是离线审计资产，不能整体提前挂给在线 Agent。
当前回调同进程，完整进程隔离仍待实现；Oracle 特权与普通 V 权限必须分别报告。

## 6. 只读重审与后续

`audit` 不调用模型或 renderer：从原始 JSON 响应复原动作，重建消息/历史/图像白名单，在原网格连续重放实际动作并核对位姿、碰撞、距离和朝向。
独立重数实例 mask 像素，校验源实例身份、诊断 pose、标签恢复声明及同位姿的诊断 RGB/公共 RGB。
不信任 `metrics.json`/`summary.json`；末尾保存整个运行目录检查前后的哈希快照。
该审计不独立重新渲染图像，不提供相机来源认证；像素与世界的对应依赖可信采集器，正式隐藏评测隔离仍须完善。

最新验收说明见[实测结果](hssd_v_middle_results_2026-11-13.md)。下一步实现 10 个基础任务、60 个 middle 分支的统一批量与汇总，另准备代表视频导出和人工复核；满足来源绑定时复用现有六分支。统一批量入口尚待实现，之后再覆盖 early/late；当前不增加新模型或时机机制。
