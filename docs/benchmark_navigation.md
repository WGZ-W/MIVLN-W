# 已审语言驱动的三类 V 导航入口

协议：`active_v_navigation_v1`。当前交付真实 HSSD 配置准备、可替换 backend 的 V 循环、CPU 合成验证与原始日志只读重审。
真实 Habitat backend 已由[独立 middle 入口](hssd_visual_navigation.md)接入，六分支 GPU 验收通过；视觉模型适配、进程隔离和正式统一榜单仍待完成。
入口：`python -B -m scripts.benchmark_navigation`，标准库即可运行，无 GPU/模型调用。

## 1. 任务及输入边界

覆盖目标替换、子目标插入与恢复、取消任务并立即 STOP，不包含禁区。
读取 `outputs/benchmark_language_annotations_002/` 已批准的 40 条模板语言，共享 10 个基础任务。
检查全部记录的审核、内容/上下文哈希、父版本链、来源实例和 RGB；未批准不能准备导航任务。

| 组件 | 可以获得的信息 |
| --- | --- |
| 普通 V 回调 | 初始语言、当前 RGB、采样历史 RGB、全部实际动作/碰撞、当前及过去已投递消息、相机与动作参数 |
| 宿主/评价器 | 上述信息及私有几何、正确更新、事件时机、实测位姿和实例诊断 |
| 本次 CPU 脚本策略 | 显式额外持有正确更新、几何和合成位姿，仅作特权接线测试 |

模型回调只接收 `(request, assets)`，输出原始 `{"action":"..."}`，不要求输出结构化状态。
图像字节按本次请求白名单取得。真值状态、诊断、目标 ID、未来介入和路径不进入 V 正文。
同进程回调还未被操作系统隔离；不能把参数白名单称为完整防作弊环境。

## 2. HSSD 准备与新触发口径

`prepare` 保留审核版本、源配置几何、初始位姿、相机参数、种子和原预算。
重新检查无介入 A 参考的实走距离、终点实例诊断及文件哈希，为每个基础任务建立三类 × 三时机，共 **90 个私有配置**。
参考有问题时保留全部相应计划槽位并报错，不能删除任务来提高通过率。

新入口使用 `external_step_from_cumulative_reference_travel_v1`：

- 在冻结 A 参考的累计实走路程达到 25%/50%/75% 后选择观测下标 k。
- 观测 k 是动作 k 之前的状态；消息恰在该步请求之前投递。
- 三类及不同策略共享外生 k，不根据参与方法的路程重新选择触发。
- 提前 STOP 未收到介入属于任务失败；动作预算不重置。

此口径仅属于新版本入口，不改旧替换任务的剩余距离口径及历史配置。
这些准备配置尚未执行 HSSD；正式发布的触发及场景划分仍需冻结。
不能把新配置直接交给旧 `run_navigation` 或旧 U 模型更新入口。

## 3. CPU 验证明确使用合成几何

CPU 将已审文本及私有标签绑定到 `synthetic_free_plane_v1`：无障碍平面，A=(0,0,-2) m，B=(1,0,-1) m，出发点原点、面朝 -Z。
前进 0.25 m、转向 90°，成功半径 0.13 m，64 个动作预算；8 次前进的合成参考对应 k=2/4/6。
**这些数值替代 HSSD 几何和时机，仅用于 CPU fixture。** 原 HSSD 值仍保存在准备包中，不被回写。
RGB 为无文字的位姿相关测试色块，距离为无障碍欧氏距离，实例可见性是合成朝向检查；不是真实物体图像或 HSSD 可行性证明。

| 条件 | 替换 | 插入恢复 | 取消 | 作用 |
| --- | --- | --- | --- | --- |
| `follow` | 成功 | 成功 | 成功 | 正确更新并完成义务 |
| `ignore` | 失败 | 失败 | 失败 | 仍走到 A，不能算服从新任务 |
| `delayed` | 成功 | 成功 | 失败 | 收到消息后先转一次；取消必须当步 STOP |
| `wrong` | 失败 | 失败 | 失败 | 投递步执行可解析但语义非法的 COMPLETE_SUBGOAL |
| `invalid` | 失败 | 失败 | 失败 | 投递步输出额外 JSON 字段，被动作契约拒绝 |

`passed` 表示正负条件均按预期被评价，不能解释为所有 episode 成功或视觉模型性能。
同一条件的早/中/晚、措辞及重复运行仍共享基础任务，不增加独立样本数。

## 4. 执行与独立评分

`mivln/benchmark_navigation.py` 的 `run_episode(config, backend, callback)` 不绑定具体模拟器。
backend 提供公开相机标定、RGB、物理动作、位姿及私有实测距离/朝向/实例诊断；世界坐标以米为单位，四元数为 XYZW，角度为度。
事件仅改变评价真值，不重置 backend、策略记忆或历史，也不替策略产生 STOP。

- 替换：已投递、在 B 实测完成且 STOP，保留 A 取消记录。
- 插入：B 的距离、朝向和实例证据有效时，COMPLETE_SUBGOAL 才恢复评价器 A；随后须在 A STOP。
- 取消：复用 `cancel_task_v1` 独立评价，投递动作必须 STOP，位姿保持，不要求目标距离/可见性。
- 错误完成动作仍执行和计步，保留违规；后面到达目标也不能抹除违规。
- 无效 JSON、回调失败不修复、不重试、不补 Oracle 动作；保留失败请求、原始响应及已经执行的轨迹。
- backend 执行失败保留最后输入/响应，未完整记录的物理转移不授予有效证据。

`audit` 从保存的原始响应重新执行 CPU backend，重建 V 请求、历史、事件、动作、位姿、诊断和 PNG 字节。
不调用策略，不信任缓存 `metrics.json`/`summary.json`。动作和终点状态须与重放一致，缺失分支仍留在分母。
Ignore 还须确实到 A；Wrong 须确实执行投递步的非法完成；Invalid 须保留该步失败请求，不能用其他故障冒充预期负对照。
重审报告记录整个 CPU 运行目录前后的哈希快照。

## 5. 可执行命令与产物

已验收目录见下一节。下列命令用于另开一个单任务、中期小批次，输出目录必须全新：

```bash
python -B -m scripts.benchmark_navigation prepare \
  --annotations-dir outputs/benchmark_language_annotations_002 \
  --output-dir outputs/benchmark_v_navigation_prepare_003

python -B -m scripts.benchmark_navigation cpu \
  --input-dir outputs/benchmark_v_navigation_prepare_003 \
  --base-task task_001 --timing middle \
  --output-dir outputs/benchmark_v_navigation_cpu_003

python -B -m scripts.benchmark_navigation audit \
  --input-dir outputs/benchmark_v_navigation_cpu_003 \
  --output-dir outputs/benchmark_v_navigation_audit_003

python -B -m unittest tests.test_benchmark_navigation -v
```

小批次为 3 类 × 5 条件 = 15 个计划 episode。全批可选择 `--base-task all --timing all`，为 90 × 5 = 450 个 **CPU 合成** episode。
源文件或这三个新模块改变后，准备包会拒绝旧代码/源绑定；保留旧包，使用新目录重新准备。

| 位置 | 内容 |
| --- | --- |
| 准备包 `manifest.json`、`configs/` | 私有 HSSD 配置、全部计划槽位、语言/几何/参考/代码绑定 |
| CPU `manifest.json`、`summary.json` | 选择范围、完整条件分母及逐项验收 |
| `episodes/<case>/<condition>/public/` | 实际请求、合法动作封装、本轮实际发送过的 PNG |
| 同分支 `private/` | fixture 配置、原始响应、实测轨迹/诊断、事件及评价状态 |
| 审计包 `audit.json`、`source_bindings.json` | 从原始证据重算的结果及检查前后快照 |

所有输出都在 Git 忽略的 `outputs/`。完整 `public/` 是离线审计资产，也包含后续消息/图像，不能整体挂给在线 Agent 预读。

## 6. 本次验收和下一完整步骤

本次使用已批准的 40 条原模板，保持其文本、审核者和版本来源不变：

- `outputs/benchmark_v_navigation_prepare_002/`：10 基础任务、90/90 HSSD 配置准备完成，实际 HSSD 导航执行数 0。
- `outputs/benchmark_v_navigation_cpu_002/`：全任务、全时机 CPU 合成验证，450/450 计划槽位按预期验收，配对错误 0；其中 150 个任务成功，其余 300 个是应当失败的负条件。
- Follow 90/90 任务成功；Ignore 0/90；Delayed 60/90（取消延迟 30 次均失败）；Wrong 0/90；Invalid 0/90。
- `outputs/benchmark_v_navigation_audit_002/`：只读重审通过，整个运行目录前后不变；125 份来源/证据绑定及旧代码绑定核验通过。
- 20 项新回归覆盖三类/三时机/五条件、权限与时间链、早停/超时/失败、实例完成、来源绑定、防改写、固定分母及拒绝空批次。
- 全量 378 项在 Habitat 环境全部通过；核心环境同套件通过，按原生依赖跳过 74 项。实际未创建 Habitat Simulator、未调用模型或 GPU。

初轮 `_001` 产物保留不覆盖；补充空批次拒绝校验后，按新代码哈希重新准备、执行和重审，当前验收以 `_002` 为准。

[真实 HSSD middle 入口](hssd_visual_navigation.md)已复用本循环接入实际 RGB/动作/碰撞、网格距离、终点朝向及实例诊断；原生参考前缀检查通过。
其 `task_001 + middle` 三类 Oracle/Ignore 六分支已真实运行并只读重审通过，见[实测结果](hssd_v_middle_results_2026-11-13.md)。下一步实现 10 任务 middle 批量入口，另准备代表视频人工核对，再扩展时机。
本页原 `scripts.benchmark_navigation` 仍仅保留 prepare/cpu/audit，真实运行使用新增 `scripts.hssd_visual_navigation run`，不改旧模块哈希。上述 CPU 成绩不替代真实 GPU 验收。
