# V 视觉导航：逐步输入、动作输出与标记可观察性契约

日期：2026-11-12；契约：`mivln_visual_navigation_v1`，`schema_version=1`。
本步骤落实初始指令、当前观测、历史观测/动作和介入消息的数据接口，不接入真实模型或 Habitat 渲染。

范围更新：当前主任务为替换、插入恢复、取消并停止，使用 `V_RGB_v1`；见 [当前采集范围](benchmark_active_scope.md)。禁区和 marked 配置保留为后置实验。
取消使用独立 [CancellationVisualSession](cancel_task.md)，复用本页 RGB/动作 schema；原 VisualInputSession 的操作范围保持冻结。
下文含禁区的三场景 demo 是原接口回归，不是当前主任务新版数据集。

## 1. 交付范围

| 文件 | 作用 |
| --- | --- |
| `mivln/visual_frames.py` | 采集原始 RGB、编码无元数据 PNG、绑定资源及红色像素标记证据 |
| `mivln/visual_messages.py` | V 请求与动作 schema、因果对齐、原始动作 JSON 解析 |
| `mivln/visual_inputs.py` | 步进会话、历史采样、一次介入、图像访问白名单、CPU 可测回调 |
| `schemas/benchmark_visual_{request,action}_v1.schema.json` | 公开输入及宿主绑定的动作封装 |
| `scripts/visual_inputs.py` | schema 导出、三类合成接口场景、PNG/动作/历史独立复核 |
| `tests/test_visual_inputs.py` | 时间、权限、实际图片、失败保留与历史防改写回归 |

模块独立于已冻结的 U/G 契约及旧导航入口，不改旧代码哈希或原实验结果。
已实现的是 **V 数据契约和合成 CPU 回调闭环**；真实视觉模型、实际导航接线、三维地面标记渲染、完整评分及进程隔离仍待完成。

## 2. 每步给模型什么

| 字段 | 内容 |
| --- | --- |
| `initial_instruction` | 原始自然语言指令，每次请求保留 |
| `observation` | 当前 RGB 资源及上一步碰撞反馈，无精确世界位姿 |
| `history.observations` | 按冻结策略选择的过去 RGB，包含采集步号 |
| `history.actions` | 本分支全部已执行动作及碰撞反馈，动作从 0 连续编号 |
| `current_intervention` | 本步新投递的消息和实际投递步；没有则为 null |
| `message_history` | 已实际投递且已执行过对应动作的消息；不含未来消息 |
| `camera` | 公共相机尺寸及 fx/fy/cx/cy 标定 |
| `history_policy` | 固定历史帧窗口和采样间隔 |
| `observation_profile` | `V_RGB_v1` 或 `V_RGB_MARKED_v1`；不同输入配置分别报告 |
| `actions` / `limits` | 动作定义、总预算、剩余预算、前进/转角/停止点阈值 |
| 身份字段 | V 契约、宿主分配的会话 ID、仅由公共正文计算的请求 ID、当前步 |

默认不含候选列表、初始符号目标、目标坐标、地图、位姿、TaskState、正确更新、诊断掩码或评价器完成反馈。
这套 V schema 目前不接受深度、语义图或额外候选；扩展需单独版本及权限声明。
图像资源只有不透明 asset ID、PNG 哈希、尺寸和类型，没有原始路径、实例 ID 或 EXIF/text 标签。
`assets_for(request_id)` 只提供这次请求选中的实际 PNG 字节；未来模型适配器还需把字节转换为视觉 API 图像附件。

## 3. 时间、历史与动作

观测步 t 是动作 t **之前**的状态：先投递本步新消息，再构造请求，最后选择并执行动作。
动作 t 的实际结果形成观测 t+1，动作记录仍标为 t。初始化观测为 0，不计动作。
消息在 k 步进入 `current_intervention`；动作 k 完成后进入 `message_history`，k+1 步的新消息字段为空。
同一选择槽重复读取请求返回相同快照，不重复投递；请求发出后不能再追加本步事件或替换图片。

默认历史采样为 `recent_stride_v1`：在过去观测步 `<t` 中选步号能被 stride 整除者，保留最近 max_frames 帧。
默认 max_frames=8、stride=1；动作和已投递消息历史完整保留。max_frames=0 只关闭历史图像重发。
采样规则在 episode 中固定；它限制每次重新发送的帧，不要求方法清除自身先前记忆。不同采样配置须记录并公平比较。

模型原始输出严格为：

```json
{"action": "turn_left"}
```

允许 `move_forward`、`turn_left`、`turn_right`、`STOP`、`COMPLETE_SUBGOAL`。
宿主将动作绑定到当前 session/request/step，拒绝过期响应和跨分支动作。完整动作能力固定公开，不根据私有真值动态屏蔽动作泄漏答案。
`COMPLETE_SUBGOAL` 的任务合法性、是否真的完成 B 和何时恢复 A 由评价器判定，动作接口本身不给正确完成反馈。
STOP 计一步并结束；中间完成计一步但不推进 RGB 物理观测；耗尽预算后不再选择动作。接口合法结束不等于任务成功。
无效 JSON、额外字段、未知动作、调用失败均记录并结束接口试验，不修复、不重试、不用 Oracle 动作补救。

## 4. 禁区怎样在实际输入中可见

首个方案采用 **V_RGB_MARKED_v1：红色地面标记辅助 RGB**，与纯 RGB 配置分开报告。
环境应从私有区域几何生成投影 mask，再把标记画进实际给模型的 RGB；原始诊断掩码与区域坐标不发送给模型。
本步实现后半段：宿主传入像素 mask，`mark_frame` 生成实际 PNG；投递时核对原图/标记图哈希、会话、步号、区域身份、标记颜色及变化像素。

该版本要求红色 `(255,0,0)`，至少 16 个像素确实由原色改成红色；这是合成接口的标记可见检查，不是目标实例的 64 像素成功指标。
真实数据发布前需要标定可辨识性、审核描述与标记的一致性，阈值调整应版本化。
没有标记的历史 HSSD RGB、仅导出视频加框、错帧证据、错误区域或改色/篡改的 PNG 都不能通过该投递检查。

**边界：**像素 mask 的世界来源仍由宿主提供。本检查不证明它正确投影了三维地面区域，不证明区域可从所有后续视角持续识别。
CPU 示例用人工矩形像素 mask，明确记录 `world_projection_verified=false`；后续需要真实投影、遮挡和持续标记渲染验收。
因此历史禁区仍只支持 Oracle 可行性证据，尚不能被标记为已完成视觉禁区导航。

## 5. 可执行入口与产物

在仓库根目录使用普通 Python 运行，输出目录必须全新：

```bash
python -B -m scripts.visual_inputs demo \
  --output-dir outputs/benchmark_visual_contract_002

python -B -m scripts.visual_inputs validate \
  --requests outputs/benchmark_visual_contract_002/public/requests.jsonl \
  --assets-dir outputs/benchmark_visual_contract_002/public/assets \
  --actions outputs/benchmark_visual_contract_002/public/actions.jsonl

python -B -m scripts.visual_inputs schema --output-dir outputs/benchmark_visual_schemas_001
python -B -m unittest tests.test_visual_inputs -v
```

demo 固定三个合成场景：替换、禁区、插入恢复。脚本策略只识别固定短语并执行示意动作，不理解图像、不证明任何真实任务成功。
公共部分保存请求、绑定动作和实际送出的 PNG；私有部分保存案例映射、原始响应、失败、标记证据及原图。
失败案例保留计划分母、已发送图像和无效响应，不能因无有效动作就丢失模型输入。
`validate` 独立核查正文、PNG 哈希/尺寸/CRC、历史图像与先前实际输入对应、消息历史保留，以及可选完整动作链。
不提供 `--actions` 时仅检查输入链和资源，不声称检查动作执行；此检查也不证明机器人物理运动或导航成功。

完整公共包是离线审计资产，不能整体挂载给在线 Agent 预读未来帧或消息；在线适配应使用逐请求的图像白名单。
同进程回调仍可能通过闭包或文件系统获得额外信息，完整隔离未实现。PNG 校验只能剥离元数据，不能发现像素里绘制的隐写答案。

## 6. 本次验收与下一完整步骤

首次产物在 `outputs/benchmark_visual_contract_001/`：三个合成接口场景、11 次请求、11 个绑定动作，独立输入/资源/动作链校验通过。
22 项新回归覆盖一次投递、历史采样、错误输出、过期动作、字节级标记证据、失败输入保留及防止回写历史。
全量 323 项测试在 Habitat 环境全部通过，核心环境按原生依赖条件跳过 74 项。
报告明确不调用真实模型、不渲染 HSSD、不评估导航成功，也不证明三维投影。

在原契约之后，V 会话和 RGB 资源收集已接入下述新版本导航循环，并完成 CPU 模拟器替身与独立日志验收。
实际 HSSD 接线仍须确认物理动作与图像时序一致、模型只得到获准图像/消息、错误动作按实执行并由独立评价器评分。
首批三类语言已审核；新增[已审语言三类导航入口](benchmark_navigation.md)已实现 backend/回调循环、HSSD 配置准备和五条件 CPU 合成验证，旧 V 契约及本页历史 demo 不变。
下一步实现真实 Habitat backend 的 RGB、动作、网格距离及实例诊断，再验收一个基础任务的三类 Oracle/Ignore。真实运行命令与完整进程隔离仍待实现；B1 尚未全部完成。禁区真实投影与持续标记继续后置。
