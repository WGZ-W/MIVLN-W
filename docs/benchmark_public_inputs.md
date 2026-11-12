# 公开消息 schema、白名单构造与 CPU 验收

日期：2026-11-12；消息契约：`mivln_public_messages_v1`，外层 `schema_version=1`。
本步骤实现 [输入权限表](benchmark_input_permissions_v1.md) 中 U 和最小 G 的数据边界，不运行模型或导航。

后续采集先采用 [两类任务范围](benchmark_active_scope.md)，G 区域消息和禁区示例保留作后置兼容性验证。

## 1. 已实现范围

| 组件 | 作用 |
| --- | --- |
| `mivln/public_messages.py` | 可导出 JSON Schema、标准库结构校验、引用/几何/四元数检查与公共正文校验和 |
| `schemas/benchmark_public_messages_v1.schema.json` | 生成并纳入版本控制的机器可读 schema，与运行时定义做一致性回归 |
| `mivln/public_inputs.py` | U/G 白名单构造器、会话时钟、一次投递、分支/时间戳核对、预算与 STOP 后关闭 |
| `scripts/public_inputs.py` | schema 导出、现有目标替换 U 请求只读导出、合成 G 消息示例、消息文件结构检查 |
| `tests/test_public_inputs.py` | 私有信息不干扰公共消息、错误状态保留、越权字段、投递边界及来源不变性回归 |

本页 U/G 契约不支持 V。后续已有独立 [V RGB 输入与动作契约](benchmark_visual_inputs.md)，深度、实际地图资源服务、真实导航接线和进程隔离仍待完成。
最小 G 只有精确位姿/碰撞观测及显式几何，不代表已完成视觉语言导航或禁区感知。
已有导航循环和模型客户端保持原入口；本步骤验证新视图可与旧回调对齐，尚未将新会话构造器接入真实导航循环。

## 2. 外层消息与分阶段字段

每条消息只接受以下顶层字段，所有嵌套对象也拒绝未授权字段：

```text
schema_version, contract, track, phase,
session_id, message_id, step, payload
```

`session_id` 是宿主新分配的 UUID，不从私有 task/event ID 推导。
`message_id` 仅由公共正文计算；它是可复算校验和，不是签名或语义内容审核。
序列化规则复用 `mivln.task_update_prompts.fingerprint`（排序键、UTF-8 JSON、紧凑分隔符）；客户端须保留相同数值表示后重算。
`step=k` 表示动作 k 前的状态；执行动作 k 后的观测为 step k+1。初始化为 step 0。

| track / phase | payload 内容 | 特别限制 |
| --- | --- | --- |
| U / initialize | mode、上下文来源声明、目标/区域候选 | 不带初始目标、几何或初始语言，避免 latest 模式旁路获得额外上下文 |
| U / intervention | `request`，复用已有 U 七字段及 mode 专属 context | state 必须显式传入实际策略状态；不从配置或评价器自动补正确状态 |
| G / initialize | 初始语言、候选、停止点、地图资源引用、起始位姿、预算/动作规格 | 地图仅宿主分配的 asset ID + SHA256，不是文件路径；没有未来事件和初始符号答案 |
| G / observation | 位姿、碰撞反馈、刚执行的动作 | 宿主包必须绑定当前会话及准确的下一观测步；拒绝 RGB、深度、诊断或真值字段 |
| G / intervention | 自然语言及显式发布的区域目录（描述/别名/矩形几何） | 不带 operation、激活标志或正确更新；区域包须绑定当前会话和投递步 |

U 不产生普通观测消息，只记录宿主实际执行的动作以推进投递时钟。G 的几何目录是该输入配置明确允许的辅助信息。
JSON Schema 检查形状；运行时还检查引用对应、ID 唯一、区域上下界、单位四元数、mode 来源及正文校验和。
会话构造器额外检查初始化、动作预算、投递顺序、一次介入及 STOP 后不可继续；单条消息校验不能替代这些时序检查。
运行时拒绝非有限/不可表示数字、布尔数值以及类型强制转换；外部 JSON Schema 工具还需执行相应语义检查。

## 3. Python 使用方式

```python
from mivln.public_inputs import PublicInputSession
from mivln.task_state import TaskState

session = PublicInputSession(
    "U", config["max_steps"],
    context_source="provided_initial_goal_then_policy_rollout",
)
header = session.initialize(config, approved_goal_candidates)

# 由宿主在每个实际执行动作后推进，不能虚构动作让未来事件提前生效。
for action in actually_executed_actions_before_delivery:
    session.advance(action)

# actual_policy_state 必须来自被测分支；此处仅展示传参，不以参考状态替代。
public_message = session.intervention(delivered_event, policy_state=actual_policy_state)
model_request = public_message["payload"]["request"]
```

上例变量来自宿主，属于 API 示意，不是可直接执行的完整导航脚本。
state 的上下文来源须声明为公开语言构建、给定初始符号后实际滚动，或模型实际滚动；来源声明仍须由宿主和审计核查。
`advance` 只登记已经执行的动作，不调用模拟器、不应用任务更新，也不证明动作物理有效。
`COMPLETE_SUBGOAL` 只在此层计步；完成证据和合法性仍由任务协议/评价器检查，不能由消息构造器授予成功。
纯函数 `update_request` 可复用已有 U 的 latest/history/state 契约；零区域的 state 请求与旧 `navigation_request` 正文一致。

## 4. 可执行命令

从仓库根目录使用普通 Python 核心环境执行，均无需 Habitat/GPU/模型。输出目录必须全新。

```bash
# 只读导出现有 10 基础任务 × 3 时机的固定上下文请求。
python -B -m scripts.public_inputs export-replacement \
  --source-dir outputs/hssd_dataset_facing_001 \
  --output-dir outputs/benchmark_public_inputs_002

# 合成几何区域消息：初始化、动作后观测、当前介入。
python -B -m scripts.public_inputs demo \
  --output-dir outputs/benchmark_geometry_messages_002

# 校验已生成消息的结构、引用、几何及正文校验和。
python -B -m scripts.public_inputs validate \
  --messages outputs/benchmark_public_inputs_002/public/messages.jsonl

# 仅生成 schema，以及边界回归。
python -B -m scripts.public_inputs schema --output-dir outputs/benchmark_message_schema_001
python -B -m unittest tests.test_public_inputs -v
```

`export-replacement` 读取旧 manifest、配置、参考及审核目录，核对冻结时机和语义；只按参考的实际前缀推进时钟，不对模型宣称产生了新轨迹。
固定上下文使用明确给定的初始符号目标，30 条请求各自独立，不能作为连续模型状态传递实验。
来源错误、缺失时机或输入改变会使整个导出失败，不能输出一个过滤后子集并声称完整。原计划和旧实验产物不变。

## 5. 公共包与私有审计

```text
输出目录/
  public/schema.json          # 消息 schema
  public/messages.jsonl       # U 初始化及投递消息
  public/requests.jsonl       # 旧回调形状的纯 U 请求
  public/manifest.json        # 公共配置、数量和公共文件哈希
  private/case_index.jsonl    # session → 原任务/配置/时机，仅供审计
  private/source_bindings.json
  private/code_bindings.json
  summary.json               # 私有来源与本次导出结论
```

只将 `public/` 视为候选公共包，不能把整个输出目录交给 Agent。
`requests.jsonl` 与旧回调正文相同，但该目录不具备 `predict_task_updates --requests-dir` 所要求的批次 manifest，不能直接冒充该入口的数据集。
公共包可包含一次试验中的多条消息；正式在线 Agent 必须经逐时投递接口访问，不能挂载整包预读未来消息或获知未来触发时刻。
纯请求 ID 对相同公共输入可能复用；跨会话身份以新会话 UUID 和消息 ID 为准。

新导出校验读取的输入文件前后 SHA256，并记录代码绑定；这是本次消费文件的检查，不是 B0 的全部帧/视频树哈希审计。
修改私有答案、路径或未来事件而保持获准输入相同，已构造消息保持相同；候选/文本须来自独立审核的允许输入。
字段结构校验不能识别自由文本中的隐写答案，也不能阻止同进程读私有文件；图像真实性、宿主时间戳、描述来源和运行隔离仍须后续验证。

## 6. 首次验收与下一步

首次产物：`outputs/benchmark_public_inputs_001/` 和 `outputs/benchmark_geometry_messages_001/`。
已导出 30 个 U 会话、60 条消息及 30 条请求，逐条与旧回调正文相等；82 个消费输入文件 SHA256 前后相同。
公共消息没有已知私有来源路径或参考/坐标/诊断字段。G 合成示例三条消息通过校验，不分发真实地图、不证明 HSSD 可行性。
新增 18 项回归通过；全量 301 项在 Habitat 环境全部通过，核心环境按原生依赖条件跳过 74 项。

后续已有 [V RGB 数据契约及合成回调验收](benchmark_visual_inputs.md)，覆盖实际 PNG、历史与动作接口，U/G 资产继续作为受控基线保留。
当前下一完整步骤是以已验收目标替换任务为起点，接入 V 的实际 RGB 收集、动作执行和公共输入日志，先完成 CPU 模拟器替身与独立验收。
几何辅助禁区控制、三维标记、跨进程隔离及完整协议评分仍待完成；本步骤不自动启动新 GPU 实验。
