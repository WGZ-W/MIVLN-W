# MI-VLN：支持随时介入的室内服务视觉语言导航 Benchmark 实施方案

## 1. 项目目标

本项目构建一个面向室内服务机器人的视觉语言导航 Benchmark。机器人根据初始自然语言指令自主导航；在人类无需持续观察或控制机器人的前提下，系统在整个执行过程中保持语言介入通道开放。人类可以异步地补充、修正、暂停或覆盖当前任务，机器人必须从当前状态继续执行，而不能重置 episode。

本项目暂定名为：

> **MI-VLN: Mixed-Initiative Vision-and-Language Navigation with Anytime Human Intervention**

中文名称：

> **支持随时人类介入的混合主动式视觉语言导航**

核心研究问题是：

> 智能体能否在连续执行过程中，将动态到达的语言介入与当前视觉观测、历史轨迹和已有任务状态结合，正确更新目标与约束，并在线修复后续导航行为？

---

## 2. 第一版研究边界

第一版只实现能够支撑论文核心结论的最小闭环。

### 2.1 包含内容

- 单机器人；
- 室内家庭服务场景；
- 连续空间中的短距离离散动作；
- 第一人称 RGB-D 视觉输入；
- 初始自然语言任务；
- 执行中异步到达的文本介入；
- 0 或 1 次介入；
- 从当前位置继续执行；
- 自动触发、可重复评测；
- 目标、约束和时序要求的结构化评价。

### 2.2 暂不包含内容

- 语音识别；
- 物体抓取和放置；
- 多机器人协作；
- 机器人主动提问；
- 多用户权限冲突；
- 真实机器人部署；
- 真人实时 HITL 主排行榜；
- 宠物、儿童等复杂社会行为建模；
- 连续速度控制和长 action chunk 抢占。

这些内容在基础 Benchmark 稳定后作为扩展轨道加入。

---

## 3. 场景与仿真底座

### 3.1 主平台

- Habitat-Sim：场景渲染、传感器、物理和导航；
- Habitat-Lab：任务、episode、动作、指标和训练接口；
- HSSD-Habitat：家庭场景；
- Bullet：碰撞和后续动态环境支持。

### 3.2 运行环境

- Linux 负责 Habitat、轨迹生成、渲染、训练和评测；
- Windows 负责代码编辑、数据检查和离线语言标注；
- 所有 Habitat-Sim、Habitat-Lab 和 HSSD 版本必须固定；
- 第一版不安装 Habitat-HITL，真人实验阶段再加入。

### 3.3 数据集许可原则

- Habitat-Sim 和 Habitat-Lab 代码按其 MIT 许可证使用；
- HSSD-Habitat 按 CC BY-NC 4.0 使用，仅用于研究和非商业用途；
- 发布 Benchmark 时只发布 episode、语言、结构化任务、评价函数和下载脚本；
- 不重新打包发布 HSSD 网格、纹理和物体资产。

---

## 4. Agent 设置

### 4.1 观测

主轨道中智能体每一步接收：

```text
RGB 图像
Depth 图像
上一动作
初始指令
截至当前时刻收到的介入历史
可选的相对位姿/里程计
```

不向模型提供：

- 完整语义地图；
- 目标精确坐标；
- 房间和物体的 simulator ID；
- 结构化 task update；
- Oracle 路径。

### 4.2 动作空间

第一版采用：

```text
MOVE_FORWARD：前进 0.25 m
TURN_LEFT：左转 15°
TURN_RIGHT：右转 15°
STOP：声明任务完成
```

介入在两个原子动作之间到达。第一版暂不研究动作执行到一半时的抢占。

### 4.3 Episode 不重置

介入发生时必须保持：

- 当前机器人位置和朝向；
- 已观察到的视觉历史；
- 已执行轨迹；
- 已完成子任务；
- 仍然有效的旧目标和约束。

介入不能通过创建新 episode 或把机器人传送到新起点实现。

---

## 5. 动态任务状态

系统需要维护显式的动态任务状态，而不是只拼接全部文本。

```python
TaskState:
    active_goal
    suspended_goals
    active_constraints
    completed_subgoals
    cancelled_goals
    status
```

其中：

```text
active_goal：当前应完成的目标
suspended_goals：暂时挂起、之后需要恢复的目标栈
active_constraints：当前仍然有效的路径或停止约束
completed_subgoals：已完成的中间目标
cancelled_goals：被新指令明确覆盖或取消的目标
status：active / paused / completed / failed
```

评价器直接根据结构化 task update 更新 ground-truth TaskState。模型只能看到对应的自然语言表达。

---

## 6. 第一版介入类型

### 6.1 目标替换 Goal Replacement

示例：

```text
初始：去厨房，停在冰箱旁边。
介入：不用去厨房了，改去书房门口。
```

语义更新：

```json
{
  "operation": "replace_goal",
  "old_goal": "near(refrigerator)",
  "new_goal": "entrance(study_room)"
}
```

要求：旧目标永久失效，机器人从当前位置前往新目标。

### 6.2 新增路径约束 Route Constraint

示例：

```text
初始：穿过客厅去厨房。
介入：不要经过客厅，从餐厅绕过去。
```

语义更新：

```json
{
  "operation": "add_constraint",
  "constraint": {
    "type": "avoid_region",
    "target": "living_room"
  }
}
```

要求：最终目标保持不变，但机器人不得进入新增禁区。

### 6.3 子任务插入与恢复 Subgoal Insertion and Resume

示例：

```text
初始：去厨房。
介入：先去玄关门口，然后继续去厨房。
```

语义更新：

```json
{
  "operation": "insert_subgoal",
  "new_active_goal": "entrance(foyer)",
  "suspend_current_goal": true,
  "resume_policy": "after_subgoal_success"
}
```

要求：先完成临时目标，再恢复原目标。

### 6.4 无介入控制组

同样的场景、起点和初始任务不发送任何介入，用于测量：

- 原始 VLN 能力；
- 新架构是否损害无介入导航；
- 介入是否真的改变行为。

---

## 7. 介入时机

第一版主排行榜只使用策略无关、外生任务更新，不根据模型是否犯错决定是否发送介入。

### 7.1 触发原则

优先使用语义状态触发，而不是固定第 N 步触发。

```text
房间边界前
关键岔路口前
完成原路径 20%—80% 时
目标首次可见前后
进入受限区域前
```

### 7.2 三类介入的触发窗口

#### 目标替换

- Early：原路径进度 20%—35%；
- Middle：原路径进度 45%—60%；
- Late：原路径进度 75%—90%，但尚未完成。

三档均衡采样，用于测试目标惯性。

#### 新增禁区

- 机器人尚未进入受限区域；
- 受限区域出现在原计划剩余路径上；
- 从当前位置存在不经过禁区的合法替代路径；
- 介入后最优路径与介入前路径存在实质差异。

#### 插入子任务

- 原目标尚未完成；
- 临时目标和原目标均可达；
- 插入目标会实质改变剩余路线；
- 完成临时目标后仍可恢复原任务。

### 7.3 触发格式

```json
{
  "type": "before_entering_region",
  "region": "living_room",
  "start_distance": 4.0,
  "end_condition": "agent_entered_region"
}
```

介入使用触发窗口而非精确坐标。进入窗口后按预先采样的进度或距离发送，以兼容不同策略的轨迹差异。

---

## 8. Episode 数据格式

```json
{
  "episode_id": "hssd_102344280_000001_branch_avoid",
  "scene_id": "102344280",
  "start": {
    "position": [1.2, 0.0, -3.5],
    "rotation": [0.0, 0.0, 0.0, 1.0]
  },
  "initial_instruction": "去厨房，停在冰箱旁边。",
  "initial_task": {
    "goal": {
      "type": "object_near",
      "object_category": "refrigerator",
      "success_radius": 1.0
    },
    "constraints": []
  },
  "interventions": [
    {
      "intervention_id": "int_000001",
      "trigger": {
        "type": "before_entering_region",
        "region": "living_room",
        "start_distance": 4.0
      },
      "utterance": "不要经过客厅，从餐厅那边绕过去。",
      "task_update": {
        "operation": "add_constraint",
        "constraint": {
          "type": "avoid_region",
          "target": "living_room"
        }
      },
      "priority": "normal",
      "preempt_current_action": false
    }
  ],
  "evaluation": {
    "final_goal": "near(refrigerator)",
    "forbidden_regions": ["living_room"],
    "required_subgoal_sequence": [],
    "max_steps": 500
  }
}
```

自然语言和结构化更新同时保存，但结构化更新仅供数据生成、Oracle 和评价器使用。

---

## 9. 数据生成流程

### 9.1 基础任务采样

从 HSSD 场景中采样：

- 可导航起点；
- 目标房间或目标物体；
- 合法停止区域；
- 起点到目标的 Oracle 路径；
- 路径经过的房间、边界、岔路口和地标。

过滤：

- 最短路径过短；
- 目标无法从导航区域观察；
- 起点与目标在同一局部区域且不需要决策；
- 场景语义缺失；
- Oracle 无法生成稳定路径。

### 9.2 构造反事实分支

同一基础任务尽量生成：

```text
Branch A：无介入
Branch B：目标替换
Branch C：新增禁区
Branch D：插入子任务
```

不同分支共享：

- 场景；
- 起点；
- 初始指令；
- 介入前参考轨迹前缀。

这使语言介入成为主要变量。

### 9.3 Oracle 过滤

每条分支必须通过：

1. 介入前任务可行；
2. 介入后任务可行；
3. 存在合法停止区域；
4. 新约束不会切断所有路径；
5. 介入会实质改变目标、合法路径或任务顺序；
6. 机器人拥有合理响应窗口；
7. 任务不依赖未向标注者展示的信息。

### 9.4 路径差异过滤

路径约束样本应满足介入前后 Oracle 剩余路径差异超过设定阈值，避免“虽然新增了约束，但最优路线完全没有变化”的无效样本。

---

## 10. 语言标注流程

### 10.1 Linux 侧预渲染

每条候选样本输出：

```text
trajectory.mp4
current_frame.jpg
history_keyframes/
poses.json
base_task.json
candidate_event.json
oracle_paths.json
```

### 10.2 Windows 标注界面

标注者看到：

- 初始指令；
- 机器人第一人称轨迹视频；
- 当前关键帧；
- 需要向机器人传达的任务变化；
- 可选俯视轨迹，仅供标注者理解；
- 介入候选时间窗口。

标注者提交：

- 自然语言介入；
- 最合适的介入时刻；
- 指令是否立即生效；
- 指令是否清楚且唯一；
- 是否需要回退或恢复旧任务；
- 自信度。

### 10.3 双人验证

第二名标注者只看初始指令、视频和介入文本，回答：

- 当前目标是什么；
- 哪些旧目标仍然有效；
- 是否存在新增禁区；
- 是否需要完成临时目标后恢复原任务；
- 最终应该在哪里停止。

两名标注者理解不一致的样本进入复审或删除。

---

## 11. 数据规模与划分

### 11.1 原型集

- 5 个 HSSD 场景；
- 100 个基础任务；
- 每个任务最多 4 个反事实分支；
- 约 300—400 个 episode；
- 自己完成第一轮语言标注；
- 目标是验证完整数据闭环，不用于最终论文结论。

### 11.2 Benchmark v1 目标规模

优先使用公开的 articulated HSSD 场景，并进行 scene-disjoint 划分：

- 100 个训练场景；
- 20 个未见验证场景；
- 其余公开场景作为未见测试场景；
- 在训练场景中额外生成 val-seen episode；
- 约 5,000 个基础训练任务；
- 平均每个基础任务生成 3 个有效分支；
- 目标约 15,000 个训练 episode；
- 验证和测试规模按场景数量同比例生成。

最终划分需要遵守 HSSD 官方可用场景列表，并记录每个 scene ID。

### 11.3 语言与组合泛化

除场景划分外，测试集还应标记：

- 未见语言表达；
- Early / Middle / Late 介入；
- 不同目标类别；
- 不同房间拓扑；
- 介入前后路径差异；
- 后续版本中的多介入组合。

---

## 12. 评测轨道

### 12.1 Task-Update Track

提供可靠的 Oracle/强导航后端，主要测试：

- 新旧指令融合；
- 动态目标更新；
- 路径约束更新；
- 子任务插入和恢复。

此轨道用于隔离语言任务更新能力，避免低层导航失败掩盖主要问题。

### 12.2 End-to-End MI-VLN Track

智能体仅使用 RGB-D、历史观测、动作和语言完成任务，测试完整导航能力。

### 12.3 后续 Human Track

真人观察模型行为并决定是否、何时介入。该轨道只用于最终用户研究，不作为第一版主排行榜。

---

## 13. 评价指标

### 13.1 Dynamic Task Success，主指标

机器人是否满足最终仍然有效的目标、约束和时序要求。

### 13.2 Intervention Compliance

```text
被满足的有效介入要求数 / 全部有效介入要求数
```

### 13.3 Constraint Violation Rate

记录机器人是否进入新增禁区或违反停止条件。

### 13.4 Post-Intervention SPL

仅从介入时刻开始计算更新后任务的路径效率：

```text
PI-SPL = Success × L*post / max(L*post, Lpost)
```

### 13.5 Response Latency

```text
收到介入到首次产生符合新任务行为之间的步数
```

### 13.6 Resume Success

插入任务完成后，机器人是否恢复原来的挂起任务。

### 13.7 Task Retention

新指令只增加约束时，机器人是否保留原目标，而不是错误地遗忘整个任务。

### 13.8 标准导航指标

- Success Rate；
- SPL；
- Navigation Error；
- Path Length；
- nDTW，可选。

所有结果必须按照介入类型和 Early/Middle/Late 时机分别报告。

---

## 14. Baseline

### 14.1 Ignore Intervention

完全忽略介入，继续执行初始任务。

### 14.2 Latest Instruction Only

只保留最新指令，丢弃初始任务和历史。

### 14.3 Concatenate History

```text
初始指令 [SEP] 介入1 [SEP] 介入2
```

直接输入标准 VLN 模型。

### 14.4 Recurrent Dialogue Memory

使用语言与视觉历史记忆，但不显式维护任务状态。

### 14.5 Structured Task-State Model

将语言解析为目标、约束、暂停任务和恢复任务，再调用导航策略。

### 14.6 Oracle Dynamic Planner

访问结构化 task update 和完整地图，作为性能上界。

必须同时报告无介入标准 VLN 性能，确认新模型没有依赖“等待介入”或牺牲基础导航能力。

---

## 15. 推荐代码结构

```text
MIVLN/
├── configs/
│   ├── simulator/
│   ├── tasks/
│   └── experiments/
├── mivln/
│   ├── simulator/
│   │   ├── habitat_env.py
│   │   ├── event_engine.py
│   │   └── task_state.py
│   ├── generation/
│   │   ├── sample_base_tasks.py
│   │   ├── sample_interventions.py
│   │   ├── oracle_validation.py
│   │   └── render_episodes.py
│   ├── evaluation/
│   │   ├── measures.py
│   │   └── evaluator.py
│   └── schemas/
├── annotation/
│   ├── web/
│   └── guidelines/
├── baselines/
│   ├── ignore_intervention/
│   ├── latest_only/
│   ├── concat_history/
│   └── structured_state/
├── data/
│   ├── episodes/
│   ├── annotations/
│   └── rendered/
├── scripts/
└── tests/
```

HSSD 原始资产不提交到 Git 仓库。

---

## 16. 实施里程碑

### M0：冻结任务规范

完成条件：

- 明确输入、动作和传感器；
- 明确三类介入；
- 明确介入不重置 episode；
- 固定 episode schema；
- 固定主指标。

### M1：跑通 Habitat 最小环境

完成条件：

- 加载官方测试场景；
- 保存 RGB-D 图像；
- 控制机器人移动和停止；
- 获取碰撞和位姿；
- 生成一段导航视频。

### M2：跑通一个 HSSD 普通导航任务

完成条件：

- 加载一个 HSSD 场景；
- 手工设置起点和目标；
- Oracle 生成路径；
- 机器人到达目标并正确停止；
- SR、SPL 和路径长度计算正确。

### M3：跑通第一个介入闭环

固定演示：

```text
初始：去厨房
机器人执行若干步
介入：不用去厨房，改去书房
机器人从当前位置转向书房
机器人正确停止
评价器判定成功
```

完成条件：

- 介入由事件引擎触发；
- 模型输入包含新语言；
- ground-truth TaskState 正确更新；
- episode 不重置；
- 更新后 Oracle 路径和评价正确。

### M4：支持三类介入

完成条件：

- 目标替换；
- 新增禁区；
- 子任务插入与恢复；
- 每类至少通过 20 个自动测试 episode。

### M5：完成原型数据集

完成条件：

- 5 个场景；
- 100 个基础任务；
- 300—400 个分支 episode；
- 视频和 JSON 可在 Windows 上查看；
- 人工复查失败样本并修正规则。

### M6：完成标注工具和指南

完成条件：

- Windows 浏览器可播放视频和选择介入时刻；
- 可提交自然语言及标签；
- 支持第二人验证；
- 标注结果可自动导回 episode。

### M7：完成 baseline 和主指标

完成条件：

- Ignore；
- Latest-only；
- Concatenation；
- Structured State；
- Oracle；
- 自动生成按类型和时机分组的结果表。

### M8：扩展到 Benchmark v1

完成条件：

- scene-disjoint 划分；
- 大规模 episode 生成；
- 人类语言标注和一致性检查；
- 隐藏测试集；
- 数据卡、许可证说明和复现实验脚本。

---

## 17. 第一个端到端原型

首个原型只实现以下单例：

```text
场景：一个 HSSD 多房间住宅
起点：卧室走廊
初始目标：厨房冰箱附近
初始指令：去厨房，停在冰箱旁边
触发时机：机器人完成原路径约 50%
介入指令：不用去厨房了，改去书房门口
新目标：书房门口停止区域
```

需要保存：

```text
介入前轨迹
介入时 RGB-D
介入后轨迹
全部位姿
初始和更新后 Oracle 路径
动态 TaskState 变化
最终评价结果
完整 episode JSON
可播放视频
```

首个原型成功标准：

1. 机器人没有被重置或传送；
2. 收到介入后不再前往厨房；
3. 从当前状态重新规划到书房；
4. 在书房门口合法区域执行 STOP；
5. Dynamic Task Success 为 1；
6. 无介入反事实分支仍然到达厨房；
7. 两个分支在介入前拥有相同轨迹前缀。

---

## 18. 第一版论文可以回答的问题

1. 标准单轮 VLN 模型能否处理执行期间到达的新语言？
2. 简单拼接历史是否足够，还是必须维护结构化动态任务状态？
3. 模型能否区分“覆盖旧目标”和“暂时插入子任务”？
4. 介入发生在 Early、Middle 和 Late 阶段时，性能如何变化？
5. 模型是否会执行新约束但遗忘仍然有效的原目标？
6. 介入是否能提高动态任务成功率，以及会增加多少恢复路径成本？
7. 在未见家庭环境中，动态任务更新能力能否泛化？

---

## 19. 主要风险与控制方式

### 风险一：任务退化成文本任务规划

控制：使用视觉目标、相似家具实例、房间边界和未见场景；主轨道不提供完整语义地图。

### 风险二：介入语言对行为没有实际影响

控制：Oracle 检查介入前后目标、路线或顺序必须发生实质变化。

### 风险三：介入时机泄漏答案

控制：同一种介入覆盖 Early/Middle/Late；加入无介入分支和保持原目标的约束分支。

### 风险四：导航失败掩盖语言理解失败

控制：同时设置 Task-Update Track 和 End-to-End Track。

### 风险五：数据规模过早膨胀

控制：先完成 5 场景、100 基础任务的原型集，通过人工复查后再扩展。

### 风险六：Habitat 工程占用全部研究时间

控制：第一版保持静态场景、短离散动作、单次介入，不启用完整 HITL、虚拟人和物体操作。

---

## 20. 当前立即执行清单

- [ ] 建立 Linux Habitat 环境；
- [ ] 固定 Habitat-Sim、Habitat-Lab 和 HSSD 版本；
- [ ] 跑通 Habitat 官方测试场景；
- [ ] 下载并加载一个 HSSD 场景；
- [ ] 保存第一张 RGB-D 图像；
- [ ] 完成一个无介入导航 episode；
- [ ] 实现 `TaskState`；
- [ ] 实现 `InterventionEvent`；
- [ ] 跑通“厨房改为书房”的首个介入样例；
- [ ] 导出视频和 episode JSON；
- [ ] 构造对应的无介入反事实分支；
- [ ] 验证两个分支介入前轨迹相同、介入后目标不同。

只有首个端到端原型完成后，才开始批量生成任务和开发 Windows 标注界面。
