# 改写与逆序候选：代表性导航验证

入口：`python -m scripts.language_navigation_representatives`。
前一阶段 60 条离线更新已经独立复算通过：更新/状态 60/60，顺序配对 30/30、措辞组 20/20、任务组 10/10 均正确。
本阶段将固定的 `paraphrase_2` 与 `reversed` 接入真实导航，验证动作前更新、策略执行和实测终点是否一致。
模型参数、候选描述、起点、朝向、种子、触发步号和评分保持不变。

## 1. 预先固定的任务与分母

选择规则是：每个审核目的地选择 `task_id` 最小的基础任务，按目的地 ID 排序。
规则不读取预测正确性、成功率或路程；即使离线预测失败，该任务也不更换。
本场景五个目的地恰好对应五类物体：

| 顺序 | 任务 | 原目标 → 新目标 | 事件前动作步号 |
| --- | --- | --- | --- |
| 1 | `task_008` | 浴缸 → 金属框床 `target_020` | 39 |
| 2 | `task_006` | 岛台 → 浴缸 `target_047` | 16 |
| 3 | `task_002` | 沙发 → 岛台 `target_051` | 37 |
| 4 | `task_001` | 岛台 → 深色沙发 `target_054` | 22 |
| 5 | `task_003` | 沙发 → 餐桌 `target_112` | 26 |

每任务新运行 Oracle/model/Ignore，固定 **5 个任务、15 个新 episode、5 次新模型调用**。
使用第二种英文改写，例如沙发消息为：

> Do not continue to the previous destination. Your new destination is the couch that is dark in color.

五候选均按 `target_112/054/051/047/020` 排列，原审核描述和别名不变。
不是重新播放离线预测：model 在事件投递时调用真实服务，原始错误按实际结果应用或拒绝，绝不回填离线答案。
导航仍使用特权坐标、网格和实例中心，不验证视觉定位或学习型导航。

## 2. 已完成的 CPU 准备

准备使用 `habitat` 环境，核验离线公开请求、封存的真实调用、来源导航证据和可达性，禁止创建 Simulator 或调用模型。
以下命令已经执行；`_001` 已存在，不需要再次准备：

```bash
python -m scripts.language_navigation_representatives prepare \
  --prepared-dir outputs/hssd_language_robustness_dev_001 \
  --offline-run outputs/hssd_language_robustness_qwen_001 \
  --output-dir outputs/hssd_language_representatives_001

python -m scripts.language_navigation_representatives check \
  --output-dir outputs/hssd_language_representatives_001
```

新配置只改变 episode 名称、当前消息和候选列表顺序，独立重建核验其余字段未变。
导航请求沿用 `nav_` 不透明 ID，离线使用 `rob_` ID；除 ID 外，消息、候选顺序、模式和旧策略状态必须完全相等。
这不等于两次调用的整个 HTTP 字节内容相同。
保存 `configs/`、代码/源文件哈希清单、公开 `request_previews.jsonl` 和 `cpu_check.json`。
原执行/评分脚本没有修改，因此既有两候选、五候选和离线结果仍能按原哈希复核。

## 3. 下一步执行：真实三条件导航

前提：维持原 Qwen3.5-9B 权重与服务配置，别名 `qwen3.5-9b`、地址 `http://127.0.0.1:8000/v1`。
参数继承已冻结离线运行：非思考、temperature=0、max_tokens=512、timeout=120。
将 GPU 2 改为分配给本次 Habitat 的编号。

```bash
cd /mnt/sdc/weiguanzhao/MIVLN-W
source /mnt/sdc/weiguanzhao/miniconda3/etc/profile.d/conda.sh
conda activate habitat

python -m scripts.language_navigation_representatives run \
  --output-dir outputs/hssd_language_representatives_001 --gpu-id 2
```

入口先核验所有来源和准备结果，再按固定顺序启动五个独立任务子进程，日志位于 `run_logs/task_008.log` 等文件。
`run-task` 是内部子命令；常规使用只执行批量 `run`，以保存完整批次状态和 GPU 来源。
每个子进程顺序运行三个条件，仅 model 调用一次服务。
非零返回仍继续其余任务；中断保留未运行槽位。当前不支持续跑、选择性重试或导入旧轨迹。
运行器拒绝已启动的目录；保留失败，避免覆盖结果或缩小分母。

## 4. 独立汇总与视频

```bash
python -m scripts.language_navigation_representatives summarize \
  --output-dir outputs/hssd_language_representatives_001

python -m scripts.language_navigation_representatives videos \
  --output-dir outputs/hssd_language_representatives_001 --fps 10
```

汇总需要 Habitat 的 CPU PathFinder、NumPy/Pillow，不构造渲染器、不发模型请求。
视频需要 FFmpeg，写入新的 `videos/task_008/oracle.mp4` 等路径；可加 `--task-id task_008`，已有视频不覆盖。

| 文件/目录 | 作用 |
| --- | --- |
| `manifest.json`、`configs/` | 固定选择规则、十五个计划位置、参数和配置来源 |
| `batch_run.json`、`run_status.json` | 批次 GPU、时间和逐任务状态/返回码 |
| `tasks/task_008/` 等 | 新三条件轨迹、图像、原始请求/响应、策略更新和逐任务审核 |
| `summary.json` | 固定分母的更新/闭环结果、阶段失败、离线/在线与物理控制对照 |
| `cases.csv/jsonl` | 始终十五行，逐任务/条件的实测导航指标或证据错误 |
| `comparison.csv`、`tasks.jsonl` | 始终五行，离线/在线预测及状态是否一致、Oracle/Ignore 是否保持原轨迹 |

新日志的版本化加载器允许逆序候选，逐分支仍调用原 `audit_episode` 和 `replay_model`。
严格核验真实请求体、一次投递、动作前应用、位姿连续、事件前观测、STOP 距离、目标朝向和实例可见性。
Oracle 与 Ignore 的全程物理轨迹对照原五候选批次；跨批次观测哈希另列，避免把 GPU 渲染差异混为模型错误。

验收首先看 `audited_tasks=5`、`audited_episodes=15`、`results_provisional=false`，再检查：

- Oracle 5/5 联合成功，Ignore 5/5 按预期失败；原始物理控制匹配 5/5。
- 新模型更新和导航闭环分别统计，以 5 为固定分母，真实错误保留。
- `offline_online_same_count` 记录预测及结果状态与离线是否相同；不相同时单独分析，不以离线答案修复。
- `comparison_complete=true` 表示完整证据与物理控制成立；`passed=true` 进一步要求三条件全部符合预期。

模型错误或调用失败可具有完整可信的证据。返回 1 时先检查阶段错误和日志，不覆盖重跑。
准备阶段显示的 0/5 只是未运行进度，不是模型准确率。

## 5. 本地验证与提交

新增 17 项测试，覆盖固定选择、候选逆序、仅改语言/顺序、离线原始证据绑定、固定分母、失败/中断、防覆盖与控制轨迹变化。
其中四项使用原生依赖加假模拟器/假 HTTP，实际运行导航循环，检查回调收到逆序改写、一次调用、不重置、
接口失败不回填，以及合法错误目标真正进入自己的重规划路径。
Habitat 全量 **271 项通过**；核心环境发现 271 项并跳过 74 项原生依赖测试。
真实来源的 CPU 准备通过，代表性十五个 GPU episode 尚未执行。

建议本轮入口、测试和文档一次提交：`feat: add representative paraphrase navigation validation`。
`outputs/`、场景、权重和凭据不提交。
