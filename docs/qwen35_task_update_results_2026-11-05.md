# Qwen3.5-9B 首次任务更新开发评测结果

检查日期：2026-11-05。结论：真实模型请求、原始预测和评分链路已跑通，
54 条记录逐条复算一致，完整更新及状态正确 46/54。
这是固定上下文下的合成开发样例检查，尚不是留出语言测试或导航实验。

## 1. 输入、设置与产物

- 公共输入：`outputs/task_update_requests_003/`，18 个样例 × latest/history/state，合计 54 条记录、13 个语义组。
- 生成目录：`outputs/task_update_qwen35_001/`；评分目录：`outputs/task_update_qwen35_eval_001/`。
- 服务报告的模型名：`qwen3.5-9b`；本地部署说明中的权重路径为 `/mnt/sdc/weiguanzhao/Qwen3.5-9B`。
- 客户端设置：`temperature=0`、`max_tokens=512`、`timeout=120`，
  `thinking_mode=off`、`chat_template_kwargs.enable_thinking=false`，不共享对话、不修复输出。
- 运行时间：北京时间 17:06:10–17:07:50，端到端约 99.8 秒，包含客户端和 HTTP 开销，不能作为纯模型推理耗时。
- 当前环境核对：vLLM 0.17.1、Torch 2.10.0+cu129、Transformers 4.57.6、CuTe DSL 4.4.0、QuACK 0.2.7。
  这是检查时的环境版本；本批客户端日志未冻结完整服务启动参数、GPU 编号和权重校验和，不能从模型别名推断这些信息。

## 2. 完整性与复算

`run.json` 记录 planned/attempted/received 均为 54，failed=0；
54 条响应全部 `finish_reason=stop`，无调用错误和长度截断标记。
所有输出均为合法契约 JSON。公共输入、索引、系统提示与预测文件哈希一致，
请求 ID 和响应 ID 无重复、缺失或错位；54 条评分 JSONL、CSV 和各项汇总均已在内存中重新计算并逐项核对。

预测文件 SHA256：

```text
34ddbc52d2ed0e4962e301be650f976b020baa9a02221fcdd5ded2926618759f
```

没有重跑真实模型，也没有修改原预测、评分文件或参考答案。

## 3. 结果

| 指标 | latest | history | state |
| --- | ---: | ---: | ---: |
| 格式合法 | 18/18 | 18/18 | 18/18 |
| 可由执行器应用（含合法拒绝） | 16/18 | 17/18 | 18/18 |
| 替换目标正确 | 3/5 | 4/5 | 5/5 |
| 新增禁区正确 | 5/5 | 5/5 | 5/5 |
| 插入子目标正确 | 5/5 | 5/5 | 5/5 |
| 应拒绝猜测时正确拒绝 | 2/3 | 1/3 | 1/3 |
| 预测及完整新状态均正确 | **15/18（83.3%）** | **15/18（83.3%）** | **16/18（88.9%）** |

总体完整正确 46/54（85.2%）。45 条应执行更新中，操作类型全部正确，
参数和最终状态正确 42/45；9 条应拒绝记录仅正确 4/9。
统计单位为样例/模式记录，不能把同一消息的三种模式当作独立任务。

`summary.json` 的 `completed_without_input_errors=false` 和 `invalid_or_missing_count=3`
来自三条目标一致性检查失败，不是 HTTP 调用失败、缺失响应或 JSON 解析失败。
现有评分器将状态应用异常也计入该错误字段，并因此返回非零状态。

## 4. 八条错误的解释

| 记录 | 实际行为 | 解释 |
| --- | --- | --- |
| `replace_zh_implicit__latest` | 把 bed→island 作为更新 | 当前消息“改去金属床”，新旧目标颠倒；latest 也缺少原目标信息 |
| `replace_zh_implicit__history` | 同样预测 bed→island | 历史已提供原目标 island，仍发生新旧目标颠倒 |
| `replace_zh_prior_change__latest` | 预测 island→table | 实际原目标已变为 bed，latest 看不到这段历史 |
| `ambiguous_sofa__latest/history/state` | 均直接选择深色沙发 | 两张沙发都符合泛称，应拒绝猜测；共三条错误 |
| `ambiguous_region__history/state` | 均直接选择红色区域 | 消息未说明红色或蓝色，应拒绝猜测；共两条错误 |

前三条被共享执行器拒绝，输入策略状态不被错误更新。
后五条虽然格式合法、可应用，语义仍然错误：执行器不能替代模型判断语言是否足够明确。
“书架不在候选中”的拒绝在三种模式下均正确。

## 5. 能支持的结论与限制

1. 已证明本地服务可以完成这批公共请求，原始输出能进入严格评分；格式与记录完整性通过。
2. 在这批样例中，state 的 15 条明确更新全部正确，但在两条歧义消息上仍直接猜测。
   总体只比另两模式多正确一条，样例少且未经留出，尚不足以得出结构化状态普遍优于历史的结论。
3. latest 的四条正确记录中，旧目标或恢复目标并不能从其允许输入唯一确定：
   `replace_zh_keep_constraint`、`insert_zh_implicit_resume`、`insert_zh_keep_constraint`、`insert_en_implicit_resume`。
   模型在这些请求中填入 island 并恰好命中；这是缺少依据的命中，不能解释为记住了隐藏历史。
4. 旧约束保留由共享状态执行器保证；只预测区域标识的模型并未因此证明自己维护了全部约束。
5. prior-state 仍由固定公共语言规则构建，模型没有在本实验中连续维护自己的历史状态。
   无图像输入、无 HSSD 导航闭环，也没有验证视觉定位或最终任务成功率。

## 6. 下一步与已有命令

下一步先建设成对上下文样例，而不是扩大当前模板的重复次数：

- 对同一句“先去深色沙发，再继续原任务”，分别设置 island、bed、table 为原目标，要求恢复目标相应变化。
- 对同一句“改去餐桌”，通过不同历史改变当前目标，检查 old_goal 是否随历史变化。
- 平衡原目标、恢复目标及候选顺序，减少固定选择 island 或首个候选也能命中的机会。
- 明确标注当前模式的信息是否足够，将合理的缺失上下文拒绝和歧义误执行分开报告。
- 按语义组划分开发与留出集；先人工核验标签，再冻结请求与参数。

成对配置和专项分组评价尚待实现，当前没有对应的新 GPU 批次命令。
现有 `export --dataset`、预测与评分入口可以复用。
本批已完成验收，无需重新生成；如需独立 CPU 评分副本，以下命令已经可用，需使用新输出目录：

```bash
conda activate mivln
python -m scripts.task_update score \
  --requests-dir outputs/task_update_requests_003 \
  --predictions outputs/task_update_qwen35_001/predictions.jsonl \
  --updater-name qwen3.5-9b-nonthinking \
  --output-dir outputs/task_update_qwen35_eval_002
```

预期保留 54 条记录并得到相同结果；三条状态应用错误会使评分命令返回 1，但评分文件仍完整写出。

## 7. 提交建议

客户端、测试、部署说明和本结果记录可作为同一个完整步骤提交：
`feat: add non-thinking Qwen3.5 inference and development results`。
相关 README、任务更新和项目框架说明一起更新；`outputs/`、模型权重和环境目录保持排除。
用户单独编写的 `docs/understand_framework.md` 不纳入本步骤的提交建议。
