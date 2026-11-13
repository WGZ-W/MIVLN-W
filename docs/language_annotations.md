# 当前三类任务的语言标注、审核与版本化导出

入口：`python -B -m scripts.language_annotations`。仅使用标准库，不调用模型或 GPU。
目标是先建立可追溯的语言试采集流程；不要求参与方法预测结构化状态。

## 1. 固定范围与来源

从 `outputs/hssd_dataset_facing_001/` 的 10 个基础任务读取已绑定的初始目标 A、替换目标 B、来源配置及审核实例 RGB。
每个基础任务生成 4 条记录：共享的初始指令，以及替换、插入恢复、取消各一条介入消息，共 **40 条语言、30 个任务案例**。
三类分别标注 A→B、先 B 后恢复 A、取消并立即 STOP；不加入禁区。
30 是语言案例数量，不是新的独立基础任务数，也不是已执行 episode 数。

初始/替换消息沿用来源文本，插入/取消消息由模板生成。所有记录均为 `template_derived`，初始状态 `pending`。
已有物体实例的审核仅证明来源实例，**不继承为新语言审核**。人工改写或明确采纳会另记作者，仍保留模板来源。
原配置、历史运行和既有代码哈希不变；旧几何验收不能代替新措辞的导航验收。

## 2. 生成和检查草稿

已生成的首批目录为 `outputs/benchmark_language_annotations_001/`，无需重复准备。
下例仅用于另开批次，所有输出目录必须尚不存在：

```bash
python -B -m scripts.language_annotations prepare \
  --source-dir outputs/hssd_dataset_facing_001 \
  --output-dir outputs/benchmark_language_annotations_new_001

python -B -m scripts.language_annotations status \
  --input-dir outputs/benchmark_language_annotations_001

python -B -m scripts.language_annotations inspect \
  --input-dir outputs/benchmark_language_annotations_001 \
  --annotation-id task_001_initial
```

| 文件 | 用途 |
| --- | --- |
| `annotations.json`、`manifest.json` | 权威内容、事件历史、来源哈希与父版本绑定；不得直接编辑 |
| `annotation_sheet.csv` | 可编辑工作表；保留全部 40 行及固定身份列 |
| `review_decisions.json` | 当前版本的空白审核模板；默认 `decision=null`，不会自动批准 |
| `review_index.md` | 私有审核索引、实例 RGB 链接、语义标签及必看证据 ID |
| `status.json` | 生成时的状态快照；实际检查以 `status` 重放结果为准 |

读取、审核和导出都会重新核对来源配置、目录清单、实例描述、停止点及图片哈希。
源目录和图片必须仍可访问，不能只搬走一个标注 JSON 后宣称证据有效。

## 3. 人工改写或采纳

先打开审核索引及 RGB，检查是否能唯一指代对应实例，再复制 CSV 编辑：

```bash
cp outputs/benchmark_language_annotations_001/annotation_sheet.csv /tmp/mivln_annotations.csv
```

使用文本编辑器或表格工具修改 `/tmp/mivln_annotations.csv`。仅修改 `text`、`language`、`author`、`note`；
`language` 支持 `en`、`zh-CN`，每条记录保存一个语言版本。修改或明确采纳模板都必须填写真实作者和具体说明。
未处理行保持原文且作者/说明留空；不能删除行、改变基础任务身份或改写真值标签。
**实际完成上述编辑后**执行：

```bash
python -B -m scripts.language_annotations revise \
  --input-dir outputs/benchmark_language_annotations_001 \
  --sheet /tmp/mivln_annotations.csv \
  --output-dir outputs/benchmark_language_annotations_edit_001
```

输出新版本，不覆盖旧版本。修改初始指令会将该任务三类介入的审核全部置回待审；修改介入只使该条审核失效。
新版本必须追加父版本事件，不能替换历史声明。CSV、审核模板是工作文件，手改它们不等于修改权威状态。

## 4. 独立人工复核

从**当前版本**复制 `review_decisions.json`，例如：

```bash
cp outputs/benchmark_language_annotations_edit_001/review_decisions.json /tmp/mivln_reviews.json
```

审核者结合该版本的 `review_index.md` 和 RGB，填写需要处理的行：

- `decision`：`approve`、`reject` 或 `needs_revision`；未处理行保持空白模板。
- `reviewer`、`note`：真实姓名/稳定署名和具体核查意见。批准者须与该条记录全部人工贡献者不同。
- `checks`：批准必须全部为 `true`；对应实例绑定、唯一指代、更新语义、无私有答案、自然语言质量。
- `checked_evidence_ids`：列出该条记录的全部必看证据 ID；不能用缺失或其他实例的证据代替。
- 保留内容/初始上下文哈希、数据集和版本 ID，不要手工“修复”过期审核文件。

先审核共享初始指令，再审核依赖它的三条介入；可按模板顺序在同一次导入中完成，也可逐批产生新版本。
语义核查须确认替换后 A 被取消、插入后仍恢复 A、取消不是暂时暂停。
如果图片或表达不足以唯一指代，选择 `needs_revision`，不能靠私有坐标补足面向 Agent 的语言。
**真实复核填写完成后**执行：

```bash
python -B -m scripts.language_annotations review \
  --input-dir outputs/benchmark_language_annotations_edit_001 \
  --decisions /tmp/mivln_reviews.json \
  --output-dir outputs/benchmark_language_annotations_review_001
```

不需要改写时，可直接对 `_001` 复核并输出新的版本；每次继续使用最新版本的模板。
空白模板、占位姓名、自我批准、缺少核查项或证据、过期版本/文本均被拒绝，整批失败时不写新版本。
审核是署名声明，工具没有身份认证，也不能证明审核者真的看过图片。正则只拦截已知 ID/路径/答案字段，语义歧义与其他泄漏仍需人工判断。

## 5. 只读审计与导出门槛

```bash
python -B -m scripts.language_annotations audit \
  --input-dir outputs/benchmark_language_annotations_review_001 \
  --output-dir outputs/benchmark_language_annotation_audit_next_001

python -B -m scripts.language_annotations export \
  --input-dir outputs/benchmark_language_annotations_review_001 \
  --output-dir outputs/benchmark_language_export_next_001
```

审计重放记录、核查版本链及源证据，并记录两份权威文件检查前后的哈希；**审计通过不表示人工审核通过**。
导出要求 40 条必需记录全部批准，30 个案例全部就绪，不默默筛掉待审或失败案例。
未达标返回码为 1，仅写 `export_report.json` 和私有的完整计划清单，不创建 `public/`。

通过后 `public/language.jsonl` 仅含随机样本 ID、初始/介入文本及语言；
私有文件另存任务类型、真值标签、来源配置、分组和审核版本关联。**整个标注包、图片路径及 `private/` 都不能交给普通 Agent。**
公开语料是离线文本资产，包含将来介入消息，不能作为在线 V 请求整体提前传入。
实际运行仍须逐步按事件投递。此入口不生成可运行的导航配置；新版本时机、场景划分和新语言导航验收尚未冻结，`release_ready=false`，`split=null`。

## 6. 本次验收与下一步

- 原草稿 `_001`：10 基础任务、40 条待审语言、30 个计划案例；保留原始状态，不回写审核。
- 原待审审计 `outputs/benchmark_language_annotation_audit_001/audit.json` 通过；`outputs/benchmark_language_export_check_001/` 保留审核前导出阻断的验证。
- 用户确认“我已审核好文件，采用原模版即可”“40条文本均可，RGB已验证”后，以已有署名 `WGZ` 转录 40 条审核声明。模板原文、作者及来源保持不变，不生成虚假的人工改写事件。
- 新审核版本 `outputs/benchmark_language_annotations_002/`：40/40 approved、30/30 language-ready；`outputs/benchmark_language_review_submission_001/` 保留决策、用户确认来源和核验报告。审核属于用户声明，不是工具自动完成的语义/视觉判断或身份认证。
- `outputs/benchmark_language_annotation_audit_002/audit.json`：审核版本只读审计通过；原草稿包及 51 份绑定来源文件前后不变。
- `outputs/benchmark_language_export_001/`：已导出 30 个离线语言案例，替换/插入/取消各 10 个，共享 10 个基础任务；公开字段及私有映射核验通过。
- 新增 18 项回归覆盖修订、独立审核、上下文失效、过期哈希、版本链、来源变动、完整分母和公开字段。
- 全量 358 项在 Habitat 环境通过；核心环境同套件通过，按原生依赖跳过 74 项。未启动渲染或真实模型。

可以直接检查当前审核版本，无须重复导入：

```bash
python -B -m scripts.language_annotations status \
  --input-dir outputs/benchmark_language_annotations_002
```

[三类 V 导航入口](benchmark_navigation.md)已实现已审语言绑定、私有 HSSD 配置准备和五条件 CPU 合成闭环，使用明确的新累计路程参考触发版本；原语言版本不变。
下一步实现真实 Habitat backend，接入实际 RGB、动作及独立实例诊断，再安排单基础任务 HSSD 验证；不能将完整离线语料提前交给在线 Agent。
首批只完成单场景模板语言审核，尚无新语言导航结果或多场景/scene-disjoint 划分，正式发布状态仍为 `release_ready=false`。
