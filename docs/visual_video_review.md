# V 视频导出、逐帧来源与人工播放审核

入口：`python -B -m scripts.export_visual_videos`，支持 export、validate、review、status。
只处理已有真实 middle 批次的已通过审计证据，不调用模型、Habitat Simulator 或 GPU。
普通 Python 加 FFmpeg/FFprobe 即可；配置好的 Habitat 环境已经提供这两个工具。

## 1. 本次已完成

`outputs/hssd_v_middle_videos_001/` 已生成 8 段 H.264 MP4、同名 SRT、逐帧 JSONL 和默认待审的审核表。
按 10 fps 播放，共 **573 帧、57.3 秒**。编码与来源检查通过，**人工批准 0 条，待审 8 条**。

| 案例 | 条件 | 帧数 | 时长 / s | 介入视频时间 / s |
| --- | --- | ---: | ---: | ---: |
| task_001 替换 | Oracle / Ignore | 51 / 43 | 5.1 / 4.3 | 2.2 |
| task_001 插入恢复 | Oracle / Ignore | 88 / 43 | 8.8 / 4.3 | 2.2 |
| task_001 取消 | Oracle / Ignore | 23 / 43 | 2.3 / 4.3 | 2.2 |
| task_007 插入恢复 | Oracle / Ignore | 200 / 82 | 20.0 / 8.2 | 4.4 |

task_007 覆盖本批最长的插入恢复 Oracle 路线；视频选样仅用于质检，不改变 60 个评测分支的分母。
全部原始运行/准备/复用/审计源目录在导出前后快照一致，没有向旧源目录添加视频。
新增 16 项回归及全量 422 项在 Habitat 环境通过；核心环境按原生/编码依赖跳过 90 项。

## 2. 视频帧和字幕表示什么

视频严格按 `public/requests.jsonl` 的 step 排序，每次取当步实际发送的当前 RGB，不按随机 UUID 文件名排序，也不重复采样历史图像。
每帧保留 request ID、asset ID、源 PNG 路径/哈希、动作、投递消息及视频时间，见 `frames/*.jsonl`。
每个视频帧是**动作之前的观测**；SRT 的 next action 是该观测之后将执行的动作。
最后一帧对应实际 STOP 前的位姿，STOP 不移动，不添加不存在的动作后新观测。

原始 RGB 没有烧录文字，字幕为同名 UTF-8 SRT，精确标记动作、当步介入和之后已经收到的消息。
使用支持外置字幕的播放器同时打开 MP4/SRT；普通浏览器或 PPT 不一定自动加载 SRT。
可在字幕中搜索 `COMPLETE_SUBGOAL`、`STOP`，结合逐帧索引暂停核对；10 fps 只是播放速度，不是模型耗时或模拟器壁钟时间。
这些视频、字幕及完整索引是离线审核资产，包含未来观测/消息，不应整体提前提供给在线 Agent。

## 3. 现在需要做的人工步骤

打开 `outputs/hssd_v_middle_videos_001/review_index.md`。先完整播放 task_001 的三类/两条件，再播放 task_007 的两个插入恢复分支。
重点检查可见实例与描述、消息先于动作、B 完成后恢复 A、取消当步 STOP，以及 Ignore 忽略新要求的实际表现。
字幕和程序评分用于定位证据，不能代替完整播放。

复制工作表后编辑：

```bash
cp -n outputs/hssd_v_middle_videos_001/review_sheet.csv outputs/mivln_video_reviews.csv
```

编辑项目内的 `outputs/mivln_video_reviews.csv`；`cp -n` 避免覆盖已有填写结果。工作副本放在原视频包目录之外，以保持来源绑定有效。保留全部八行，video_id、任务/条件、原任务成功标签及视频哈希均不能修改。

| 字段 | 填写要求 |
| --- | --- |
| `decision` | approve / reject / needs_revision；尚未处理行保持整个默认 pending 行 |
| `reviewer` | 真实署名，不能留空或使用占位姓名 |
| `note` | 可留空；建议有问题时注明时间与具体表现 |
| `watched_full_video` | 实际完整播放后填 true |
| `rgb_quality` | 核对画面清晰度/可用性 |
| `intervention_timing` | 核对投递及动作时序 |
| `matches_recorded_actions` | 核对画面/字幕与实际动作记录一致 |
| `target_identity` | 替换/插入任务核对相关实例；取消不要求到达/看见原目标，保持 N/A |
| 类型行为项 | 仅填写适用的 replacement_behavior、subgoal_and_resumption 或 cancel_behavior；其他保持 N/A |

适用检查项必须填小写 true/false；批准要求全部适用项为 true。
对于 Ignore，批准表示**视频忠实展示了已审计的预期失败**，不是改判任务成功。未到 B 或未立即停恰好是需要确认的负对照行为。
如果画面不足以判断、帧序不清或存在异常，填 needs_revision/reject 并说明，不为完成流程强行批准。

实际核对并填写完成后导入新审核目录：

```bash
python -B -m scripts.export_visual_videos review \
  --input-dir outputs/hssd_v_middle_videos_001 \
  --sheet outputs/mivln_video_reviews.csv \
  --output-dir outputs/hssd_v_middle_video_review_001

python -B -m scripts.export_visual_videos status \
  --input-dir outputs/hssd_v_middle_videos_001 \
  --review-dir outputs/hssd_v_middle_video_review_001
```

可分批填写，未处理行保持默认；不能删除视频行。修改 CSV 本身不等于审核入库。
每次导入必须用新目录，不修改原视频包；重新审核时保留旧目录作为记录。
不指定 review-dir 的 status 始终显示原包的待审状态，避免自动猜测哪个审核版本生效。
审核是署名声明，不是身份认证或播放器观看证明；工具不会自动批准任何真实视频。

## 4. 自动校验及复现导出

已执行以下验证，返回 encoding_passed=true、pending=8、all_playback_approved=false：

```bash
conda activate habitat
python -B -m scripts.export_visual_videos validate \
  --input-dir outputs/hssd_v_middle_videos_001
```

validate 核对当前源目录与原独立审计快照一致，再核对导出帧序、字幕、文件哈希以及 FFprobe 的帧数、分辨率、帧率和时长。
它复用已完成的导航审计，不重新执行原生导航评价、不重新渲染图像；导航评分源文件或代码发生变化会被拒绝。
MP4 是兼容播放的有损 yuv420p，自动检查不声明其像素与 PNG 逐位一致，也不代表人工播放通过。

如需重导出，使用全新目录；已有 `_001` 不覆盖：

```bash
python -B -m scripts.export_visual_videos export \
  --input-dir outputs/hssd_v_middle_batch_run_001 \
  --audit-dir outputs/hssd_v_middle_batch_audit_001 \
  --selection representative --fps 10 \
  --output-dir outputs/hssd_v_middle_videos_002
```

`--selection all` 可导出全部 60 个分支，仍不增加基础任务数；默认 representative 固定上述八段。
工具优先从 PATH、其次从当前 Python 环境 bin 查找 FFmpeg/FFprobe，也可用 --ffmpeg/--ffprobe 显式指定。
编码失败保留完整计划及失败表，返回 1，不生成可审核 manifest；不能默默跳过坏视频后宣称全批通过。

## 5. 产物与边界

- `manifest.json`：八段视频身份、原任务标签、时间信息、MP4/SRT/逐帧索引哈希及代码绑定。
- `export_report.json`：完整导出计划、编码结果和源目录未改变的检查。
- `private/source_bindings.json`：原运行/准备/审计目录及完整文件快照绑定。
- `review_index.md`、`videos/`、`frames/`：人工查看入口、MP4/SRT 与逐帧来源。
- `review_sheet.csv`：可编辑工作表；新审核目录的 `review.json` 绑定精确视频包版本和人工声明。

导出校验、人工视频审核与导航任务成功分别报告。直接更改待审标签、使用过期包/视频哈希、缺少观看/核查项或删行均不能形成有效批准。
不能将视频输出写入受保护的运行、复用、准备或审计目录；源目录新增文件也可能使原绑定失效。
所有视频/审核产物在 Git 忽略的 outputs 中；只提交入口、测试和说明。

下一完整步骤是人工播放并导入八条审核记录，验收 all_playback_approved=true 或保留具体修订项。
质检完成后再推进第二场景、scene-disjoint 分组与 early/late 扩展；烧录字幕/PPT 拼接不是当前质检通过的前提。
