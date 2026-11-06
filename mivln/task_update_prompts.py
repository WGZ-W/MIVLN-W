"""Public-information-only prompt conditions for task-update ablations.

This layer changes presentation/instructions, never candidate contents, labels,
state construction, or evaluation. No evaluator or reference module is imported.
"""

from __future__ import annotations

import hashlib
import json

from .task_update import validate_request


PROTOCOL = "task_update_prompt_ablation_v1"
VARIANTS = ("baseline", "history_timeline", "refusal_check")
OUTPUT_SUFFIX = (
    "\n本次调用只输出 prediction 对应的 JSON 对象。外层 request_id、request_sha256、"
    "prediction 包装由客户端生成，不要在模型回答中输出外层包装。\n")
REFUSAL_INSTRUCTIONS = """

输出前进行可回答性检查，仍然只输出规定的 JSON，不输出检查过程：
1. 当前消息引用的物体或区域必须有候选描述支持。没有匹配候选时，按引用类型输出 unknown_goal 或 unknown_region。
2. 存在多个符合描述的候选，或区域的区分特征未指定时，输出 ambiguous_goal 或 ambiguous_region。
   候选列表的先后顺序、同一类别或相近颜色都不能用来补足未给出的区分信息。
3. 更新所需的 old_goal 或 resume_goal 必须能由当前模式允许的输入唯一确定；缺少依据时输出 missing_context。
   不猜测常见目标，不借用其他请求，不把候选名称凭空作为旧目标或恢复目标。
4. 信息充分且引用唯一时执行对应更新；不要对明确可解释的消息一律拒绝。
拒绝仍使用 status=abstain 和相应 reason；成功仍使用原来的三类 task_update 格式。
"""


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def fingerprint(value):
    """Hash the semantic JSON body, independently of whitespace in a file."""
    canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_variant(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown task-update prompt variant")


def system_prompt(instructions, variant="baseline"):
    check_variant(variant)
    # Preserve legacy baseline bytes, including the final newline and envelope
    # clarification. This allows comparison to already completed model batches.
    extra = REFUSAL_INSTRUCTIONS if variant == "refusal_check" else ""
    return instructions + extra + OUTPUT_SUFFIX


def user_prompt(request, variant="baseline"):
    check_variant(variant)
    request = validate_request(request)
    if variant != "history_timeline" or request["mode"] != "history":
        return encode(request)
    # Only re-present permitted facts. In particular, do not call the rule
    # updater to compute an active goal or read a correct state from references.
    public = {key: value for key, value in request.items() if key not in ("message", "context")}
    context = request["context"]
    history = context["message_history"]
    lines = ["公共候选与请求标识（候选内容和顺序保持原样）：", encode(public),
        "初始指令（最早收到；不表示导航目标已经完成）：", encode(context["initial_instruction"]),
        "此前介入消息（按接收顺序排列，任务变更已生效；这些不是本轮新消息，也不表示导航已完成）："]
    lines.extend(f"{index}. {encode(message)}" for index, message in enumerate(history, 1))
    if not history:
        lines.append("[]")
    lines.extend(["当前消息（只预测这一条新介入；依据此前仍有效的任务解释原任务）：", encode(request["message"])])
    return "\n".join(lines)
