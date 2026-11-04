"""Render the current group-meeting framework as portable vector and PNG assets.

This is a documentation-only illustration, not an experiment or a simulator
entry. Coordinates use a 1920 x 1080 slide canvas, with y increasing downward.
Matplotlib and a Chinese font are needed only when regenerating the figure.
The SVG outlines glyphs for portability; the PDF embeds its font subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as DrawingPath


INK = "#152B45"
MUTED = "#53677D"
BLUE = "#2C66BA"
TEAL = "#087F86"
ORANGE = "#C06C1E"
LINE = "#6B8199"


def render(output_dir: Path, font_path: Path) -> None:
    """Save a 16:9 figure, keeping execution and evaluation information separate."""
    font = FontProperties(fname=str(font_path))
    matplotlib.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis("off")

    def text(x, y, value, size=24, color=INK, weight="normal", align="left"):
        # Convert desired pixels at the base DPI into Matplotlib point units.
        return ax.text(x, y, value, fontsize=size * .72, color=color,
                       fontproperties=font, weight=weight, ha=align, va="center")

    def box(x, y, width, height, fill="white", edge="#D8E2ED", radius=18):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill, edgecolor=edge, linewidth=1.2))

    def arrow(points, color=LINE, dashed=False):
        """Draw orthogonal connectors with the arrow only on the final segment."""
        if len(points) > 2:
            path = DrawingPath(points[:-1], [DrawingPath.MOVETO] +
                               [DrawingPath.LINETO] * (len(points) - 2))
            ax.add_patch(PathPatch(path, fill=False, color=color, linewidth=1.7,
                                   linestyle="--" if dashed else "-"))
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=16,
            linewidth=1.7, color=color, linestyle="--" if dashed else "-",
            shrinkA=0, shrinkB=3))

    def module(x, y, width, height, title, lines, color=BLUE, fill="#F2F7FE"):
        box(x, y, width, height, fill, color)
        text(x + 24, y + 37, title, 27, color, "bold")
        for index, line in enumerate(lines):
            text(x + 24, y + 83 + 36 * index, line, 23, MUTED)

    # The data strip describes offline task construction. It is deliberately
    # not an unlabelled input to a future nonprivileged language policy.
    text(64, 62, "动态任务更新与连续导航评测框架", 43, weight="bold")
    text(64, 111, "HSSD 家庭场景  /  目标替换 + 新增禁区  /  当前阶段：特权 Oracle 验证", 25, MUTED)
    box(64, 151, 1792, 67, "#F4F7FA", "#E2E9F0", 13)
    text(86, 184, "可复现任务基础", 25, BLUE, "bold")
    text(330, 184, "审核实例与可达停止点    ·    固定场景 / 导航网格    ·    同起点 / 朝向 / 种子    ·    输入哈希", 24, MUTED)

    text(64, 244, "介入输入", 21, MUTED)
    text(444, 244, "策略执行通道", 21, BLUE)
    module(64, 267, 300, 180, "② 单次事件调度", [
        "替换目标：A → B",
        "新增禁区：A + 约束",
        "动作前投递，仅生效一次",
    ], ORANGE, "#FFF8EF")
    module(444, 267, 360, 180, "① 显式策略任务状态", [
        "有效目标 / 已取消目标",
        "仍然有效的空间约束",
        "Follow 更新；Ignore 保持",
    ])
    module(884, 267, 410, 180, "Oracle 导航与重规划", [
        "坐标 / 网格 / 结构化更新",
        "目标替换或受约束绕行",
        "从当前位置继续选择动作",
    ])
    module(1374, 267, 482, 180, "Habitat-Sim 连续执行", [
        "前进 / 转向 / 最终 STOP",
        "介入不重置位姿与执行历史",
        "记录轨迹、RGB-D 与事件日志",
    ])
    arrow([(364, 351), (444, 351)], ORANGE)
    arrow([(804, 351), (884, 351)], BLUE)
    arrow([(1294, 351), (1374, 351)], BLUE)

    # Feedback is pose information for the current privileged planner. RGB-D
    # is recorded, not presented as visual perception by the Oracle.
    arrow([(1515, 447), (1515, 503), (1089, 503), (1089, 447)], BLUE)
    text(1305, 482, "位姿反馈", 21, BLUE, align="center")
    arrow([(1740, 447), (1740, 555)], TEAL)
    text(1720, 502, "实测证据", 21, TEAL, align="right")

    # Evaluation receives its own true task state even when Ignore retains the
    # old policy state; no evaluation diagnostic is routed back into the policy.
    text(444, 533, "独立评价通道", 21, TEAL)
    module(444, 555, 360, 159, "评价器任务真值", [
        "按事件更新目标与约束",
        "不依赖策略是否响应介入",
    ], TEAL, "#EFF9F7")
    arrow([(404, 351), (404, 635), (444, 635)], ORANGE)
    module(884, 555, 972, 159, "③ 独立联合验收", [
        "有效目标 STOP  +  实测距离 / 朝向  +  指定实例可见",
        "事件后运动线段违规检测；以任务真值判定成功与失败",
    ], TEAL, "#EFF9F7")
    arrow([(804, 635), (884, 635)], TEAL)

    # This panel describes matched runs, not three concurrent physical agents.
    box(64, 518, 300, 196, "#F7F9FC", "#D8E2ED")
    text(88, 552, "同任务配对对照", 25, INK, "bold")
    for y, label in [(595, "Control：无介入"),
                      (632, "Follow：响应更新"),
                      (669, "Ignore：保持原策略")]:
        text(88, y, label, 23, MUTED)

    text(64, 768, "本阶段方法贡献", 27, INK, "bold")
    contributions = [
        (64, BLUE, "#F2F7FE", "① 目标与约束的显式更新", [
            "区分目标取消与约束累加",
            "更新受影响部分，保留仍有效任务",
        ]),
        (672, ORANGE, "#FFF8EF", "② 介入后的状态连续执行", [
            "动作前一次投递，保留位姿与历史",
            "从当前位置重规划，保持 episode 连续",
        ]),
        (1280, TEAL, "#EFF9F7", "③ 配对负对照与独立验收", [
            "分离策略状态与评价真值",
            "识别错误终点与“到达但违规”",
        ]),
    ]
    for x, color, fill, heading, lines in contributions:
        box(x, 806, 576, 158, fill, color)
        text(x + 24, 844, heading, 28, color, "bold")
        for index, line in enumerate(lines):
            text(x + 24, 891 + index * 35, line, 24, MUTED)

    text(64, 1020, "当前边界：Oracle 使用特权坐标、导航网格与结构化更新；本图展示任务与评测机制，语言理解尚待接入。", 23, MUTED)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output_dir / f"method_framework.{extension}",
                    dpi=160, facecolor="white", metadata={"Creator": "MIVLN-W"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--font", type=Path,
                        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("Chinese font not found; provide a valid font file with --font")
    render(args.output_dir, args.font)
