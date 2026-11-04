"""绘制整个项目的基础框架与三项创新思路，导出中文矢量图和高清图片。

本版用于先说明研究主线，省略实现细节和实验指标。画布为 1920×1080，
原点在左上角。绘图不依赖导航程序，不创建模拟器或运行实验。
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


INK = "#24364B"
MUTED = "#647180"
BLUE = "#4D729B"
LINE = "#8797A9"


def render(output_dir: Path, font_path: Path) -> None:
    """仅保留五个模块，图中文字均使用简体中文。"""
    font = FontProperties(fname=str(font_path))
    matplotlib.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis("off")
    labels = []

    def text(x, y, value, size=28, color=INK, align="center"):
        # 字号按基础画布像素换算为点；提高导出分辨率时布局不变。
        label = ax.text(x, y, value, fontsize=size * .72,
                        fontproperties=font, color=color, ha=align, va="center")
        labels.append(label)

    def box(x, y, width, height, title, lines=(), fill="#F5F8FC"):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0,rounding_size=12",
            facecolor=fill, edgecolor="#BCC9D7", linewidth=1.3))
        text(x + width / 2, y + 43, title, 32)
        for index, line in enumerate(lines):
            text(x + width / 2, y + 91 + index * 33, line, 24, MUTED)

    def arrow(points, dashed=False):
        """折线绕开模块，保留箭头方向，避免文字和连线叠在一起。"""
        style = (0, (5, 4)) if dashed else "-"
        if len(points) > 2:
            ax.add_patch(PathPatch(
                DrawingPath(points[:-1], [DrawingPath.MOVETO] +
                            [DrawingPath.LINETO] * (len(points) - 2)),
                fill=False, color=LINE, linewidth=1.8, linestyle=style))
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=17,
            color=LINE, linewidth=1.8, linestyle=style, shrinkA=0, shrinkB=5))

    text(80, 83, "项目方法框架（基础版）", 43, align="left")
    text(80, 137, "理解任务变化，在交互中持续完成导航", 27, MUTED, "left")

    # 一条主线说明整个项目，不把当前特权导航实现当作完整语言方法。
    box(80, 310, 300, 165, "人类指令", ["初始要求与后续介入"])
    box(490, 310, 350, 165, "任务更新", ["目标、约束与子目标", "结合当前观测与历史"])
    box(950, 310, 350, 165, "导航决策", ["行动、询问或等待"])
    box(1410, 310, 400, 165, "环境执行", ["执行动作，返回观测"])
    for start, end in ((380, 490), (840, 950), (1300, 1410)):
        arrow([(start, 392), (end, 392)])

    # 求助是整体研究方向，用虚线配合页脚说明其尚待实现。
    arrow([(1125, 310), (1125, 235), (230, 235), (230, 310)], dashed=True)
    text(685, 210, "主动求助", 25, BLUE)

    # 环境观测回到任务更新模块；历史信息由智能体维护。
    arrow([(1510, 475), (1510, 550), (665, 550), (665, 475)])
    text(1080, 526, "观测反馈", 25, BLUE)

    # 评价单独接收实际执行结果，不画真值回流到语言策略的箭头。
    box(1410, 625, 400, 145, "独立评价", ["判断任务完成与交互代价"], fill="#F4F8F6")
    arrow([(1710, 475), (1710, 625)])
    text(1688, 579, "执行结果", 24, MUTED, "right")

    text(80, 766, "创新思路（待验证）", 29, align="left")
    # 每项只保留一句话，后续根据实验逐步细化，不预设性能优势。
    cards = [
        (80, "动态任务更新", "要求变化时，保留仍有效的任务"),
        (670, "双向主动交互", "人可以介入，机器人可以求助"),
        (1260, "连续执行与独立评价", "保留执行历史，独立判断完成情况"),
    ]
    for x, title, description in cards:
        box(x, 810, 550, 135, title, [description], fill="white")

    text(80, 1017, "整体研究思路：语言理解与主动求助仍待实现。", 23, MUTED, "left")

    # 自动检查文字是否超出画布；模块内部留白与箭头另看导出的图片。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in labels:
        extent = label.get_window_extent(renderer)
        if not (0 <= extent.x0 <= extent.x1 <= 1920 and 0 <= extent.y0 <= extent.y1 <= 1080):
            raise ValueError(f"文字超出画布：{label.get_text()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(output_dir / f"project_method_framework.{extension}",
                    dpi=200, facecolor="white", metadata={"Creator": "MIVLN-W"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--font", type=Path,
                        default=Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"))
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("未找到中文字体，请用 --font 指定字体文件")
    render(args.output_dir, args.font)
