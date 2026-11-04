"""按任务与基准、方法、实验与分析三类贡献绘制项目整体框架。

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
    """用三个区域串联研究主线，图中文字均使用简体中文。"""
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

    text(60, 83, "项目方法框架与三类贡献", 43, align="left")
    text(60, 137, "面向动态人类意图与异步帮助的连续导航", 27, MUTED, "left")

    # 第一类贡献说明问题、任务和数据。策略可见输入与评价真值
    # 沿不同路径传递，避免把结构化正确答案画成语言模型的输入。
    box(60, 265, 510, 535, "一、任务与基准贡献", fill="#F3F7FC")
    text(315, 356, "动态意图 · 双向发起 · 异步回复", 25, MUTED)
    text(315, 435, "任务与事件协议", 29, BLUE)
    text(315, 480, "目标、约束与子目标", 25, MUTED)
    text(315, 560, "可复现任务数据", 29, BLUE)
    text(315, 605, "语言、场景划分与独立真值", 25, MUTED)
    text(315, 750, "成果：统一协议与数据集", 25)

    # 第二类贡献保留两个核心方法模块和环境闭环，具体模型与
    # 训练策略留待后续细化，不在初版图中堆叠网络或损失函数。
    box(650, 265, 680, 535, "二、方法贡献", fill="#F4F8F6")
    text(990, 356, "协同维护记忆，按需更新与交互", 25, MUTED)
    box(680, 410, 275, 150, "任务与空间记忆",
        ["语言理解与增量更新", "保留仍有效的要求"], fill="white")
    box(1025, 410, 275, 150, "交互与导航决策",
        ["权衡收益与交互代价", "行动、询问或等待"], fill="white")
    arrow([(955, 485), (1025, 485)])
    box(825, 655, 330, 90, "环境执行", fill="white")
    arrow([(1162, 560), (1162, 610), (1100, 610), (1100, 655)])
    text(1230, 600, "执行动作", 22, MUTED)
    arrow([(880, 655), (880, 610), (817, 610), (817, 560)])
    text(750, 590, "观测反馈", 22, MUTED)
    text(990, 775, "保留历史，持续执行", 23)

    # 第三类贡献强调控制变量和验证问题，不能把规划中的方法
    # 优势写成既有实验结论；未来通过配对、消融和跨场景检验。
    box(1410, 265, 450, 535, "三、实验与分析贡献", fill="#FCF8F2")
    text(1635, 356, "统一任务与人类帮助预算", 25, MUTED)
    text(1635, 435, "配对实验与模块消融", 29, BLUE)
    text(1635, 480, "比较记忆、求助与回复时机", 23, MUTED)
    text(1635, 560, "联合评价与失败分析", 29, BLUE)
    text(1635, 605, "任务完成、恢复效率与交互成本", 23, MUTED)
    text(1635, 648, "区分理解、落地、交互与执行错误", 22, MUTED)
    text(1635, 750, "成果：方法收益与失败归因", 25)

    # 区域之间的箭头表示研究流程；下方单独的真值路径只进入
    # 实验评价区域，不与策略的任务输入路径合并。
    arrow([(570, 490), (650, 490)])
    text(610, 460, "任务输入", 18, MUTED)
    arrow([(1330, 490), (1410, 490)])
    text(1370, 460, "运行记录", 18, MUTED)
    arrow([(315, 800), (315, 870), (1635, 870), (1635, 800)])
    text(975, 846, "独立任务真值，仅用于评价", 25, MUTED)

    text(960, 952, "定义问题 → 提出方法 → 验证效果", 31)
    text(60, 1020, "整体研究设计：语言任务更新、学习型记忆与主动交互仍需实现和验证。", 23, MUTED, "left")

    # 自动检查文字是否超出画布；模块内部留白与箭头另看导出的图片。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in labels:
        extent = label.get_window_extent(renderer)
        if not (0 <= extent.x0 <= extent.x1 <= 1920 and 0 <= extent.y0 <= extent.y1 <= 1080):
            raise ValueError(f"文字超出画布：{label.get_text()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        path = output_dir / f"project_method_framework.{extension}"
        fig.savefig(path,
                    dpi=200, facecolor="white", metadata={"Creator": "MIVLN-W"})
        if extension == "svg":
            # 导出器在多行路径坐标后附加空格；换行仍保留为坐标分隔，
            # 去掉行尾空白不会改变图形，并使生成文件通过版本检查。
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n",
                            encoding="utf-8")
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
