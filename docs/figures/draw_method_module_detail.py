"""绘制项目中间方法模块的中文细化图。

只展开信息流和核心决策，不预设模型、训练损失或已验证的性能优势。
画布为 1920×1080，原点在左上角。生成图稿不运行导航或模拟器。
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
LINE = "#8797A9"
BLUE = "#4D729B"


def render(output_dir: Path, font_path: Path) -> None:
    """导出具有任务、观测和人类交互闭环的方法设计。"""
    font = FontProperties(fname=str(font_path))
    matplotlib.rcParams.update({"svg.fonttype": "path", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1920)
    ax.set_ylim(1080, 0)
    ax.axis("off")
    labels = []

    def text(x, y, value, size=25, color=INK, align="center"):
        label = ax.text(x, y, value, fontsize=size * .72, color=color,
                        fontproperties=font, ha=align, va="center")
        labels.append(label)

    def box(x, y, width, height, title, lines, fill="#F5F8FC"):
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0,rounding_size=12",
            facecolor=fill, edgecolor="#BCC9D7", linewidth=1.3))
        text(x + width / 2, y + 40, title, 30)
        for index, line in enumerate(lines):
            text(x + width / 2, y + 89 + index * 34, line, 24, MUTED)

    def arrow(points, dashed=False):
        """让主流程、已有状态回传与观测反馈各有独立连线。"""
        style = (0, (5, 4)) if dashed else "-"
        if len(points) > 2:
            ax.add_patch(PathPatch(
                DrawingPath(points[:-1], [DrawingPath.MOVETO] +
                            [DrawingPath.LINETO] * (len(points) - 2)),
                fill=False, color=LINE, linewidth=1.8, linestyle=style))
        ax.add_patch(FancyArrowPatch(
            points[-2], points[-1], arrowstyle="-|>", mutation_scale=17,
            linewidth=1.8, color=LINE, linestyle=style, shrinkA=0, shrinkB=5))

    text(60, 75, "方法模块细化：记忆、任务更新与交互决策", 41, align="left")
    text(60, 132, "对应整体框架中的“方法贡献”，展示拟研究的信息流与决策闭环", 26, MUTED, "left")

    # 人类主动介入和对求助的回复都从语言通道进入。回复可能
    # 延迟，不能默认最后到达的消息必然覆盖当前有效任务。
    box(60, 300, 300, 180, "人类消息", ["初始指令与新介入", "即时或延迟的人类回复"])
    box(450, 300, 400, 180, "语言理解与增量更新", [
        "解析意图与目标指代", "判断取消、保留或暂挂"], fill="#F3F7FC")
    box(940, 300, 400, 180, "动态任务状态", [
        "目标、约束及任务恢复状态", "关联任务版本与待答问题"], fill="#F4F8F6")
    box(1430, 300, 400, 180, "交互决策", [
        "评估不确定性、风险与收益", "权衡提问与等待代价"], fill="#FCF8F2")

    arrow([(360, 390), (450, 390)])
    text(405, 365, "新消息", 20, MUTED)
    arrow([(850, 390), (940, 390)])
    text(895, 365, "更新结果", 18, MUTED)
    arrow([(940, 445), (850, 445)])
    text(895, 470, "已有状态", 18, MUTED)
    arrow([(1340, 390), (1430, 390)])
    text(1385, 365, "有效任务", 18, MUTED)

    # 底部构成连续导航闭环：决策产生行动，环境返回观测，
    # 记忆为下一次指代落地、任务更新和求助判断提供上下文。
    box(450, 650, 400, 160, "空间与历史记忆", [
        "积累地标、轨迹与可通行信息", "支持指代落地与任务恢复"], fill="#F3F7FC")
    box(940, 650, 400, 160, "环境执行", [
        "执行动作，返回视觉观测", "记录位置与任务进展"])
    box(1430, 650, 400, 160, "导航与等待", [
        "按当前任务继续执行", "保留位姿与执行历史"], fill="#F4F8F6")
    arrow([(1630, 480), (1630, 650)])
    text(1654, 560, "行动或等待", 24, MUTED, "left")
    arrow([(1430, 730), (1340, 730)])
    text(1385, 705, "动作", 22, MUTED)
    arrow([(940, 730), (850, 730)])
    text(895, 705, "观测", 22, MUTED)
    arrow([(650, 650), (650, 480)])
    text(620, 570, "空间与历史证据", 24, MUTED, "right")
    arrow([(800, 650), (800, 560), (1510, 560), (1510, 480)])
    text(1150, 535, "空间信息与执行进展", 24, BLUE)

    # 主动求助回到人类；人类回复随后从同一消息入口进入更新。
    # 虚线只突出交互路径，不代表该模块已经完成实现。
    arrow([(1630, 300), (1630, 220), (210, 220), (210, 300)], dashed=True)
    text(920, 195, "按需提问", 25, BLUE)

    text(960, 907, "增量更新任务 · 保留有效上下文 · 权衡交互代价", 30)
    text(60, 1008, "拟研究方法：现有任务状态与执行后端提供基础；语言更新、学习型记忆和主动交互仍待实现。",
         22, MUTED, "left")

    # 文字边界自动检查，箭头关系与节点内留白另在图片中查看。
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for label in labels:
        extent = label.get_window_extent(renderer)
        if not (0 <= extent.x0 <= extent.x1 <= 1920 and 0 <= extent.y0 <= extent.y1 <= 1080):
            raise ValueError(f"文字超出画布：{label.get_text()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "pdf", "png"):
        path = output_dir / f"method_module_detail.{extension}"
        fig.savefig(path, dpi=200, facecolor="white", metadata={"Creator": "MIVLN-W"})
        if extension == "svg":
            # 换行仍作为路径坐标分隔；清理导出器行尾空白供版本检查。
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
