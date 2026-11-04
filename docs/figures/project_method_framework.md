# 项目方法框架与创新思路：基础版

更新日期：2026-11-04。本图覆盖整个项目的研究方向，用简体中文先讲清基本流程。

## 框架

**人类指令 → 任务更新 → 导航决策 → 环境执行**

环境返回观测，机器人结合当前信息与执行历史继续更新任务；需要帮助时向人求助。
独立评价模块根据实际执行情况判断任务完成与交互代价。

## 三条创新思路

1. **动态任务更新**：要求变化时，保留仍有效的任务。
2. **双向主动交互**：人可以介入，机器人可以求助。
3. **连续执行与独立评价**：保留执行历史，独立判断完成情况。

这些是后续要验证的研究思路。当前已有三类结构化任务更新及导航评价基础，语言理解与主动求助仍待实现。

## 图稿

- [高清图片](project_method_framework.png)：3840×2160，适合插入演示文稿。
- [矢量图](project_method_framework.svg)：适合缩放。
- [打印版](project_method_framework.pdf)。
- [绘图源码](draw_project_method_framework.py)。

## 下一步

先用这一版确认模块关系和三条创新思路；确认后，每次只细化一个模块，例如任务更新。
放映时检查主线是否一眼能读懂、文字是否清晰、创新思路是否各用一句话说清楚。

需要修改后重新出图时，在仓库根目录执行：

```bash
MPLCONFIGDIR=/tmp/mivln-framework-mpl PYTHONDONTWRITEBYTECODE=1 \
  /mnt/sdc/weiguanzhao/miniconda3/envs/habitat/bin/python \
  docs/figures/draw_project_method_framework.py
```

以上命令仅生成图稿。本文、绘图源码和三种图稿可作为文档修改提交：
`docs: simplify project framework diagram in Chinese`。
