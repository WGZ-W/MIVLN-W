# 项目方法框架：三类贡献

更新日期：2026-11-04。本图依据 [项目完整贡献说明](../paper_contribution.md)，以简体中文展示整个项目的三类拟形成贡献。

## 框架

**任务与基准 → 方法设计 → 实验与分析**

方法部分保留“任务与空间记忆 → 交互与导航决策 → 环境执行”的核心流程，环境观测反馈给记忆模块。
任务描述、语言与允许的观测供策略使用；独立任务真值沿图中下方的路径仅送入评价部分。

## 三类贡献

1. **任务与基准贡献**：定义动态意图、双向发起和异步回复条件，交付统一协议及可复现数据集。
2. **方法贡献**：结合任务状态与空间记忆进行增量更新，权衡收益和交互代价，连续选择行动、询问或等待。
3. **实验与分析贡献**：在统一任务和人类帮助预算下，通过配对与消融分析完成情况、恢复效率、交互成本及失败原因。

三类贡献分别回答“研究什么”“怎样解决”“怎样证明有效”。这是完整研究目标；当前已有结构化任务更新和导航评价基础，
语言任务更新、学习型记忆与主动交互仍需实现和验证。图中“成果”表示预期研究产物。

## 图稿

- [高清图片](project_method_framework.png)：3840×2160，适合插入演示文稿。
- [矢量图](project_method_framework.svg)：适合缩放。
- [打印版](project_method_framework.pdf)。
- [绘图源码](draw_project_method_framework.py)。

## 下一步

先将本图用于项目总体介绍，依次讲清任务协议、核心方法和验证设计。
下一步只细化方法区域的语言任务更新模块，明确输入、输出和与历史拼接等基线的比较。
验收关注三类贡献是否对应明确产物、方法闭环是否清晰，以及评价真值是否独立。

需要修改后重新出图时，在仓库根目录执行：

```bash
MPLCONFIGDIR=/tmp/mivln-framework-mpl PYTHONDONTWRITEBYTECODE=1 \
  /mnt/sdc/weiguanzhao/miniconda3/envs/habitat/bin/python \
  docs/figures/draw_project_method_framework.py
```

以上命令仅生成图稿。本文、绘图源码和三种图稿可作为文档修改提交：
`docs: organize project framework around three contribution types`。
