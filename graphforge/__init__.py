"""GraphForge —— 图表示学习与链接预测系统。

忠实实现 node2vec（Grover & Leskovec, KDD 2016）：
biased random walk + skip-gram negative sampling + 边特征算子 + 下游分类器。

作者: 晨星
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "晨星"
__all__ = ["__version__", "__author__"]
