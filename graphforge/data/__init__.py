"""GraphForge 数据层：合成图生成、载入、正负样本划分。"""

from __future__ import annotations

from .loader import build_graph, from_edge_list, from_networkx, load_builtin
from .split import split_edges
from .synthetic import (
    degree_corrected_sbm,
    karate_club,
    lfr_graph,
    sbm_from_avg_degree,
    stochastic_block_model,
)

__all__ = [
    "build_graph",
    "degree_corrected_sbm",
    "from_edge_list",
    "from_networkx",
    "karate_club",
    "lfr_graph",
    "load_builtin",
    "sbm_from_avg_degree",
    "split_edges",
    "stochastic_block_model",
]
