"""pytest 公共夹具。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphforge.core.seed import derive_seed, make_rng  # noqa: E402
from graphforge.data.split import split_edges  # noqa: E402
from graphforge.data.synthetic import karate_club, stochastic_block_model  # noqa: E402


@pytest.fixture(scope="session")
def karate_graph():
    return karate_club()


@pytest.fixture(scope="session")
def karate_split(karate_graph):
    return split_edges(
        karate_graph, 0.7, 0.1, 0.2, 1.0, make_rng(derive_seed(0, "split"))
    )


@pytest.fixture(scope="session")
def sbm_graph():
    return stochastic_block_model(120, n_communities=3, p_in=0.18, p_out=0.02,
                                  rng=make_rng(11))


@pytest.fixture(scope="session")
def sbm_split(sbm_graph):
    return split_edges(sbm_graph, 0.7, 0.1, 0.2, 1.0, make_rng(derive_seed(3, "split")))


@pytest.fixture
def triangle_graph():
    """三条边的三角形 + 一条悬挂边的最小玩具图。"""
    adjacency = np.zeros((4, 4), dtype=np.float64)
    for u, v in ((0, 1), (1, 2), (0, 2), (2, 3)):
        adjacency[u, v] = 1.0
        adjacency[v, u] = 1.0
    from graphforge.core.types import GraphData

    return GraphData(adjacency=adjacency, name="toy")
