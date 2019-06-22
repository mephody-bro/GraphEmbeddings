import pytest

import networkx as nx
from embedders.hope import Hope
from numpy.linalg import norm


@pytest.fixture
def embedding():
    graph = nx.Graph([])
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    dimensions = 2

    embedder = Hope(graph, dimensions)

    return embedder.fit()


def test_output_shape(embedding):
    assert embedding.shape == (6, 2)


def test_distance(embedding):
    assert norm(embedding[0] - embedding[1]) < norm(embedding[0] - embedding[2])
