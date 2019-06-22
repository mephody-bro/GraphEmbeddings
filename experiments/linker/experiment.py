import numpy as np

from .graph_sampler import GraphSampler
from .metric import calc_link_prediction_score


class LinkerExperiment:
    """
    Experiment to evaluate embedder algorithms for link prediction task
    Remove part of the vertices and expects embedding would help to correctly predicts missing
    """
    def __init__(self, description, embedder, graph, ratio):
        """
        :param description: is used when printing results of the experiment
        :param embedder: how we'll embed graph to vectors
        :param graph
        :param ratio: which ratio of vertices to be deleted
        """
        self.description = description
        self.embedder = embedder
        self.graph = graph
        self.ratio = ratio

    def run(self):
        a = self._calculate()
        print("{}: {}".format(self.description, str(a)))
        return a

    def _calculate(self, seed=43):
        edges = self.graph.edges()
        nodes = self.graph.nodes()

        train_graph = GraphSampler(self.graph, self.ratio).fit_transform()

        E = self.embedder.fit()

        train_edges = train_graph.edges()
        edges_set = set(edges)
        train_edges_set = set(train_edges)

        test_edges_set = edges_set - train_edges_set

        np.random.seed(seed)
        test_neg_edges_set = self._generate_negative_set(nodes, test_edges_set)
        train_neg_edges_set = self._generate_negative_set(nodes, train_edges_set)

        return calc_link_prediction_score(E,
                                          train_edges,
                                          list(train_neg_edges_set),
                                          list(test_edges_set),
                                          list(test_neg_edges_set),
                                          score='roc-auc')

    @staticmethod
    def _generate_negative_set(nodes, positive_set):
        generated = set()
        while len(generated) < len(positive_set):
            edge = np.random.choice(nodes), np.random.choice(nodes)
            if edge not in positive_set:
                generated.add(edge)
        return generated
