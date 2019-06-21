import numpy as np

from .graph_sampler import GraphSampler
from .metric import calc_link_prediction_score


class LinkerExperiment:
    def __init__(self, config, embedder, graph):
        self.config = config
        self.embedder = embedder
        self.graph = graph

    def run(self):
        method, name, dim = self.config['embedder'], self.config['dataset'], self.config['dimension']
        a = self._calculate()
        print("{} {} {}: {}".format(method, name, str(dim), str(a)))
        return a

    def _calculate(self, ratio=0.5, seed=43):
        edges = self.graph.edges()
        nodes = self.graph.nodes()

        train_graph = GraphSampler(self.graph, ratio).fit_transform()

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

