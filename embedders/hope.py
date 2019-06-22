import networkx as nx
import numpy as np
import scipy.sparse.linalg as lg


class Hope:
    """
    Implementation of HOPE graph embedding algorithm
    """
    def __init__(self, graph, dimension, name="", beta=0.01, show_error=False):
        """
        :param graph: graph to be embedded; it's supposed to be networkx graph object
        :param dimension: size of the target vector for node
        :param name: name of the graph
        """
        self.beta = beta
        self.graph = graph

        self.graph_name = name
        self.dim = dimension
        self.representation = None
        self.show_error = show_error

    def fit(self):
        if self.representation is None:
            self.representation = self.learn_embedding()
        return self.representation

    def learn_embedding(self):
        if not self.graph:
            raise Exception('graph is None')

        A = nx.to_numpy_matrix(self.graph)
        M_g = np.eye(self.graph.number_of_nodes()) - self.beta * A
        M_l = self.beta * A
        S = np.dot(np.linalg.inv(M_g), M_l)

        u, s, vt = lg.svds(S, k=self.dim // 2)
        X1 = np.dot(u, np.diag(np.sqrt(s)))
        X2 = np.dot(vt.T, np.diag(np.sqrt(s)))
        E = np.concatenate((X1, X2), axis=1)

        p_d_p_t = np.dot(u, np.dot(np.diag(s), vt))
        eig_err = np.linalg.norm(p_d_p_t - S)
        if self.show_error:
            print('SVD error (low rank): %f' % eig_err)
        return E
