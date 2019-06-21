import networkx as nx


graphs_info = {
    'football': {
        'path': 'datasets/cluster-datasets/football.txt',
        'weighted': False
    }
}


def get_graph(dataset_name):
    graph_info = graphs_info[dataset_name]
    graph = nx.read_edgelist(graph_info['path'], nodetype=int)
    if graph_info['weighted']:
        for edge in graph.edges():
            graph[edge[0]][edge[1]]['weight'] = 1
    return graph

