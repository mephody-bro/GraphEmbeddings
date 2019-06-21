from datasets import get_graph
from embedders import get_embedder
from experiments import get_experiment


def start_experiment(config):
    graph = get_graph(config['dataset'])
    embedder = get_embedder(config, graph)
    experiment = get_experiment(config, embedder, graph)
    experiment.run()


def parse_config(parser):
    parser.add_argument('--experiment', type=str, default="linker")
    parser.add_argument('--embedder', type=str, default="deepwalk")
    parser.add_argument('--dataset', type=str, default="football")
    # parser.add_argument('--dimensions', nargs='*', type=int, default=[4, 8, 16])
    parser.add_argument('--dimension', type=int, default=8)
    parser.add_argument('--path_to_dumps', type=str, default="dumps")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parse_config(parser)
    args = parser.parse_args()

    start_experiment(args.__dict__)