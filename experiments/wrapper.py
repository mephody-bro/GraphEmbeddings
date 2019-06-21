
from .linker.experiment import LinkerExperiment


def get_experiment(config, embedder, graph):
    return LinkerExperiment(config, embedder, graph)
