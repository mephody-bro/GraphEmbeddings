from .linker.experiment import LinkerExperiment


def get_experiment(config, embedder, graph):
    description = generate_description(config)

    if config['experiment'] == 'linker':
        experiment = LinkerExperiment(description, embedder, graph, ratio=config['link_predict_ratio'])
    else:
        raise RuntimeError("Unknown experiment")
    return experiment


def generate_description(config):
    return "{} {} {}".format(config['embedder'], config['dataset'], config['dimension'])
