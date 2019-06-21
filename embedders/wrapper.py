from .hope import Hope

#
def get_embedder(config, graph):
    if config['embedder'] == 'hope':
        return Hope(graph, config)
    else:
        raise RuntimeError('wrong embedder type')

