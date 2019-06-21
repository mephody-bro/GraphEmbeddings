


def start_experiment(config):
    print(config)
    # embedder = get_embedder(config)
    # dataset = get_dataset(config['dataset'])
    # experiment = Experiment(embedder, dataset, config)
    # experiment.launch()



def parse_config(parser):
    parser.add_argument('--experiment', type=str, default="linker")
    parser.add_argument('--embedder', type=str, default="deepwalk")
    # parser.add_argument('--embedder', type=str, default="deepwalk")
    parser.add_argument('--dimensions', nargs='*', type=int, default=[4, 8, 16])





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parse_config(parser)
    args = parser.parse_args()

    start_experiment(args.__dict__)