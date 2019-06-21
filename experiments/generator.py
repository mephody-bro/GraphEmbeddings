from copy import deepcopy


class ConfigGenerator:
    def __init__(self, config):
        self.config = config

    def generate(self):
        configs = []

        for dimension in self.config['dimensions']:
            new_config = deepcopy(self.config)
            new_config['dimension'] = dimension
            configs.append(new_config)

        return configs

