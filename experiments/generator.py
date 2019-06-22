from copy import deepcopy


class ConfigGenerator:
    """
    Generates multiple configs from single configuration by iterating list of parameters
        I.e. instead of one config with 'dimensions' [4, 8, 16, 32]
        we'll get the four configs with 'dimension' 4, 8, 16, 32 respectively
    """
    def __init__(self, config):
        self.config = config

    def generate(self):
        configs = []

        for dimension in self.config['dimensions']:
            new_config = deepcopy(self.config)
            new_config['dimension'] = dimension
            configs.append(new_config)

        return configs

