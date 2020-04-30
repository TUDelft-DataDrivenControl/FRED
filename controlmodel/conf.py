import yaml
import numpy as np


class ControlModelParameters:
    """
    Load parameters from .yaml file.
    """

    def __init__(self):
        self._config = None

    def load(self, file):
        try:
            self._load_configuration_from_yaml(file)
        except KeyError as ke:
            raise KeyError("Missing definition in config file, did not find {}".format(ke))

    def _load_configuration_from_yaml(self, file):
        stream = open(file, "r")
        self._config = yaml.load(stream=stream, Loader=yaml.SafeLoader)

    def print(self):
        print(yaml.dump(self._config))
