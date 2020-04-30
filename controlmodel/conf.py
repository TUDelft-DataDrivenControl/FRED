import yaml
import numpy as np


class ControlModelParameters:
    """
    Load parameters from .yaml file.
    """

    def __init__(self):
        self._config = None

    def load(self, file):
        self._load_configuration_from_yaml(file)
        try:
           self._assign_configuration()
        except KeyError as ke:
            raise KeyError("Missing definition in config file, did not find {}".format(ke))

    def _load_configuration_from_yaml(self, file):
        stream = open(file, "r")
        self._config = yaml.load(stream=stream, Loader=yaml.SafeLoader)

    def print(self):
        print(yaml.dump(self._config))

    def _assign_configuration(self):
        self.turbine = self.Turbine(self._config["turbine"])

    class Turbine:
        """
        Turbine configuration class
        """
        def __init__(self,config_dict):
            self.axial_induction = config_dict["axial_induction"]
            self.diameter = config_dict["diameter"]
            self.radius = self.diameter / 2
            self.thickness = config_dict["thickness"]
            self.hub_height = config_dict["hub_height"]


if __name__ == '__main__':
    par = ControlModelParameters()
    par.load("../config/test_config.yaml")
    # par.print()
    # par.turbine.print()
