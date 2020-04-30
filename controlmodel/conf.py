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
        self.wind_farm = self.WindFarm(self._config["wind_farm"])
        self.turbine = self.Turbine(self._config["turbine"])

    class WindFarm:
        def __init__(self, config_dict):
            self.size = config_dict["size"]
            self.cells = config_dict["cells"]
            self.positions = config_dict["positions"]
            self.yaw_angles = config_dict["yaw_angles"]
            self.yaw_angles = [np.array(x) for x in self.yaw_angles]
            self.do_refine_turbines = config_dict["do_refine_turbines"]
            if self.do_refine_turbines:
                self.refine_radius = config_dict["refine_radius"]
            else:
                self.refine_radius = None

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
