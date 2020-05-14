import yaml
import numpy as np
import logging

logger = logging.getLogger("cm.conf")


class ControlModelParameters:
    """
    Load parameters from .yaml file.
    """

    def __init__(self):
        self._config = None
        self.wind_farm = None
        self.turbine = None
        self.simulation = None
        self.flow = None

    def load(self, file):
        logger.info("Loading configuration from: {}".format(file))
        self._load_configuration_from_yaml(file)
        try:
            self._assign_configuration()
        except KeyError as ke:
            message = "Missing definition in config file, did not find {}".format(ke)
            logger.error(message, exc_info=1)
            raise KeyError("Missing definition in config file, did not find {}".format(ke))
        logger.info("Loaded configuration.")

    def _load_configuration_from_yaml(self, file):
        stream = open(file, "r")
        self._config = yaml.load(stream=stream, Loader=yaml.SafeLoader)

    def print(self):
        print(yaml.dump(self._config))

    def _assign_configuration(self):
        self.wind_farm = self.WindFarm(self._config["wind_farm"])
        self.turbine = self.Turbine(self._config["turbine"])
        self.simulation = self.Simulation(self._config["simulation"])
        self.flow = self.Flow(self._config["flow"])

    class WindFarm:
        def __init__(self, config_dict):
            self.size = config_dict["size"]
            self.cells = config_dict["cells"]
            self.positions = config_dict["positions"]
            self.yaw_angles = config_dict["yaw_angles"]
            # self.yaw_angles = [np.array(x) for x in self.yaw_angles]
            self.do_refine_turbines = config_dict["do_refine_turbines"]
            if self.do_refine_turbines:
                self.refine_radius = config_dict["refine_radius"]
            else:
                self.refine_radius = None
            self.controller = self.FarmController(config_dict["farm_controller"])

        class FarmController:
            def __init__(self, config_dict):
                self.type = config_dict["type"]
                self.control_discretisation = config_dict["control_discretisation"]
                if self.type == "series":
                    self.yaw_series = np.array(config_dict["yaw_series"])

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

    class Simulation:
        def __init__(self, config_dict):
            self.is_dynamic = config_dict["is_dynamic"]
            # if not self.is_dynamic:
            #     raise NotImplementedError("Steady flow currently not implemented")
            if self.is_dynamic:
                self.total_time = config_dict["total_time"]
                self.time_step = config_dict["time_step"]
                self.write_time_step = config_dict["write_time_step"]
            self.name = config_dict["name"]
            self.save_logs = config_dict["save_logs"]
            self.dimensions = config_dict["dimensions"]

    class Flow:
        def __init__(self, config_dict):
            self.kinematic_viscosity = config_dict["kinematic_viscosity"]
            self.tuning_viscosity = config_dict["tuning_viscosity"]
            self.density = config_dict["density"]
            self.mixing_length = config_dict["mixing_length"]
            self.type = config_dict["type"]
            if self.type == "steady":
                self.inflow_velocity = config_dict["inflow_velocity"]
            elif self.type == "series":
                self.inflow_velocity_series = np.array(config_dict["inflow_velocity_series"])
                self.inflow_velocity = self.inflow_velocity_series[0, 1:3]


par = ControlModelParameters()
wind_farm = par.wind_farm
turbine = par.turbine
flow = par.flow
simulation = par.simulation

with_adjoint = True

if __name__ == '__main__':
    par = ControlModelParameters()
    par.load("../config/test_config.yaml")
    # par.print()
    # par.turbine.print()
