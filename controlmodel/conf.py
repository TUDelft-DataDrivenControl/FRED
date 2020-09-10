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
        self.ssc = None
        self.mode = None

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
        self.mode = self._config["mode"]
        if self.mode == "simulation":
            self.wind_farm = self.WindFarm(self._config["wind_farm"])
            self.turbine = self.Turbine(self._config["turbine"])
            self.simulation = self.Simulation(self._config["simulation"])
            self.flow = self.Flow(self._config["flow"])

        if self.mode == "supercontroller":
            self.ssc = self.SSC(self._config["ssc"])
            self.turbine = self.Turbine(self._config["turbine"])
            if self.ssc.type == "gradient_step":
                self.wind_farm = self.WindFarm(self._config["wind_farm"])
                self.simulation = self.Simulation(self._config["simulation"])
                self.flow = self.Flow(self._config["flow"])

    class WindFarm:
        def __init__(self, config_dict):
            self.size = config_dict["size"]
            self.cells = config_dict["cells"]
            self.positions = config_dict["positions"]
            self.yaw_angles = np.deg2rad(config_dict["yaw_angles"])
            # self.yaw_angles = [np.array(x) for x in self.yaw_angles]
            self.do_refine_turbines = config_dict["do_refine_turbines"]
            if self.do_refine_turbines:
                self.refine_radius = config_dict["refine_radius"]
            else:
                self.refine_radius = None
            self.controller = self.FarmController(config_dict["farm_controller"])

        class FarmController:
            def __init__(self, config_dict):
                self.yaw_control_type = config_dict["yaw_control_type"]
                self.control_discretisation = config_dict["control_discretisation"]
                if self.yaw_control_type == "series":
                    self.yaw_series = np.array(config_dict["yaw_series"])
                    self.yaw_series[:, 1:] = np.deg2rad(self.yaw_series[:, 1:])
                if self.yaw_control_type == "external":
                    self.port = config_dict["port"]

                self.axial_induction_control_type = config_dict.get("axial_induction_control_type", "fixed")
                if self.axial_induction_control_type == "series":
                    self.axial_induction_series = np.array(config_dict["axial_induction_series"])

                # todo: refine control settings
                self.pitch_control_type = config_dict.get("pitch_control_type", "none")
                self.torque_control_type = config_dict.get("torque_control_type", "none")


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
            self.kernel = config_dict["kernel"]
            self.deflection_scale = config_dict["deflection_scale"]
            self.yaw_rate_limit = np.deg2rad(config_dict.get("yaw_rate_limit",-1))
            self.coefficients = config_dict.get("coefficients", "induction")
            self.pitch = config_dict.get("pitch", 0.)
            self.torque = config_dict.get("torque", 0.)

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
            self.wake_mixing_length = config_dict["wake_mixing_length"]
            self.wake_mixing_width = config_dict["wake_mixing_width"]
            self.wake_mixing_offset = config_dict["wake_mixing_offset"]
            self.wake_mixing_ml_max = config_dict["wake_mixing_ml_max"]
            self.continuity_correction = config_dict["continuity_correction"]
            self.type = config_dict["type"]
            if self.type == "steady":
                self.inflow_velocity = config_dict["inflow_velocity"]
            elif self.type == "series":
                self.inflow_velocity_series = np.array(config_dict["inflow_velocity_series"])
                self.inflow_velocity = self.inflow_velocity_series[0, 1:3]

    class SSC:
        def __init__(self, config_dict):
            self.port = config_dict["port"]
            self.type = config_dict["type"]
            self.control_discretisation = config_dict["control_discretisation"]
            if self.type == "fixed":
                self.yaw_angles = np.deg2rad(config_dict["yaw_angles"])
            if self.type == "series":
                self.yaw_series = np.array(config_dict["yaw_series"])
                self.yaw_series[:,1:] = np.deg2rad(self.yaw_series[:,1:])
                self.yaw_angles = self.yaw_series[0, 1:]
                self.axial_induction_series = np.array(config_dict["axial_induction_series"])
            if self.type == "gradient_step":
                self.yaw_angles = np.deg2rad(config_dict["yaw_angles"])
                self.prediction_horizon = config_dict["prediction_horizon"]
                self.control_horizon = config_dict["control_horizon"]
                self.transient_time = config_dict.get("transient_time",-1)
                self.objective = config_dict["objective"]
                if self.objective == "tracking":
                    self.power_reference = np.array(config_dict["power_reference"])
                    self.power_reference[:, 1] *= 1e6


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
