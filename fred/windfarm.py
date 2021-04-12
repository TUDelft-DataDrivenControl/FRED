from fenics import *
import fred.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
from fred.turbine import Turbine
from fred.controller import Controller
import logging
logger = logging.getLogger("cm.windfarm")


class WindFarm:
    """Wind farm class with turbine objects and a farm controller.

     """

    def __init__(self):
        logger.info("Setting up wind farm")
        self._size = conf.par.wind_farm.size

        self._turbines = []
        turbine_yaw = [Constant(x) for x in conf.par.wind_farm.yaw_angles]
        turbine_positions = conf.par.wind_farm.positions
        self._turbines = [Turbine(x, y) for (x, y) in zip(turbine_positions, turbine_yaw)]
        logger.info("Added {} turbines to farm".format(len(self._turbines)))

        self._controller = Controller(self)
        logger.info("Added controller to farm")

    def get_turbines(self):
        """Returns list of turbine objects in wind farm."""
        return self._turbines

    def apply_controller(self, simulation_time):
        """Update wind farm controls for current time point in simulation

        Args:
            simulation_time (float): time instance in simulation

        """
        self._controller.control(simulation_time)

    def get_yaw_controls(self):
        """Get a list of yaw control signals from controller

        Returns:
            list: yaw controls over simulation time segment

        """
        return self._controller.get_yaw_controls()

    def get_axial_induction_controls(self):
        """Get a list of axial induction control signals from controller

        Returns:
            list: yaw controls over simulation time segment

        """
        return self._controller.get_axial_induction_controls()

    def get_pitch_controls(self):
        return self._controller.get_pitch_controls()

    def get_torque_controls(self):
        return self._controller.get_torque_controls()

    def get_controls_list(self, name):
        return  self._controller.get_controls_list(name)

    def set_control_reference_series(self, name, time_series, reference_series):
        self._controller.set_control_reference_series(name, time_series, reference_series)

    def clear_controls(self):
        """Clear the recorded list of control signals from the wind farm controller"""
        self._controller.clear_controls()
