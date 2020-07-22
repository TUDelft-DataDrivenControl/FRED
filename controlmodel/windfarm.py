from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
from controlmodel.turbine import Turbine
from controlmodel.controller import Controller
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

    def clear_controls(self):
        """Clear the recorded list of control signals from the wind farm controller"""
        self._controller.clear_controls()