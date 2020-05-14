from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
from controlmodel.turbine import Turbine
from controlmodel.controller import Controller
import logging
logger = logging.getLogger("cm.windfarm")

class WindFarm:

    def __init__(self):
        logger.info("Setting up wind farm")
        self._size = conf.par.wind_farm.size
        self._turbines = []
        turbine_yaw = [Constant(x) for x in conf.par.wind_farm.yaw_angles]
        turbine_positions = conf.par.wind_farm.positions
        self._turbines = [Turbine(x, y) for (x, y) in zip(turbine_positions, turbine_yaw)]

        self._controller = Controller(self)

    def get_turbines(self):
        return self._turbines

    def apply_controller(self, simulation_time):
        self._controller.control_yaw(simulation_time)

    def get_controls(self):
        return self._controller.get_controls()
