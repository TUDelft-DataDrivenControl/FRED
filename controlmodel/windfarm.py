from fenics import *
from fenics_adjoint import *
import controlmodel.conf as conf
from controlmodel.turbine import Turbine
from controlmodel.controller import Controller


class WindFarm:

    def __init__(self):
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
