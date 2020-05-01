import controlmodel.conf as conf
import numpy as np


class Controller:

    def __init__(self, wind_farm):
        self._control_type = conf.par.wind_farm.controller.type
        self._wind_farm = wind_farm
        self._turbines = wind_farm.get_turbines()
        self._yaw_ref = conf.par.wind_farm.yaw_angles
        if self._control_type == "series":
            self._time_series = conf.par.wind_farm.controller.yaw_series[:,0]
            self._yaw_series = conf.par.wind_farm.controller.yaw_series[:,1:]

    def control_yaw(self, simulation_time):
        switcher = {
            "fixed": self._fixed_yaw,
            "series": self._fixed_time_series
        }
        controller_function = switcher.get(self._control_type)
        controller_function(simulation_time)
        self._update_yaw()

    def _fixed_yaw(self, simulation_time):
        self._yaw_ref = conf.par.wind_farm.yaw_angles

    def _fixed_time_series(self, simulation_time):
        for idx in range(len(self._turbines)):
            self._yaw_ref[idx] = np.interp(simulation_time, self._time_series, self._yaw_series[:,idx])

    def _update_yaw(self):
        if len(self._yaw_ref) != len(self._turbines):
            raise ValueError(
                "Computed yaw reference (length {:d}) does not match {:d} turbines in farm."
                .format(len(self._yaw_ref), len(self._turbines)))

        for (wt, y) in zip(self._turbines, self._yaw_ref):
            if wt.get_yaw() != y:
                wt.set_yaw(y)
        # [wt.set_yaw(y) for (wt,y) in zip(self._turbines, self._yaw_ref)]
