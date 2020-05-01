import controlmodel.conf as conf


class Controller:

    def __init__(self, wind_farm):
        self._control_type = conf.par.wind_farm.controller.type
        self._wind_farm = wind_farm
        self._turbines = wind_farm.get_turbines()
        self._yaw_ref = []
        print(self._control_type)

    def control_yaw(self, simulation_time):
        self._fixed_yaw(simulation_time)
        self._update_yaw()

    def _fixed_yaw(self, simulation_time):
        self._yaw_ref = conf.par.wind_farm.yaw_angles

    def _update_yaw(self):
        if len(self._yaw_ref) != len(self._turbines):
            raise ValueError(
                "Computed yaw reference (length {:d}) does not match {:d} turbines in farm."
                .format(len(self._yaw_ref), len(self._turbines)))

        [wt.set_yaw(y) for (wt,y) in zip(self._turbines, self._yaw_ref)]
