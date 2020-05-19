import numpy as np
import controlmodel.conf as conf

from controlmodel.zmqserver import ZmqServer
import logging
logger = logging.getLogger("cm.ssc")


class SuperController:

    def __init__(self):
        self._control_type = conf.par.ssc.type
        self._server = None  # ZmqServer(conf.par.wind_farm.controller.port)
        self._yaw_reference = conf.par.wind_farm.yaw_angles.copy()
        # todo: pitch reference may be useful later for work with SOWFA
        self._pitch_reference = np.zeros_like(self._yaw_reference)
        logger.info("SSC initialised")

        if self._control_type == "series":
            self._time_series = conf.par.ssc.yaw_series[:, 0]
            self._yaw_series = conf.par.ssc.yaw_series[:, 1:]

    def start(self):
        self._server = ZmqServer(conf.par.wind_farm.controller.port)
        logger.info("SSC started")

        while True:
            sim_time, measurements = self._server.receive()
            if sim_time % conf.par.ssc.control_discretisation < conf.par.simulation.time_step:
                self._set_yaw_reference(simulation_time=sim_time)
            self._server.send(self._yaw_reference, self._pitch_reference)
            logger.info("Sent control signals for time: {:.2f}".format(sim_time))

    def _set_yaw_reference(self, simulation_time):
        switcher = {
            "fixed": self._fixed_reference,
            "series": self._time_series_reference
        }
        control_function = switcher[self._control_type]
        control_function(simulation_time)

    def _fixed_reference(self, simulation_time):
        return self._yaw_reference

    def _time_series_reference(self, simulation_time):
        for idx in range(len(self._yaw_reference)):
            self._yaw_reference[idx] = np.interp(simulation_time, self._time_series, self._yaw_series[:, idx])
