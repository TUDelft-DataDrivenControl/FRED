import numpy as np
import controlmodel.conf as conf

from controlmodel.zmqserver import ZmqServer
import logging
logger = logging.getLogger("cm.ssc")


class SuperController:

    def __init__(self):
        self._server = None  # ZmqServer(conf.par.wind_farm.controller.port)
        self._yaw_reference = conf.par.wind_farm.yaw_angles.copy()
        # todo: pitch reference may be useful later for work with SOWFA
        self._pitch_reference = np.zeros_like(self._yaw_reference)
        logger.info("SSC initialised")

    def start(self):
        self._server = ZmqServer(conf.par.wind_farm.controller.port)
        self._yaw_reference = conf.par.wind_farm.yaw_angles.copy()
        # todo: pitch reference may be useful later for work with SOWFA
        self._pitch_reference = np.zeros_like(self._yaw_reference)
        logger.info("SSC started")
        while True:
            sim_time, measurements = self._server.receive()
            self._server.send(self._yaw_reference, self._pitch_reference)
            logger.info("Sent control signals for time: {:.2f}".format(sim_time))
