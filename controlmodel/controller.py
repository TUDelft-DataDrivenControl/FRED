from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
import zmq
from io import StringIO
import logging
logger = logging.getLogger("cm.controller")


class Controller:

    def __init__(self, wind_farm):
        self._control_type = conf.par.wind_farm.controller.type
        logger.info("Setting up controller of type: {}".format(self._control_type))
        self._wind_farm = wind_farm
        self._turbines = wind_farm.get_turbines()
        self._yaw_ref = []
        self._time_last_updated = 0

        if self._control_type == "series":
            self._time_series = conf.par.wind_farm.controller.yaw_series[:, 0]
            self._yaw_series = conf.par.wind_farm.controller.yaw_series[:, 1:]

        if self._control_type == "external":
            logger.info("Initialising ZMQ communication")
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            address = "tcp://localhost:{}".format(conf.par.wind_farm.controller.port)
            self._socket.connect(address)
            logger.info("Connected to: {}".format(address))

    def control_yaw(self, simulation_time):
        if (simulation_time - self._time_last_updated >= conf.par.wind_farm.controller.control_discretisation)\
                or self._yaw_ref == []:
            switcher = {
                "fixed": self._fixed_yaw,
                "series": self._fixed_time_series,
                "external": self._external_controller
            }
            controller_function = switcher.get(self._control_type)
            new_ref = controller_function(simulation_time)
            self._update_yaw(new_ref)
            self._time_last_updated = simulation_time

    def _fixed_yaw(self, simulation_time):
        new_ref = conf.par.wind_farm.yaw_angles.copy()
        return new_ref

    def _fixed_time_series(self, simulation_time):
        self._time_series = conf.par.wind_farm.controller.yaw_series[:, 0]
        self._yaw_series = conf.par.wind_farm.controller.yaw_series[:, 1:]
        new_ref = conf.par.wind_farm.yaw_angles.copy()
        for idx in range(len(self._turbines)):
            new_ref[idx] = np.interp(simulation_time, self._time_series, self._yaw_series[:,idx])
        return new_ref

    def _external_controller(self, simulation_time):
        # todo: measurements
        measurements = np.linspace(0, 6, 7)
        measurements[0] = simulation_time
        measurement_string = " ".join(["{:.6f}".format(x) for x in measurements]).encode()
        logger.warning("Real measurements not implemented yet")
        logger.info("Sending: {}".format(measurement_string))
        self._socket.send(measurement_string)

        message = self._socket.recv()
        # raw message contains a long useless tail with b'\x00' characters  (at least if from sowfa)
        # split off the tail before decoding into a Python unicode string
        json_data = message.split(b'\x00', 1)[0].decode()
        received_data = np.loadtxt(StringIO(json_data), delimiter=' ')
        logger.info("Received controls: {}".format(json_data))
        new_ref = received_data[0::2]

        return new_ref

    def _update_yaw(self, new_ref):
        if len(new_ref) != len(self._turbines):
            raise ValueError(
                "Computed yaw reference (length {:d}) does not match {:d} turbines in farm."
                .format(len(new_ref), len(self._turbines)))

        self._yaw_ref.append([Constant(y) for y in new_ref])
        [wt.set_yaw(y) for (wt, y) in zip(self._turbines, self._yaw_ref[-1])]

    def get_controls(self):
        return self._yaw_ref

    def clear_controls(self):
        self._yaw_ref = []
