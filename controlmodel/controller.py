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
        self._yaw_control_type = conf.par.wind_farm.controller.yaw_control_type
        logger.info("Setting up yaw controller of type: {}".format(self._yaw_control_type))

        self._axial_induction_control_type = conf.par.wind_farm.controller.axial_induction_control_type
        logger.info("Setting up induction controller of type: {}".format(self._axial_induction_control_type))

        self._wind_farm = wind_farm
        self._turbines = wind_farm.get_turbines()

        self._yaw_ref = []
        self._axial_induction_ref = []

        self._time_last_updated_yaw = 0
        self._time_last_updated_induction = 0

        if self._yaw_control_type == "series":
            self._yaw_time_series = conf.par.wind_farm.controller.yaw_series[:, 0]
            self._yaw_series = conf.par.wind_farm.controller.yaw_series[:, 1:]

        if self._axial_induction_control_type == "series":
            self._axial_induction_time_series = conf.par.wind_farm.controller.axial_induction_series[:, 0]
            self._axial_induction_series = conf.par.wind_farm.controller.axial_induction_series[:,1:]

        if self._yaw_control_type == "external":
            self._received_data = []
            logger.info("Initialising ZMQ communication")
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            address = "tcp://localhost:{}".format(conf.par.wind_farm.controller.port)
            self._socket.connect(address)
            logger.info("Connected to: {}".format(address))

    def control(self, simulation_time):
        self.control_yaw(simulation_time)
        self.control_axial_induction(simulation_time)


    def control_yaw(self, simulation_time):
        if (simulation_time - self._time_last_updated_yaw >= conf.par.wind_farm.controller.control_discretisation)\
                or self._yaw_ref == []:
            switcher = {
                "fixed": self._fixed_yaw,
                "series": self._yaw_series_control,
                "external": self._external_controller
            }
            yaw_controller_function = switcher.get(self._yaw_control_type)
            new_ref = yaw_controller_function(simulation_time)
            self._update_yaw(new_ref)
            self._time_last_updated_yaw = simulation_time

    def control_axial_induction(self, simulation_time):
        if (simulation_time - self._time_last_updated_induction >= conf.par.wind_farm.controller.control_discretisation)\
                or self._axial_induction_ref == []:
            switcher = {
                "fixed": self._fixed_induction,
                "series": self._induction_series_control,
                "external": self._external_induction_controller
            }
            induction_controller_function = switcher.get(self._axial_induction_control_type)
            new_ref = induction_controller_function(simulation_time)
            self._update_axial_induction(new_ref)
            self._time_last_updated_induction = simulation_time

    def _fixed_yaw(self, simulation_time):
        new_ref = conf.par.wind_farm.yaw_angles.copy()
        return new_ref

    def _fixed_induction(self, simulation_time):
        new_ref = [wt.get_axial_induction() for wt in self._wind_farm.get_turbines()]
        return new_ref

    def _yaw_series_control(self, simulation_time):
        self._yaw_time_series = conf.par.wind_farm.controller.yaw_series[:, 0]
        self._yaw_series = conf.par.wind_farm.controller.yaw_series[:, 1:]
        new_ref = conf.par.wind_farm.yaw_angles.copy()
        for idx in range(len(self._turbines)):
            new_ref[idx] = np.interp(simulation_time, self._yaw_time_series, self._yaw_series[:, idx])
        return new_ref

    def _induction_series_control(self, simulation_time):
        self._axial_induction_time_series = conf.par.wind_farm.controller.axial_induction_series[:, 0]
        self._axial_induction_series = conf.par.wind_farm.controller.axial_induction_series[:, 1:]
        new_ref = []
        for idx in range(len(self._turbines)):
            new_ref.append(np.interp(simulation_time, self._axial_induction_time_series, self._axial_induction_series[:, idx]))
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
        self._received_data = np.loadtxt(StringIO(json_data), delimiter=' ')
        logger.info("Received controls: {}".format(json_data))
        new_ref = self._received_data[0::2]

        return new_ref

    def _external_induction_controller(self, simulation_time):
        logger.warning("External induction controller only works if yaw controller implemented as well")
        new_ref = self._received_data[1::2]
        return new_ref

    def _update_yaw(self, new_ref):
        if len(new_ref) != len(self._turbines):
            raise ValueError(
                "Computed yaw reference (length {:d}) does not match {:d} turbines in farm."
                .format(len(new_ref), len(self._turbines)))

        new_ref = self._apply_yaw_rate_limit(new_ref)

        self._yaw_ref.append([Constant(y) for y in new_ref])
        [wt.set_yaw_ref(y) for (wt, y) in zip(self._turbines, self._yaw_ref[-1])]
        logger.info("Set yaw to {}".format(new_ref))

    def _update_axial_induction(self, new_ref):
        if len(new_ref) != len(self._turbines):
            raise ValueError(
                "Computed axial induction reference (length {:d}) does not match {:d} turbines in farm."
                .format(len(new_ref), len(self._turbines)))

        self._axial_induction_ref.append([Constant(a) for a in new_ref])
        [wt.set_axial_induction(a) for (wt, a) in zip(self._turbines, self._axial_induction_ref[-1])]
        logger.info("Set axial induction to {}".format(new_ref))

    def _apply_yaw_rate_limit(self, new_ref):
        if len(self._yaw_ref)>0:
            yaw_rate_limit = conf.par.turbine.yaw_rate_limit
            time_step = conf.par.simulation.time_step
            new_ref = np.array(new_ref)
            prev_ref = np.array([float(y) for y in self._yaw_ref[-1]])
            delta_ref = new_ref - prev_ref
            if yaw_rate_limit > 0:
                delta_ref = np.min((yaw_rate_limit * time_step * np.ones_like(new_ref), delta_ref),0)
                delta_ref = np.max((-yaw_rate_limit * time_step * np.ones_like(new_ref), delta_ref),0)
            new_ref = prev_ref + delta_ref
        return new_ref

    def get_yaw_controls(self):
        return self._yaw_ref

    def get_axial_induction_controls(self):
        return self._axial_induction_ref

    def clear_controls(self):
        self._yaw_ref = []
        self._axial_induction_ref = []
