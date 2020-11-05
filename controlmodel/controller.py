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
        self._wind_farm = wind_farm
        self._turbines = wind_farm.get_turbines()

        self._controls = {}
        for control in conf.par.wind_farm.controller.controls:
            self._controls[control] = (Control(name=control,
                                          control_type=conf.par.wind_farm.controller.controls[control]["type"],
                                          value=np.array(conf.par.wind_farm.controller.controls[control]["values"])))
            # self._controls_dict[control] = self._controls[-1]
            logger.info("Setting up {:s} controller of type: {}".format(control, conf.par.wind_farm.controller.controls[control]["type"]))

        # todo: reattach external controller
        if "self._yaw_control_type" == "external":
            self._received_data = []
            logger.info("Initialising ZMQ communication")
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.REQ)
            address = "tcp://localhost:{}".format(conf.par.wind_farm.controller.port)
            self._socket.connect(address)
            logger.info("Connected to: {}".format(address))

    def control(self, simulation_time):
        for control in self._controls.values():
            control.do_control(simulation_time)
            [wt.set_control(control.get_name(), c) for (wt, c) in zip(self._turbines, control.get_reference())]

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
        if self._axial_induction_control_type == "external":
            new_ref = self._received_data[0::2]
        else:
            new_ref = np.deg2rad(self._received_data[1::3])

        return new_ref

    def _external_induction_controller(self, simulation_time):
        logger.warning("External induction controller only works if yaw controller implemented as well")
        new_ref = self._received_data[1::2]
        return new_ref

    def _external_pitch_controller(self, simulation_time):
        logger.warning("External pitch_torque controller only works if yaw controller implemented as well")
        new_ref = self._received_data[2::3]
        return new_ref

    def _external_torque_controller(self, simulation_time):
        new_ref = self._received_data[0::3]
        return new_ref

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

    def get_controls(self, name):
        return self._controls[name].get_controls_list()

    def clear_controls(self):
        for control in self._controls.values():
            control.clear_control_list()


class Control:
    def __init__(self, name, control_type, value):
        self._name = name
        switcher = {
            "fixed": self._fixed_control,
            "series": self._series_control,
            "external": self._external_control
        }
        self._control_function = switcher.get(control_type, "fixed")
        self._reference = []
        self._time_last_updated = 0
        self._time_series = None
        self._reference_series = None

        if control_type == "series":
            self._time_series = value[:, 0]
            self._reference_series = value[:, 1:]
            if name=="yaw":
                self._reference_series = np.deg2rad(self._reference_series)

    def get_name(self):
        return self._name

    def get_control_list(self):
        return self._reference

    def clear_control_list(self):
        self._reference = []

    def get_reference(self):
        return self._reference[-1]

    def do_control(self, simulation_time):
        if (simulation_time - self._time_last_updated >= conf.par.wind_farm.controller.control_discretisation) \
                or self._reference == []:
            # execute the control function from init.
            new_reference = self._control_function(simulation_time)
            # todo: rate limit
            # new_reference = self._apply_rate_limit(new_reference)
            self._update_reference(new_reference)
            self._time_last_updated = simulation_time

    def _fixed_control(self, simulation_time):
        new_reference = conf.par.wind_farm.yaw_angles.copy()
        return new_reference

    def _series_control(self, simulation_time):
        new_reference = conf.par.wind_farm.yaw_angles.copy()
        for idx in range(len(conf.par.wind_farm.positions)):
            new_reference[idx] = np.interp(simulation_time, self._time_series, self._reference_series[:, idx])
        return new_reference
        # raise NotImplementedError("Series control not implemented in Control class")

    def _external_control(self, simulation_time):
        raise NotImplementedError("External control not implemented in Control class")

    def _update_reference(self, new_reference):
        if len(new_reference) != len(conf.par.wind_farm.positions):
            raise ValueError(
                "Computed yaw reference (length {:d}) does not match {:d} turbines in farm."
                    .format(len(new_reference), len(conf.par.wind_farm.positions)))
        self._reference.append([Constant(y) for y in new_reference])


