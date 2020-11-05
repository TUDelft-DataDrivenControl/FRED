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

        self._pitch_control_type = conf.par.wind_farm.controller.pitch_control_type
        logger.info("Setting up pitch controller of type: {}".format(self._pitch_control_type))

        self._torque_control_type = conf.par.wind_farm.controller.torque_control_type
        logger.info("Setting up torque controller of type: {}".format(self._torque_control_type))

        self._wind_farm = wind_farm
        self._turbines = wind_farm.get_turbines()

        self._yaw_control = Control(name='yaw',
                                    control_type=conf.par.wind_farm.controller.yaw_control_type,
                                    value=conf.par.wind_farm.controller.yaw_series)

        self._yaw_ref = []
        self._axial_induction_ref = []
        self._pitch_ref = []
        self._torque_ref = []

        self._time_last_updated_yaw = 0
        self._time_last_updated_induction = 0
        self._time_last_updated_pitch = 0
        self._time_last_updated_torque = 0

        # todo: create series controller class?
        if self._yaw_control_type == "series":
            self._yaw_time_series = conf.par.wind_farm.controller.yaw_series[:, 0]
            self._yaw_series = conf.par.wind_farm.controller.yaw_series[:, 1:]

        if self._axial_induction_control_type == "series":
            self._axial_induction_time_series = conf.par.wind_farm.controller.axial_induction_series[:, 0]
            self._axial_induction_series = conf.par.wind_farm.controller.axial_induction_series[:, 1:]

        if self._pitch_control_type == "series":
            self._pitch_time_series = conf.par.wind_farm.controller.pitch_series[:, 0]
            self._pitch_series = conf.par.wind_farm.controller.pitch_series[:, 1:]

        if self._torque_control_type == "series":
            self._torque_time_series = conf.par.wind_farm.controller.torque_series[:, 0]
            self._torque_series = conf.par.wind_farm.controller.torque_series[:, 1:]

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
        if self._axial_induction_control_type != "none":
            self.control_axial_induction(simulation_time)
        self.control_pitch_and_torque(simulation_time)

    def control_yaw(self, simulation_time):
        self._yaw_control.do_control(simulation_time)
        [wt.set_yaw_ref(y) for (wt, y) in zip(self._turbines, self._yaw_control.get_reference())]
        logger.info("Set {:s} to {}".format(self._yaw_control.get_name(), self._yaw_control.get_reference()))
        # if (simulation_time - self._time_last_updated_yaw >= conf.par.wind_farm.controller.control_discretisation)\
        #         or self._yaw_ref == []:
        #     switcher = {
        #         "fixed": self._fixed_yaw,
        #         "series": self._yaw_series_control,
        #         "external": self._external_controller
        #     }
        #     yaw_controller_function = switcher.get(self._yaw_control_type)
        #     new_ref = yaw_controller_function(simulation_time)
        #     self._update_yaw(new_ref)
        #     self._time_last_updated_yaw = simulation_time

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

    def control_pitch_and_torque(self, simulation_time):
        if (simulation_time - self._time_last_updated_pitch >= conf.par.wind_farm.controller.control_discretisation)\
                or self._pitch_ref == []:
            switcher_pitch = {
                "fixed": self._fixed_pitch,
                "series": self._pitch_series_control,
                "external": self._external_pitch_controller
            }
            pitch_controller_function = switcher_pitch.get(self._pitch_control_type)
            new_pitch_ref = pitch_controller_function(simulation_time)

            switcher_torque = {
                "fixed": self._fixed_torque,
                "series": self._torque_series_control,
                "external": self._external_torque_controller
            }
            torque_controller_function = switcher_torque.get(self._torque_control_type)
            new_torque_ref = torque_controller_function(simulation_time)

            self._update_pitch_and_torque(new_pitch_ref, new_torque_ref)
            self._time_last_updated_pitch = simulation_time

    def _fixed_yaw(self, simulation_time):
        new_ref = conf.par.wind_farm.yaw_angles.copy()
        return new_ref

    def _fixed_induction(self, simulation_time):
        new_ref = [wt.get_axial_induction() for wt in self._turbines]
        return new_ref

    def _fixed_pitch(self, simulation_time):
        new_ref = [wt.get_pitch() for wt in self._turbines]
        return new_ref

    def _fixed_torque(self, simulation_time):
        new_ref = [wt.get_torque() for wt in self._turbines]
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

    def _pitch_series_control(self, simulation_time):
        self._pitch_time_series = conf.par.wind_farm.controller.pitch_series[:, 0]
        self._pitch_series = conf.par.wind_farm.controller.pitch_series[:, 1:]
        new_ref = []
        for idx in range(len(self._turbines)):
            new_ref.append(np.interp(simulation_time, self._pitch_time_series, self._pitch_series[:,idx]))
        return new_ref

    def _torque_series_control(self, simulation_time):
        self._torque_time_series = conf.par.wind_farm.controller.torque_series[:, 0]
        self._torque_series = conf.par.wind_farm.controller.torque_series[:, 1:]
        new_ref = []
        for idx in range(len(self._turbines)):
            new_ref.append(np.interp(simulation_time, self._torque_time_series, self._torque_series[:,idx]))
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

    def _update_pitch_and_torque(self, new_pitch_ref, new_torque_ref):
        for ref in [new_pitch_ref, new_torque_ref]:
            if len(ref) != len(self._turbines):
                raise ValueError(
                    "Computed reference (length {:d}) does not match {:d} turbines in farm."
                    .format(len(ref), len(self._turbines)))

        self._pitch_ref.append([Constant(b) for b in new_pitch_ref])
        self._torque_ref.append([Constant(q) for q in new_torque_ref])
        [wt.set_pitch_and_torque(b,q) for (wt, b, q) in zip(self._turbines, self._pitch_ref[-1], self._torque_ref[-1])]
        logger.info("Set pitch to {}".format(new_pitch_ref))
        logger.info("Set torque to {}".format(new_torque_ref))

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

    def get_pitch_controls(self):
        return self._pitch_ref

    def get_torque_controls(self):
        return self._torque_ref

    def clear_controls(self):
        self._yaw_ref = []
        self._axial_induction_ref = []
        self._pitch_ref = []
        self._torque_ref = []


class Control:
    def __init__(self, name, control_type, value):
        self._name = name
        switcher = {
            "fixed": self._fixed_control,
            "series": self._series_control,
            "external": self._external_control
        }
        self._control_function = switcher.get(control_type, "fixed")
        # control reference list
        self._reference = []
        self._time_last_updated = 0
        # self._initialise(value)
        self._time_series = None
        self._reference_series = None

        # def _initialise(self, value):
        # if self._type == "fixed":
        #     self._reference =
        # el
        if control_type == "series":
            self._time_series = value[:, 0]
            self._reference_series = value[:, 1:]

    def get_name(self):
        return self._name

    def get_control_list(self):
        return self._reference

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
        raise NotImplementedError("Series control not implemented in Control class")

    def _external_control(self, simulation_time):
        raise NotImplementedError("External control not implemented in Control class")

    def _update_reference(self, new_reference):
        if len(new_reference) != len(conf.par.wind_farm.positions):
            raise ValueError(
                "Computed yaw reference (length {:d}) does not match {:d} turbines in farm."
                    .format(len(new_reference), len(conf.par.wind_farm.positions)))
        self._reference.append([Constant(y) for y in new_reference])


