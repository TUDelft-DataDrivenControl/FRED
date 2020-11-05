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
        # self._yaw_control_type = conf.par.wind_farm.controller.yaw_control_type
        # logger.info("Setting up yaw controller of type: {}".format(self._yaw_control_type))
        #
        # self._axial_induction_control_type = conf.par.wind_farm.controller.axial_induction_control_type
        # logger.info("Setting up induction controller of type: {}".format(self._axial_induction_control_type))
        #
        # self._pitch_control_type = conf.par.wind_farm.controller.pitch_control_type
        # logger.info("Setting up pitch controller of type: {}".format(self._pitch_control_type))
        #
        # self._torque_control_type = conf.par.wind_farm.controller.torque_control_type
        # logger.info("Setting up torque controller of type: {}".format(self._torque_control_type))

        self._wind_farm = wind_farm
        self._turbines = wind_farm.get_turbines()

        self._controls = []
        for control in conf.par.wind_farm.controller.controls:
            self._controls.append(Control(name=control,
                                          control_type=conf.par.wind_farm.controller.controls[control]["type"],
                                          value=np.array(conf.par.wind_farm.controller.controls[control]["values"])))
            logger.info("Setting up {:s} controller of type: {}".format(control, conf.par.wind_farm.controller.controls[control]["type"]))

        # self._yaw_control = Control(name='yaw',
        #                             control_type=conf.par.wind_farm.controller.yaw_control_type,
        #                             value=conf.par.wind_farm.controller.yaw_series)
        #
        # self._axial_induction_control = Control(name="axial_induction",
        #                                         control_type=conf.par.wind_farm.controller.axial_induction_control_type,
        #                                         value=conf.par.wind_farm.controller.axial_induction_series)
        #
        # self._pitch_control = Control(name="pitch",
        #                               control_type=conf.par.wind_farm.controller.pitch_control_type,
        #                               value=conf.par.wind_farm.controller.pitch_series)
        #
        # self._torque_control = Control(name="torque",
        #                                control_type=conf.par.wind_farm.controller.torque_control_type,
        #                                value=conf.par.wind_farm.controller.torque_series)
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
        for control in self._controls:
            control.do_control(simulation_time)
            # [wt.set_yaw_ref(y) for (wt, y) in zip(self._turbines, self._yaw_control.get_reference())]
            [wt.set_control(control.get_name(), c) for (wt, c) in zip(self._turbines, control.get_reference())]

        # self.control_yaw(simulation_time)
        # if self._axial_induction_control_type != "none":
        #     self.control_axial_induction(simulation_time)
        # self.control_pitch_and_torque(simulation_time)

    def control_yaw(self, simulation_time):
        self._yaw_control.do_control(simulation_time)
        [wt.set_yaw_ref(y) for (wt, y) in zip(self._turbines, self._yaw_control.get_reference())]
        logger.info("Set {:s} to {}".format(self._yaw_control.get_name(), self._yaw_control.get_reference()))

    def control_axial_induction(self, simulation_time):
        self._axial_induction_control.do_control(simulation_time)
        [wt.set_axial_induction(a) for (wt, a) in zip(self._turbines, self._axial_induction_control.get_reference())]
        logger.info("Set axial induction to {}".format(self._axial_induction_control.get_reference()))

    def control_pitch_and_torque(self, simulation_time):
        self._pitch_control.do_control(simulation_time)
        new_pitch_reference = self._pitch_control.get_reference()

        self._torque_control.do_control(simulation_time)
        new_torque_reference = self._torque_control.get_reference()

        [wt.set_pitch_and_torque(b,q) for (wt, b, q) in zip(self._turbines, new_pitch_reference, new_torque_reference)]
        logger.info("Set pitch to {}".format(new_pitch_reference))
        logger.info("Set torque to {}".format(new_torque_reference))



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

    def get_yaw_controls(self):
        return self._yaw_control.get_control_list()

    def get_axial_induction_controls(self):
        return self._axial_induction_control.get_control_list()

    def get_pitch_controls(self):
        return self._pitch_control.get_control_list()

    def get_torque_controls(self):
        return self._torque_control.get_control_list()

    def clear_controls(self):
        self._yaw_control.clear_control_list()
        self._axial_induction_control.clear_control_list()
        self._pitch_control.clear_control_list()
        self._torque_control.clear_control_list()


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


