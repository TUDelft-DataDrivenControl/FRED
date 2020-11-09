from fenics import *
import controlmodel.conf as conf

if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
import os
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver

from controlmodel.zmqserver import ZmqServer

from tools.tsrtracker import TorqueController

import logging

logger = logging.getLogger("cm.ssc")


class SuperController:

    def __init__(self):
        self._control_mode = conf.par.ssc.mode
        self._external_controls = conf.par.ssc.external_controls
        self._external_measurements = conf.par.ssc.external_measurements
        self._plant = conf.par.ssc.plant
        self._server = None
        self._controls = {}
        self._gradient_controls = {}
        self._with_gradient_step = False
        for control in conf.par.ssc.controls:
            self._controls[control] = ControlParameter(name=control,
                                                       config=conf.par.ssc.controls[control])
            logger.info("Setting up {:s} controller of type: {}"
                        .format(control, conf.par.ssc.controls[control]["type"]))
            if conf.par.ssc.controls[control]["type"] == "gradient_step":
                self._gradient_controls[control] = self._controls[control]
                self._with_gradient_step = True

        logger.info("SSC initialised")
        self._data_file = None
        self._measurements = {}
        self._sim_time = None
        self._tracker_torque_reference = None
        self._num_turbines = len(conf.par.wind_farm.positions)

        # todo: re-attach to sowfa
        if self._with_gradient_step:
            self._wind_farm = WindFarm()
            self._dynamic_flow_problem = DynamicFlowProblem(self._wind_farm)
            self._dynamic_flow_solver = DynamicFlowSolver(self._dynamic_flow_problem, ssc=self)
            self._time_last_optimised = -1.

        if self._plant == "sowfa":
            self._tsr_tracker = TorqueController(len(conf.par.wind_farm.positions), conf.par.ssc.sowfa_time_step)

    def start(self):
        self._server = ZmqServer(conf.par.ssc.port)
        logger.info("SSC started")
        self._run_control()

    def _run_control(self):
        self._setup_output_file()
        while True:
            sim_time, measurements = self._server.receive()
            self._sim_time = sim_time
            self._assign_measurements(measurements)

            for control in self._controls.values():
                control.do_control(sim_time)
            if self._with_gradient_step:
                if (sim_time - self._time_last_optimised >= conf.par.ssc.control_horizon) \
                        or (self._time_last_optimised < 0):
                    self._compute_gradients(sim_time)

            send_controls = []
            if self._plant == "cm":
                for control in self._external_controls:
                    send_controls += [self._controls[control].get_reference()]

            elif self._plant == "sowfa":
                # convert tsr to torque signal
                self._tsr_tracker.run_estimator(measured_rotor_speed=np.array([self._measurements["rotorSpeed"]]),
                                                measured_generator_torque=np.array(
                                                    [self._measurements["generatorTorque"]]),
                                                measured_blade_pitch=np.array([self._measurements["bladePitch"]]))
                torque_set_point = self._tsr_tracker.generate_torque_reference(
                    tsr_desired=self._controls['torque'].get_reference())
                self._tracker_torque_reference = np.array(torque_set_point).squeeze()
                self._server.send(self._yaw_reference, self._pitch_reference, self._tracker_torque_reference)
                for control in self._external_controls:
                    if control == "torque":
                        send_controls += [self._tracker_torque_reference]
                    else:
                        send_controls += [self._controls[control].get_reference()]

            self._server.send(send_controls)
            self._write_output_file()

    def _setup_output_file(self):
        results_dir = "./results/" + conf.par.simulation.name
        os.makedirs(results_dir, exist_ok=True)
        self._data_file = results_dir + "/log_ssc.csv"
        with open(self._data_file, 'w') as log:
            log.write("time")
            m = self._external_measurements
            c = [c for c in self._controls]
            if self._plant == "sowfa":
                t = ["filtered_rotor_speed", "wind_speed_estimate", "tsr_estimate"]
            else:
                t = []
            for idx in range(self._num_turbines):
                for var in m + c + t:
                    log.write(",{:s}_{:03n}".format(var, idx))
            log.write("\r\n")

    def _write_output_file(self):
        with open(self._data_file, 'a') as log:
            log.write("{:.6f}".format(self._sim_time))
            m = [m for m in self._measurements.values()]
            c = [c.get_reference() for c in self._controls.values()]
            # t = ["filtered_rotor_speed", "wind_speed_estimate", "tsr_estimate"]
            if self._plant == "sowfa":
                t = [self._tsr_tracker._estimator._rotor_speed_filtered, self._tsr_tracker._estimator._wind_speed,
                     self._tsr_tracker._estimator._rotor_speed_filtered * (
                             np.pi / 30) * conf.par.turbine.radius / self._tsr_tracker._estimator._wind_speed]
            else:
                t = []
            # t = [np.array(x).squeeze() for x in t]
            for idx in range(self._num_turbines):
                for var in m + c + t:
                    log.write(",{:.6f}".format(var[idx]))
            log.write("\r\n")

    def _assign_measurements(self, measurements):
        for idx in range(len(self._external_measurements)):
            self._measurements[self._external_controls[idx]] = measurements[idx::len(self._external_measurements)]

    def _compute_gradients(self, simulation_time):
        if self._time_last_optimised >= 0:
            for control in self._gradient_controls.values():
                control.shift_reference_series(simulation_time)

        self._time_last_optimised = simulation_time
        self._dynamic_flow_solver.save_checkpoint()

        if simulation_time > conf.par.ssc.transient_time:

            self._set_wind_farm_reference_series()
            self._do_forward_simulation()
            controls_list, num_controls_dict = self._get_control_lists()
            tracking_functional, tracking_functional_array = self._construct_cost_functional()
            gradient = self._calculate_gradients(tracking_functional, controls_list)

            start_index = 0
            for control in self._gradient_controls:
                end_index = start_index + num_controls_dict[control]
                print("{} : {}".format(start_index, end_index))
                self._gradient_controls[control].update_reference_with_gradient(tracking_functional_array,
                                                                                gradient[start_index:end_index])
                start_index = end_index

        self._dynamic_flow_solver.reset_checkpoint()
        self._dynamic_flow_solver.solve_segment(conf.par.ssc.control_horizon)

    def _do_forward_simulation(self):
        time_horizon = conf.par.ssc.prediction_horizon
        logger.info("Forward simulation over time horizon {:.2f}".format(time_horizon))
        self._dynamic_flow_solver.solve_segment(time_horizon)

    def _set_wind_farm_reference_series(self):
        for control in self._gradient_controls.values():
            self._wind_farm.set_control_reference_series(name=control.get_name(),
                                                         time_series=control.get_time_series(),
                                                         reference_series=control.get_reference_series())

    def _get_control_lists(self):
        controls_list = []
        num_controls_dict = {}
        for control in self._gradient_controls:
            controls_list += (self._wind_farm.get_controls_list(name=control))
            num_controls_dict[control] = len(self._wind_farm.get_controls_list(name=control))
        return controls_list, num_controls_dict

    def _construct_cost_functional(self):
        power = self._dynamic_flow_solver.get_power_functional_list()
        total_power = [sum(x) * 1e-6 for x in power]
        time = np.arange(self._sim_time, self._sim_time + conf.par.ssc.prediction_horizon,
                         conf.par.simulation.time_step)
        t_ref_array = conf.par.ssc.power_reference[:, 0]
        p_ref_array = conf.par.ssc.power_reference[:, 1] * 1e-6
        p_ref = np.interp(time, t_ref_array, p_ref_array)
        logger.info("power reference: {}".format(p_ref))

        power_difference_squared = [(p - pr) * (p - pr) for p, pr in zip(total_power, p_ref)]
        # control_difference_squared = [1e4 * assemble((c1[0] - c0[0]) * (c1[0] - c0[0]) * dx(UnitIntervalMesh(1))) for
        #                               c0, c1 in zip(yaw_controls[:-1], yaw_controls[1:])]

        logger.info("length power tracking: {:d}".format(len(power_difference_squared)))
        logger.info("Power cost:   {:.2e}".format(sum(power_difference_squared)))
        # logger.info("Control cost: {:.2e}".format(sum(control_difference_squared)))
        tracking_functional = sum(power_difference_squared)  # + \
        # sum(control_difference_squared)
        return tracking_functional, power_difference_squared

    def _calculate_gradients(self, tracking_functional, controls_list):
        # J = tracking_functional
        # todo: control on both turbines?
        m = [Control(c[0]) for c in controls_list]
        gradient = compute_gradient(tracking_functional, m)
        scale = 1.
        gradient = np.array([scale * float(g) for g in gradient])
        return gradient

    def get_power_reference(self, simulation_time):
        t_ref_array = conf.par.ssc.power_reference[:, 0]
        p_ref_array = conf.par.ssc.power_reference[:, 1]
        return np.interp(simulation_time, t_ref_array, p_ref_array)


""" MAXIMISATION
        total_power = [sum(x) for x in power]
                    power_squared = [(p - 10e6) * (p - 10e6) for p in total_power]
                    control_difference_squared = [
                        1e1 * assemble((c1[0] - c0[0]) * (c1[0] - c0[0]) * dx(UnitIntervalMesh(1))) for c0, c1 in
                        zip(yaw_controls[:-1], yaw_controls[1:])]

                    logger.info("Power cost:   {:.2e}".format(sum(power_squared)))
                    logger.info("Control cost: {:.2e}".format(sum(control_difference_squared)))
                    tracking_functional = sum(power_squared) + \
                                          sum(control_difference_squared)

                    m = [Control(x[0]) for x in yaw_controls]
                    m = m + [Control(x[0]) for x in pitch_controls]
                    m = m + [Control(x[0]) for x in torque_controls]

                    gradient = compute_gradient(tracking_functional, m)

                    mdot = [Constant(1.) for x in m]
                    logger.info("mdot: {}".format([float(md) for md in mdot]))
                    hessian = compute_hessian(tracking_functional, m, mdot)

                    gradient = np.array([float(g) for g in gradient])
                    yaw_gradient = gradient[:len(yaw_controls)]
                    pitch_gradient = gradient[len(yaw_controls):2*len(yaw_controls)]
                    torque_gradient = gradient[2*len(yaw_controls):]
                    logger.info("Computed gradient: {}".format(gradient))

                    hessian = np.array([float(h) for h in hessian])
                    yaw_hessian = hessian[:len(yaw_controls)]
                    pitch_hessian = hessian[len(yaw_controls):2*len(yaw_controls)]
                    torque_hessian = hessian[2*len(yaw_controls):]
                    logger.info("Computed Hessian {}".format(hessian))
"""


class ControlParameter:
    # def __init__(self, name, control_type, values):
    def __init__(self, name, config):
        self._name = name
        switcher = {
            "fixed": self._fixed_control,
            "series": self._series_control,
            "gradient_step": self._gradient_step_control
        }
        self._control_type = config["type"]
        self._control_function = switcher.get(self._control_type, "fixed")
        self._reference = []
        values = np.array(config["values"])

        self._limits = config.get("range", None)
        self._rate_limit = config.get("rate_limit", None)

        if self._control_type == "fixed":
            self._reference = values
        if self._control_type == "series":  # or self._control_type == "gradient_step":
            self._reference = values[0, 1:].copy()
            self._time_series = values[:, 0]
            self._reference_series = values[:, 1:]

        if self._control_type == "gradient_step":
            self._reference = values[0, 1:].copy()
            self._time_series = np.arange(0, conf.par.ssc.prediction_horizon,
                                          conf.par.ssc.control_discretisation) + conf.par.simulation.time_step
            self._reference_series = np.ones((len(self._time_series), len(self._reference)))
            for idx in range(len(self._reference)):
                self._reference_series[:, idx] = np.interp(self._time_series, values[:, 0], values[:, idx + 1])
        #     self._yaw_reference_series = np.ones((len(self._time_reference_series), len(conf.par.ssc.yaw_angles)+1))
        #     self._yaw_reference_series[:, 0] = self._time_reference_series
        #     self._yaw_reference_series[:, 1:] = self._yaw_reference * self._yaw_reference_series[:,1:]

    def get_name(self):
        return self._name

    def do_control(self, simulation_time):
        self._control_function(simulation_time)
        print("{}: {}".format(self._name, self._reference))

    def _fixed_control(self, simulation_time):
        self._reference = self._reference
        # logger.error("Fixed control not implemented in SSC")

    def _series_control(self, simulation_time):
        reference_index = int((simulation_time - conf.par.simulation.time_step)
                              % conf.par.ssc.control_horizon // conf.par.ssc.control_discretisation)
        self._reference = self._reference_series[reference_index, :]

    def _gradient_step_control(self, simulation_time):
        for idx in range(len(self._reference)):
            self._reference[idx] = np.interp(simulation_time, self._time_series, self._reference_series[:, idx])
        logger.error("gradient step control not implemented in SSC")

    def get_reference(self):
        return self._reference

    def get_time_series(self):
        return self._time_series

    def get_reference_series(self):
        return self._reference_series

    def shift_reference_series(self, simulation_time):
        new_start_index = int(conf.par.ssc.control_horizon // conf.par.ssc.control_discretisation)
        new_time_reference = np.arange(self._time_series[new_start_index],
                                       self._time_series[new_start_index] + conf.par.ssc.prediction_horizon,
                                       conf.par.ssc.control_discretisation)
        for idx in range(len(self._reference)):
            self._reference_series[:, idx] = np.interp(new_time_reference,
                                                       self._time_series,
                                                       self._reference_series[:, idx])
        self._time_series = new_time_reference

    def update_reference_with_gradient(self, tracking_functional_array, gradient):
        logger.info("{} gradient = {}".format(self._name, gradient))
        step = -1 * (conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
               (tracking_functional_array / gradient)
        max_step = 5.
        step = np.sign(step) * np.min((np.abs(step), max_step * np.ones_like(gradient)), 0)
        step[0] = 0.
        if self._name == "yaw":
            step = np.rad2deg(step)
        # todo: min/max step
        # todo: multiple turbines
        print(self._reference_series.shape)
        self._reference_series[:, 0] = self._reference_series[:, 0] + step
        self._apply_limits()

    def _apply_limits(self):
        # ..
        # logger.error("No rate/min/max limits implemented")
        # # todo: min/max limits
        #
        # # todo: rate limits
        if self._rate_limit is not None:
            dt = np.diff(self._time_series)
            dy_limit = self._rate_limit * dt
            dy = np.diff(self._reference_series[:, 0])
            dy = np.max((-1 * dy_limit, np.min((np.abs(dy), dy_limit), axis=0)), axis=0)
            self._reference_series[1:, 0] = self._reference_series[0, 0] + np.cumsum(dy)

        if self._limits is not None:
            self._reference_series[:, 0] = \
                np.min((self._limits[1] * np.ones_like(self._reference_series[:, 0]),
                        np.max((self._limits[0] * np.ones_like(self._reference_series[:, 0]),
                                self._reference_series[:, 0]), 0)), 0)

        #     # np.diff(self._reference_series[:,0]) / np.diff(self._time_series)
        # if conf.par.turbine.yaw_rate_limit > 0 and self._name == "yaw":
        #     yaw_rate_limit = conf.par.turbine.yaw_rate_limit
        #     for idx in range(len(self._reference_series) - 1):
        #         dyaw = self._reference_series[idx + 1, 0] - self._reference_series[idx, 0]
        #         dt = self._time_series[idx + 1] - self._time_series[idx]
        #         dmax = yaw_rate_limit * dt
        #         dyaw = np.max((-dmax, np.min((dmax, dyaw))))
        #         self._reference_series[idx + 1, 0] = self._reference_series[idx, 0] + dyaw
        #
        # if self._name == "pitch":
        #     self._reference_series[:, 0] = \
        #     np.min((25 * np.ones_like(self._reference_series[:, 0]),
        #             np.max((-1 * np.ones_like(self._reference_series[:, 0]),
        #                     self._reference_series[:, 0]), 0)), 0)
