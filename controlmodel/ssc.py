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
        self._control_type = conf.par.ssc.type
        self._control_mode = conf.par.ssc.mode
        self._plant = conf.par.ssc.plant
        self._server = None
        self._yaw_reference = conf.par.ssc.yaw_angles.copy()
        # todo: pitch reference may be useful later for work with SOWFA
        self._axial_induction_reference = conf.par.turbine.axial_induction * np.ones_like(self._yaw_reference)
        self._pitch_reference = conf.par.turbine.pitch * np.ones_like(self._yaw_reference)
        self._torque_reference = conf.par.turbine.torque * np.ones_like(self._yaw_reference)
        logger.info("SSC initialised")
        self._data_file = None
        self._measurements = None
        self._sim_time = None
        self._tracker_torque_reference = None

        if self._control_type == "series":
            self._yaw_time_series = conf.par.ssc.yaw_series[:, 0]
            self._yaw_series = conf.par.ssc.yaw_series[:, 1:]
            if self._control_mode == "induction":
                self._axial_induction_time_series = conf.par.ssc.axial_induction_series[:, 0]
                self._axial_induction_series = conf.par.ssc.axial_induction_series[:, 1:]
            elif self._control_mode == "pitch_torque":
                self._pitch_time_series = conf.par.ssc.pitch_series[:, 0]
                self._pitch_series = conf.par.ssc.pitch_series[:,1:]
                self._torque_time_series = conf.par.ssc.torque_series[:, 0]
                self._torque_series = conf.par.ssc.torque_series[:, 1:]

        if self._control_type == "gradient_step":
            self._wind_farm = WindFarm()
            self._dynamic_flow_problem = DynamicFlowProblem(self._wind_farm)
            self._dynamic_flow_solver = DynamicFlowSolver(self._dynamic_flow_problem, ssc=self)

            self._time_last_optimised = -1.
            self._time_reference_series = np.arange(0, conf.par.ssc.prediction_horizon, conf.par.ssc.control_discretisation) + conf.par.simulation.time_step

            self._yaw_reference_series = np.ones((len(self._time_reference_series), len(conf.par.ssc.yaw_angles)+1))
            self._yaw_reference_series[:, 0] = self._time_reference_series
            self._yaw_reference_series[:, 1:] = self._yaw_reference * self._yaw_reference_series[:,1:]

            if self._control_mode == "induction":
                self._axial_induction_reference_series =   np.ones_like(self._yaw_reference_series)
                self._axial_induction_reference_series[:, 0] = self._time_reference_series
                self._axial_induction_reference_series[:, 1:] = self._axial_induction_reference * self._axial_induction_reference_series[:, 1:]
            elif self._control_mode == "pitch_torque":
                self._pitch_reference_series = np.ones_like(self._yaw_reference_series)
                self._pitch_reference_series[:, 0] = self._time_reference_series
                self._pitch_reference_series[:, 1:] = self._pitch_reference * self._pitch_reference_series[:, 1:]
                self._torque_reference_series = np.ones_like(self._yaw_reference_series)
                self._torque_reference_series[:, 0] = self._time_reference_series
                self._torque_reference_series[:, 1:] = self._torque_reference * self._torque_reference_series[:, 1:]

        if self._plant == "sowfa":
            self._tsr_tracker = TorqueController(len(conf.par.wind_farm.positions), conf.par.ssc.sowfa_time_step)

    def start(self):
        self._server = ZmqServer(conf.par.ssc.port)
        logger.info("SSC started")
        if self._control_mode == "induction":
            self._run_yaw_induction_control()
        elif self._control_mode == "pitch_torque":
            self._run_yaw_pitch_torque_control()

    def _run_yaw_induction_control(self):
        while True:
            sim_time, measurements = self._server.receive()
            self._set_yaw_induction_reference(simulation_time=sim_time)
            self._server.send_yaw_induction(self._yaw_reference, self._axial_induction_reference)
            logger.info("Sent yaw and induction control signals for time: {:.2f}".format(sim_time))

    def _run_yaw_pitch_torque_control(self):
        self._setup_output_file()
        while True:
            sim_time, measurements = self._server.receive()
            self._sim_time = sim_time
            self._measurements = measurements
            self._set_yaw_pitch_torque_reference(simulation_time=sim_time)
            # self._server.send(self._yaw_reference, self._pitch_reference, self._torque_reference)
            if self._plant=="cm":
                self._server.send(self._yaw_reference, self._pitch_reference, self._torque_reference)
                logger.info("Sent yaw, pitch, torque control signals for time: {:.2f}".format(sim_time))
                logger.info("Yaw: {:.2f}, pitch {:.2f}, torque {:.2f}".format(self._yaw_reference[0], self._pitch_reference[0], self._torque_reference[0]))
            elif self._plant=="sowfa":
                logger.warning("TSR tracker not yet connected!")
                # todo: make automatic from sowfa specs
                # measured_rotor_speed = mea[1::8]
                # measured_generator_torque = []
                # measured_blade_pitch =
                self._tsr_tracker.run_estimator(measured_rotor_speed=np.array([measurements[1::8]]),
                                                measured_generator_torque=np.array([measurements[5::8]]),
                                                measured_blade_pitch=np.array([measurements[7::8]]))
                torque_set_point = self._tsr_tracker.generate_torque_reference(tsr_desired=self._torque_reference)
                self._tracker_torque_reference = np.array(torque_set_point).squeeze()
                self._server.send(self._yaw_reference, self._pitch_reference, self._tracker_torque_reference)
                logger.info("Sent yaw, pitch, torque control signals for time: {:.2f}".format(sim_time))
                logger.info(
                    "Yaw: {:.2f}, pitch {:.2f}, torque {:.2f}".format(self._yaw_reference[0], self._pitch_reference[0],
                                                                          self._torque_reference[0]))
                self._write_output_file()

    def _setup_output_file(self):
        results_dir = "./results/" + conf.par.simulation.name
        os.makedirs(results_dir, exist_ok=True)
        self._data_file = results_dir + "/log_ssc.csv"
        with open(self._data_file, 'w') as log:
            log.write("time")
            m = ["measured_rotor_speed", "measured_generator_torque", "measured_blade_pitch"]
            c = ["yaw_reference", "pitch_reference", "tsr_reference", "torque_reference"]
            t = ["filtered_rotor_speed", "wind_speed_estimate", "tsr_estimate"]
            for idx in range(len(self._yaw_reference)):
                for var in m+c+t:
                    log.write(",{:s}_{:03n}".format(var,idx))
                log.write("\r\n")

    def _write_output_file(self):
        with open(self._data_file, 'a') as log:
            log.write("{:.6f}".format(self._sim_time))
            # m = ["measured_rotor_speed", "measured_generator_torque", "measured_blade_pitch"]
            m = [self._measurements[1::8], self._measurements[5::8], self._measurements[7::8]]
            # c = ["yaw_reference", "pitch_reference", "tsr_reference", "torque_reference"]
            c = [self._yaw_reference, self._pitch_reference, self._torque_reference, self._tracker_torque_reference]
            # t = ["filtered_rotor_speed", "wind_speed_estimate", "tsr_estimate"]
            t = [self._tsr_tracker._estimator._rotor_speed_filtered, self._tsr_tracker._estimator._wind_speed, self._tsr_tracker._estimator._rotor_speed_filtered*(np.pi/30)*conf.par.turbine.radius/self._tsr_tracker._estimator._wind_speed]
            t = [np.array(x).squeeze() for x in t]
            for idx in range(len(self._yaw_reference)):
                for var in m+c+t:
                    log.write(",{:.6f}".format(var[idx]))
                log.write("\r\n")

    def _set_yaw_induction_reference(self, simulation_time):
        switcher = {
            "fixed": self._fixed_reference,
            "series": self._yaw_induction_time_series_reference,
            "gradient_step": self._yaw_induction_gradient_step_reference
        }
        control_function = switcher[self._control_type]
        control_function(simulation_time)

    def _set_yaw_pitch_torque_reference(self, simulation_time):
        switcher = {
            "fixed": self._fixed_reference,
            "series": self._yaw_pitch_torque_time_series_reference,
            "gradient_step": self._yaw_pitch_torque_gradient_step_reference
            #todo: "gradient_step": self._gradient_step_reference
        }
        control_function = switcher[self._control_type]
        control_function(simulation_time)

    def _fixed_reference(self, simulation_time):
        return None

    def _yaw_induction_time_series_reference(self, simulation_time):
        for idx in range(len(self._yaw_reference)):
            self._yaw_reference[idx] = np.interp(simulation_time, self._yaw_time_series, self._yaw_series[:, idx])
            self._axial_induction_reference[idx] = np.interp(simulation_time, self._axial_induction_time_series, self._axial_induction_series[:, idx])

    def _yaw_pitch_torque_time_series_reference(self, simulation_time):
        for idx in range(len(self._yaw_reference)):
            self._yaw_reference[idx] = np.interp(simulation_time, self._yaw_time_series, self._yaw_series[:, idx])
            self._pitch_reference[idx] = np.interp(simulation_time, self._pitch_time_series, self._pitch_series[:, idx])
            self._torque_reference[idx] = np.interp(simulation_time, self._torque_time_series, self._torque_series[:, idx])

    def _yaw_induction_gradient_step_reference(self, simulation_time):
        # t0 = simulation_time

        # todo: store history up_prev etc...
        if (simulation_time - self._time_last_optimised >= conf.par.ssc.control_horizon)\
                or (self._time_last_optimised < 0):

            # run forward simulation and gradient sensitivity
            if self._time_last_optimised >= 0:
                ch_idx = int(conf.par.ssc.control_horizon // conf.par.ssc.control_discretisation)
                # end_idx = len(conf.par.ssc.prediction_horizon // conf.par.ssc.control_discretisation)
                new_time_reference = np.arange(self._yaw_reference_series[ch_idx,0],
                                               self._yaw_reference_series[ch_idx,0]+conf.par.ssc.prediction_horizon,
                                               conf.par.ssc.control_discretisation)
                for idx in range(len(self._yaw_reference)):
                    self._yaw_reference_series[:,idx+1] = np.interp(
                        new_time_reference,
                        self._yaw_reference_series[:,0],
                        self._yaw_reference_series[:,idx+1])
                    self._axial_induction_reference_series[:,idx+1] = np.interp(
                        new_time_reference,
                        self._axial_induction_reference_series[:,0],
                        self._axial_induction_reference_series[:,idx+1])

                self._yaw_reference_series[:,0] = new_time_reference
                self._axial_induction_reference_series[:,0] = new_time_reference



            logger.debug("Yaw ref series {}".format(self._yaw_reference_series))
            self._time_last_optimised = simulation_time
            self._dynamic_flow_solver.save_checkpoint()

            if simulation_time > conf.par.ssc.transient_time:
                time_horizon = conf.par.ssc.prediction_horizon
                logger.info("Forward simulation over time horizon {:.2f}".format(time_horizon))
                self._dynamic_flow_solver.solve_segment(time_horizon)
                # set yaw reference series in conf
                conf.par.wind_farm.controller.yaw_series = self._yaw_reference_series
                conf.par.wind_farm.controller._axial_induction_series = self._axial_induction_reference_series

                # if simulation_time > 200:
                # Get the relevant controls and power series over the time segment of the forward simulation
                yaw_controls = self._wind_farm.get_yaw_controls()
                axial_induction_controls = self._wind_farm.get_axial_induction_controls()
                power = self._dynamic_flow_solver.get_power_functional_list()

                logger.debug("Yaw controls: {}".format(len(yaw_controls)))
                logger.debug("Induction controls: {}".format(len(axial_induction_controls)))
                logger.debug("Functional: {}".format(len(power)))

                if conf.par.ssc.objective == "maximisation":
                    total_power = [sum(x) for x in power]
                    power_squared = [(p-10e6)*(p-10e6) for p in total_power]
                    control_difference_squared = [1e1 * assemble((c1[0]-c0[0]) * (c1[0]-c0[0])*dx(UnitIntervalMesh(1)))  for c0,c1 in zip(yaw_controls[:-1],yaw_controls[1:])]


                    logger.info("Power cost:   {:.2e}".format(sum(power_squared)))
                    logger.info("Control cost: {:.2e}".format(sum(control_difference_squared)))
                    tracking_functional = sum(power_squared) + \
                                          sum(control_difference_squared)

                    m = [Control(x[0]) for x in yaw_controls]
                    m = m + [Control(x[0]) for x in axial_induction_controls]
                    gradient = compute_gradient(tracking_functional, m)
                    # mdot = [x.get_derivative() for x in m]
                    mdot = [Constant(1.) for x in m]
                    logger.info("mdot: {}".format([float(md) for md in mdot]))
                    hessian = compute_hessian(tracking_functional, m, mdot)

                    gradient = np.array([float(g) for g in gradient])
                    yaw_gradient = gradient[:len(yaw_controls)]
                    axial_induction_gradient = gradient[len(yaw_controls):]
                    logger.info("Computed gradient: {}".format(gradient))

                    hessian = np.array([float(h) for h in hessian])
                    yaw_hessian = hessian[:len(yaw_controls)]
                    axial_induction_hessian = hessian[len(yaw_controls):]
                    logger.info("Computed Hessian {}".format(hessian))

                    if conf.par.turbine.yaw_rate_limit < 0:
                        max_yaw_step = np.deg2rad(5.)
                    else:
                        max_yaw_step = conf.par.turbine.yaw_rate_limit * conf.par.ssc.control_horizon
                    max_axial_induction_step = 0.1

                    logger.info("Functional: {:.2e}".format(tracking_functional))
                    # tracking_functional_array = np.array(power_squared)
                    scale = 1e-1 * np.ones_like(yaw_gradient)
                    yaw_step = -1 * scale * (conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                           (yaw_gradient / yaw_hessian)
                    axial_induction_step = -1 * scale * (conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                           (axial_induction_gradient / axial_induction_hessian)

                    logger.info("yaw_step: {}".format(yaw_step))
                    yaw_step = np.sign(yaw_step) * np.min((np.abs(yaw_step), max_yaw_step*np.ones_like(yaw_gradient)), 0)
                    axial_induction_step = np.sign(axial_induction_step) * np.min((np.abs(axial_induction_step), max_axial_induction_step * np.ones_like(axial_induction_gradient)), 0)


                elif conf.par.ssc.objective == "tracking":
                    total_power = [sum(x)*1e-6 for x in power]
                    time = np.arange(simulation_time, simulation_time + time_horizon, conf.par.simulation.time_step)
                    t_ref_array = conf.par.ssc.power_reference[:,0]
                    p_ref_array = conf.par.ssc.power_reference[:,1] * 1e-6
                    p_ref = np.interp(time,t_ref_array, p_ref_array)
                    logger.info("power reference: {}".format(p_ref))

                    power_difference_squared = [(p-pr)*(p-pr) for p,pr in zip(total_power, p_ref)]
                    control_difference_squared = [1e4 * assemble((c1[0]-c0[0]) * (c1[0]-c0[0])*dx(UnitIntervalMesh(1)))  for c0,c1 in zip(yaw_controls[:-1],yaw_controls[1:])]

                    logger.info("length power tracking: {:d}".format(len(power_difference_squared)))
                    logger.info("Power cost:   {:.2e}".format(sum(power_difference_squared)))
                    logger.info("Control cost: {:.2e}".format(sum(control_difference_squared)))
                    tracking_functional = sum(power_difference_squared)  + \
                                          sum(control_difference_squared)
                    m = [Control(x[0]) for x in yaw_controls]
                    m = m + [Control(x[0]) for x in axial_induction_controls]
                    gradient = compute_gradient(tracking_functional, m)
                    scale = 1.

                    gradient = np.array([scale*float(g) for g in gradient])
                    yaw_gradient = gradient[:len(yaw_controls)]
                    axial_induction_gradient = gradient[len(yaw_controls):]

                    logger.info("Computed yaw gradient: {}".format(yaw_gradient))
                    logger.info("Computed axial_induction gradient: {}".format(axial_induction_gradient))
                    # step_magnitude = np.abs(scale*gradient)
                    if conf.par.turbine.yaw_rate_limit < 0:
                        max_yaw_step = np.deg2rad(5.)
                    else:
                        max_yaw_step = conf.par.turbine.yaw_rate_limit * conf.par.ssc.control_horizon
                    max_axial_induction_step = 0.1

                    logger.info("Functional: {:.2e}".format(tracking_functional))
                    tracking_functional_array = np.array(power_difference_squared) # + cds
                    scale = 1e0 * np.ones_like(yaw_gradient)
                    yaw_step = -1 * scale * (conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                           (tracking_functional_array / yaw_gradient)
                    axial_induction_step = -1 * scale * (conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                           (tracking_functional_array / axial_induction_gradient)

                    logger.info("yaw_step: {}".format(yaw_step))
                    yaw_step = np.sign(yaw_step) * np.min((np.abs(yaw_step), max_yaw_step*np.ones_like(yaw_gradient)), 0)
                    axial_induction_step = np.sign(axial_induction_step) * np.min((np.abs(axial_induction_step), max_axial_induction_step * np.ones_like(axial_induction_gradient)), 0)

            else:
                yaw_step = np.zeros_like(self._yaw_reference_series[:,1])
                axial_induction_step = np.zeros_like(self._yaw_reference_series[:, 1])
            logger.info("Applied yaw step: {}".format(yaw_step))
            logger.info("Applied axial induction step: {}".format(axial_induction_step))

            if conf.par.turbine.yaw_rate_limit > 0:
                yaw_step[0] = 0.
            self._yaw_reference_series[:, 1] += yaw_step
            self._axial_induction_reference_series[:, 1] += axial_induction_step

            # apply limits
            self._axial_induction_reference_series[:,1] = \
                np.min((0.33*np.ones_like(self._axial_induction_reference_series[:,1]),
                    np.max((np.zeros_like(self._axial_induction_reference_series[:,1]), self._axial_induction_reference_series[:,1]),0)),0)

            if conf.par.turbine.yaw_rate_limit > 0:
                yaw_rate_limit = conf.par.turbine.yaw_rate_limit
                for idx in range(len(self._yaw_reference_series)-1):
                    dyaw = self._yaw_reference_series[idx+1,1]-self._yaw_reference_series[idx,1]
                    dt = self._yaw_reference_series[idx+1,0]-self._yaw_reference_series[idx,0]
                    dmax= yaw_rate_limit*dt
                    dyaw = np.max((-dmax, np.min((dmax,dyaw))))
                    self._yaw_reference_series[idx+1,1] = self._yaw_reference_series[idx,1]+dyaw

            conf.par.wind_farm.controller.yaw_series = self._yaw_reference_series
            conf.par.wind_farm.controller.axial_induction_series = self._axial_induction_reference_series
            self._dynamic_flow_solver.reset_checkpoint()
            self._dynamic_flow_solver.solve_segment(conf.par.ssc.control_horizon)

        # else:
        # send saved signal
        reference_index = int((simulation_time - conf.par.simulation.time_step)
                              % conf.par.ssc.control_horizon // conf.par.ssc.control_discretisation)
        logger.debug("Sending reference_index: {:5d}".format(reference_index))
        self._yaw_reference = self._yaw_reference_series[reference_index, 1:]
        self._axial_induction_reference = self._axial_induction_reference_series[reference_index, 1:]

    def _yaw_pitch_torque_gradient_step_reference(self, simulation_time):
        if (simulation_time - self._time_last_optimised >= conf.par.ssc.control_horizon) \
                or (self._time_last_optimised < 0):

            # run forward simulation and gradient sensitivity
            if self._time_last_optimised >= 0:
                ch_idx = int(conf.par.ssc.control_horizon // conf.par.ssc.control_discretisation)
                new_time_reference = np.arange(self._yaw_reference_series[ch_idx, 0],
                                               self._yaw_reference_series[ch_idx, 0] + conf.par.ssc.prediction_horizon,
                                               conf.par.ssc.control_discretisation)
                for idx in range(len(self._yaw_reference)):
                    self._yaw_reference_series[:, idx + 1] = np.interp(
                        new_time_reference,
                        self._yaw_reference_series[:, 0],
                        self._yaw_reference_series[:, idx + 1])
                    self._pitch_reference_series[:, idx + 1] = np.interp(
                        new_time_reference,
                        self._pitch_reference_series[:, 0],
                        self._pitch_reference_series[:, idx + 1])
                    self._torque_reference_series[:, idx + 1] = np.interp(
                        new_time_reference,
                        self._torque_reference_series[:, 0],
                        self._torque_reference_series[:, idx + 1])


                self._yaw_reference_series[:, 0] = new_time_reference
                self._pitch_reference_series[:, 0] = new_time_reference
                self._torque_reference_series[:, 0] = new_time_reference

            logger.debug("Yaw ref series {}".format(self._yaw_reference_series))
            self._time_last_optimised = simulation_time
            self._dynamic_flow_solver.save_checkpoint()

            if simulation_time > conf.par.ssc.transient_time:
                time_horizon = conf.par.ssc.prediction_horizon
                logger.info("Forward simulation over time horizon {:.2f}".format(time_horizon))
                self._dynamic_flow_solver.solve_segment(time_horizon)
                # set yaw reference series in conf
                conf.par.wind_farm.controller.yaw_series = self._yaw_reference_series
                conf.par.wind_farm.controller._pitch_series = self._pitch_reference_series
                conf.par.wind_farm.controller._torque_series = self._torque_reference_series

                # if simulation_time > 200:
                # Get the relevant controls and power series over the time segment of the forward simulation
                yaw_controls = self._wind_farm.get_yaw_controls()
                pitch_controls = self._wind_farm.get_pitch_controls()
                torque_controls = self._wind_farm.get_torque_controls()
                power = self._dynamic_flow_solver.get_power_functional_list()

                logger.debug("Yaw controls: {}".format(len(yaw_controls)))
                logger.debug("Pitch controls: {}".format(len(pitch_controls)))
                logger.debug("Torque controls: {}".format(len(torque_controls)))

                logger.debug("Functional: {}".format(len(power)))

                if conf.par.ssc.objective == "maximisation":
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

                    if conf.par.turbine.yaw_rate_limit < 0:
                        max_yaw_step = np.deg2rad(5.)
                    else:
                        max_yaw_step = conf.par.turbine.yaw_rate_limit * conf.par.ssc.control_horizon
                    max_pitch_step = 5.
                    max_torque_step = 5.

                    logger.info("Functional: {:.2e}".format(tracking_functional))
                    # tracking_functional_array = np.array(power_squared)
                    scale = 1e-1 * np.ones_like(yaw_gradient)
                    yaw_step = -1 * scale * (conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                               (yaw_gradient / yaw_hessian)
                    pitch_step = -1 * scale * (
                        conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                                 (pitch_gradient / pitch_hessian)
                    torque_step = -1 * scale * (
                        conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                                 (torque_gradient / torque_hessian)

                    logger.info("yaw_step: {}".format(yaw_step))
                    yaw_step = np.sign(yaw_step) * np.min((np.abs(yaw_step), max_yaw_step * np.ones_like(yaw_gradient)),
                                                          0)

                    pitch_step = np.sign(pitch_step) * np.min((np.abs(pitch_step),
                                                               max_pitch_step * np.ones_like(pitch_gradient)), 0)
                    torque_step = np.sign(torque_step) * np.min((np.abs(torque_step),
                                                               max_torque_step * np.ones_like(torque_gradient)), 0)

                elif conf.par.ssc.objective == "tracking":
                    total_power = [sum(x) * 1e-6 for x in power]
                    time = np.arange(simulation_time, simulation_time + time_horizon, conf.par.simulation.time_step)
                    t_ref_array = conf.par.ssc.power_reference[:, 0]
                    p_ref_array = conf.par.ssc.power_reference[:, 1] * 1e-6
                    p_ref = np.interp(time, t_ref_array, p_ref_array)
                    logger.info("power reference: {}".format(p_ref))

                    power_difference_squared = [(p - pr) * (p - pr) for p, pr in zip(total_power, p_ref)]
                    control_difference_squared = [
                        1e4 * assemble((c1[0] - c0[0]) * (c1[0] - c0[0]) * dx(UnitIntervalMesh(1))) for c0, c1 in
                        zip(yaw_controls[:-1], yaw_controls[1:])]

                    logger.info("length power tracking: {:d}".format(len(power_difference_squared)))
                    logger.info("Power cost:   {:.2e}".format(sum(power_difference_squared)))
                    logger.info("Control cost: {:.2e}".format(sum(control_difference_squared)))
                    tracking_functional = sum(power_difference_squared) + \
                                          sum(control_difference_squared)
                    m = [Control(x[0]) for x in yaw_controls]
                    m = m + [Control(x[0]) for x in pitch_controls]
                    m = m + [Control(x[0]) for x in torque_controls]
                    gradient = compute_gradient(tracking_functional, m)
                    scale = 1.

                    gradient = np.array([scale * float(g) for g in gradient])
                    yaw_gradient = gradient[:len(yaw_controls)]
                    pitch_gradient = gradient[len(yaw_controls):2*len(yaw_controls)]
                    torque_gradient = gradient[2*len(yaw_controls):]

                    logger.info("Computed yaw gradient: {}".format(yaw_gradient))
                    logger.info("Computed pitch gradient: {}".format(pitch_gradient))
                    logger.info("Computed torque gradient: {}".format(torque_gradient))
                    # step_magnitude = np.abs(scale*gradient)
                    if conf.par.turbine.yaw_rate_limit < 0:
                        max_yaw_step = np.deg2rad(5.)
                    else:
                        max_yaw_step = conf.par.turbine.yaw_rate_limit * conf.par.ssc.control_horizon
                    max_pitch_step = 5.
                    max_torque_step = 5.

                    logger.info("Functional: {:.2e}".format(tracking_functional))
                    tracking_functional_array = np.array(power_difference_squared)  # + cds
                    scale = 1e0 * np.ones_like(yaw_gradient)
                    yaw_step = -1 * scale * (conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                               (tracking_functional_array / yaw_gradient)
                    pitch_step = -1 * scale * (
                            conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                                 (tracking_functional_array / pitch_gradient)
                    torque_step = -1 * scale * (
                            conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                                  (tracking_functional_array / torque_gradient)

                    logger.info("yaw_step: {}".format(yaw_step))
                    yaw_step = np.sign(yaw_step) * np.min((np.abs(yaw_step), max_yaw_step * np.ones_like(yaw_gradient)),
                                                          0)
                    pitch_step = np.sign(pitch_step) * np.min((np.abs(pitch_step),
                                                               max_pitch_step * np.ones_like(pitch_gradient)), 0)
                    torque_step = np.sign(torque_step) * np.min((np.abs(torque_step),
                                                                 max_torque_step * np.ones_like(torque_gradient)), 0)

            else:
                yaw_step = np.zeros_like(self._yaw_reference_series[:, 1])
                pitch_step = np.zeros_like(self._yaw_reference_series[:, 1])
                torque_step = np.zeros_like(self._yaw_reference_series[:, 1])
            logger.info("Applied yaw step: {}".format(yaw_step))
            logger.info("Applied pitch step: {}".format(pitch_step))
            logger.info("Applied torque step: {}".format(torque_step))

            if conf.par.turbine.yaw_rate_limit > 0:
                yaw_step[0] = 0.
            self._yaw_reference_series[:, 1] += yaw_step
            self._pitch_reference_series[:, 1] += pitch_step
            self._torque_reference_series[:, 1] += torque_step

            # apply limits
            self._pitch_reference_series[:, 1] = \
                np.min((25 * np.ones_like(self._pitch_reference_series[:, 1]),
                        np.max((-1*np.ones_like(self._pitch_reference_series[:, 1]),
                                self._pitch_reference_series[:, 1]), 0)), 0)

            self._torque_reference_series[:, 1] = \
                np.min((14 * np.ones_like(self._torque_reference_series[:, 1]),
                        np.max((3*np.ones_like(self._torque_reference_series[:, 1]),
                                self._torque_reference_series[:, 1]), 0)), 0)

            if conf.par.turbine.yaw_rate_limit > 0:
                yaw_rate_limit = conf.par.turbine.yaw_rate_limit
                for idx in range(len(self._yaw_reference_series) - 1):
                    dyaw = self._yaw_reference_series[idx + 1, 1] - self._yaw_reference_series[idx, 1]
                    dt = self._yaw_reference_series[idx + 1, 0] - self._yaw_reference_series[idx, 0]
                    dmax = yaw_rate_limit * dt
                    dyaw = np.max((-dmax, np.min((dmax, dyaw))))
                    self._yaw_reference_series[idx + 1, 1] = self._yaw_reference_series[idx, 1] + dyaw

            conf.par.wind_farm.controller.yaw_series = self._yaw_reference_series
            conf.par.wind_farm.controller.pitch_series = self._pitch_reference_series
            conf.par.wind_farm.controller.torque_series = self._torque_reference_series
            self._dynamic_flow_solver.reset_checkpoint()
            self._dynamic_flow_solver.solve_segment(conf.par.ssc.control_horizon)

        # else:
        # send saved signal
        reference_index = int((simulation_time - conf.par.simulation.time_step)
                              % conf.par.ssc.control_horizon // conf.par.ssc.control_discretisation)
        logger.debug("Sending reference_index: {:5d}".format(reference_index))
        self._yaw_reference = self._yaw_reference_series[reference_index, 1:]
        self._pitch_reference = self._pitch_reference_series[reference_index, 1:]
        self._torque_reference = self._torque_reference_series[reference_index, 1:]


    def get_power_reference(self, simulation_time):
        t_ref_array = conf.par.ssc.power_reference[:, 0]
        p_ref_array = conf.par.ssc.power_reference[:, 1]
        return np.interp(simulation_time, t_ref_array, p_ref_array)