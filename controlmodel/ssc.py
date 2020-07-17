from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver


from controlmodel.zmqserver import ZmqServer
import logging
logger = logging.getLogger("cm.ssc")


class SuperController:

    def __init__(self):
        self._control_type = conf.par.ssc.type
        self._server = None
        self._yaw_reference = conf.par.ssc.yaw_angles.copy()
        # todo: pitch reference may be useful later for work with SOWFA
        self._axial_induction_reference = 0.2 * np.ones_like(self._yaw_reference)
        logger.info("SSC initialised")

        if self._control_type == "series":
            self._time_series = conf.par.ssc.yaw_series[:, 0]
            self._yaw_series = conf.par.ssc.yaw_series[:, 1:]

        if self._control_type == "gradient_step":
            self._wind_farm = WindFarm()
            self._dynamic_flow_problem = DynamicFlowProblem(self._wind_farm)
            self._dynamic_flow_solver = DynamicFlowSolver(self._dynamic_flow_problem, ssc=self)
            self._time_last_optimised = -1.
            self._time_reference_series = np.arange(0, conf.par.ssc.prediction_horizon, conf.par.ssc.control_discretisation) + conf.par.simulation.time_step
            self._yaw_reference_series = np.ones((len(self._time_reference_series), len(conf.par.ssc.yaw_angles)+1))
            self._yaw_reference_series[:, 0] = self._time_reference_series
            self._yaw_reference_series[:, 1:] = self._yaw_reference * self._yaw_reference_series[:,1:]


    def start(self):
        self._server = ZmqServer(conf.par.ssc.port)
        logger.info("SSC started")

        while True:
            sim_time, measurements = self._server.receive()
            # if sim_time % conf.par.ssc.control_discretisation < conf.par.simulation.time_step:
            self._set_yaw_reference(simulation_time=sim_time)
            self._server.send(self._yaw_reference, self._axial_induction_reference)
            logger.info("Sent control signals for time: {:.2f}".format(sim_time))

    def _set_yaw_reference(self, simulation_time):
        switcher = {
            "fixed": self._fixed_reference,
            "series": self._time_series_reference,
            "gradient_step": self._gradient_step_reference
        }
        control_function = switcher[self._control_type]
        control_function(simulation_time)

    def _fixed_reference(self, simulation_time):
        return self._yaw_reference

    def _time_series_reference(self, simulation_time):
        for idx in range(len(self._yaw_reference)):
            self._yaw_reference[idx] = np.interp(simulation_time, self._time_series, self._yaw_series[:, idx])

    def _gradient_step_reference(self, simulation_time):
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
                    self._yaw_reference_series[:,idx+1] = np.interp(new_time_reference,self._yaw_reference_series[:,0], self._yaw_reference_series[:,idx+1]) #, right=conf.par.ssc.yaw_angles[idx])
                self._yaw_reference_series[:,0] = new_time_reference
            logger.debug("Yaw ref series {}".format(self._yaw_reference_series))
            self._time_last_optimised = simulation_time
            self._dynamic_flow_solver.save_checkpoint()

            if simulation_time > conf.par.ssc.transient_time:
                time_horizon = conf.par.ssc.prediction_horizon
                logger.info("Forward simulation over time horizon {:.2f}".format(time_horizon))
                self._dynamic_flow_solver.solve_segment(time_horizon)
                # set yaw reference series in conf
                conf.par.wind_farm.controller.yaw_series = self._yaw_reference_series

                # if simulation_time > 200:
                # Get the relevant controls and power series over the time segment of the forward simulation
                controls = self._wind_farm.get_controls()
                power = self._dynamic_flow_solver.get_power_functional_list()

                logger.debug("Controls: {}".format(len(controls)))
                logger.debug("Functional: {}".format(len(power)))

                if conf.par.ssc.objective == "maximisation":
                    total_power = [sum(x) for x in power]
                    power_squared = [(p-10e6)*(p-10e6) for p in total_power]
                    control_difference_squared = [1e1 * assemble((c1[0]-c0[0]) * (c1[0]-c0[0])*dx(UnitIntervalMesh(1)))  for c0,c1 in zip(controls[:-1],controls[1:])]
                    # print(control_difference_squared)
                    logger.info("Power cost:   {:.2e}".format(sum(power_squared)))
                    logger.info("Control cost: {:.2e}".format(sum(control_difference_squared)))
                    tracking_functional = sum(power_squared) + \
                                          sum(control_difference_squared)
                    m = [Control(x[0]) for x in controls]
                    gradient = compute_gradient(tracking_functional, m)
                    # mdot = [x.get_derivative() for x in m]
                    mdot = [Constant(1.) for x in m]
                    logger.info("mdot: {}".format([float(md) for md in mdot]))
                    hessian = compute_hessian(tracking_functional, m, mdot)

                    gradient = np.array([float(g) for g in gradient])
                    logger.info("Computed gradient: {}".format(gradient))
                    # [print(float(h)) for h in hessian]
                    hessian = np.array([float(h) for h in hessian])
                    logger.info("Computed Hessian {}".format(hessian))


                    max_step = np.deg2rad(5.)
                    logger.info("Functional: {:.2e}".format(tracking_functional))
                    # tracking_functional_array = np.array(power_squared)
                    scale = 1e-1 * np.ones_like(gradient)
                    step = -1 * scale * (conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                           (gradient / hessian)
                    logger.info("Step: {}".format(step))
                    step = np.sign(step) * np.min((np.abs(step), max_step*np.ones_like(gradient)), 0)

                elif conf.par.ssc.objective == "tracking":
                    total_power = [sum(x)*1e-6 for x in power]
                    time = np.arange(simulation_time, simulation_time + time_horizon, conf.par.simulation.time_step)
                    t_ref_array = conf.par.ssc.power_reference[:,0]
                    p_ref_array = conf.par.ssc.power_reference[:,1] * 1e-6
                    p_ref = np.interp(time,t_ref_array, p_ref_array)
                    logger.info("power reference: {}".format(p_ref))
                    power_difference_squared = [(p-pr)*(p-pr) for p,pr in zip(total_power, p_ref)]
                    control_difference_squared = [1e1 * assemble((c1[0]-c0[0]) * (c1[0]-c0[0])*dx(UnitIntervalMesh(1)))  for c0,c1 in zip(controls[:-1],controls[1:])]
                    # print(control_difference_squared)
                    logger.info("Power cost:   {:.2e}".format(sum(power_difference_squared)))
                    logger.info("Control cost: {:.2e}".format(sum(control_difference_squared)))
                    tracking_functional = sum(power_difference_squared)  + \
                                          sum(control_difference_squared)
                    m = [Control(x[0]) for x in controls]
                    gradient = compute_gradient(tracking_functional, m)
                    scale = 1.

                    gradient = np.array([scale*float(g) for g in gradient])
                    logger.info("Computed gradient: {}".format(gradient))
                    # step_magnitude = np.abs(scale*gradient)
                    if conf.par.turbine.yaw_rate_limit < 0:
                        max_step = np.deg2rad(5.)
                    else:
                        max_step = conf.par.turbine.yaw_rate_limit * conf.par.ssc.control_discretisation
                    # step = np.sign(gradient) * np.min((np.abs(gradient), max_step*np.ones_like(gradient)),0)
                    logger.info("Functional: {:.2e}".format(tracking_functional))
                    # cds = np.zeros_like(power_difference_squared)
                    # cds[:-1] = np.array(control_difference_squared)
                    tracking_functional_array = np.array(power_difference_squared) # + cds
                    scale = 1e0 * np.ones_like(gradient)
                    step = -1 * scale * (conf.par.simulation.time_step / conf.par.ssc.control_discretisation) * \
                           (tracking_functional_array / gradient)
                    logger.info("Step: {}".format(step))
                    step = np.sign(step) * np.min((np.abs(step), max_step*np.ones_like(gradient)), 0)

            else:
                step = np.zeros_like(self._yaw_reference_series[:,1])
            logger.info("Applied step: {}".format(step))

            if conf.par.turbine.yaw_rate_limit > 0:
                step[0] = 0.
            self._yaw_reference_series[:, 1] += step

            if conf.par.turbine.yaw_rate_limit > 0:
                yaw_rate_limit = conf.par.turbine.yaw_rate_limit
                for idx in range(len(self._yaw_reference_series)-1):
                    dyaw = self._yaw_reference_series[idx+1,1]-self._yaw_reference_series[idx,1]
                    dt = self._yaw_reference_series[idx+1,0]-self._yaw_reference_series[idx,0]
                    dmax= yaw_rate_limit*dt
                    dyaw = np.max((-dmax, np.min((dmax,dyaw))))
                    self._yaw_reference_series[idx+1,1] = self._yaw_reference_series[idx,1]+dyaw


            conf.par.wind_farm.controller.yaw_series = self._yaw_reference_series
            self._dynamic_flow_solver.reset_checkpoint()
            self._dynamic_flow_solver.solve_segment(conf.par.ssc.control_horizon)

        # else:
        # send saved signal
        reference_index = int((simulation_time - conf.par.simulation.time_step)
                              % conf.par.ssc.control_horizon // conf.par.ssc.control_discretisation)
        logger.debug("Sending reference_index: {:5d}".format(reference_index))
        self._yaw_reference = self._yaw_reference_series[reference_index, 1:]


    def get_power_reference(self, simulation_time):
        t_ref_array = conf.par.ssc.power_reference[:, 0]
        p_ref_array = conf.par.ssc.power_reference[:, 1]
        return np.interp(simulation_time, t_ref_array, p_ref_array)