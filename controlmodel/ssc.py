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
        self._pitch_reference = np.zeros_like(self._yaw_reference)
        logger.info("SSC initialised")

        if self._control_type == "series":
            self._time_series = conf.par.ssc.yaw_series[:, 0]
            self._yaw_series = conf.par.ssc.yaw_series[:, 1:]

        if self._control_type == "gradient_step":
            self._wind_farm = WindFarm()
            self._dynamic_flow_problem = DynamicFlowProblem(self._wind_farm)
            self._dynamic_flow_solver = DynamicFlowSolver(self._dynamic_flow_problem)

    def start(self):
        self._server = ZmqServer(conf.par.ssc.port)
        logger.info("SSC started")

        while True:
            sim_time, measurements = self._server.receive()
            # if sim_time % conf.par.ssc.control_discretisation < conf.par.simulation.time_step:
            self._set_yaw_reference(simulation_time=sim_time)
            self._server.send(self._yaw_reference, self._pitch_reference)
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
        # todo: define time horizon in configuration
        # todo: store history up_prev etc...
        time_horizon = 10.
        logger.info("Forward simulation over time horizon {:.2f}".format(time_horizon))
        self._dynamic_flow_solver.save_checkpoint()
        self._dynamic_flow_solver.solve_segment(time_horizon)
        self._dynamic_flow_solver.reset_checkpoint()

        # Get the relevant controls and power series over the time segment of the forward simulation
        controls = self._wind_farm.get_controls()
        power = self._dynamic_flow_solver.get_power_functional_list()

        logger.debug("Controls: {}".format(len(controls)))
        logger.debug("Functional: {}".format(len(power)))

        total_power = sum([sum(x) for x in power])
        average_power = total_power / len(power)
        # average power does not affect scaling if horizon is changed
        m = [Control(x[0]) for x in controls]
        gradient = compute_gradient(average_power, m)
        logger.info("Computed gradient: {}".format([float(x) for x in gradient]))

        scale = 1e-8
        gradient = float(gradient[0])
        step_magnitude = np.abs(scale*gradient)
        step = -1 * np.sign(gradient) * np.min((step_magnitude, 0.1))
        logger.info("Step magnitude: {:.2f}, applied step: {:.2f}".format(step_magnitude, step))
        self._yaw_reference[0] += step
        conf.par.wind_farm.yaw_angles = self._yaw_reference.copy()
        # todo: pass yaw reference to the local controller

