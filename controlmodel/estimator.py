from fenics import *
import controlmodel.conf as conf

if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
import os
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver

from tools.data import *
from tools.probes import *

set_log_active(False)
import logging

logger = logging.getLogger("cm.estimator")


class Estimator:
    """
    State estimation

    """

    def __init__(self):
        self._estimator_type = conf.par.estimator.estimation_type
        if self._estimator_type != "offline":
            logger.error("Only `offline` estimation is implemented")

        self._assimilation_window = conf.par.estimator.assimilation_window
        self._transient_period = conf.par.estimator.transient_period
        self._prediction_period = conf.par.estimator.prediction_period
        self._forward_step = conf.par.estimator.forward_step

        self._cost_function_weights = conf.par.estimator.cost_function_weights
        for key, val in self._cost_function_weights.items():
            self._cost_function_weights[key] = float(val)

        self._wind_farm = WindFarm()
        self._dynamic_flow_problem = DynamicFlowProblem(self._wind_farm)
        self._dynamic_flow_solver = DynamicFlowSolver(self._dynamic_flow_problem)
        # self._time_last_optimised = -1.

        self._time_measured = []
        self._stored_measurements = {}
        self._model_measurements = {}
        # todo: generalise in config file
        self._model_measurements["flow"] = []
        for idx in range(self._assimilation_window):
            self._model_measurements["flow"]  += [Function(self._dynamic_flow_problem.get_vector_space())]
        self._model_measurements["power"] = [[wt.get_power()
                                              for wt in self._dynamic_flow_problem.get_wind_farm().get_turbines()]
                                              for m in self._model_measurements["flow"]]
        self._state_update_parameters = []

        data_dir = conf.par.estimator.data["dir"]
        self._power_file = data_dir + conf.par.estimator.data["power"]
        self._yaw_file = data_dir + conf.par.estimator.data["yaw"]
        self._probe_file = data_dir + conf.par.estimator.data["probe"]

    def run(self):
        self._load_measurements()
        self._run_transient()
        for steps in range(3): # todo: properly specify range from config
            self._run_forward_model()
            self._optimise_state_update_parameters()

    def _load_measurements(self):
        self._load_measurements_from_sowfa()

    def _load_measurements_from_sowfa(self):
        logger.info("Loading measurement data from SOWFA files")
        t, p, nt = read_power_sowfa(self._power_file)
        t, y, nt = read_power_sowfa(self._yaw_file)
        if len(self._wind_farm.get_turbines()) != nt:
            logger.error("Data has {:d} turbines but estimator is initialised with {:d}".format(nt, len(
                self._wind_farm.get_turbines())))
        logger.info("Loaded power and yaw data")

        logger.info("Resampling data to simulation time step")
        # todo: convert below code to work in class
        time_vec = np.arange(0, t[-1], conf.par.simulation.time_step)
        self._stored_measurements["time"] = time_vec
        num_measurements = len(time_vec)
        self._stored_measurements["power"] = np.zeros((num_measurements, nt))
        self._stored_measurements["yaw"] = np.zeros((num_measurements, nt))
        for idx in range(nt):
            self._stored_measurements["power"][:, idx] = np.interp(time_vec, t, p[:, idx])
            self._stored_measurements["yaw"][:, idx] = np.interp(time_vec, t, y[:, idx])

        probe_positions, t, probe_data = read_probe_data(self._probe_file)
        # # probe_measurement_points = []
        #
        probe_data = probe_data[t % 1 <= 0.01, :, 0:2]
        # todo: move this to flow problem?
        cells = conf.par.wind_farm.cells
        measurement_mesh = RectangleMesh(Point([0., 0.]), Point(conf.par.wind_farm.size), cells[0], cells[1],
                                         diagonal='left/right')
        V_m = VectorElement("CG", measurement_mesh.ufl_cell(), 1)
        measurement_function_space = FunctionSpace(measurement_mesh, V_m)
        coords = measurement_function_space.sub(0).collapse().tabulate_dof_coordinates()
        points = probe_positions
        indices = []
        for coord in coords:
            # print(point)
            idx = int(np.logical_and(points[:, 0] == coord[0], points[:, 1] == coord[1]).nonzero()[0])
            indices.append(idx)

        velocity_measurements = []
        for idx in range(len(self._stored_measurements["power"])):
            velocity_measurements += [Function(measurement_function_space)]

        for n in range(len(velocity_measurements) - 1):
            velocity_measurements[n + 1].vector()[:] = probe_data[n + 1, indices, :].ravel()
        velocity_measurements[0].assign(velocity_measurements[1])

        velocity_measurements = [project(vm, self._dynamic_flow_problem.get_vector_space()) for vm in
                                 velocity_measurements]

        self._stored_measurements["probes"] = velocity_measurements

        for control in ["yaw"]:
            self._wind_farm.set_control_reference_series(name=control,
                                                         time_series=self._stored_measurements["time"],
                                                         reference_series=self._stored_measurements[control])

    def _run_transient(self):
        logger.info("Running transient part of simulation over {:.0f}s".format(conf.par.estimator.transient_period))
        with stop_annotating():
            transient_time = conf.par.estimator.transient_period - self._forward_step
            self._dynamic_flow_solver.solve_segment(transient_time)
            self._dynamic_flow_solver.save_checkpoint()

    def _run_estimation_step(self):
        logger.info("Running estimation step")

        self._dynamic_flow_solver.reset_checkpoint()
        start_step = self._dynamic_flow_solver.get_simulation_step()
        logger.warning("Note that assimilation window is used in steps, not seconds")
        end_step = start_step + self._assimilation_window

        # todo: refine forward run into separate function
        # todo: fix access to private functions
        self._model_measurements["start_step"] = start_step
        for idx in range(end_step - start_step):
            self._dynamic_flow_solver._solve_step()
            self._model_measurements["power"][idx] = \
                [wt.get_power() for wt in self._dynamic_flow_problem.get_wind_farm().get_turbines()]
            self._model_measurements["flow"][idx].assign(project(self._dynamic_flow_solver.get_velocity_solution(),
                                                        self._dynamic_flow_problem.get_vector_space()))
        self._state_update_parameters = self._dynamic_flow_solver.get_state_update_parameters(start_step, end_step)

    def _optimise_state_update_parameters(self):
        J = self._compute_objective_function()

        m = [Control(c) for c in self._state_update_parameters]
        Jhat = ReducedFunctional(J, m)

        m_opt = minimize(Jhat, "L-BFGS-B", options={"maxiter": 1, "disp": False}, tol=1e-3)
        [c.assign(co) for c, co in zip(self._state_update_parameters, m_opt)]

    def _run_forward_model(self):
        # set checkpoint
        self._dynamic_flow_solver.reset_checkpoint()
        with stop_annotating():
            self._dynamic_flow_solver.reset_checkpoint()
            self._dynamic_flow_solver.solve_segment(self._forward_step)
            self._dynamic_flow_solver.save_checkpoint()
            # self._dynamic_flow_solver.solve_segment(horizon - self._forward_step)
        self._run_estimation_step()
        with stop_annotating():
            self._dynamic_flow_solver.solve_segment(self._prediction_period)

    def _compute_objective_function(self):
        start_step = self._model_measurements["start_step"]
        objective_function_value = AdjFloat(0.)
        for idx in range(self._assimilation_window):
            flow_difference = self._stored_measurements["probes"][start_step + idx] - self._model_measurements["flow"][idx]
            cost_flow = self._cost_function_weights["velocity"] * \
                        assemble(0.5 * inner(flow_difference, flow_difference) * dx)

            cost_power = AdjFloat(0.)
            power_difference_list = [(p0 - p1) * 1e-6 for p0, p1
                                     in zip(self._stored_measurements["power"][start_step + idx],
                                            self._model_measurements["power"][idx])]
            for power_difference in power_difference_list:
                cost_power += self._cost_function_weights["power"] * 0.5 * power_difference ** 2

            control = self._state_update_parameters[idx]
            cost_input = self._cost_function_weights["input"] * assemble(0.5 * inner(control, control) * dx)

            cost_regularisation = self._cost_function_weights["regularisation"] * \
                                  assemble(0.5 * inner(grad(control), grad(control)) * dx)

            objective_function_value += cost_flow + cost_power + cost_input + cost_regularisation

        return objective_function_value
