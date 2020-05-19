from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
import time
import os
import logging
logger = logging.getLogger("cm.flowsolver")

class FlowSolver:

    def __init__(self, flow_problem):
        self._flow_problem = flow_problem

        self._left, self._right = self._flow_problem.get_linear_system()
        self._up_next, self._up_prev, self._up_prev2 = self._flow_problem.get_state_vectors()
        self._forcing = self._flow_problem.get_forcing()
        self._force_space = self._flow_problem.get_force_space()

        self._solver = 'petsc'
        self._preconditioner = 'none'

        self._vtk_file_u = None
        self._vtk_file_p = None
        self._vtk_file_f = None
        self._data_file = None

        self._setup_output_files()

        self._functional_list = []

    def _setup_output_files(self):

        results_dir = "./results/" + conf.par.simulation.name
        os.makedirs(results_dir, exist_ok=True)

        self._vtk_file_u = File(results_dir + "/U.pvd")
        self._vtk_file_p = File(results_dir + "/p.pvd")
        self._vtk_file_f = File(results_dir + "/f.pvd")
        self._data_file = results_dir + "/log.csv"

        # write headers for csv data log file
        with open(self._data_file, 'w') as log:
            log.write("time")
            for idx in range(len(self._flow_problem.get_wind_farm().get_turbines())):
                log.write(",yaw_{0:03n}".format(idx))
                log.write(",force_x_{0:03n}".format(idx))
                log.write(",force_y_{0:03n}".format(idx))
                log.write(",power_{0:03n}".format(idx))
            log.write("\r\n")

    def get_power_functional_list(self):
        return self._functional_list

    def get_flow_problem(self):
        return self._flow_problem


class SteadyFlowSolver(FlowSolver):
    def __init__(self, flow_problem):
        logger.info("Starting steady flow solver")
        FlowSolver.__init__(self, flow_problem)

    def solve(self):
        logger.info("Starting solution of steady flow problem")
        bcs = self._flow_problem.get_boundary_conditions()

        solver_parameters = {"nonlinear_solver": "snes",
                             "snes_solver": {
                                 "linear_solver": "petsc",
                                 "maximum_iterations": 40,
                                 "error_on_nonconvergence": True,
                                 "line_search": "bt",
                             }}
        solve(self._flow_problem.get_variational_form() == 0,
              self._up_next,
              bcs=bcs,
              solver_parameters=solver_parameters
              )
        self._functional_list.append([wt.get_power() for wt in self._flow_problem.get_wind_farm().get_turbines()])

        # write output
        u_sol, p_sol = self._up_next.split()
        self._vtk_file_u.write(u_sol)
        self._vtk_file_p.write(p_sol)
        self._vtk_file_f.write(
            project(self._forcing, self._force_space,
                    annotate=False))


class DynamicFlowSolver(FlowSolver):

    def __init__(self, flow_problem):
        logger.info("Starting dynamic flow solver")
        FlowSolver.__init__(self, flow_problem)

        self._simulation_time = 0.0
        self._time_start = 0.

        self._simulation_time_checkpoint = None
        self._up_prev_checkpoint = None
        self._up_prev2_checkpoint = None

    def solve(self):
        logger.info("Starting dynamic flow solution")
        num_steps = int(conf.par.simulation.total_time // conf.par.simulation.time_step + 1)

        self._time_start = time.time()  # runtime timing

        for n in range(num_steps):
            self._solve_step()
            # append individual turbine power
            self._functional_list.append([wt.get_power() for wt in self._flow_problem.get_wind_farm().get_turbines()])
        logger.info("Ran dynamic flow solution until t={:.2f}".format(self._simulation_time))

    def solve_segment(self, time_horizon):
        logger.info("Starting dynamic flow solution from t={:.2f} to t={:.2f}"
                    .format(self._simulation_time, self._simulation_time+time_horizon))
        num_steps = int(time_horizon // conf.par.simulation.time_step)

        self._functional_list = []
        self._time_start = time.time()
        self._flow_problem.get_wind_farm().clear_controls()

        for n in range(num_steps):
            self._solve_step()
            # append individual turbine power
            self._functional_list.append([wt.get_power() for wt in self._flow_problem.get_wind_farm().get_turbines()])

        logger.info("Finished segment dynamic flow solution to t={:.2f}"
                    .format(self._simulation_time))

    def _solve_step(self):
        self._simulation_time += conf.par.simulation.time_step
        self._flow_problem.update_inflow(self._simulation_time)
        self._flow_problem.get_wind_farm().apply_controller(self._simulation_time)

        A = assemble(self._left)
        b = assemble(self._right)
        x = self._up_next.vector()
        for bc in self._flow_problem.get_boundary_conditions():
            bc.apply(A, b)
        solve(A, x, b,
              self._solver, self._preconditioner)

        logger.info(
            "{:.2f} seconds sim-time in {:.2f} seconds real-time".format(self._simulation_time,
                                                                         time.time() - self._time_start))
        self._up_prev2.assign(self._up_prev)
        self._up_prev.assign(self._up_next)

        self._write_step_data()

    def _write_step_data(self):
        with open(self._data_file, 'a') as log:
            log.write("{:.6f}".format(self._simulation_time))
            # for idx in range(num_turbines):
            for wt in self._flow_problem.get_wind_farm().get_turbines():
                log.write(",{:.6f}".format(np.rad2deg(float(wt.get_yaw()))))
                force = wt.get_force()
                log.write(",{:.6f}".format(force[0]))
                log.write(",{:.6f}".format(force[1]))
                power = wt.get_power()
                log.write(",{:.6f}".format(power))
            log.write("\r\n")

        if self._simulation_time % conf.par.simulation.write_time_step <= DOLFIN_EPS_LARGE:
            u_sol, p_sol = self._up_next.split()
            self._vtk_file_u.write(u_sol)
            self._vtk_file_p.write(p_sol)
            self._vtk_file_f.write(
                project(self._forcing, self._force_space,
                        annotate=False))

    def save_checkpoint(self):
        logger.info("Saving checkpoint at t={:.2f}".format(self._simulation_time))
        self._simulation_time_checkpoint = self._simulation_time
        self._up_prev_checkpoint = self._up_prev.copy(deepcopy=True)
        self._up_prev2_checkpoint = self._up_prev2.copy(deepcopy=True)

    def reset_checkpoint(self):
        logger.info("Restoring state to checkpoint at t={:.2f}".format(self._simulation_time_checkpoint))
        self._simulation_time = self._simulation_time_checkpoint
        self._up_prev.assign(self._up_prev_checkpoint)
        self._up_prev2.assign(self._up_prev2_checkpoint)


