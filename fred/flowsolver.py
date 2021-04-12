from fenics import *
import fred.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
import time
import os
import logging
logger = logging.getLogger("cm.flowsolver")

class FlowSolver:

    def __init__(self, flow_problem, ssc=None):
        self._flow_problem = flow_problem

        self._left, self._right = self._flow_problem.get_linear_system()
        self._up_next, self._up_prev, self._up_prev2 = self._flow_problem.get_state_vectors()
        self._u_sol, self._p_sol = self._up_next.split()
        self._boundary_field = self._flow_problem.get_boundary_field()
        self._forcing = self._flow_problem.get_forcing()
        self._force_space = self._flow_problem.get_vector_space()
        self._nu_turbulent = self._flow_problem.get_nu_turbulent()
        self._scalar_space = self._flow_problem.get_scalar_space()

        self._solver = 'petsc'
        self._preconditioner = 'none'

        self._vtk_file_u = None
        self._vtk_file_p = None
        self._vtk_file_f = None
        self._data_file = None
        self._probes = conf.par.simulation.probes

        self._supercontroller = ssc
        self._setup_output_files()

        self._functional_list = []
        self._state_update_parameters = []


    def _setup_output_files(self):

        results_dir = "./results/" + conf.par.simulation.name
        os.makedirs(results_dir, exist_ok=True)

        self._vtk_file_u = File(results_dir + "/U.pvd")
        self._vtk_file_p = File(results_dir + "/p.pvd")
        self._vtk_file_f = File(results_dir + "/f.pvd")
        # self._vtk_file_nu = File(results_dir + "/nu.pvd")
        # self._vtk_file_pe = File(results_dir + "/Pe.pvd")
        self._data_file = results_dir + "/log.csv"

        # write headers for csv data log file
        with open(self._data_file, 'w') as log:
            log.write("time")
            for idx in range(len(self._flow_problem.get_wind_farm().get_turbines())):
                log.write(",yaw_{0:03n}".format(idx))
                log.write(",induction_{0:03n}".format(idx))
                log.write(",pitch_{0:03n}".format(idx))
                log.write(",torque_{0:03n}".format(idx))
                log.write(",tsr_{0:03n}".format(idx))
                log.write(",force_x_{0:03n}".format(idx))
                log.write(",force_y_{0:03n}".format(idx))
                log.write(",power_{0:03n}".format(idx))
                log.write(",velocity_x_{0:03n}".format(idx))
                log.write(",velocity_y_{0:03n}".format(idx))
                log.write(",ud_{0:03n}".format(idx))
                log.write(",kernel_{0:03n}".format(idx))
            if self._supercontroller is not None:
                log.write(",power_ref")
            for idx in range(len(self._probes)):
                log.write(",probe_{0:03n}".format(idx))
            log.write("\r\n")

    def get_power_functional_list(self):
        return self._functional_list

    def get_flow_problem(self):
        return self._flow_problem

    def get_velocity_solution(self):
        return self._u_sol

    def get_pressure_solution(self):
        return self._p_sol

    def get_segment_state_update_parameters(self):
        return self._segment_state_update_parameters

    def get_state_update_parameters(self, idx0=0, idx1=None):
        return self._state_update_parameters[idx0:idx1]

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
        self._u_sol, self._p_sol = self._up_next.split()
        self._vtk_file_u.write(self._u_sol)
        self._vtk_file_p.write(self._p_sol)
        self._vtk_file_f.write(
            project(self._forcing, self._force_space,
                    annotate=False))
        # nu_t =  project(self._nu_turbulent, self._scalar_space,
        #             annotate=False)
        # self._vtk_file_nu.write(nu_t)
        # deltax = conf.par.wind_farm.size[0]/conf.par.wind_farm.cells[0]
        # # peclet_condition = u_sol.sub(0)*deltax/self._nu_turbulent
        # peclet_condition = 2 * self._nu_turbulent / u_sol.sub(0)
        # pe = project(peclet_condition, self._scalar_space,
        #              annotate=False)
        # self._vtk_file_pe.write(pe)

class DynamicFlowSolver(FlowSolver):

    def __init__(self, flow_problem, ssc=None):
        logger.info("Starting dynamic flow solver")
        FlowSolver.__init__(self, flow_problem, ssc)

        self._simulation_time = 0.0
        self._step_number = 0
        self._time_start = 0.

        self._simulation_time_checkpoint = None
        self._step_number_checkpoint = None
        self._up_prev_checkpoint = None
        self._up_prev2_checkpoint = None
        self._boundary_field_checkpoint = None

        # self._supercontroller = ssc

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
        segment_start_step = self._step_number
        self._functional_list = [] # store power for every turbine over time-steps
        self._time_start = time.time()
        self._flow_problem.get_wind_farm().clear_controls()

        for n in range(num_steps):
            self._solve_step()
            # append individual turbine power
            self._functional_list.append([wt.get_power() for wt in self._flow_problem.get_wind_farm().get_turbines()])

        self._segment_state_update_parameters = self._state_update_parameters[segment_start_step:
                                                                              segment_start_step+num_steps]
        logger.info("Finished segment dynamic flow solution to t={:.2f}"
                    .format(self._simulation_time))

    def _solve_step(self):
        self._simulation_time += conf.par.simulation.time_step

        self._flow_problem.update_inflow(self._simulation_time)
        self._flow_problem.get_wind_farm().apply_controller(self._simulation_time)

        logger.info("Insert state update parameter")
        if len(self._state_update_parameters) <= self._step_number:
            self._state_update_parameters += [Function(self._flow_problem.get_full_function_space())]
        state_update = self._state_update_parameters[self._step_number]
        self._up_prev2.assign(project(self._up_prev + state_update,
                                      self._flow_problem.get_full_function_space()))
        self._up_prev.assign(project(self._up_next + state_update,
                                     self._flow_problem.get_full_function_space()))
        # self._state_update_parameters[self._step_number].sub(0)
        # todo: explain why this does not work and .split() does
        velocity_update, pressure_update   = state_update.split()
        self._boundary_field.assign(project(self._boundary_field + velocity_update,
                                            self._flow_problem.get_vector_space()))

        # A = assemble(self._left)
        # b = assemble(self._right)
        # x = self._up_next.vector()
        # for bc in self._flow_problem.get_boundary_conditions():
        #     bc.apply(A, b)
        # solve(A, x, b,
        #       self._solver, self._preconditioner)


        logger.info("adjusted solver to a==L format")
        solve(self._left==self._right, self._up_next, self._flow_problem.get_boundary_conditions())
        self._u_sol, self._p_sol = self._up_next.split()
        logger.info(
            "{:.2f} seconds sim-time in {:.2f} seconds real-time".format(self._simulation_time,
                                                                         time.time() - self._time_start))
        # self._up_prev2.assign(self._up_prev)
        # self._up_prev.assign(self._up_next)



        self._step_number += 1

        self._write_step_data()

    def _write_step_data(self):
        with open(self._data_file, 'a') as log:
            log.write("{:.6f}".format(self._simulation_time))
            # for idx in range(num_turbines):
            for wt in self._flow_problem.get_wind_farm().get_turbines():
                yaw = wt.get_yaw()
                log.write(",{:.6f}".format(np.rad2deg(yaw)))
                a = wt.get_axial_induction()
                log.write(",{:.6f}".format(a))
                pitch = wt.get_pitch()
                log.write(",{:.6f}".format(pitch))
                torque = wt.get_torque()
                log.write(",{:.6f}".format(torque))
                tsr = wt.get_tip_speed_ratio()
                log.write(",{:.6f}".format(tsr))
                force = wt.get_force()
                log.write(",{:.6f}".format(np.sqrt(force[0]**2+force[1]**2)))
                log.write(",{:.6f}".format(np.rad2deg(np.arctan2(force[0], force[1])) % 360.))
                power = wt.get_power()
                log.write(",{:.6f}".format(power))
                velocity = wt.get_velocity()
                log.write(",{:.6f}".format(velocity[0]))
                log.write(",{:.6f}".format(velocity[1]))
                ud = velocity[0] * - sin(yaw) + velocity[1] * - cos(yaw)
                log.write(",{:.6f}".format(ud))
                kernel = wt.get_kernel()
                log.write(",{:.6f}".format(kernel))
            if self._supercontroller is not None:
                log.write(",{:.6f}".format(self._supercontroller.get_power_reference(self._simulation_time)))
            for idx in range(len(self._probes)):
                log.write(",{:.6f}".format(self.get_probe_measurement(idx)))
            log.write("\r\n")

        if (self._simulation_time % conf.par.simulation.write_time_step <= DOLFIN_EPS_LARGE)\
                and conf.par.simulation.save_logs:
            u_sol, p_sol = self._up_next.split()
            self._vtk_file_u.write(u_sol)
            self._vtk_file_p.write(p_sol)
            self._vtk_file_f.write(
                project(self._forcing, self._force_space,
                        annotate=False))
            # nut = project(self._nu_turbulent, self._scalar_space,
            #             annotate=False)
            # self._vtk_file_nu.write(nut)

    def save_checkpoint(self):
        logger.info("Saving checkpoint at t={:.2f}".format(self._simulation_time))
        self._simulation_time_checkpoint = self._simulation_time
        self._step_number_checkpoint = self._step_number
        self._up_prev_checkpoint = self._up_prev.copy(deepcopy=True)
        self._up_prev2_checkpoint = self._up_prev2.copy(deepcopy=True)
        self._boundary_field_checkpoint = self._boundary_field.copy(deepcopy=True)
        set_working_tape(Tape())

    def reset_checkpoint(self):
        logger.info("Restoring state to checkpoint at t={:.2f}".format(self._simulation_time_checkpoint))
        self._simulation_time = self._simulation_time_checkpoint
        self._step_number = self._step_number_checkpoint
        self._up_prev.assign(self._up_prev_checkpoint)
        self._up_prev2.assign(self._up_prev2_checkpoint)
        self._boundary_field.assign(self._boundary_field_checkpoint)
        set_working_tape(Tape())

    def get_checkpoint(self):
        logger.info("Returning checkpoint at t={:.2f}".format(self._simulation_time))
        up_1 = Function(self._up_prev.function_space())
        up_2  = Function(self._up_prev.function_space())
        gu = Function(self._flow_problem.get_vector_space())
        up_1.assign(self._up_prev)
        up_2.assign(self._up_prev2)
        gu.assign(self._boundary_field)
        # return self._simulation_time, self._up_prev.copy(deepcopy=True), self._up_prev2.copy(deepcopy=True)
        return self._simulation_time, self._step_number, up_1, up_2, gu

    def set_checkpoint(self, checkpoint):
        simulation_time, step_number, up_prev, up_prev2, gu = checkpoint
        self._simulation_time = simulation_time
        self._step_number = step_number
        self._up_prev.assign(up_prev)
        self._up_prev2.assign(up_prev2)
        self._boundary_field.assign(gu)
        set_working_tape(Tape())

    def get_simulation_step(self):
        return self._step_number

    def get_simulation_time(self):
        return self._simulation_time

    def get_probe_measurement(self,idx):
        velocity = self._u_sol(self._probes[idx])
        velocity_magnitude = np.sqrt(velocity[0]*velocity[0] + velocity[1]*velocity[1])
        return velocity_magnitude
