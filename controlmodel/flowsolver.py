from fenics import *
from fenics_adjoint import *
import controlmodel.conf as conf
import time

class DynamicFlowSolver():

    def __init__(self, flow_problem):
        self._flow_problem = flow_problem

        self._left, self._right = self._flow_problem.get_linear_system()
        self._up_next, self._up_prev, self._up_prev2 = self._flow_problem.get_state_vectors()

        # todo: set up output files

        self._solver = 'petsc'
        self._preconditioner = 'none'

        results_dir = "./results/" + conf.par.simulation.name
        self._vtk_file_u = File(results_dir + "_U.pvd")
        self._vtk_file_p = File(results_dir + "_p.pvd")

    def solve(self):
        num_steps = int(conf.par.simulation.total_time // conf.par.simulation.time_step + 1)
        simulation_time = 0.0
        time_start = time.time()

        # # initialise a cost functional for adjoint methods
        # functional_list = []
        # controls = []

        for n in range(num_steps):
            # update_yaw(simulation_time, turbine_yaw, yaw_series)
            A = assemble(self._left)
            b = assemble(self._right)
            x = self._up_next.vector()
            for bc in self._flow_problem.get_boundary_conditions(conf.par.flow.inflow_velocity):
                bc.apply(A, b)
            solve(A, x, b,
                  self._solver, self._preconditioner)

            print(
                "{:.2f} seconds sim-time in {:.2f} seconds real-time".format(simulation_time, time.time() - time_start))
            simulation_time += conf.par.simulation.time_step
            self._up_prev2.assign(self._up_prev)
            self._up_prev.assign(self._up_next)

            # write_data(data_file, num_turbines,
            #            sim_time=simulation_time,
            #            yaw=turbine_yaw,
            #            force_list=force_list,
            #            power_list=power_list
            #            )

            if simulation_time % conf.par.simulation.write_time_step <= DOLFIN_EPS_LARGE:
                u_sol, p_sol = self._up_next.split()
                self._vtk_file_u.write(u_sol)
                self._vtk_file_p.write(p_sol)
                # vtk_file_f.write(
                #     project(f, force_space,
                #             annotate=False))

    # todo: def _solve_step
