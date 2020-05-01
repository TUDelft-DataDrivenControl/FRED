from fenics import *
from fenics_adjoint import *

import controlmodel.conf as conf
from controlmodel.turbine import Turbine
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem

import numpy as np
import time
import matplotlib.pyplot as plt

parameters["form_compiler"]["quadrature_degree"] = 8
parameters["form_compiler"]["optimize"] = True

# parameters["reorder_dofs_serial"] = False


def plot_matrix(A):
    plt.imshow(A.array(), vmin=0., vmax=0.001, cmap='binary')
    plt.show()


def generate_mesh(domain, cells):
    #
    southwest_corner = Point([0.0, 0.0])
    northeast_corner = Point(domain)
    farm_mesh = RectangleMesh(southwest_corner, northeast_corner, cells[0], cells[1], diagonal='crossed')
    #  diagonal = “left”, “right”, “left/right”, “crossed”
    return farm_mesh


def refine_mesh(farm_mesh, turbine_positions, refine_radius):
    #
    cell_markers = MeshFunction("bool", farm_mesh, 2)
    cell_markers.set_all(False)

    def _is_near_turbine(cell, pos, radius):
        # check if cell midpoint within refinement radius around turbine
        in_rx = np.abs(cell.midpoint().x() - pos[0]) <= radius
        in_ry = np.abs(cell.midpoint().y() - pos[1]) <= radius
        return in_rx and in_ry

    for position in turbine_positions:
        for cell in cells(farm_mesh):
            if _is_near_turbine(cell, position, refine_radius):
                cell_markers[cell] = True

    return refine(farm_mesh, cell_markers)


def setup_function_space(farm_mesh):
    vector_element = VectorElement("Lagrange", farm_mesh.ufl_cell(), 2)
    finite_element = FiniteElement("Lagrange", farm_mesh.ufl_cell(), 1)
    taylor_hood_element = vector_element * finite_element
    mixed_function_space = FunctionSpace(farm_mesh, taylor_hood_element)
    return mixed_function_space


def setup_boundary_conditions(domain, inflow_velocity, mixed_function_space):
    bound_margin = 1.

    def wall_boundary_west(x, on_boundary):
        return x[0] <= 0. + bound_margin and on_boundary

    def wall_boundary_north(x, on_boundary):
        return x[1] >= domain[1] - bound_margin and on_boundary

    def wall_boundary_south(x, on_boundary):
        return x[1] <= 0. + bound_margin and on_boundary

    def wall_boundary_east(x, on_boundary):
        return x[0] >= domain[0] - bound_margin and on_boundary

    # make empty list for inflow functions
    inflow = []
    outflow = []
    epsilon = 1e-14

    if inflow_velocity[0] >= epsilon:  # east in positive x direction
        inflow.append(wall_boundary_west)
    else:
        outflow.append(wall_boundary_west)

    if inflow_velocity[0] <= -epsilon:
        inflow.append(wall_boundary_east)
    else:
        outflow.append(wall_boundary_east)
    # neither west and east boundaries are active within a conf.epsilon margin around 0.

    if inflow_velocity[1] >= epsilon:  # north in positive y direction.
        inflow.append(wall_boundary_south)
    else:
        outflow.append(wall_boundary_south)

    if inflow_velocity[1] <= -epsilon:
        inflow.append(wall_boundary_north)
    else:
        outflow.append(wall_boundary_north)
    # neither north and south boundaries are active within a conf.epsilon margin around 0.

    bcs = []  # make empty list for boundary conditions
    for wall in inflow:
        bc = DirichletBC(mixed_function_space.sub(0), inflow_velocity, wall)
        bcs.append(bc)
    return bcs


def update_yaw_with_series(simulation_time, turbine_yaw, yaw_series):
    t = yaw_series[:, 0]
    for idx in range(len(turbine_yaw)):
        y = yaw_series[:, idx + 1]
        turbine_yaw[idx].assign(np.interp(simulation_time, t, y))
        # print(float(turbine_yaw[idx]))


def write_headers(data_file, num_turbines):
    # write headers for csv data log file
    with open(data_file, 'w') as log:
        log.write("time")
        for idx in range(num_turbines):
            log.write(",yaw_{0:03n}".format(idx))
            log.write(",force_x_{0:03n}".format(idx))
            log.write(",force_y_{0:03n}".format(idx))
            log.write(",power_{0:03n}".format(idx))
        log.write("\r\n")


def write_data(data_file, num_turbines, sim_time, yaw, force_list, power_list):
    # write data line to file

    with open(data_file, 'a') as log:
        log.write("{:.6f}".format(sim_time))
        for idx in range(num_turbines):
            # integrate force distribution
            force = [assemble(force_list[idx][0] * dx), assemble(force_list[idx][1] * dx)]
            power = assemble(power_list[idx]*dx)
            log.write(",{:.6f}".format(float(yaw[idx])))
            log.write(",{:.6f}".format(force[0]))
            log.write(",{:.6f}".format(force[1]))
            log.write(",{:.6f}".format(power))
        log.write("\r\n")


def main():
    conf.par.load("./config/test_config.yaml")

    solver = 'petsc'
    preconditioner = 'none'

    time_start = time.time()

    # domain = conf.par.wind_farm.size  # m
    # cells = conf.par.wind_farm.cells


    num_turbines = len(conf.par.wind_farm.positions)
    #
    # # make yaw angles Dolfin constants for later adjustment
    # turbine_yaw = [Constant(x) for x in conf.par.wind_farm.yaw_angles]
    # turbine_positions = conf.par.wind_farm.positions
    # turbines = [Turbine(x, y) for (x, y) in zip(turbine_positions, turbine_yaw)]
    #
    # refine_radius = conf.par.wind_farm.refine_radius  # m

    epsilon = DOLFIN_EPS_LARGE

    results_dir = "./results/"+conf.par.simulation.name
    vtk_file_u = File(results_dir + "_U.pvd")
    vtk_file_p = File(results_dir + "_p.pvd")
    # vtk_file_f = File(results_dir + "_f.pvd")
    data_file = results_dir + "_log.csv"
    write_headers(data_file, num_turbines)

    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    left, right = dfp.get_linear_system()
    up_next, up_prev, up_prev2 = dfp.get_state_vectors()

    # boundary_conditions = setup_boundary_conditions(domain, conf.par.flow.inflow_velocity, mixed_function_space)

    num_steps = int(conf.par.simulation.total_time // conf.par.simulation.time_step + 1)
    simulation_time = 0.0

    # # initialise a cost functional for adjoint methods
    # functional_list = []
    # controls = []

    for n in range(num_steps):
        # update_yaw(simulation_time, turbine_yaw, yaw_series)
        A = assemble(left)
        b = assemble(right)
        x = up_next.vector()
        for bc in dfp.get_boundary_conditions(conf.par.flow.inflow_velocity):
            bc.apply(A, b)
        solve(A, x, b,
              solver, preconditioner)

        print("{:.2f} seconds sim-time in {:.2f} seconds real-time".format(simulation_time, time.time() - time_start))
        simulation_time += conf.par.simulation.time_step
        up_prev2.assign(up_prev)
        up_prev.assign(up_next)

        # write_data(data_file, num_turbines,
        #            sim_time=simulation_time,
        #            yaw=turbine_yaw,
        #            force_list=force_list,
        #            power_list=power_list
        #            )

        if simulation_time % conf.par.simulation.write_time_step <= epsilon:
            u_sol, p_sol = up_next.split()
            vtk_file_u.write(u_sol)
            vtk_file_p.write(p_sol)
            # vtk_file_f.write(
            #     project(f, force_space,
            #             annotate=False))

    time_end = time.time()
    print("Total time: {:.2f} seconds".format(time_end - time_start))


if __name__ == '__main__':
    main()
