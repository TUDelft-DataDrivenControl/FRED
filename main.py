from fenics import *
from fenics_adjoint import *

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


def compute_turbine_forcing_two_dim_old(u, position, yaw, radius):
    """
        Computes two-dimensional turbine forcing based on Actuator-Disk Model.
        The force is distributed using a kernel similar to [King2017].

        :param u: two-dimensional vector velocity field
        :return: two-dimensional vector force field
        forcing - flow forcing field
        force - scaled force to three-d turbine
        power - power scaled to three-d turbine
        """
    area = np.pi * radius ** 2
    thickness = 0.2 * radius

    def _compute_thrust_coefficient_prime(axial_induction):
        ct = 4 * axial_induction * (1 - axial_induction)
        return ct / (1 - axial_induction) ** 2

    thrust_coefficient_prime = _compute_thrust_coefficient_prime(axial_induction=0.33)
    force = 0.5 * area * thrust_coefficient_prime
    ud = u[0] * cos(yaw) + u[1] * sin(yaw)

    # Compute forcing kernel with shifted space.
    # 1.85544, 2.91452 are magic numbers that make kernel integrate to 1.
    # Need this monstrosity because dolfin-adjoint won't rotate the kernel if the yaw constant
    # is buried three layers deep in a nested expression... :(
    forcing = Expression(("-f * cos(yaw) *"
                          "exp(-1 * pow((cos(yaw) * (x[0]-xt) + sin(yaw) * (x[1]-yt)) / w, gamma)) "
                          "/ (1.85544 * w)"
                          "* exp(-1 * pow(pow((-sin(yaw) * (x[0]-xt) + cos(yaw) * (x[1]-yt)) / r, 2), gamma)) "
                          "/ (2.91452* pow(r, 2))",
                          "-f * sin(yaw) *"
                          "exp(-1 * pow((cos(yaw) * (x[0]-xt) + sin(yaw) * (x[1]-yt)) / w, gamma)) "
                          "/ (1.85544 * w)"
                          "* exp(-1 * pow(pow((-sin(yaw) * (x[0]-xt) + cos(yaw) * (x[1]-yt)) / r, 2), gamma)) "
                          "/ (2.91452* pow(r, 2))")
                         ,
                         xt=position[0], yt=position[1],
                         gamma=6,
                         yaw=yaw, f=force,
                         w=float(thickness), r=float(radius),
                         degree=2
                         ) * ud ** 2
    # forcing = forcing_expr

    # The above computation yields a two-dimensional body force.
    # This is scaled to a 3D equivalent for output.
    fscale = pi * 0.5 * radius
    force = forcing * fscale
    power = -dot(force, u)
    # power = force * fscale * kernel * ud**3

    return forcing, force, power

def compute_turbine_forcing_two_dim(u, farm_mesh, position, yaw, radius):
    """
        Computes two-dimensional turbine forcing based on Actuator-Disk Model.
        The force is distributed using a kernel similar to [King2017].

        :param u: two-dimensional vector velocity field
        :return: two-dimensional vector force field
        forcing - flow forcing field
        force - scaled force to three-d turbine
        power - power scaled to three-d turbine
        """
    area = np.pi * radius ** 2
    thickness = 0.2 * radius

    def _compute_thrust_coefficient_prime(axial_induction):
        ct = 4 * axial_induction * (1 - axial_induction)
        return ct / (1 - axial_induction) ** 2

    thrust_coefficient_prime = _compute_thrust_coefficient_prime(axial_induction=0.33)
    force = 0.5 * area * thrust_coefficient_prime
    ud = u[0] * cos(yaw) + u[1] * sin(yaw)

    x = SpatialCoordinate(farm_mesh)
    # turbine position
    xt = position[0]
    yt = position[1]
    # shift spatial coordinate
    xs = x[0] - xt
    ys = x[1] - yt
    # rotate spatial coordinate
    xr = cos(yaw) * xs + sin(yaw) * ys
    yr = -sin(yaw) * xs + cos(yaw) * ys
    # formulate forcing kernel
    # 1.85544, 2.91452 are magic numbers that make kernel integrate to 1.
    r = radius
    w = thickness
    gamma = 6
    kernel = exp(-1 * pow(xr / w, gamma)) / (1.85544 * w) \
             * exp(-1 * pow(pow(yr / r, 2), gamma)) / (2.91452 * pow(r, 2))
    # compute forcing function with kernel
    forcing = -1*force * kernel * as_vector((cos(yaw), sin(yaw))) * ud ** 2

    # The above computation yields a two-dimensional body force.
    # This is scaled to a 3D equivalent for output.
    fscale = pi * 0.5 * radius
    force = forcing * fscale
    power = -dot(force, u)
    # power = force * fscale * kernel * ud**3

    return forcing, force, power


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
    solver = 'petsc'
    preconditioner = 'none'
    # solver = 'minres'
    # preconditioner = 'hypre_amg'

    # controller = 'series'
    controller = 'gradient_step'

    time_start = time.time()
    domain = [2000., 1000.]  # m
    cells = [20, 10]
    turbine_positions = [
        [400., 500.],
        [1300., 500.]
    ]  # m
    num_turbines = len(turbine_positions)
    turbine_yaw_values = [
        0.0,
        0.0
    ]  # rad
    # make yaw angles Dolfin constants for later adjustment
    turbine_yaw = [Constant(x) for x in turbine_yaw_values]
    # turbine_yaw = [x.assign(y) for (x,y) in zip(turbine_yaw, turbine_yaw_values)]

    step = np.deg2rad(20.)
    # yaw_series = np.array([
    #     [0., 0., 0.],
    #     [30., step, 0.],
    #     [999.9, step, 0.]
    # ])
    yaw_series = np.array([
        [0., 0.0, 0.],
        [299.9, 0., 0.],
        [300.0, step, 0.],
        [599.9, step, 0.],
        [600.0, 0., 0.],
        [999.9, 0, 0.]
    ])

    turbine_radius = [
        89.,
        89.
    ]
    refine_radius = 100.  # m

    time_step = 0.5  # s

    final_time = 999.  # s
    inflow_velocity = [8., 0.]  # m.s^-1
    kinematic_viscosity = 0.000015
    tuning_viscosity = 1.
    mixing_length = 20.

    control_discretisation = 120.

    epsilon = 1e-14
    write_time_step = 10.
    results_dir = "./results/two"
    vtk_file_u = File(results_dir + "_U.pvd")
    vtk_file_p = File(results_dir + "_p.pvd")
    vtk_file_f = File(results_dir + "_f.pvd")
    data_file = results_dir + "_log.csv"
    write_headers(data_file, num_turbines)

    # set up mesh with refinements
    farm_mesh = generate_mesh(domain, cells)
    farm_mesh = refine_mesh(farm_mesh, turbine_positions, 2 * refine_radius)
    farm_mesh = refine_mesh(farm_mesh, turbine_positions, refine_radius)

    # set up Taylor-Hood function space over the mesh
    mixed_function_space = setup_function_space(farm_mesh)
    force_space = mixed_function_space.sub(0).collapse()

    (u, p) = TrialFunctions(mixed_function_space)  # the velocity and pressure solution functions
    (v, q) = TestFunctions(mixed_function_space)  # velocity and pressure test functions for weak form

    up_next = Function(mixed_function_space)  # the solution function, step n

    up_prev = Function(mixed_function_space)  # previous solution, step n-1
    up_prev2 = Function(mixed_function_space)  # previous solution step n-2

    u_prev, p_prev = split(up_prev)
    u_prev2, p_prev2 = split(up_prev2)

    # Set initial conditions for the numerical simulation
    initial_condition = Constant([inflow_velocity[0], inflow_velocity[1], 0.])  # velocity and pressure
    up_prev.assign(interpolate(initial_condition, mixed_function_space))

    # specify time discretisation of Navier-Stokes solutions.
    u_tilde = 1.5 * u_prev - 0.5 * u_prev2  # (1-alpha)*u+alpha*u_prev
    u_bar = 0.5 * (u + u_prev)  # (1-alpha)*u+alpha*u_prev

    dt = Constant(time_step)
    nu = Constant(kinematic_viscosity)

    # Take the combination of all turbine forcing kernels to add into the flow
    forcing_list, force_list, power_list = [], [], []
    for idx in range(len(turbine_positions)):
        forcing, force, power = compute_turbine_forcing_two_dim(u_prev, farm_mesh, turbine_positions[idx], turbine_yaw[idx],
                                                                turbine_radius[idx])
        forcing_list.append(forcing)
        force_list.append(force)
        power_list.append(power)
    f = sum(forcing_list)

    # Turbulence modelling with a mixing length model.
    if mixing_length > 1e-14:
        ml = Constant(mixing_length)
        grad_u = grad(u_prev)
        b = grad_u + grad_u.T
        s = sqrt(0.5 * inner(b, b))
        # s = sqrt(2*inner(0.5*(grad(u_prev)+grad(u_prev).T),0.5*(grad(u_prev)+grad(u_prev).T)))
        nu_turbulent = ml ** 2 * s
    else:
        nu_turbulent = Constant(0.)

    # Tuning viscosity may be used instead of a mixing length model
    nu_tuning = Constant(tuning_viscosity)

    # skew-symmetric formulation of the convective term
    convective_term = 0.5 * (inner(dot(u_tilde, nabla_grad(u_bar)), v)
                             + inner(div(outer(u_tilde, u_bar)), v))

    variational_form = inner(u - u_prev, v) * dx \
                       + dt * (nu + nu_tuning + nu_turbulent) * inner(nabla_grad(u_bar), nabla_grad(v)) * dx \
                       + dt * convective_term * dx \
                       - dt * inner(f, v) * dx \
                       - dt * inner(div(v), p) * dx \
                       + inner(div(u), q) * dx

    left = lhs(variational_form)
    right = rhs(variational_form)

    boundary_conditions = setup_boundary_conditions(domain, inflow_velocity, mixed_function_space)

    num_steps = int(final_time // time_step + 1)
    simulation_time = 0.0

    # initialise a cost functional for adjoint methods
    functional_list = []
    controls = []

    for n in range(num_steps):
        if controller == 'gradient_step':
            if simulation_time % control_discretisation <= epsilon:
                if len(functional_list) >= 1:
                    # gradient step update to yaw
                    m = [Control(x) for x in controls[-len(turbine_yaw):]]
                    J = functional_list[-1]
                    # rf = ReducedFunctional(J,m)
                    # rf.derivative()
                    gradient = compute_gradient(J,m)

                    # print([float(x) for x in gradient])
                    scale = 1e-7/control_discretisation
                    new_yaw = [Constant(float(x) + scale * float(y)) for (x, y) in zip(turbine_yaw, gradient)]
                    # new_yaw = [Constant(0.) for x in turbine_yaw]
                    # [x.assign(float(y)) for (x,y) in zip(new_yaw,turbine_yaw)]
                    # new_yaw = [Constant(float(x) - 0.15) for x in turbine_yaw]
                else:
                    new_yaw = [Constant(float(x)) for x in turbine_yaw]
                set_working_tape(Tape())
                for idx in range(len(turbine_yaw)):
                    turbine_yaw[idx].assign(new_yaw[idx])
                controls += new_yaw
                functional_list += [0.]
            print_string = "[{:6.2f}s] - step: {:04n}: \n".format(simulation_time, n)
            for idx in range(len(turbine_positions)):
                print_string += "  - wt{:n}.yaw: {:.6f} ({}) [control: {}],\n ".format(idx, float(turbine_yaw[idx]),
                                                                                       turbine_yaw[idx], controls[-len(turbine_positions) + idx])
            print(print_string)
        elif controller == 'series':
            update_yaw_with_series(simulation_time, turbine_yaw, yaw_series)




        # update_yaw(simulation_time, turbine_yaw, yaw_series)
        A = assemble(left)
        b = assemble(right)
        x = up_next.vector()
        for bc in boundary_conditions:
            bc.apply(A, b)
        solve(A, x, b,
              solver, preconditioner)
        if controller == 'gradient_step':
            for idx in range(len(turbine_positions)):
                functional_list[-1] += assemble((dt*power_list[idx])*dx)

        # print("J = {}\n".format(functional))

        print("{:.2f} seconds sim-time in {:.2f} seconds real-time".format(simulation_time, time.time() - time_start))
        simulation_time += time_step
        up_prev2.assign(up_prev)
        up_prev.assign(up_next)

        write_data(data_file, num_turbines,
                   sim_time=simulation_time,
                   yaw=turbine_yaw,
                   force_list=force_list,
                   power_list=power_list
                   )



        if simulation_time % write_time_step <= epsilon:
            u_sol, p_sol = up_next.split()
            vtk_file_u.write(u_sol)
            vtk_file_p.write(p_sol)
            vtk_file_f.write(
                project(f, force_space,
                        annotate=False))
                        # solver_type='gmres', annotate=False))


    # ctrls = [Control(x) for x in turbine_yaw]

    # # jacobian matrix gradients
    # set_log_level(LogLevel.ERROR)
    # m = [Control(x) for x in controls]
    # for turbine in range(len(turbine_positions)):
    #     for idx in range(len(functional_list)):
    #         gradient = compute_gradient(functional_list[idx], m[turbine::len(turbine_positions)])
    #         print([float(x) for x in gradient])
    #     print("\n")

    # # Taylor Test
    # rf = ReducedFunctional(functional, ctrls)
    # h = [Constant(0.1), Constant(0.1)]
    # conv_rate = taylor_test(rf,turbine_yaw,h)


    # set_log_level(LogLevel.ERROR)
    # y = minimize(rf, bounds=[[-0.5, -0.5], [0.5, 0.5]], method="BFGS", tol=1e-3,
    #              options={"maxiter": 5, "disp": False})
    # print(y)
    time_end = time.time()
    print("Total time: {:.2f} seconds".format(time_end - time_start))
    # plot_matrix(A)
    # plt.figure()
    # plt.plot(mixed_function_space.tabulate_dof_coordinates()[:, 0],
    #          mixed_function_space.tabulate_dof_coordinates()[:, 1], '.')

    print("a")


if __name__ == '__main__':
    main()
