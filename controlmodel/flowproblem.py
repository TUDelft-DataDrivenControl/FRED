from fenics import *
from fenics_adjoint import *
import controlmodel.conf as conf
from controlmodel.turbine import Turbine

class DynamicFlowProblem:

    def __init__(self, wind_farm):
        self._wind_farm = wind_farm

        self._mesh = None
        self._generate_mesh()
        if conf.par.wind_farm.do_refine_turbines:
            self._refine_mesh()

        self._mixed_function_space = None
        self._setup_function_space()

        self._boundary_conditions = []
        self._inflow_velocity = conf.par.flow.inflow_velocity
        self._setup_boundary_conditions()


        self._variational_form = None
        self._up_next = None
        self._up_prev = None
        self._up_prev2 = None
        self._construct_variational_form()
        self._lhs = None
        self._rhs = None
        self._split_variational_form()

    def _generate_mesh(self):
        southwest_corner = Point([0.0, 0.0])
        northeast_corner = Point(conf.par.wind_farm.size)
        cells = conf.par.wind_farm.cells
        self._mesh = RectangleMesh(southwest_corner, northeast_corner, cells[0], cells[1], diagonal='crossed')
        #  diagonal = “left”, “right”, “left/right”, “crossed”

    def _refine_mesh(self):
        #
        cell_markers = MeshFunction("bool", self._mesh, 2)
        cell_markers.set_all(False)

        def _is_near_turbine(cell, pos, radius):
            # check if cell midpoint within refinement radius around turbine
            in_rx = abs(cell.midpoint().x() - pos[0]) <= radius
            in_ry = abs(cell.midpoint().y() - pos[1]) <= radius
            return in_rx and in_ry

        for position in conf.par.wind_farm.positions:
            for cell in cells(self._mesh):
                if _is_near_turbine(cell, position, conf.par.wind_farm.refine_radius):
                    cell_markers[cell] = True

        self._mesh = refine(self._mesh, cell_markers)

    def _setup_function_space(self):
        vector_element = VectorElement("Lagrange", self._mesh.ufl_cell(), 2)
        finite_element = FiniteElement("Lagrange", self._mesh.ufl_cell(), 1)
        taylor_hood_element = vector_element * finite_element
        mixed_function_space = FunctionSpace(self._mesh, taylor_hood_element)
        self._mixed_function_space = mixed_function_space

    def _setup_boundary_conditions(self):
        bound_margin = 1.

        def wall_boundary_north(x, on_boundary):
            return x[1] >= conf.par.wind_farm.size[1] - bound_margin and on_boundary

        def wall_boundary_east(x, on_boundary):
            return x[0] >= conf.par.wind_farm.size[0] - bound_margin and on_boundary

        def wall_boundary_south(x, on_boundary):
            return x[1] <= 0. + bound_margin and on_boundary

        def wall_boundary_west(x, on_boundary):
            return x[0] <= 0. + bound_margin and on_boundary


        boundaries = [wall_boundary_north,
                      wall_boundary_east,
                      wall_boundary_south,
                      wall_boundary_west]
        bcs = [DirichletBC(self._mixed_function_space.sub(0), self._inflow_velocity, b) for b in boundaries]
        self._boundary_conditions = bcs

    def get_boundary_conditions(self, current_inflow):
        idx_N = 0
        idx_E = 1
        idx_S = 2
        idx_W = 3

        # make empty list for boundary indices
        indices  = []
        epsilon = DOLFIN_EPS_LARGE
        if current_inflow[1] <= -epsilon:
            indices.append(idx_N)

        if current_inflow[1] >= epsilon:  # north in positive y direction.
            indices.append(idx_S)

        if current_inflow[0] <= -epsilon:
            indices.append(idx_E)

        if current_inflow[0] >= epsilon:  # east in positive x direction
            indices.append(idx_W)

        # neither west and east boundaries are active within a conf.epsilon margin around 0.
        # neither north and south boundaries are active within a conf.epsilon margin around 0.

        return [self._boundary_conditions[x] for x in indices]

    def _construct_variational_form(self):
        (u, p) = TrialFunctions(self._mixed_function_space)  # the velocity and pressure solution functions
        (v, q) = TestFunctions(self._mixed_function_space)  # velocity and pressure test functions for weak form

        self._up_next = Function(self._mixed_function_space)  # the solution function, step n

        self._up_prev = Function(self._mixed_function_space)  # previous solution, step n-1
        self._up_prev2 = Function(self._mixed_function_space)  # previous solution step n-2

        u_prev, p_prev = split(self._up_prev)
        u_prev2, p_prev2 = split(self._up_prev2)

        # Set initial conditions for the numerical simulation
        initial_condition = Constant(
            [conf.par.flow.inflow_velocity[0], conf.par.flow.inflow_velocity[1], 0.])  # velocity and pressure
        self._up_prev.assign(interpolate(initial_condition, self._mixed_function_space))

        # specify time discretisation of Navier-Stokes solutions.
        u_tilde = 1.5 * u_prev - 0.5 * u_prev2  # (1-alpha)*u+alpha*u_prev
        u_bar = 0.5 * (u + u_prev)  # (1-alpha)*u+alpha*u_prev

        dt = Constant(conf.par.simulation.time_step)
        nu = Constant(conf.par.flow.kinematic_viscosity)

        # Take the combination of all turbine forcing kernels to add into the flow
        forcing_list, force_list, power_list = [], [], []
        turbines = self._wind_farm.get_turbines()

        for idx in range(len(turbines)):
            forcing, force, power = turbines[idx].compute_forcing(u_prev)
            forcing_list.append(forcing)
            force_list.append(force)
            power_list.append(power)
        f = sum(forcing_list)

        # Turbulence modelling with a mixing length model.
        if conf.par.flow.mixing_length > 1e-14:
            ml = Constant(conf.par.flow.mixing_length)
            grad_u = grad(u_prev)
            b = grad_u + grad_u.T
            s = sqrt(0.5 * inner(b, b))
            nu_turbulent = ml ** 2 * s
        else:
            nu_turbulent = Constant(0.)

        # Tuning viscosity may be used instead of a mixing length model
        nu_tuning = Constant(conf.par.flow.tuning_viscosity)

        # skew-symmetric formulation of the convective term
        convective_term = 0.5 * (inner(dot(u_tilde, nabla_grad(u_bar)), v)
                                 + inner(div(outer(u_tilde, u_bar)), v))

        variational_form = inner(u - u_prev, v) * dx \
                           + dt * (nu + nu_tuning + nu_turbulent) * inner(nabla_grad(u_bar), nabla_grad(v)) * dx \
                           + dt * convective_term * dx \
                           - dt * inner(f, v) * dx \
                           - dt * inner(div(v), p) * dx \
                           + inner(div(u), q) * dx

        self._variational_form = variational_form

    def _split_variational_form(self):
        self._lhs = lhs(self._variational_form)
        self._rhs = rhs(self._variational_form)

    def get_linear_system(self):
        return self._lhs, self._rhs

    def get_state_vectors(self):
        return self._up_next, self._up_prev, self._up_prev2

    # variational form


