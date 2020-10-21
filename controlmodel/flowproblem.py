from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
import logging
logger = logging.getLogger("cm.flowproblem")


class FlowProblem:
    """
    Base class for flow problem definition
    """

    def __init__(self, wind_farm):
        self._wind_farm = wind_farm
        self._mesh = None
        self._generate_mesh()
        for step in range(conf.par.wind_farm.do_refine_turbines):
            self._refine_mesh(step+1)

        self._mixed_function_space = None
        self._setup_function_space()
        self._force_space = self._mixed_function_space.sub(0).collapse()
        self._scalar_space = self._mixed_function_space.sub(1).collapse()

        self._boundary_conditions = []

        self._u_mag = conf.par.flow.inflow_velocity[0]
        self._theta = conf.par.flow.inflow_velocity[1]
        theta = self._theta * pi / 180.
        self._inflow_velocity = Constant((-self._u_mag * sin(theta), -self._u_mag * cos(theta)))
        # self._inflow_velocity = Constant((8.,0.))
        # self._u_mag = conf.par.flow.inflow_velocity[0]
        # self._theta = conf.par.flow.inflow_velocity[1]
        # self._update_inflow_velocity()

        self._setup_boundary_conditions()

        self._variational_form = None
        self._up_next = None
        self._up_prev = None  # only used in dynamic
        self._up_prev2 = None
        self._forcing = None

        self._lhs = None
        self._rhs = None

        self._nu_turbulent = None

    def _update_inflow_velocity(self):
        theta = self._theta * pi / 180.
        u = Constant((-self._u_mag * sin(theta), -self._u_mag * cos(theta)))
        self._inflow_velocity.assign(u)

    def _generate_mesh(self):
        southwest_corner = Point([0.0, 0.0])
        northeast_corner = Point(conf.par.wind_farm.size)
        cells = conf.par.wind_farm.cells
        self._mesh = RectangleMesh(southwest_corner, northeast_corner, cells[0], cells[1], diagonal='crossed')
        #  diagonal = “left”, “right”, “left/right”, “crossed”

    def _refine_mesh(self, step):
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
                if _is_near_turbine(cell, position, conf.par.wind_farm.refine_radius * step):
                    cell_markers[cell] = True

        self._mesh = refine(self._mesh, cell_markers)

    def _setup_function_space(self):
        if conf.par.flow.finite_element == "TH":
            logger.info("Setting up Taylor-Hood finite element function space")
            vector_element = VectorElement("Lagrange", self._mesh.ufl_cell(), 2)
            finite_element = FiniteElement("Lagrange", self._mesh.ufl_cell(), 1)
            taylor_hood_element = vector_element * finite_element
            mixed_function_space = FunctionSpace(self._mesh, taylor_hood_element)
        elif conf.par.flow.finite_element == "MINI":
            logger.info("Setting up MINI finite element function space")
            P1 = FiniteElement("CG", self._mesh.ufl_cell(), 1)
            B = FiniteElement("Bubble", self._mesh.ufl_cell(), self._mesh.topology().dim() + 1)
            V = VectorElement(NodalEnrichedElement(P1, B))
            Q = P1
            mixed_function_space = FunctionSpace(self._mesh, V * Q)
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

    def get_boundary_conditions(self):
        # todo: fix
        # current_inflow = [float(self._inflow_velocity[0]), float(self._inflow_velocity[1])]
        current_inflow = self._inflow_velocity.values()
        # self._inflow_velocity.eval()

        idx_N = 0
        idx_E = 1
        idx_S = 2
        idx_W = 3

        # make empty list for boundary indices
        indices = []
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

    def _split_variational_form(self):
        self._lhs = lhs(self._variational_form)
        self._rhs = rhs(self._variational_form)

    def get_linear_system(self):
        return self._lhs, self._rhs

    def get_variational_form(self):
        return self._variational_form

    def get_state_vectors(self):
        return self._up_next, self._up_prev, self._up_prev2

    def get_forcing(self):
        return self._forcing

    def get_force_space(self):
        return self._force_space

    def get_wind_farm(self):
        return self._wind_farm

    def get_nu_turbulent(self):
        return self._nu_turbulent

    def get_scalar_space(self):
        return self._scalar_space

class SteadyFlowProblem(FlowProblem):

    def __init__(self, wind_farm):
        logger.info("Constructing steady-state flow problem")
        FlowProblem.__init__(self, wind_farm)

        self._construct_variational_form()
        self._split_variational_form()

    def _construct_variational_form(self):
        self._up_next = Function(self._mixed_function_space)

        # Need initial condition for steady state because solver won't converge starting from 0_
        vx, vy = self._inflow_velocity.values()
        initial_condition = Constant((vx, vy, 0.))  # velocity and pressure
        self._up_next.assign(interpolate(initial_condition, self._mixed_function_space))

        (u, p) = split(self._up_next)
        (v, q) = TestFunctions(self._mixed_function_space)

        nu = Constant(conf.par.flow.kinematic_viscosity)
        nu_tuning = Constant(conf.par.flow.tuning_viscosity)

        if conf.par.flow.mixing_length > 1e-14:
            ml = Constant(conf.par.flow.mixing_length)
            grad_u = grad(u)
            b = grad_u + grad_u.T
            s = sqrt(0.5 * inner(b, b))
            nu_turbulent = ml ** 2 * s
        else:
            nu_turbulent = Constant(0.)

        nu_combined = nu + nu_tuning + nu_turbulent
        self._nu_turbulent = nu_combined
        # Take the combination of all turbine forcing kernels to add into the flow
        forcing_list = [wt.compute_forcing(u) for wt in self._wind_farm.get_turbines()]
        f = sum(forcing_list)
        self._forcing = f

        # variational_form = inner(grad(u) * u, v) * dx + (nu_combined * inner(grad(u), grad(v))) * dx \
        #                    - inner(div(v), p) * dx - inner(div(u), q) * dx \
        #                    - inner(f, v) * dx

        # variational_form = inner(grad(u) * u, v) * dx +  (2 * nu_combined * inner(0.5*(grad(u) + grad(u).T), 0.5* (grad(v) + grad(v).T))) * dx \
        #                    - inner(div(v), p) * dx - inner(div(u), q) * dx \
        #                    - inner(f, v) * dx

        epsilon = 0.5*(grad(u) + grad(u).T)
        variational_form = inner(2 * nu_combined * epsilon, grad(v)) * dx \
                           + inner(grad(u) * u, v) * dx\
                           - inner(div(v), p) * dx - inner(div(u), q) * dx \
                           - inner(f, v) * dx

        self._variational_form = variational_form


class DynamicFlowProblem(FlowProblem):

    def __init__(self, wind_farm):
        logger.info("Constructing dynamic flow problem")
        FlowProblem.__init__(self, wind_farm)

        self._construct_variational_form()
        self._split_variational_form()

    def _construct_variational_form(self):
        (u, p) = TrialFunctions(self._mixed_function_space)  # the velocity and pressure solution functions
        (v, q) = TestFunctions(self._mixed_function_space)  # velocity and pressure test functions for weak form

        self._up_next = Function(self._mixed_function_space)  # the solution function, step n

        self._up_prev = Function(self._mixed_function_space)  # previous solution, step n-1
        self._up_prev2 = Function(self._mixed_function_space)  # previous solution step n-2

        u_prev, p_prev = split(self._up_prev)
        u_prev2, p_prev2 = split(self._up_prev2)

        vx, vy = self._inflow_velocity.values()
        initial_condition = Constant((vx, vy, 0.))
        self._up_prev.assign(interpolate(initial_condition, self._mixed_function_space))

        # specify time discretisation of Navier-Stokes solutions.
        u_tilde = 1.5 * u_prev - 0.5 * u_prev2  # (1-alpha)*u+alpha*u_prev
        u_bar = 0.5 * (u + u_prev)  # (1-alpha)*u+alpha*u_prev

        dt = Constant(conf.par.simulation.time_step)
        nu = Constant(conf.par.flow.kinematic_viscosity)

        # Take the combination of all turbine forcing kernels to add into the flow
        forcing_list = [wt.compute_forcing(u_prev) for wt in self._wind_farm.get_turbines()]
        f = sum(forcing_list)
        self._forcing = f

        # Turbulence modelling with a mixing length model.
        def mixing_length(u):
            x = SpatialCoordinate(u)
            ml = [Constant(conf.par.flow.mixing_length)]
            for pos in conf.par.wind_farm.positions:
                # shift space
                xs = x[0] - pos[0]
                ys = x[1] - pos[1]
                # rotate space
                # todo: get wind direction from constants
                theta = np.deg2rad(270)
                logger.warning("Gaussian mixing length variation assumes fixed West wind")
                diameter = conf.par.turbine.diameter
                length = conf.par.flow.wake_mixing_length * diameter
                offset = conf.par.flow.wake_mixing_offset * diameter
                width = conf.par.flow.wake_mixing_width * diameter
                ml_max = conf.par.flow.wake_mixing_ml_max
                xr = -np.sin(theta) * xs - np.cos(theta) * ys - offset
                yr = np.cos(theta) * xs - np.sin(theta) * ys

                ml.append(ml_max * exp(- pow(xr / length, 2) - pow(yr / width, 4)))
            return sum(ml)

        if conf.par.flow.mixing_length > 1e-14:
            ml = mixing_length(u_prev)
            grad_u = grad(u_prev)
            b = grad_u + grad_u.T
            s = sqrt(0.5 * inner(b, b))
            nu_turbulent = ml ** 2 * s
        else:
            logger.warning("Not using a mixing length model.")
            nu_turbulent = Constant(0.)

        self._nu_turbulent = nu_turbulent

        # Tuning viscosity may be used instead of a mixing length model
        nu_tuning = Constant(conf.par.flow.tuning_viscosity)
        nu_combined = (nu + nu_tuning + nu_turbulent)
        # skew-symmetric formulation of the convective term
        convective_term = 0.5 * (inner(dot(u_tilde, nabla_grad(u_bar)), v)
                                 + inner(div(outer(u_tilde, u_bar)), v))

        if conf.par.flow.continuity_correction == "wfsim":
            logger.info("Applying WFSim continuity correction, valid for West-East flow")
            # attempt to modify relaxation according to Boersma2018 WFSim
            continuity_correction = 1 * u[1].dx(1)
        else:
            logger.info("Applying no continuity correction")
            continuity_correction = 0

        nu_combined = nu + nu_tuning + nu_turbulent
        epsilon = 0.5 * (grad(u) + grad(u).T)
        variational_form = inner(u - u_prev, v) * dx \
                           + dt * inner(2 * nu_combined * epsilon, grad(v)) * dx \
                           + dt * convective_term * dx \
                           - dt * inner(f, v) * dx \
                           - dt * inner(div(v), p) * dx \
                           + inner(div(u) + continuity_correction, q) * dx
        # + dt * (nu + nu_tuning + nu_turbulent) * inner(nabla_grad(u_bar), nabla_grad(v)) * dx \

            # variational_form = inner(u - u_prev, v) * dx \
        #                    + dt * (2 * nu_combined * inner(0.5 * (grad(u) + grad(u).T), 0.5 * (grad(v) + grad(v).T))) * dx \
        #                    + dt * convective_term * dx \
        #                    - dt * inner(f, v) * dx \
        #                    - dt * inner(div(v), p) * dx \
        #                    + inner(div(u) + continuity_correction, q) * dx

        self._variational_form = variational_form

    def update_inflow(self, simulation_time):
        if conf.par.flow.type == "series":
            t = conf.par.flow.inflow_velocity_series[:, 0]
            u_mag_series = conf.par.flow.inflow_velocity_series[:, 1]
            theta_series = conf.par.flow.inflow_velocity_series[:, 2]
            self._u_mag = np.interp(simulation_time, t, u_mag_series)
            self._theta = np.interp(simulation_time, t, theta_series)
            self._update_inflow_velocity()
            logger.info("Inflow spec is: [{:.2f}, {:.2f}]".format(float(self._u_mag), float(self._theta)))
