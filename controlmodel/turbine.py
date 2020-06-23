from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
import logging
logger = logging.getLogger("cm.turbine")


class Turbine:

    def __init__(self, position, yaw):
        logger.info("Initialising turbine at ({:5.0f}, {:5.0f})".format(position[0], position[1]))
        self._position = position
        self._yaw = Constant(yaw)

        self._radius = conf.par.turbine.radius
        self._diameter = conf.par.turbine.diameter
        self._area = pi * self._radius ** 2
        self._thickness = conf.par.turbine.thickness
        self._hub_height = conf.par.turbine.hub_height

        self._axial_induction = conf.par.turbine.axial_induction
        self._thrust_coefficient_prime = self._compute_ct_prime(self._axial_induction)

        self._force = None
        self._power = None
        self._velocity = None
        self._kernel = None

    def _compute_ct_prime(self, a):
        ct = 4 * a * (1 - a)
        ctp = ct / (1 - a)**2
        return ctp

    def set_yaw(self,new_yaw):
        self._yaw.assign(new_yaw)

    def compute_forcing(self, u):
        if conf.par.simulation.dimensions == 2:
            return self._compute_turbine_forcing_two_dim(u)
        elif conf.par.simulation.dimensions == 3:
            raise NotImplementedError("Three-dimensional turbine forcing not yet defined")
            # return self._compute_force_three_dim(u)
        else:
            raise ValueError("Invalid dimension.")

    def _compute_turbine_forcing_two_dim(self, u):
        """
            Computes two-dimensional turbine forcing based on Actuator-Disk Model.
            The force is distributed using a kernel similar to [King2017].

            :param u: two-dimensional vector velocity field
            :return: two-dimensional vector force field
            forcing - flow forcing field
            force - scaled force to three-d turbine
            power - power scaled to three-d turbine
            """

        force = 0.5 * self._area * self._thrust_coefficient_prime
        ud = u[0] * - sin(self._yaw) + u[1] * - cos(self._yaw)

        x = SpatialCoordinate(u)
        # turbine position
        xt = self._position[0]
        yt = self._position[1]
        # shift spatial coordinate
        xs = x[0] - xt
        ys = x[1] - yt
        # rotate spatial coordinate
        xr = -sin(self._yaw) * xs - cos(self._yaw) * ys
        yr = cos(self._yaw) * xs - sin(self._yaw) * ys
        # formulate forcing kernel
        # 1.85544, 2.91452 are magic numbers that make kernel integrate to 1.
        r = self._radius
        w = self._thickness
        gamma = 6
        kernel = exp(-1 * pow(xr / w, gamma)) / (1.85544 * w) \
                 * exp(-1 * pow(pow(yr / r, 2), gamma)) / (2.91452 * pow(r, 2))
        # compute forcing function with kernel
        forcing = -1 * force * kernel * as_vector((-sin(self._yaw), -cos(self._yaw))) * ud ** 2
        # todo: check this
        power = force * kernel * ud ** 3

        # The above computation yields a two-dimensional body force.
        # This is scaled to a 3D equivalent for output.
        fscale = pi * 0.5 * self._radius
        # todo: compute accurate 3D scaling factor
        self._force = forcing * fscale
        self._power = power * fscale
        # self._power = - self._force*ud #dot(self._force, u)
        self._kernel = kernel * fscale
        self._velocity = u * kernel * fscale

        return forcing

    def get_yaw(self):
        return self._yaw

    def get_force(self):
        return [assemble(self._force[0] * dx), assemble(self._force[1] * dx)]

    def get_power(self):
        return assemble(self._power * dx)

    def get_velocity(self):
        # return [assemble(self._velocity * dx),-1]
        return [assemble(self._velocity[0] * dx), assemble(self._velocity[1] * dx)]

    def get_kernel(self):
        return assemble(self._kernel * dx)