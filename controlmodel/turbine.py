from fenics import *
import controlmodel.conf as conf

if conf.with_adjoint:
    from fenics_adjoint import *
from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

import numpy as np
import scipy.interpolate
import logging

logger = logging.getLogger("cm.turbine")


class Turbine:
    """Wind turbine class"""

    def __init__(self, position, yaw):
        """Initialise wind turbine

        Args:
            position (list of floats): [x, y] position of turbine in the wind farm (m)
            yaw: initial turbine yaw angle (rad)

        """
        logger.info("Initialising turbine at ({:5.0f}, {:5.0f})".format(position[0], position[1]))
        self._position = position
        self._yaw_ref = Constant(yaw)
        if conf.par.turbine.yaw_rate_limit > 0:
            self._yaw_rate_limit = Constant(conf.par.turbine.yaw_rate_limit)
        else:
            logger.info("Turbine has no rate limit.")
            self._yaw_rate_limit = None
        self._yaw = self._yaw_ref

        self._radius = conf.par.turbine.radius
        self._diameter = conf.par.turbine.diameter
        self._area = pi * self._radius ** 2
        self._thickness = conf.par.turbine.thickness
        self._hub_height = conf.par.turbine.hub_height

        self._axial_induction = Constant(conf.par.turbine.axial_induction)

        if conf.par.turbine.coefficients == "induction":
            self._thrust_coefficient_prime = self._compute_ct_prime(self._axial_induction)
            self._power_coefficient_prime = self._thrust_coefficient_prime * (1 - self._axial_induction)
        elif conf.par.turbine.coefficients == "lut":
            self._pitch = Constant(conf.par.turbine.pitch)  # todo: load from config file
            # todo: implement first order turbine model for torque to tipspeed ratio
            self._torque = Constant(conf.par.turbine.torque)
            self._tip_speed_ratio = Constant(conf.par.turbine.torque)
            # load ct and cp look-up table from file
            pitch_grid, tsr_grid, ct_array, cp_array = read_rosco_curves()
            self._ct_function, self._cp_function = lookup_field(pitch_grid, tsr_grid, ct_array, cp_array)
            self._thrust_coefficient_prime = Constant(0.)
            self._power_coefficient_prime = Constant(0.)
            self._update_coefficients()
            logger.warning("Setting turbine coefficients from LUT not yet fully implemented")
        else:
            raise KeyError(
                "Invalid method for ct/cp calculations: {} is not defined".format(conf.par.turbine.coefficients))

        self._force = None
        self._power = None
        self._velocity = None
        self._kernel = None

    def _compute_ct_prime(self, a):
        """Calculate thrust coefficient from axial induction.

        Args:
             a (float): axial induction factor (-)
        """
        ct = 4 * a * (1 - a)
        ctp = ct / pow((1 - a), 2)
        return ctp

    def _update_coefficients(self):
        ct = get_coefficient(self._ct_function, self._pitch, self._tip_speed_ratio)
        cp = get_coefficient(self._cp_function, self._pitch, self._tip_speed_ratio)
        a = 0.5 - 0.5 * sqrt(1-ct)
        print(ct)
        print(a)
        self.set_axial_induction(a)
        self._thrust_coefficient_prime.assign(ct / (1 - a))
        self._power_coefficient_prime.assign(cp / pow((1 - a), 2))

    def set_yaw_ref(self, new_yaw_ref):
        """Set the turbine to  new yaw reference angle.

        Assigns the specified values to the Dolfin `Constant` storing the yaw angle.

        Args:
            new_yaw_ref (float): new turbine yaw angle (rad)

        """
        self._yaw_ref.assign(new_yaw_ref)

    def compute_forcing(self, u):
        """Calculate the turbine forcing effect on the flow.

        Args:
            u (Function): vector velocity field
        """
        if conf.par.simulation.dimensions == 2:
            return self._compute_turbine_forcing_two_dim(u)
        elif conf.par.simulation.dimensions == 3:
            raise NotImplementedError("Three-dimensional turbine forcing not yet defined")
            # return self._compute_force_three_dim(u)
        else:
            raise ValueError("Invalid dimension.")

    def _compute_turbine_forcing_two_dim(self, u):
        """Computes two-dimensional turbine forcing based on Actuator-Disk Model.

        Depending on the specification in the configuration file, the force is distributed using a kernel similar to
        the work by R. King (2017), or using a conventional Gaussian kernel.

        Args:
            u (Function): two-dimensional vectory velocity field

        Returns:
            Function: two-dimensional vector force field.

        """

        force_constant = 0.5 * conf.par.flow.density * self._area * self._thrust_coefficient_prime
        power_constant = 0.5 * conf.par.flow.density * self._area * self._power_coefficient_prime
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
        if conf.par.turbine.kernel == "king":
            logger.info("Turbine forcing kernel as in work by R. King (2017)")
            kernel = exp(-1 * pow(xr / w, gamma)) / (1.85544 * w) * \
                     exp(-1 * pow(pow(yr / r, 2), gamma)) / (2.91452 * pow(r, 2))
        elif conf.par.turbine.kernel == "gaussian":
            logger.info("Turbine forcing with gaussian distribution")
            r = self._radius * 0.6
            w = self._thickness
            zr = 0
            kernel = (exp(-1.0 * pow(xr / w, 6)) / (w * 1.85544)) * \
                     (exp(-0.5 * pow(yr / r, 2)) / (r * sqrt(2 * pi))) * \
                     (exp(-0.5 * pow(zr / r, 2)) / (r * sqrt(2 * pi)))

        # compute forcing function with kernel
        scale = conf.par.turbine.deflection_scale
        logger.info("Scaling force for wake deflection by factor {:.1f}".format(scale))
        forcing = -1 * force_constant * kernel * as_vector((-sin(self._yaw), -scale * cos(self._yaw))) * ud ** 2
        # todo: check this
        power = power_constant * kernel * ud ** 3

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
        """Get turbine yaw angle.

        Returns:
            float: turbine yaw angle (rad)

        """
        return float(self._yaw)



    def get_force(self):
        """Get current turbine force.

        Performs integration of turbine forcing kernel over wind farm domain.

        Returns:
            list of floats: x and y component of turbine force (N)

        """
        return [assemble(self._force[0] * dx), assemble(self._force[1] * dx)]

    def get_power(self):
        """Get current turbine power.

        Performs integration of power kernel over wind farm domain.

        Returns:
            float: turbine power (W)
        """
        return assemble(self._power * dx)

    def get_velocity(self):
        # return [assemble(self._velocity * dx),-1]
        return [assemble(self._velocity[0] * dx), assemble(self._velocity[1] * dx)]

    def get_kernel(self):
        """Get the integrated kernel value.

        Perform integration of kernel over wind farm domain. Kernel should integrate to 1.

        Returns:
            float: kernel size (-)
        """
        return assemble(self._kernel * dx)

    def get_axial_induction(self):
        """Get the axial induction factor.

        Returns:
            float: turbine axial induction factor (-)
        """
        return float(self._axial_induction)

    def set_axial_induction(self, new_axial_induction):
        """Set the turbine to the new axial induction factor.

        Args:
            new_axial_induction (float): new axial induction factor (-)

        """
        self._axial_induction.assign(new_axial_induction)

    def get_pitch(self):
        return float(self._pitch)

    def get_torque(self):
        return float(self._torque)

    def set_pitch_and_torque(self, new_pitch=0., new_torque=0.):
        self._pitch.assign(new_pitch)
        self._torque.assign(new_torque)
        logger.warning("Linking torque control directly to TSR because turbine model not yet implemented")
        #todo: implement first order turbine model
        self._tip_speed_ratio.assign(new_torque)
        self._update_coefficients()

    def get_tip_speed_ratio(self):
        return float(self._tip_speed_ratio)


def read_rosco_curves():
    filename = "./config/Cp_Ct_Cq.DTU10MW.txt"
    with open(filename, "r") as f:
        datafile = f.readlines()
    for idx in range(len(datafile)):
        if "Pitch angle" in datafile[idx]:
            pitch_array = np.loadtxt(filename, skiprows=idx + 1, max_rows=1)
        if "TSR vector" in datafile[idx]:
            tsr_array = np.loadtxt(filename, skiprows=idx + 1, max_rows=1)
        if "Wind speed" in datafile[idx]:
            wind_speed = np.loadtxt(filename, skiprows=idx + 1, max_rows=1)

        if "Power coefficient" in datafile[idx]:
            cp_array = np.loadtxt(filename, skiprows=idx + 2, max_rows=len(tsr_array))
        if "Thrust coefficient" in datafile[idx]:
            ct_array = np.loadtxt(filename, skiprows=idx + 2, max_rows=len(tsr_array))
        if "Torque coefficent" in datafile[idx]:
            cq_array = np.loadtxt(filename, skiprows=idx + 2, max_rows=len(tsr_array))

    pitch_grid, tsr_grid = np.meshgrid(pitch_array, tsr_array)
    return pitch_grid, tsr_grid, ct_array, cp_array


def lookup_field(pitch_grid, tsr_grid, ct_array, cp_array):
    # construct function space
    sw_corner = Point(np.min(pitch_grid), np.min(tsr_grid))
    ne_corner = Point(np.max(pitch_grid), np.max(tsr_grid))
    (n_tsr, n_pitch) = pitch_grid.shape
    # set function in function space
    m = RectangleMesh(sw_corner, ne_corner, n_pitch + 1, n_tsr + 1)
    fe = FiniteElement("Lagrange", m.ufl_cell(), 1)
    fs = FunctionSpace(m, fe)

    # assign values to function
    dof_coords = fs.tabulate_dof_coordinates()

    ct = Function(fs)
    ct_interp = scipy.interpolate.interp2d(pitch_grid[0, :], tsr_grid[:, 0], ct_array, kind='linear')
    ct_values = ct.vector().get_local()

    cp = Function(fs)
    cp_interp = scipy.interpolate.interp2d(pitch_grid[0, :], tsr_grid[:, 0], cp_array, kind='linear')
    cp_values = cp.vector().get_local()
    for idx in range(len(dof_coords)):
        pitch, tsr = dof_coords[idx]
        logger.warning("Limiting 0<=ct<=1 for axial induction calculations")
        ct_values[idx] = np.min((np.max((ct_interp(pitch, tsr),0.)),1.))
        cp_values[idx] = np.min((np.max((cp_interp(pitch, tsr),0.)),1.))
    ct.vector().set_local(ct_values)
    cp.vector().set_local(cp_values)

    # write ct and cp field to output file for visual inspection
    # ct_file = File("ct.pvd")
    # cp_file = File("cp.pvd")
    # ct_file.write(ct)
    # cp_file.write(cp)

    return ct, cp


def get_coefficient(func, pitch, tsr):
    return func(pitch, tsr)


backend_get_coefficient = get_coefficient


class CoefficientBlock(Block):
    def __init__(self, func, pitch, tsr, **kwargs):
        super(CoefficientBlock, self).__init__()
        self.kwargs = kwargs
        self.func = func
        self.add_dependency(pitch)
        self.add_dependency(tsr)
        degree = func.function_space().ufl_element().degree()
        family = func.function_space().ufl_element().family()
        mesh = func.function_space().mesh()
        if np.isin(family, ["CG", "Lagrange"]):
            self.V = FunctionSpace(mesh, "DG", degree - 1)
        else:
            raise NotImplementedError(
                "Not implemented for other elements than Lagrange")

    def __str__(self):
        return "CoefficientBlock"

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        # output = get_derivative(inputs[0], inputs[1], idx) * adj_inputs[0]
        grad_idx = project(self.func.dx(idx), self.V)
        output = grad_idx(inputs[0], inputs[1]) * adj_inputs[0]
        return output

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_get_coefficient(self.func, inputs[0], inputs[1])


get_coefficient = overload_function(get_coefficient, CoefficientBlock)
