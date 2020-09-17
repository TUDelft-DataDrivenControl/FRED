from fenics import *
from fenics_adjoint import *
import numpy as np
from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function
import scipy.interpolate


def read_rosco_curves():
    filename = "Cp_Ct_Cq.DTU10MW.txt"
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
    # logger.warning("Limiting 0<=ct<=1 for axial induction calculations")
    for idx in range(len(dof_coords)):
        pitch, tsr = dof_coords[idx]
        ct_values[idx] = np.min((np.max((ct_interp(pitch, tsr), 0.)), 1.))
        cp_values[idx] = np.min((np.max((cp_interp(pitch, tsr), 0.)), 1.))
        a = 0.5 - 0.5 * (np.sqrt(1 - ct_values[idx]))
        # convert to local
        ct_values[idx] = ct_values[idx] / (1 - a)
        cp_values[idx] = cp_values[idx] / (1 - a) ** 2
    ct.vector().set_local(ct_values)
    cp.vector().set_local(cp_values)

    # write ct and cp field to output file for visual inspection
    # ct_file = File("ct.pvd")
    # cp_file = File("cp.pvd")
    # ct_file.write(ct)
    # cp_file.write(cp)

    return ct, cp


def get_coefficient(func, pitch, torque, rotor_speed, power_aero, disc_velocity):
    dt = 1.
    inertia = 10.
    wn = (dt / inertia) * (power_aero / rotor_speed - torque) + rotor_speed
    radius = 90.
    tsr = (wn * radius) / disc_velocity
    # tsr = float(torque * rotor_speed)
    # tsr = torque*rotor_speed
    return func(pitch, tsr)


def get_coefficient_derivative(func_grad, pitch, torque, rotor_speed, power_aero, disc_velocity, idx=0):
    dt = 1.
    inertia = 10.
    radius = 90.
    wn = (dt / inertia) * (power_aero / rotor_speed - torque) + rotor_speed
    tsr = (wn * radius) / disc_velocity
    # tsr = torque*rotor_speed
    p1 = func_grad[int(np.min((idx, 1)))]

    if idx == 0:
        p2 = float(1.)  # pitch
    elif idx == 1:  # torque derivative
        # p2 = float(rotor_speed)
        p2 = AdjFloat((radius / disc_velocity) * (-1 * dt / inertia))
    elif idx == 2:  # rotor speed
        # p2 = float(torque)
        p2 = AdjFloat((radius / disc_velocity) * (- (dt * power_aero) / (inertia * rotor_speed ** 2) + 1.))
    elif idx == 3:
        p2 = AdjFloat((radius / disc_velocity) * (dt / (inertia * rotor_speed)))
    elif idx == 4:
        p2 = AdjFloat(-1 * (wn * radius) / disc_velocity ** 2)
    else:
        raise ValueError("Derivative index out of bounds.")

    return p1(pitch, tsr) * p2


backend_get_coefficient = get_coefficient


class CoefficientBlock(Block):
    def __init__(self, func, pitch, torque, rotor_speed, power_aero, disc_velocity, **kwargs):
        super(CoefficientBlock, self).__init__()
        self.kwargs = kwargs
        self.func = func
        self.add_dependency(pitch)
        self.add_dependency(torque)
        self.add_dependency(rotor_speed)
        self.add_dependency(power_aero)
        self.add_dependency(disc_velocity)
        # self.rotor_speed = rotor_speed
        # self.power_aero = power_aero
        # self.disc_velocity = disc_velocity
        degree = func.function_space().ufl_element().degree()
        family = func.function_space().ufl_element().family()
        mesh = func.function_space().mesh()
        if np.isin(family, ["CG", "Lagrange"]):
            self.V = FunctionSpace(mesh, "DG", degree - 1)
        else:
            raise NotImplementedError(
                "Not implemented for other elements than Lagrange")
        self.func_grad = [project(func.dx(x), self.V) for x in range(2)]

    def __str__(self):
        return "CoefficientBlock"

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        # output = get_derivative(inputs[0], inputs[1], idx) * adj_inputs[0]
        # idx2 = int(np.min((idx, 1)))
        # print(idx2)
        # grad_idx = project(self.func.dx(idx2), self.V)
        # output = grad_idx(inputs[0], inputs[1]) * adj_inputs[0]\
        #          * rotor_speed_derivative(func=self.func_grad,
        #                                                pitch=inputs[0],
        #                                                torque=inputs[1],
        #                                                rotor_speed=inputs[2],
        #                                                power_aero=self.power_aero, #inputs[3],
        #                                                disc_velocity=self.disc_velocity, #inputs[4],
        #                                                idx=idx) \
        output = get_coefficient_derivative(func_grad=self.func_grad,
                                            pitch=inputs[0],
                                            torque=inputs[1],
                                            rotor_speed=inputs[2],
                                            power_aero=inputs[3],
                                            disc_velocity=inputs[4],
                                            idx=idx) * adj_inputs[0]
        print(output)
        print([float(ix) for ix in inputs])
        return output

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_get_coefficient(func=self.func,
                                       pitch=inputs[0],
                                       torque=inputs[1],
                                       rotor_speed=inputs[2],
                                       power_aero=inputs[3],
                                       disc_velocity=inputs[4])
        # return backend_get_coefficient(self.func, inputs[0], inputs[1], inputs[2], self.power_aero, self.disc_velocity)


get_coefficient = overload_function(get_coefficient, CoefficientBlock)


pitch_grid, tsr_grid, ct_array, cp_array = read_rosco_curves()
ct, cp = lookup_field(pitch_grid, tsr_grid, ct_array, cp_array)

# float((1 / inertia) * (power_aero / rotor_speed - torque) + rotor_speed)
w = Constant(5.)
q = Constant(10.)
pa = Constant(10.)
# iner = Constant(5.)
b = Constant(0.)
ud = Constant(10.)
# tsr = w
# wn = (1 / iner) * (pa / q - q) + w

w.assign(0.8)
q.assign(1.)
pa.assign(5.)
b.assign(2.)

ctp = get_coefficient(ct, b, q, w, pa, ud)

J = ctp ** 2
controls = [b, q, w, pa, ud]
m = [Control(c) for c in controls]
Jh = ReducedFunctional(J, m)
Jh.derivative()
# print([float(g) for g in Jh.derivative()])
h = [Constant(0.01 * np.random.rand()) for c in controls]

taylor_test(Jh, controls, h)
