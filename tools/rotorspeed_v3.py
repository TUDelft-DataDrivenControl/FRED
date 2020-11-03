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


def get_coefficient(func, coord1, coord2, gradient=False, grad_idx=None):
    return func(coord1, coord2)


backend_get_coefficient = get_coefficient


class CoefficientBlock(Block):
    def __init__(self, func, coord1, coord2, **kwargs):
        super(CoefficientBlock, self).__init__()
        self.kwargs = kwargs
        self.func = func
        self.add_dependency(coord1)
        self.add_dependency(coord2)
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
        grad_idx = project(self.func.dx(idx), self.V)
        return grad_idx(inputs[0], inputs[1]) * adj_inputs[0]

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_get_coefficient(self.func, inputs[0], inputs[1])


get_coefficient = overload_function(get_coefficient, CoefficientBlock)



def main():
    pitch_grid, tsr_grid, ct_array, cp_array = read_rosco_curves()
    ct, cp = lookup_field(pitch_grid, tsr_grid, ct_array, cp_array)

    tsr = Constant(10.)
    pitch =Constant(1.)
    ctp = Constant(0.)
    ctval = get_coefficient(ct, pitch, tsr)
    ctp.assign(ctval)
    print(ctval)
    J = ctp
    controls = [pitch, tsr]
    for idx in range(3):
        new_pitch = Constant(np.random.rand())
        new_tsr = Constant(5+5*np.random.rand())
        controls = controls + [new_pitch, new_tsr]
        pitch.assign(new_pitch)
        tsr.assign(new_tsr)
        ctval = get_coefficient(ct, pitch, tsr)
        ctp.assign(ctval)
        J += ctp
        print(J)

    J = ctp
    m = [Control(c) for c in controls]
    Jhat = ReducedFunctional(J,m)
    h = [Constant(0.01*np.random.rand()) for x in controls]
    Jhat.derivative()
    taylor_test(Jhat, controls, h)

if __name__ == '__main__':
    main()
