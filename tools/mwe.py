from fenics import *
from fenics_adjoint import *
import numpy as np
from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function


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


mesh = UnitSquareMesh(10, 10)
V0 = FunctionSpace(mesh, "DG", 0)
V1 = FunctionSpace(mesh, "Lagrange", 1)

u = Function(V1)
x = SpatialCoordinate(u)
z = project(x[0]*x[1], V1)

x1 = [Constant(r) for r in np.random.rand(1)]
x2 = [Constant(r) for r in np.random.rand(1)]

# functional_list =
# for idx in range(len(x1)):
idx = 0
y = Constant(0.)

dz = Function(V1)

z.assign(project(z+dz,V1))
ct = get_coefficient(z, x1[idx], x2[idx])
# a = AdjFloat(sqrt(1-ct))

J = (ct) ** 2
# controls = x1 + x2
controls = [dz]
m = [Control(c) for c in controls]
h = [Constant(0.01*np.random.rand()) for c in controls]

Jhat = ReducedFunctional(J, m)
print(5*"\n")
Jhat.derivative()

print(5*"\n")
taylor_test(Jhat, controls, h)
