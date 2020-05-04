from fenics import *
from fenics_adjoint import *

import controlmodel.conf as conf
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver

import numpy as np
import time
import matplotlib.pyplot as plt

parameters["form_compiler"]["quadrature_degree"] = 8
parameters["form_compiler"]["optimize"] = True

# parameters["reorder_dofs_serial"] = False

def main():
    time_start = time.time()
    conf.par.load("./config/test_config.yaml")

    # this is where parameter adjustments are possible

    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)

    dfs.solve()

    functional_list = dfs.get_power_functional_list()

    # J = sum([sum(x) for x in functional_list])
    n = int(conf.par.wind_farm.controller.control_discretisation / conf.par.simulation.time_step)
    binned_J = []
    for idx in range(len(functional_list) // n):
        print(idx)
        print(functional_list[idx*n:(idx+1)*n])
        binned_J.append(sum([sum(x) for x in functional_list][idx*n:(idx+1)*n]))

    # J = sum([sum(x) for x in functional_list[0:n]])
    # J = sum([sum(x) for x in functional_list[n:2*n]])

    m = [Control(c[0]) for c in wind_farm.get_controls()]
    g = compute_gradient(J, m)


    time_end = time.time()
    print("Total time: {:.2f} seconds".format(time_end - time_start))


if __name__ == '__main__':
    main()
