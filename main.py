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

def update_yaw_with_series(simulation_time, turbine_yaw, yaw_series):
    t = yaw_series[:, 0]
    for idx in range(len(turbine_yaw)):
        y = yaw_series[:, idx + 1]
        turbine_yaw[idx].assign(np.interp(simulation_time, t, y))
        # print(float(turbine_yaw[idx]))


def main():
    conf.par.load("./config/test_config.yaml")

    time_start = time.time()

    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)

    dfs.solve()

    time_end = time.time()
    print("Total time: {:.2f} seconds".format(time_end - time_start))


if __name__ == '__main__':
    main()
