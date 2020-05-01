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
