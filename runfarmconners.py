from fenics import *
import controlmodel.conf as conf
# if conf.with_adjoint:
#     from fenics_adjoint import *
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem, SteadyFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver, SteadyFlowSolver
import numpy as np
import time

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    filename='farmcon.log',
                    filemode='w')
logger = logging.getLogger('')

parameters["form_compiler"]["quadrature_degree"] = 8
parameters["form_compiler"]["optimize"] = True


def main():
    time_start = time.time()
    # conf.par.load("config/farmconners/fc.A1.yaml")
    # run_one_turbine_cases()
    run_three_turbine_cases()
    run_nine_turbine_cases()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))

def run_one_turbine_cases():
    for yaw_offset in [0.]: #np.linspace(-30, 30, 7):
        conf.par.load("config/farmconners/fc.A4.1WT.yaml")
        conf.par.simulation.name = "fc.A4.1WT.Y{:03.0f}".format(yaw_offset)
        conf.par.wind_farm.yaw_angles += np.deg2rad(yaw_offset)
        run_case()

def run_three_turbine_cases():
    offsets = [[  0., -10., 0.],
               [  0., -30., 0.],
               [ 10.,   0., 0.],
               [-10.,   0., 0.],
               [-10., -10., 0.],
               [-10., -10., 0.],
               [-10., -30., 0.],
               [ 30,    0., 0.],
               [-30.,   0., 0.],
               [-30., -10., 0.],
               [-30., -20., 0.],
               [-30., -30., 0.]]
    for yaw_offsets in offsets[0:]:
        conf.par.load("config/farmconners/fc.A4.3WT.yaml")
        conf.par.simulation.name += ".Y{:03.0f}_Y{:03.0f}_Y{:03.0f}".format(*yaw_offsets)
        conf.par.wind_farm.yaw_angles += np.deg2rad(yaw_offsets)
        run_case()

def run_nine_turbine_cases():
    offsets = [[-10., -10., 0., -10., -20., 0., -10., -30., 0.],
               [-30., -10., 0., -30., -20., 0., -30., -30., 0.],
               [-10.,   0., 0., -20.,   0., 0., -30.,   0., 0.]]
    for yaw_offsets in offsets[0:]:
        conf.par.load("config/farmconners/fc.A4.9WT.yaml")
        for row in range(3):
            connector = "." if row == 0 else "_"
            row_sign = "-" if yaw_offsets[row] < 0. else ""
            conf.par.simulation.name += \
                "{:s}Y{:s}{:1.0f}{:1.0f}{:1.0f}".format(
                    connector,
                    row_sign,
                    *[np.abs(y/10.) for y in yaw_offsets[row::3]])
        conf.par.wind_farm.yaw_angles += np.deg2rad(yaw_offsets)
        run_case()

def run_case():
    t0 = time.time()
    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)
    dfs.solve()
    t1 = time.time()
    logger.info("Case {:s} ran in {:.2f} s".format(conf.par.simulation.name,t1-t0))

if __name__ == '__main__':
    main()