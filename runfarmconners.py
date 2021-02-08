from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
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
    for yaw_offset in np.linspace(-30, 30, 7):
        conf.par.load("config/farmconners/fc.A1.WT1.yaml")

        conf.par.simulation.name = "fc.A1.1WT.{:s}{:03.0f}".format("plus" if yaw_offset >= 0. else "min",
                                                           np.abs(yaw_offset))
        conf.par.wind_farm.yaw_angles += np.deg2rad(yaw_offset)

        wind_farm = WindFarm()
        dfp = DynamicFlowProblem(wind_farm)
        dfs = DynamicFlowSolver(dfp)
        dfs.solve()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


if __name__ == '__main__':
    main()