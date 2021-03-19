from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem, SteadyFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver, SteadyFlowSolver
from controlmodel.ssc import SuperController
from multiprocessing import Process
import controlmodel.analysis as analysis

from controlmodel.estimator import Estimator

import numpy as np
import time

from tools.plot import *
from tools.data import *

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    filename='main_estimation.log',
                    filemode='w')
logger = logging.getLogger('')

parameters["form_compiler"]["quadrature_degree"] = 8
parameters["form_compiler"]["optimize"] = True

# parameters["reorder_dofs_serial"] = False


def main():
    # load measurements
    time_start = time.time()
    # conf.par.load("./config/test_config.yaml")

    # this is where parameter adjustments are possible
    # #
    # wind_farm = WindFarm()
    # dfp = DynamicFlowProblem(wind_farm)
    # dfs = DynamicFlowSolver(dfp)
    # # # #
    # dfs.solve()

    est = Estimator()
    est.load_measurements()
    est.run_transient()
    est.run_estimation_step()
    est.run_prediction()
    # for i in steps:
    #     est.run_estimation_step()
    #     est.run_prediction()


    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))



if __name__ == '__main__':
    conf.par.load("./config/two.01.step.estimator.yaml")
    main()