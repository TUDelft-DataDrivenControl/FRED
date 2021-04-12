from fenics import *
import fred.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
from fred.windfarm import WindFarm
from fred.flowproblem import DynamicFlowProblem, SteadyFlowProblem
from fred.flowsolver import DynamicFlowSolver, SteadyFlowSolver
from fred.ssc import SuperController
from multiprocessing import Process
import fred.analysis as analysis

from fred.estimator import Estimator

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
    time_start = time.time()

    est = Estimator()
    est.run()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))



if __name__ == '__main__':
    conf.par.load("./config/two.01.step.estimator.yaml")
    main()