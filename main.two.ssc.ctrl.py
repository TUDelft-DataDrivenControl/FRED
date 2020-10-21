from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem, SteadyFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver, SteadyFlowSolver
from controlmodel.ssc import SuperController
import time
import numpy as np

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    filename='two.ctrl.log',
                    filemode='w')
logger = logging.getLogger('')

parameters["form_compiler"]["quadrature_degree"] = 8
parameters["form_compiler"]["optimize"] = True



def main():
    logger.setLevel(logging.DEBUG)
    time_start = time.time()
    conf.par.load("./config/two.ssc.ctrl.a.yaml")
    t = np.arange(0, 1000., 1.)
    pr = 8.0e6 + 0.7e6 * np.round(np.cos(t / 10))
    power_reference = np.zeros((len(t), 2))
    power_reference[:, 0] = t
    power_reference[:, 1] = pr
    conf.par.ssc.power_reference = power_reference
    ssc = SuperController()
    ssc.start()
    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


if __name__ == '__main__':
    main()
