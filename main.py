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

import numpy as np
import time

from tools.plot import *
from tools.data import *

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    filename='main.log',
                    filemode='w')
logger = logging.getLogger('')

parameters["form_compiler"]["quadrature_degree"] = 4
parameters["form_compiler"]["optimize"] = True
# parameters["reorder_dofs_serial"] = False


def main():
    time_start = time.time()
    # this is where parameter adjustments are possible

    wf = WindFarm()
    dfp = DynamicFlowProblem(wf)
    dfs = DynamicFlowSolver(dfp)
    dfs.solve()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


def main_steady():
    time_start = time.time()

    wf = WindFarm()
    sfp = SteadyFlowProblem(wf)
    sfs = SteadyFlowSolver(sfp)
    sfs.solve()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


def main_with_ssc():
    logger.setLevel(logging.DEBUG)
    time_start = time.time()

    def run_sim():
        conf.par.load("./config/one.ssc.sim.bq.yaml")
        wind_farm = WindFarm()
        dfp = DynamicFlowProblem(wind_farm)
        dfs = DynamicFlowSolver(dfp)
        dfs.solve()

    def run_ssc():
        conf.par.load("./config/one.ssc.ctrl.bq.yaml")
        t = np.arange(0,1000.,1.)
        pr = 5.0e6 + 0.7e6 *np.round(np.cos(t/10))
        power_reference = np.zeros((len(t),2))
        power_reference[:,0] = t
        power_reference[:,1] = pr
        conf.par.ssc.power_reference = power_reference
        ssc = SuperController()
        ssc.start()

    Process(target=run_sim).start()
    Process(target=run_ssc).start()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


if __name__ == '__main__':

    # conf.par.load("./config/farmconners/fc.A1.yaml")
    conf.par.load("./config/3d.1wt.yaml")
    main()
    # main_steady()
    # main_with_ssc()
    # list_timings()

