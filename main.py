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

parameters["form_compiler"]["quadrature_degree"] = 8
parameters["form_compiler"]["optimize"] = True

# parameters["reorder_dofs_serial"] = False


def main():
    time_start = time.time()
    conf.par.load("./config/test_config.yaml")

    # this is where parameter adjustments are possible
    # #
    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)
    # # #
    # dfs.solve()
    # dfs.solve_segment(300)
    # dfs.solve_segment(360)
    dfs.solve_segment(600)
    # #
    # # # analysis.construct_jacobian_matrix(dfs, turbine_idx=0)
    # analysis.construct_lti_jacobian_matrix(dfs, turbine_idx=0)
    # dj_dm = load_lti_jacobian(turbine_idx=0)
    # np.fill_diagonal(dj_dm,0.)
    # plot_jacobian(dj_dm)
    # # # plt.figure()
    # plt.plot(dj_dm[:,0])
    # # p = np.trapz(dj_dm[-1,:],dx=np.deg2rad(10))
    # # print(p)
    plt.show()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


def main_steady():
    time_start = time.time()
    conf.par.load("./config/test_config_steady.yaml")
    wind_farm = WindFarm()
    sfp = SteadyFlowProblem(wind_farm)
    sfs = SteadyFlowSolver(sfp)
    sfs.solve()

    functional_list = sfs.get_power_functional_list()
    controls = sfs.get_flow_problem().get_wind_farm().get_controls()

    J = sum(functional_list[0])
    # J = functional_list[0][1]
    m = [Control(c) for c in controls]
    # g = compute_gradient(J,m)
    Jhat = ReducedFunctional(-J,m)
    m_opt = minimize(Jhat, bounds=([-0.5,-0.5], [0.5,0.5]))
    [print(float(x)) for x in m_opt]
    [wt.set_yaw(y) for (wt,y) in zip(wind_farm.get_turbines(), m_opt)]
    sfs.solve()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


def main_rotating():
    time_start = time.time()
    conf.par.load("./config/test_config_rotating.yaml")

    # this is where parameter adjustments are possible

    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)

    dfs.solve()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


def main_with_ssc():
    logger.setLevel(logging.DEBUG)
    time_start = time.time()

    def run_sim():
        conf.par.load("./config/test_config_ssc_sim.yaml")
        wind_farm = WindFarm()
        dfp = DynamicFlowProblem(wind_farm)
        dfs = DynamicFlowSolver(dfp)
        dfs.solve()

    def run_ssc():
        conf.par.load("./config/test_config_ssc_ctrl.yaml")
        ssc = SuperController()
        ssc.start()

    Process(target=run_sim).start()
    Process(target=run_ssc).start()

    # ssc.start()
    # dfs.solve()
    #
    # analysis.construct_jacobian_matrix(dfs, turbine_idx=1)

    # dj_dm = load_jacobian(turbine_idx=0)
    # plot_jacobian(dj_dm)
    # plt.show()

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


def main_power_yaw():
    time_start = time.time()
    conf.par.load("./config/test_config_power_yaw.yaml")

    # this is where parameter adjustments are possible
    yaw_series = np.zeros((28, 2))
    yaw_series[0:2, :] = [[0, 300], [299.9, 300]]
    for idx in range(13):
        yaw_series[2 * (idx + 1), :] = [300 + 30 * idx, 300 - 5 * idx]
        yaw_series[2 * (idx + 1) + 1, :] = [300 + 30 * idx + 29.9, 300 - 5 * idx]
    yaw_series[:,1] = np.deg2rad(yaw_series[:,1])
    conf.par.wind_farm.controller.yaw_series = yaw_series

    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)

    dfs.solve_segment(800)

    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


if __name__ == '__main__':
    # main()
    # main_power_yaw()
    # main_steady()
    # main_rotating()
    main_with_ssc()
