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
    # conf.par.load("./config/test_config.yaml")

    # this is where parameter adjustments are possible
    # #
    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)
    # # #
    dfs.solve()
    # dfs.solve_segment(300)
    # dfs.solve_segment(360)
    # dfs.solve_segment(600)
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
    # plt.show()

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
        conf.par.load("./config/one.ssc.sim.yaml")
        wind_farm = WindFarm()
        dfp = DynamicFlowProblem(wind_farm)
        dfs = DynamicFlowSolver(dfp)
        dfs.solve()

    def run_ssc():
        conf.par.load("./config/one.ssc.ctrl.yaml")
        t = np.arange(0,1000.,1.)
        pr = 6.0e6 + 0.7e6 *np.round(np.cos(t/10))
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

def main_with_ssc_two():
    logger.setLevel(logging.DEBUG)
    time_start = time.time()

    def run_sim():
        conf.par.load("./config/two.ssc.sim.yaml")
        wind_farm = WindFarm()
        dfp = DynamicFlowProblem(wind_farm)
        dfs = DynamicFlowSolver(dfp)
        dfs.solve()

    def run_ssc():
        conf.par.load("./config/two.ssc.ctrl.yaml")
        t = np.arange(0,1000.,1.)
        pr = 20.0e6 + 0.7e6 *np.round(np.cos(t/10))
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


def main_step_series():
    time_start = time.time()
    for step in [0, 5, 10, 15, 20]:
        conf.par.load("./config/two.01.step.yaml")
        yaw_series = np.array([[0., 270., 270.],
                               [299.9, 270., 270.],
                               [300.0, 270. + step, 270.],
                               [1000.0, 270. + step, 270]])
        yaw_series[:,1:] = np.deg2rad(yaw_series[:,1:])
        conf.par.wind_farm.controller.yaw_series = yaw_series
        conf.par.simulation.name = "two.01.step.{:02d}".format(step)
        print("Starting: " + conf.par.simulation.name)
        wind_farm = WindFarm()
        dfp = DynamicFlowProblem(wind_farm)
        dfs = DynamicFlowSolver(dfp)
        dfs.solve()
    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))

def main_yaw_sweep():
    time_start = time.time()
    conf.par.load("./config/two.01.step.yaml")
    t = np.linspace(0, 13400, 13401)

    yaw_angle_output = 270. * np.ones((13401, 2))
    idx = 0

    angle_start = 310.
    angle_sweep = 80.
    step_length = 300.
    step_size = 2.
    time_start_sweep = 600.
    time_end_sweep = time_start_sweep + ((angle_sweep / step_size) + 1) * step_length
    for current_time in t:
        if time_start_sweep <= current_time < time_end_sweep:
            yaw_angle_output[idx, 0] = angle_start - step_size * (
                        ((step_size / step_length) * (current_time - time_start_sweep)) // step_size)

        idx += 1

    yaw_series = np.hstack((np.array([t]).transpose(), yaw_angle_output))
    yaw_series[:, 1:] = np.deg2rad(yaw_series[:, 1:])
    conf.par.wind_farm.controller.yaw_series = yaw_series
    conf.par.simulation.name = "two.02.sweep"
    conf.par.simulation.total_time = 13400.
    print("Starting: " + conf.par.simulation.name)
    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)
    dfs.solve()


    time_end = time.time()
    logger.info("Total time: {:.2f} seconds".format(time_end - time_start))

if __name__ == '__main__':
    # main()
    # main_power_yaw()
    # main_steady()
    # main_rotating()
    # main_with_ssc()
    # main_with_ssc_two()
    # main_step_series()
    # main_yaw_sweep()
    # conf.par.load("./config/one.00.steady.yaml")
    conf.par.load("./config/one.02.sweep.yaml")
    main()
    # main_with_ssc()
