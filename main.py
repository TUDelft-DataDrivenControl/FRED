from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem, SteadyFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver, SteadyFlowSolver

import controlmodel.analysis as analysis

import numpy as np
import time

from tools.plot import *
from tools.data import *

parameters["form_compiler"]["quadrature_degree"] = 8
parameters["form_compiler"]["optimize"] = True

# parameters["reorder_dofs_serial"] = False

def main():
    time_start = time.time()
    conf.par.load("./config/test_config.yaml")

    # this is where parameter adjustments are possible

    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)

    dfs.solve()

    # analysis.construct_jacobian_matrix(dfs, turbine_idx=0)

    # dj_dm = load_jacobian(turbine_idx=0)
    # plot_jacobian(dj_dm)
    # plt.show()

    time_end = time.time()
    print("Total time: {:.2f} seconds".format(time_end - time_start))


def main_steady():
    time_start = time.time()
    conf.par.load("./config/test_config_steady.yaml")
    wind_farm = WindFarm()
    sfp = SteadyFlowProblem(wind_farm)
    sfs = SteadyFlowSolver(sfp)
    sfs.solve()

    functional_list = sfs.get_power_functional_list()
    controls = sfs.get_flow_problem().get_wind_farm().get_controls()

    # J = sum(functional_list[0])
    J = functional_list[0][1]
    m = [Control(c) for c in controls]
    # g = compute_gradient(J,m)
    Jhat = ReducedFunctional(-J,m)
    m_opt = minimize(Jhat, bounds=([-0.5,-0.5], [0.5,0.5]))
    [print(float(x)) for x in m_opt]
    [wt.set_yaw(y) for (wt,y) in zip(wind_farm.get_turbines(), m_opt)]
    sfs.solve()

    time_end = time.time()
    print("Total time: {:.2f} seconds".format(time_end - time_start))


if __name__ == '__main__':
    # main()
    main_steady()
