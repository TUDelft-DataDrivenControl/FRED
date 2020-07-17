from fenics import *
import controlmodel.conf as conf
if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
import time
import logging
logger = logging.getLogger("cm.analysis")


def construct_jacobian_matrix(dfs, turbine_idx, power_idx=None):
    """
    Construct a Jacobian matrix to identify power-yaw sensitivity.
    E_i/\psi_i
    :param dfs:
    :param turbine_idx:
    :param power_idx:
    :return:
    """
    functional_list = dfs.get_power_functional_list()

    # sum functional over control discretisation steps
    n = int(conf.par.wind_farm.controller.control_discretisation / conf.par.simulation.time_step)
    binned_functional = []

    if power_idx is None:  # Take total wind farm power
        power_idx = -1
        for idx in range(len(functional_list) // n):
            binned_functional.append(
                sum([sum(x) for x in functional_list][idx * n:(idx + 1) * n]) * conf.par.simulation.time_step)
    else:  # or take power for a specific turbine
        for idx in range(len(functional_list) // n):
            binned_functional.append(
                sum([x[power_idx] for x in functional_list][idx * n:(idx + 1) * n]) * conf.par.simulation.time_step)

    controls = dfs.get_flow_problem().get_wind_farm().get_yaw_controls()

    m = [Control(c[turbine_idx]) for c in controls]

    dj_dm_list = []
    for idx in range(len(binned_functional)):
        t0 = time.time()
        J = binned_functional[idx]
        g = compute_gradient(J, m[0:idx + 1])
        logger.info("Gradient calculation took: {:.3} s".format(time.time()-t0))
        new_row = np.zeros(len(binned_functional))
        new_row[0:idx + 1] = [float(x) for x in g]
        print(new_row)
        dj_dm_list.append(new_row)

    dj_dm = np.array(dj_dm_list)

    with open("./results/" + conf.par.simulation.name + "/djdm_P{:d}_T{:d}.npy".format(power_idx,turbine_idx), "wb") as f:
        np.save(f, dj_dm)

    return dj_dm


def construct_lti_jacobian_matrix(dfs, turbine_idx):
    functional_list = dfs.get_power_functional_list()

    # sum functional over control discretisation steps
    n = int(conf.par.wind_farm.controller.control_discretisation / conf.par.simulation.time_step)

    binned_functional = []

    # Take total wind farm power and make binned averages
    for idx in range(len(functional_list) // n):
        binned_functional.append(
           sum([sum(x) for x in functional_list][idx * n:(idx + 1) * n])/n)

    controls = dfs.get_flow_problem().get_wind_farm().get_yaw_controls()

    m = [Control(c[turbine_idx]) for c in controls]

    dj_dm_list = []
    idx = -1
    t0 = time.time()
    J = binned_functional[idx]
    g = compute_gradient(J, m)
    logger.info("Gradient calculation took: {:.3} s".format(time.time()-t0))
    # new_row = np.zeros(len(binned_functional))
    new_row = np.array([float(x) for x in g])
    dj_dm = np.zeros((len(new_row), len(new_row)))
    for idx in range(len(new_row)):
        dj_dm[idx, :idx + 1] = new_row[-idx - 1:]

    with open("./results/" + conf.par.simulation.name + "/djdm_LTI_P_T{:d}.npy".format(turbine_idx), "wb") as f:
        np.save(f, dj_dm)

    return dj_dm