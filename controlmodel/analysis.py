from fenics import *
from fenics_adjoint import *
import numpy as np
import controlmodel.conf as conf


def construct_jacobian_matrix(dfs, turbine_idx, power_idx=None):
    functional_list = dfs.get_power_functional_list()

    # sum functional over control discretisation steps
    n = int(conf.par.wind_farm.controller.control_discretisation / conf.par.simulation.time_step)
    binned_functional = []

    if power_idx is None:  # Take total wind farm power
        for idx in range(len(functional_list) // n):
            binned_functional.append(
                sum([sum(x) for x in functional_list][idx * n:(idx + 1) * n]) * conf.par.simulation.time_step)
    else:  # or take power for a specific turbine
        for idx in range(len(functional_list) // n):
            binned_functional.append(
                sum([x[power_idx] for x in functional_list][idx * n:(idx + 1) * n]) * conf.par.simulation.time_step)

    controls = dfs.get_flow_problem().get_wind_farm().get_controls()

    m = [Control(c[turbine_idx]) for c in controls]

    dj_dm_list = []
    for idx in range(len(binned_functional)):
        J = binned_functional[idx]
        g = compute_gradient(J, m[0:idx + 1])
        new_row = np.zeros(len(binned_functional))
        new_row[0:idx + 1] = [float(x) for x in g]
        print(new_row)
        dj_dm_list.append(new_row)

    dj_dm = np.array(dj_dm_list)

    with open("./results/" + conf.par.simulation.name + "/djdm_T{:d}".format(turbine_idx), "wb") as f:
        np.save(f, dj_dm)

    return dj_dm
