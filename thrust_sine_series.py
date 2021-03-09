from fenics import *
import controlmodel.conf as conf

if conf.with_adjoint:
    from fenics_adjoint import *
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver

from tools.data import *
# from tools.probes import *

import time

set_log_active(False)

parameters["form_compiler"]["quadrature_degree"] = 8
parameters["form_compiler"]["optimize"] = True



import os
from multiprocessing import Process, Pool
from itertools import product

def run_sine_test(induction_amplitude=0, strouhal=0.25 ):
    print("Running amplitude {:03d} strouhal {:03d}".format(int(1000*induction_amplitude), int(1000*strouhal)))
    conf.par.load("./config/two.03.thrust_sine.yaml")
    conf.par.simulation.save_logs = False
    conf.par.simulation.name += "/{:03d}.{:03d}".format(int(1000*induction_amplitude), int(1000*strouhal))

    # strouhal = 0.25
    velocity = conf.par.flow.inflow_velocity[0]  # m.s^-1
    diameter = conf.par.turbine.radius * 2  # m
    frequency = strouhal * velocity / diameter  # s^-1
    radial_frequency = 2 * np.pi * frequency
    # induction_amplitude = 0.01

    # conf.par.wind_farm.controller.controls["induction"]["values"]
    new_time_series = np.arange(0., conf.par.simulation.total_time, conf.par.simulation.time_step)
    induction_reference_series = np.zeros((len(new_time_series), 3))
    induction_reference_series[:, 0] = new_time_series
    induction_reference_series[:, 1] = 0.33 - induction_amplitude + induction_amplitude * np.sin(
        radial_frequency * new_time_series)
    induction_reference_series[:, 2] = 0.33
    conf.par.wind_farm.controller.controls["axial_induction"]["values"] = induction_reference_series

    with stop_annotating():
        wind_farm = WindFarm()
        dfp = DynamicFlowProblem(wind_farm)
        dfs = DynamicFlowSolver(dfp)
        dfs.solve()

    logfile = "./results/" + conf.par.simulation.name + "/log.csv"
    nt = len(conf.par.wind_farm.positions)


    t, p = read_log_data(logfile, nt, var="power")

    t0 = 180
    tot = conf.par.simulation.total_time
    dt = tot-t0
    period = 1/frequency
    nperiods = dt // period
    t1 = int(t0+nperiods*period)
    mp = np.mean(np.sum(p, axis=1)[t0:t1])
    print("Mean power is: {:.3f} MW".format(mp))
    return [induction_amplitude, strouhal, mp]

def run_series():
    a = np.linspace(0.,0.1,11)
    p = np.zeros(a.shape)
    for idx in range(len(a)):
        p[idx] = run_sine_test(a[idx])

    return a,p

def run_grid():
    a = np.linspace(0,0.15,16)
    st = np.linspace(0.1,0.4,7)
    for idx0 in range(len(a)):
        for idx1 in range(len(st)):
            run_sine_test(induction_amplitude=a[idx0],
                          strouhal=st[idx1])
            Process(target=run_sine_test, args=[a[idx0],st[idx1]]).start()

def worker(procnum,pnum2):
    print('I am number %d,%d in process %d' % (procnum,pnum2, os.getpid()))
    return (os.getpid(), procnum, pnum2)


if __name__ == '__main__':
    a = np.linspace(0,0.15,16)
    st = np.linspace(0.1,0.4,13)

    pool = Pool(processes = 40)
    results = pool.starmap(run_sine_test, product(a, st))
    # print(pool.starmap(worker, product([0,2],[1,2,3]) ))
    np.savetxt("sine_grid.txt",results)


