from fenics import *
import controlmodel.conf as conf

if conf.with_adjoint:
    from fenics_adjoint import *
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem, SteadyFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver, SteadyFlowSolver
import time
import os
import sys
from tools.data import *
from tools.plot import *

from multiprocessing import Process

import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(name)-16s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M',
                    filename='./results/farmcon.log',
                    filemode='w')
logger = logging.getLogger('')

parameters["form_compiler"]["quadrature_degree"] = 8
parameters["form_compiler"]["optimize"] = True

offsets_3wt = [[0., -10., 0.],
               [0., -30., 0.],
               [10., 0., 0.],
               [-10., 0., 0.],
               [-10., -10., 0.],
               [-10., -10., 0.],
               [-10., -30., 0.],
               [30, 0., 0.],
               [-30., 0., 0.],
               [-30., -10., 0.],
               [-30., -20., 0.],
               [-30., -30., 0.]]

offsets_9wt = [[-10., -10., 0., -10., -20., 0., -10., -30., 0.],
               [-30., -10., 0., -30., -20., 0., -30., -30., 0.],
               [-10., 0., 0., -20., 0., 0., -30., 0., 0.]]


def main_run():
    with stop_annotating():
        time_start = time.time()
        # conf.par.load("config/farmconners/fc.A1.yaml")
        # run_one_turbine_cases()
        run_three_turbine_cases()  # 12
        run_nine_turbine_cases()  # 3

        time_end = time.time()
        logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


def main_plot():
    with stop_annotating():
        time_start = time.time()

        plot_three_turbine_cases()
        # plot_nine_turbine_cases()

        time_end = time.time()
        logger.info("Total time: {:.2f} seconds".format(time_end - time_start))


def run_one_turbine_cases():
    for yaw_offset in np.linspace(-30, 30, 7):
        conf.par.load("config/farmconners/fc.A4.1WT.yaml")
        conf.par.simulation.name = "fc.A4.1WT.Y{:03.0f}".format(yaw_offset)
        conf.par.wind_farm.yaw_angles += np.deg2rad(yaw_offset)
        run_case()


def run_three_turbine_cases():
    def run_3WT_case(yaw_offsets):
        conf.par.load("config/farmconners/fc.A4.3WT.yaml")
        conf.par.simulation.name += ".Y{:03.0f}_Y{:03.0f}_Y{:03.0f}".format(*yaw_offsets)
        conf.par.wind_farm.yaw_angles += np.deg2rad(yaw_offsets)
        run_case()

    for yaw_offsets in offsets_3wt[0:]:
        Process(target=run_3WT_case, args=[yaw_offsets]).start()


def run_nine_turbine_cases():
    def run_9WT_case(yaw_offsets):
        conf.par.load("config/farmconners/fc.A4.9WT.yaml")
        for row in range(3):
            connector = "." if row == 0 else "_"
            row_sign = "-" if yaw_offsets[row] < 0. else ""
            conf.par.simulation.name += \
                "{:s}Y{:s}{:1.0f}{:1.0f}{:1.0f}".format(
                    connector,
                    row_sign,
                    *[np.abs(y / 10.) for y in yaw_offsets[row::3]])
        conf.par.wind_farm.yaw_angles += np.deg2rad(yaw_offsets)
        run_case()

    for yaw_offsets in offsets_9wt[0:1]:
        Process(target=run_9WT_case, args=[yaw_offsets]).start()


def plot_flow_slice(frame):
    fig, ax = plt.subplots(1, 1, sharex='all', sharey='all')
    x, u, tri = read_vtk_unstructured("./results/" + conf.par.simulation.name + "/U{:06d}.vtu".format(frame))
    current_ax = ax
    # levels = np.linspace(np.min(u[:, 0]), np.max(u[:, 0]), 20)
    levels = np.linspace(3., 9, 21)

    a, contours = plot_contours(x, u, current_ax, levels=levels)
    plot_turbines(conf.par.wind_farm.positions,
                  conf.par.turbine.radius,
                  np.rad2deg(conf.par.wind_farm.yaw_angles),
                  current_ax)

    current_ax.set_xlabel(labels['x'])
    current_ax.set_ylabel(labels['y'])
    cb = plt.colorbar(contours, ax=current_ax)
    cb.set_label(labels["umag"])
    ax.set_title(conf.par.simulation.name)
    return fig, ax

def plot_three_turbine_cases():
    def plot_3WT_case(yaw_offsets):
        conf.par.load("config/farmconners/fc.A4.3WT.yaml")
        conf.par.simulation.name += ".Y{:03.0f}_Y{:03.0f}_Y{:03.0f}".format(*yaw_offsets)
        conf.par.wind_farm.yaw_angles += np.deg2rad(yaw_offsets)

        fig_dir = "./results/" + conf.par.simulation.name + "/figures/"
        os.makedirs(fig_dir, exist_ok=True)
        for frame in range(80):
            fig, ax = plot_flow_slice(frame)
            fig.savefig(fig_dir + "slice_{:03d}.png".format(frame), dpi=600, format="png")

    for yaw_offsets in offsets_3wt[0:]:
        Process(target=plot_3WT_case, args=[yaw_offsets]).start()

def plot_nine_turbine_cases():
    def plot_9WT_case(yaw_offsets):
        conf.par.load("config/farmconners/fc.A4.9WT.yaml")
        for row in range(3):
            connector = "." if row == 0 else "_"
            row_sign = "-" if yaw_offsets[row] < 0. else ""
            conf.par.simulation.name += \
                "{:s}Y{:s}{:1.0f}{:1.0f}{:1.0f}".format(
                    connector,
                    row_sign,
                    *[np.abs(y / 10.) for y in yaw_offsets[row::3]])
        conf.par.wind_farm.yaw_angles += np.deg2rad(yaw_offsets)

        fig_dir = "./results/" + conf.par.simulation.name + "/figures/"
        os.makedirs(fig_dir, exist_ok=True)
        for frame in range(80):
            fig, ax = plot_flow_slice(frame)
            fig.savefig(fig_dir + "slice_{:03d}.png".format(frame), dpi=600, format="png")

    for yaw_offsets in offsets_9wt[0:]:
        Process(target=plot_9WT_case, args=[yaw_offsets]).start()


def run_case():
    t0 = time.time()
    wind_farm = WindFarm()
    dfp = DynamicFlowProblem(wind_farm)
    dfs = DynamicFlowSolver(dfp)
    dfs.solve()
    t1 = time.time()
    logger.info("Case {:s} ran in {:.2f} s".format(conf.par.simulation.name, t1 - t0))


if __name__ == '__main__':
    # main()
    print(sys.argv[1])
    if len(sys.argv) > 1:
        if sys.argv[1] == "run":
            print("Starting run")
            main_run()
        elif sys.argv[1] == "plot":
            print("starting plot")
            main_plot()
        else:
            print("Choose either 'plot' or 'run' ")
    else:
        print('provide plot or run argument')
