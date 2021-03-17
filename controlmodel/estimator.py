from fenics import *
import controlmodel.conf as conf

if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
import os
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver

from tools.data import *
from tools.probes import *

import logging

logger = logging.getLogger("cm.estimator")


class Estimator:
    """
    State estimation

    """
    def __init__(self):
        self._estimator_type = conf.par.estimator.estimation_type
        if self._estimator_type != "offline":
            logger.error("Only `offline` estimation is implemented")

        self._assimilation_window = conf.par.estimator.assimilation_window
        self._transient_period = conf.par.estimator.transient_period
        self._predicition_period = conf.par.estimator.prediction_period

        self._cost_function_weights = conf.par.estimator.cost_function_weights

        self._wind_farm = WindFarm()
        self._dynamic_flow_problem = DynamicFlowProblem(self._wind_farm)
        self._dynamic_flow_solver = DynamicFlowSolver(self._dynamic_flow_problem)
        # self._time_last_optimised = -1.

        self._time_measured = []
        self._stored_measurements = {}
        self._stored_state = []
        self._stored_controls = {}
        self._stored_controls["time"] = np.zeros(self._assimilation_window)
        self._power_measured = []

        data_dir = conf.par.estimator.data["dir"]
        self._power_file = data_dir + conf.par.estimator.data["power"]
        self._yaw_file = data_dir + conf.par.estimator.data["yaw"]
        self._probe_file = data_dir + conf.par.estimator.data["probe"]


    def load_measurements(self):
        self._load_measurements_from_sowfa()

    def _load_measurements_from_sowfa(self):
        logger.info("Loading measurement data from SOWFA files")
        t, p, nt = read_power_sowfa(self._power_file)
        t, y, nt = read_power_sowfa(self._yaw_file)
        if len(self._wind_farm.get_turbines()) != nt:
            logger.error("Data has {:d} turbines but estimator is initialised with {:d}".format(nt, len(self._wind_farm.get_turbines())))
        logger.info("Loaded power and yaw data")

        logger.info("Resampling data to simulation time step")
        # todo: convert below code to work in class
        time_vec = np.arange(0, t[-1], conf.par.simulation.time_step)
        self._stored_measurements["time"] = time_vec
        num_measurements = len(time_vec)
        self._stored_measurements["power"] = np.zeros((num_measurements,nt))
        self._stored_measurements["yaw"] = np.zeros((num_measurements, nt))
        for idx in range(nt):
            self._stored_measurements["power"][:, idx] = np.interp(time_vec, t, p[:, idx])
            self._stored_measurements["yaw"][:, idx] = np.interp(time_vec, t, y[:, idx])
        self._stored_measurements["yaw"] = np.deg2rad(self._stored_measurements["yaw"])
        logger.info("Loaded nacelle yaw measurements in degrees and stored in radians")

        probe_positions, t, probe_data = read_probe_data(self._probe_file)
        # # probe_measurement_points = []
        #
        probe_data = probe_data[t % 1 <= 0.01, :, 0:2]
        # todo: move this to flow problem?
        cells = conf.par.wind_farm.cells
        measurement_mesh = RectangleMesh(Point([0., 0.]), Point(conf.par.wind_farm.size),  cells[0],cells[1], diagonal='left/right')
        V_m = VectorElement("CG", measurement_mesh.ufl_cell(), 1)
        measurement_function_space = FunctionSpace(measurement_mesh, V_m)
        coords = measurement_function_space.sub(0).collapse().tabulate_dof_coordinates()
        points = probe_positions
        indices = []
        for coord in coords:
            # print(point)
            idx = int(np.logical_and(points[:, 0] == coord[0], points[:, 1] == coord[1]).nonzero()[0])
            indices.append(idx)

        velocity_measurements = []
        for idx in range(len(self._stored_measurements["power"])):
            velocity_measurements += [Function(measurement_function_space)]

        for n in range(len(velocity_measurements) - 1):
            velocity_measurements[n + 1].vector()[:] = probe_data[n + 1, indices, :].ravel()
        velocity_measurements[0].assign(velocity_measurements[1])

        self._stored_measurements["probes"] = velocity_measurements

    def run_transient(self):
        logger.info("Running transient part of simulation over {:.0f}s".format(conf.par.estimator.transient_period))
        with stop_annotating():
            transient_time = conf.par.estimator.transient_period
            self._dynamic_flow_solver.solve_segment(transient_time)


    def store_checkpoint(self, simulation_time, checkpoint):
        self._stored_state.append(checkpoint)
        print("Storing state for t={:.2f}".format(simulation_time))

    def store_measurement(self, simulation_time, measurements):
        self._stored_measurements.append(measurements.copy())
        self._time_measured.append(simulation_time)
        # storing measured power as numpy array does not work
        self._power_measured.append(list(measurements["generatorPower"]))

    def store_controls(self, simulation_time, controls):
        # for c in controls.values():
        #     print(c.get_reference())

        self._stored_controls["time"][:-1] = self._stored_controls["time"][1:]
        self._stored_controls["time"][-1] = simulation_time
        for c in controls:
            if c not in self._stored_controls:
                self._stored_controls[c] = np.zeros((self._assimilation_window, len(controls[c].get_reference())))
            self._stored_controls[c][:-1, :] = self._stored_controls[c][1:,:]
            self._stored_controls[c][-1,:] = controls[c].get_reference()
            # self._stored_controls[c].append(controls[c].get_reference())
            # print(self._stored_controls[c])
        # print(controls)


    def do_estimation(self):
        self.run_forward_simulation()
        self.construct_functional()
        # self.compute_gradient()

    def run_forward_simulation(self):
        # with stored controls
        start_time = self._time_measured[-1-self._assimilation_window]
        for control in self._stored_controls:
            if control != "time":
                self._wind_farm.set_control_reference_series(name=control,
                                                     time_series=self._stored_controls["time"],
                                                     reference_series=self._stored_controls[control])
        self._dynamic_flow_solver.set_checkpoint(self._stored_state[-1-self._assimilation_window])
        self._dynamic_flow_solver.solve_segment(self._time_measured[-1]-start_time)
        # self._dynamic_flow_solver.reset_checkpoint()
        # self._dynamic_flow_solver.solve_segment(conf.par.ssc.control_horizon)

    def construct_functional(self):
        # todo: make private
        # todo: proper cost function
        # todo: weights from config
        # get stored power
        power_stored = self._power_measured[-self._assimilation_window:]
        # get modelled power
        power_modelled =self._dynamic_flow_solver.get_power_functional_list()
        print("power stored")
        print(power_stored)
        print("power_modelled")
        print(power_modelled)
        # print(" I need to calculate J without numpy...
        # difference = [(p_s-p_m)*(p_s-p_m) for p_s, p_m in zip(power_stored,power_modelled)]
        J = None
        # for idx in range(2): #todo: len(turbines)
        #     difference = [(p_m[idx]-p_s[idx])*(p_m[idx]-p_s[idx])*1e-6*1e-6 for p_s, p_m in zip(power_stored,power_modelled)]
        #     squared_difference = sum(difference)
        #     if J is None:
        #         J = squared_difference
        #     else:
        #         J+=squared_difference
        J = power_modelled[0][0]  #- power_stored[0][0]

        tp = [sum(x) *1e-6for x in power_modelled]
        tps = [sum(x)*1e-6 for x in power_stored]
        dsq = [(p-pr) * (p-pr) for p, pr in zip(tp, tps)]
        J = sum(dsq)
        # print(type(difference[0]))
        # print(type(squared_difference))


        # sum([sum(p_s - p_m) for p_s, p_m in zip(power_stored, power_modelled)])
            # (power_stored - power_modelled).ravel()
        # J = squared_difference
        print("Cost function value: {:.2f}".format(J))
        # print(float(J))

        t,up1,up2 = self._stored_state[-1-self._assimilation_window]
        print(type(up1))
        # print(up2)
        # clist = self._wind_farm.get_controls_list("yaw")
        m = [Control(c) for c in [up1]]
        # m = [Control(c[0]) for c in clist]
        compute_gradient(J,m)

        # J  = (p-pr)*W*(p-pr)

    # def assign_adjoint_controls(self):
    #
    # def compute_gradient(self, functional, adjoint_controls):
    #
    # def update_state_with_gradient(self):
    #
    #
    # def get_state(self):
        # return up, up_prev

        # def _do_state_estimation(self, simulation_time):
        #     assimilation_window = 20
        #
        #     if simulation_time > conf.par.ssc.transient_time:
        #         a = 1
        #         # add measurements to history
        #
        #         # forward simulate to current time
        #
        #         # construct functional
        #         measured_power =
        #         modelled_power
        #
        #         functional =
        #         # calculate gradients
        #
        #         # update state