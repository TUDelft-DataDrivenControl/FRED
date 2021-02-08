from fenics import *
import controlmodel.conf as conf

if conf.with_adjoint:
    from fenics_adjoint import *
import numpy as np
import os
from controlmodel.windfarm import WindFarm
from controlmodel.flowproblem import DynamicFlowProblem
from controlmodel.flowsolver import DynamicFlowSolver

import logging

logger = logging.getLogger("cm.estimator")


class Estimator:
    """
    State estimation

    """
    def __init__(self):
        self._assimilation_window = 20
        # self._

        self._wind_farm = WindFarm()
        self._dynamic_flow_problem = DynamicFlowProblem(self._wind_farm)
        self._dynamic_flow_solver = DynamicFlowSolver(self._dynamic_flow_problem)
        # self._time_last_optimised = -1.

        self._time_measured = []
        self._stored_measurements = []
        self._stored_state = []
        self._stored_controls = {}
        self._stored_controls["time"] = np.zeros(self._assimilation_window)
        self._power_measured = []

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