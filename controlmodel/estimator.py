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
        self._dynamic_flow_solver = DynamicFlowSolver(self._dynamic_flow_problem, ssc=self)
        # self._time_last_optimised = -1.

        self._time_measured = []
        self._measurements = []




    def store_measurement(self, measurements):


    def store_controls(self, controls):


    def


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