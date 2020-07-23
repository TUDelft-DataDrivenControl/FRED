
Introduction
============

* Wind farm control for integrative approach to optimisation of power production.
* Power maximisation over time and tracking of a specified power reference.

Goals
-----
* Flexibility in control-oriented wind farm modelling.
* Sufficient speed for real-time control applications.


Notes
-----
Explain differences between two and three dimensional implementation.


* Installation
* demos / unit tests
* folder structure explanation.
* running on HPC
* ZMQ communication with SOWFA
*

State of the project
--------------------
Two-dimensional flow model with adjoint solutions for gradient-based optimisation.
Currently implemented is gradient-step control iterating over time to produce optimal yaw and axial induction control signals.

Three-dimensional flow modelling has been tested to be functional, but the code is currently not suited for control.


link to `FEniCS <https://fenicsproject.org/>`_  and `dolfin-adjoint <http://www.dolfin-adjoint.org/>`_ project pages
ZeroMQ

If you use this project, please cite ... TORQUE2020 paper