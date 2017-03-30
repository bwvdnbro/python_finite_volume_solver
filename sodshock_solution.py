#! /usr/bin/python

################################################################################
# This file is part of python_finite_volume_solver
# Copyright (C) 2017 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
#
# python_finite_volume_solver is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# python_finite_volume_solver is distributed in the hope that it will be useful,
# but WITOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with python_finite_volume_solver. If not, see
# <http://www.gnu.org/licenses/>.
################################################################################

################################################################################
# @file sodshock_solution.py
#
# @brief Analytic solution for the 1D Sod shock problem.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
################################################################################

################################################################################
# This script can be used to obtain a reference solution for the 1D Sod shock
# problem.
#
# The Sod shock problem is in fact just a Riemann problem, which means the
# analytic solution is the same as for the corresponding Riemann problem,
# evaluated at some later time t.
# To obtain the reference solution, we solve the Riemann problem, and then set
# up an array of positions x. For a given time t, we then get the solution for
# each position by evaluating the Riemann problem solution at reference velocity
# dxdt = x / t.
#
# This script depends on the file riemannsolver.py, and will only work if this
# file is in the same folder as this script, or is in the system PATH.
#
# This file should not be included directly (if so, it just plots the solution
# of the 1D Sod shock problem at t = 0.2 and exits).
# Instead include it in another script as follows:
#  import sodshock_solution
# (make sure the folder containing the file is part of the system PATH).
#
# To get the Sod shock problem solution at a time t, run
#  xsol, rhosol, usol, psol = sodshock_solution.get_solution(t)
# The return values are numpy arrays containing respectively the positions,
# densities, fluid velocities and pressures.
################################################################################

# Import the Python numerical libraries, which we need for arange and array
import numpy as np

# Import the Riemann solver, which we need to... solve the Riemann problem.
import riemannsolver

################################################################################
# @brief Get the Sod shock problem solution at the given time.
#
# @param time Time for which we want to evaluate the solution.
# @return numpy arrays containing respectively the positions, densities, fluid
# velocities and pressures of the solution.
################################################################################
def get_solution(time):

  # create the Riemann solver instance
  solver = riemannsolver.RiemannSolver(5./3.)

  # set up the positions array
  x = np.arange(0., 1., 0.001)

  # sample the solution
  rho = np.zeros(len(x))
  u = np.zeros(len(x))
  p = np.zeros(len(x))
  for i in range(len(x)):
    rho[i], u[i], p[i], _ = solver.solve(1., 0., 1., 0.125, 0., 0.1,
                                         (x[i] - 0.5) / time)

  return x, rho, u, p

################################################################################
# @brief Default action when this file is run directly: plot the Sod shock
# problem solution at t = 0.2 and exit.
################################################################################
if __name__ == "__main__":
  print "\nThis script should not be run directly. Instead, import it into " \
        "another script and use its functionality there.\n" \
        "Now that we're running anyway, we will just show the Sod shock " \
        "problem solution at time t = 0.2.\n"

  x, rho, u, p = get_solution(0.2)

  # import matplotlib for plotting
  import pylab as pl

  pl.plot(x, rho, "r-")
  pl.show()
