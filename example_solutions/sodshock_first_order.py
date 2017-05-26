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
# @file sodshock_first_order.py
#
# @brief First order finite volume solution for the 1D Sod shock problem.
#
# This script is only intended as an example; many improvements are possible,
# and you should definitely try to write your own before looking at this file.
#
# @author Bert Vandenbroucke (b7@st-andrews.ac.uk)
################################################################################

# Import the Riemann solver library. Note that this will only work if the file
# 'riemannsolver.py' is in the same directory as this script.
import riemannsolver as riemann
# Import the Python numerical libraries, as we need them for arange.
import numpy as np
# Import Matplotlib, which we will use to plot the results.
import pylab as pl

################################################################################
# some global definitions
################################################################################

# the constant adiabatic index
GAMMA = 5./3.

# the Riemann solver
solver = riemann.RiemannSolver(GAMMA)

# the constant time step
timestep = 0.001

# number of steps
numstep = 200

# number of cells
numcell = 100

################################################################################
# the actual program
################################################################################

##
# @ brief The Cell class.
##
class Cell:
  ##
  # @brief Constructor.
  #
  # This method initializes some variables. This is not really necessary for
  # most of them, but it provides a nice overview of which variables are
  # actually part of the class (because Python does not have nice and clear
  # class definitions like C++).
  ##
  def __init__(self):
    self._midpoint = 0.
    self._volume = 0.
    self._mass = 0.
    self._momentum = 0.
    self._energy = 0.
    self._density = 0.
    self._velocity = 0.
    self._pressure = 0.

    self._right_ngb = None
    # Note: the surface area is not really necessary in the 1D case
    self._surface_area = 1.

# set up the cells
cells = []
for i in range(numcell):

  cell = Cell()
  cell._midpoint = (i + 0.5) / numcell
  cell._volume = 1. / numcell
  if i < numcell/2:
    cell._mass = cell._volume
    cell._energy = cell._volume / (GAMMA - 1.)
  else:
    cell._mass = 0.125 * cell._volume
    cell._energy = 0.1 * cell._volume / (GAMMA - 1.)
  cell._momentum = 0.

  # set the neighbour of the previous cell (only if there is a previous cell)
  if len(cells) > 0:
    cells[-1]._right_ngb = cell

  cells.append(cell)

# do the actual time integration loop
for i in range(numstep):

  # convert conserved into primitive variables
  for cell in cells:
    volume = cell._volume
    mass = cell._mass
    momentum = cell._momentum
    energy = cell._energy

    density = mass / volume
    velocity = momentum / mass
    pressure = (GAMMA - 1.) * (energy / volume - 0.5 * density * velocity**2)

    cell._density = density
    cell._velocity = velocity
    cell._pressure = pressure

  # solve the Riemann problem and do the flux exchanges
  for cell in cells:
    densityL = cell._density
    velocityL = cell._velocity
    pressureL = cell._pressure

    cell_right = cell._right_ngb
    if not cell_right:
      # the last cell does not have a right neighbour: impose reflective
      # boundary conditions
      densityR = densityL
      velocityR = -velocityR
      pressureR = pressureL
    else:
      densityR = cell_right._density
      velocityR = cell_right._velocity
      pressureR = cell_right._pressure

    # now feed everything to the Riemann solver (we ignore the last return
    # value)
    densitysol, velocitysol, pressuresol, _ = \
      solver.solve(densityL, velocityL, pressureL,
                   densityR, velocityR, pressureR)

    # get the fluxes
    flux_mass = densitysol * velocitysol
    flux_momentum = densitysol * velocitysol**2 + pressuresol
    flux_energy = (pressuresol * GAMMA / (GAMMA - 1.) + \
                   0.5 * densitysol * velocitysol**2) * velocitysol

    # do the flux exchange
    A = cell._surface_area

    cell._mass -= flux_mass * A * timestep
    cell._momentum -= flux_momentum * A * timestep
    cell._energy -= flux_energy * A * timestep

    if cell_right:
      cell_right._mass += flux_mass * A * timestep
      cell_right._momentum += flux_momentum * A * timestep
      cell_right._energy += flux_energy * A * timestep

  # we need to do something special for the left boundary of the first cell
  # (we will just impose reflective boundary conditions)
  densityL = cells[0]._density
  velocityL = -cells[0]._velocity
  pressureL = cells[0]._pressure
  densityR = cells[0]._density
  velocityR = cells[0]._velocity
  pressureR = cells[0]._pressure
  # call the Riemann solver
  densitysol, velocitysol, pressuresol, _ = \
    solver.solve(densityL, velocityL, pressureL, densityR, velocityR, pressureR)
  # get the fluxes
  flux_mass = densitysol * velocitysol
  flux_momentum = densitysol * velocitysol**2 + pressuresol
  flux_energy = (pressuresol * GAMMA / (GAMMA - 1.) + \
                 0.5 * densitysol * velocitysol**2) * velocitysol
  # do the flux exchange
  A = cells[0]._surface_area
  cells[0]._mass += flux_mass * A * timestep
  cells[0]._momentum += flux_momentum * A * timestep
  cells[0]._energy += flux_energy * A * timestep

# reference solution: as the Sod shock problem is in fact a Riemann problem,
# this is just the actual solution of the Riemann problem, evaluated at the
# final time of the simulation.
xref = np.arange(0., 1., 0.001)
rhoref = [solver.solve(1., 0., 1., 0.125, 0., 0.1,
                       (x - 0.5) / (timestep * numstep))[0] \
          for x in xref]

# plot the reference solution and the actual solution
pl.plot(xref, rhoref, "r-")
pl.plot([cell._midpoint for cell in cells],
        [cell._density for cell in cells],
        "k.")
pl.ylim(0., 1.1)
pl.xlabel("Position")
pl.ylabel("Density")
pl.tight_layout()
# save the plot as a PNG image
pl.savefig("sodshock_first_order.png")
