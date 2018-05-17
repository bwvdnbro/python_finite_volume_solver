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
# @file sodshock_lagrangian.py
#
# @brief Second order Lagrangian finite volume solutio for the 1D Sod shock
# problem.
#
# This script is only intended as an example; many improvements are possible,
# and you should definitely try to write your own before looking at this file.
#
# Note also that Lagrangian schemes work best if every cell has a similar mass
# resolution, which is not the case in the specific setup used here. As a result
# the overall accuracy of this particular solution is not really better than the
# corresponding Eulerian version.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
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
    self._centroid = 0.
    self._volume = 0.
    self._mass = 0.
    self._momentum = 0.
    self._energy = 0.
    self._density = 0.
    self._velocity = 0.
    self._pressure = 0.

    self._gradient_density = 0.
    self._gradient_velocity = 0.
    self._gradient_pressure = 0.

    self._right_ngb = None
    # Note: the surface area is not really necessary in the 1D case
    self._surface_area = 1.

    # NEW: velocity with which the cell itself moves
    self._cell_velocity = 0.

    # velocities of the faces
    self._left_velocity = 0.
    self._right_velocity = 0.
    self._left_face = 0.
    self._right_face = 0.

# set up the cells
cells = []
for i in range(numcell):

  cell = Cell()
  cell._left_face = (i + 0.) / numcell
  cell._right_face = (i + 1.) / numcell
  cell._volume = 1. / numcell
  if i < numcell/2:
    cell._mass = cell._volume
    cell._energy = cell._volume / (GAMMA - 1.)
  else:
    cell._mass = 0.125 * cell._volume
    cell._energy = 0.1 * cell._volume / (GAMMA - 1.)
  cell._momentum = 0.

  # set the neighbour of the previous cell (only if there is a previous cell
  if len(cells) > 0:
    cells[-1]._right_ngb = cell

  cells.append(cell)

# do the actual time integration loop
for i in range(numstep):

  # NEW: move the cells
  for cell in cells:
    cell._left_face += timestep * cell._left_velocity
    cell._right_face += timestep * cell._right_velocity

  # NEW: recompute cell volumes
  for icell in range(numcell):
    xmin = cells[icell]._left_face
    xplu = cells[icell]._right_face
    cells[icell]._volume = (xplu - xmin)
    cells[icell]._centroid = 0.5 * (xmin + xplu)

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

  # reset cell velocities
  for cell in cells:
    cell._cell_velocity = 0.5 * (cell._left_velocity + cell._right_velocity)

  # compute gradients for the primitive variables
  for icell in range(numcell):
    xcell = cells[icell]._centroid
    densitycell = cells[icell]._density
    velocitycell = cells[icell]._velocity
    pressurecell = cells[icell]._pressure

    xmin = -cells[icell]._centroid
    densitymin = cells[icell]._density
    velocitymin = -cells[icell]._velocity
    pressuremin = cells[icell]._pressure
    if icell > 0:
      xmin = cells[icell - 1]._centroid
      densitymin = cells[icell - 1]._density
      velocitymin = cells[icell - 1]._velocity
      pressuremin = cells[icell - 1]._pressure

    xplu = 2. - cells[icell]._centroid
    densityplu = cells[icell]._density
    velocityplu = -cells[icell]._velocity
    pressureplu = cells[icell]._pressure
    if icell < numcell - 1:
      xplu = cells[icell + 1]._centroid
      densityplu = cells[icell + 1]._density
      velocityplu = cells[icell + 1]._velocity
      pressureplu = cells[icell + 1]._pressure

    dxplu = xplu - xcell
    gradient_density_plu = (densityplu - densitycell) / dxplu
    gradient_velocity_plu = (velocityplu - velocitycell) / dxplu
    gradient_pressure_plu = (pressureplu - pressurecell) / dxplu

    dxmin = xcell - xmin
    gradient_density_min = (densitycell - densitymin) / dxmin
    gradient_velocity_min = (velocitycell - velocitymin) / dxmin
    gradient_pressure_min = (pressurecell - pressuremin) / dxmin

    if abs(gradient_density_min) < abs(gradient_density_plu):
      cells[icell]._gradient_density = gradient_density_min
    else:
      cells[icell]._gradient_density = gradient_density_plu

    if abs(gradient_velocity_min) < abs(gradient_velocity_plu):
      cells[icell]._gradient_velocity = gradient_velocity_min
    else:
      cells[icell]._gradient_velocity = gradient_velocity_plu

    if abs(gradient_pressure_min) < abs(gradient_pressure_plu):
      cells[icell]._gradient_pressure = gradient_pressure_min
    else:
      cells[icell]._gradient_pressure = gradient_pressure_plu

  # solve the Riemann problem and do the flux exchanges
  for cell in cells:
    densityL = cell._density
    velocityL = cell._velocity
    pressureL = cell._pressure
    xL = cell._centroid
    gradient_densityL = cell._gradient_density
    gradient_velocityL = cell._gradient_velocity
    gradient_pressureL = cell._gradient_pressure
    # NEW: get the cell velocity
    cell_velocityL = cell._cell_velocity

    cell_right = cell._right_ngb
    if not cell_right:
      # the last cell does not have a right neigbhour: impose reflective
      # boundary conditions
      densityR = densityL
      velocityR = -velocityR
      pressureR = pressureL
      xR = 2. - cell._centroid
      gradient_densityR = -cell._gradient_density
      gradient_velocityR = cell._gradient_velocity
      gradient_pressureR = -cell._gradient_pressure
      cell_velocityR = -cell._cell_velocity
    else:
      densityR = cell_right._density
      velocityR = cell_right._velocity
      pressureR = cell_right._pressure
      xR = cell_right._centroid
      gradient_densityR = cell_right._gradient_density
      gradient_velocityR = cell_right._gradient_velocity
      gradient_pressureR = cell_right._gradient_pressure
      cell_velocityR = cell_right._cell_velocity

    # extrapolate the variables from the cell centroid position to the position
    # of the face
    dx = 0.5 * (xR - xL)
    densityL_ext = densityL + dx * gradient_densityL
    velocityL_ext = velocityL + dx * gradient_velocityL
    pressureL_ext = pressureL + dx * gradient_pressureL
    densityR_ext = densityR - dx * gradient_densityR
    velocityR_ext = velocityR - dx * gradient_velocityR
    pressureR_ext = pressureR - dx * gradient_pressureR

    # predict variables forward in time for half a time step
    densityL_ext -= 0.5 * timestep * (densityL * gradient_velocityL + \
                                      velocityL * gradient_densityL)
    velocityL_ext -= 0.5 * timestep * (velocityL * gradient_velocityL + \
                                       gradient_pressureL / densityL)
    pressureL_ext -= 0.5 * timestep * (velocityL * gradient_pressureL + \
                                       GAMMA * pressureL * gradient_velocityL)
    densityR_ext -= 0.5 * timestep * (densityR * gradient_velocityR + \
                                      velocityR * gradient_densityR)
    velocityR_ext -= 0.5 * timestep * (velocityR * gradient_velocityR + \
                                       gradient_pressureR / densityR)
    pressureR_ext -= 0.5 * timestep * (velocityR * gradient_pressureR + \
                                       GAMMA * pressureR * gradient_velocityR)

    # overwrite the left and right state with the extrapolated values
    densityL = densityL_ext
    velocityL = velocityL_ext
    pressureL = pressureL_ext
    densityR = densityR_ext
    velocityR = velocityR_ext
    pressureR = pressureR_ext

    if densityL < 0. or pressureL < 0.:
      print "Negative density or pressure!"
      print "Density:", densityL
      print "Pressure:", pressureL
      exit()
    if densityR < 0. or pressureR < 0.:
      print "Negative density or pressure!"
      print "Density:", densityR
      print "Pressure:", pressureR
      exit()

    # NEW: boost to a frame moving with the face velocity
#    vface = 0.5 * (cell_velocityL + cell_velocityR)
#    velocityL -= vface
#    velocityR -= vface

    # now feed everything to the Riemann solver (we ignore the last return
    # value)
    velocitysol, pressuresol = solver.solve_middle_state(
      densityL, velocityL, pressureL, densityR, velocityR, pressureR)

    # NEW: deboost back to the lab frame
#    velocitysol += vface
    densitysol = 0.
    vface = velocitysol
    cell._right_velocity = vface

    # get the fluxes
    # NEW: correction terms due to the movement of the face
    flux_mass = densitysol * (velocitysol - vface)
    flux_momentum = densitysol * velocitysol * (velocitysol - vface) + \
                    pressuresol
    flux_energy = pressuresol * velocitysol + \
                  (pressuresol / (GAMMA - 1.) + \
                   0.5 * densitysol * velocitysol**2) * (velocitysol - vface)

    # do the flux exchange
    A = cell._surface_area

    cell._mass -= flux_mass * A * timestep
    cell._momentum -= flux_momentum * A * timestep
    cell._energy -= flux_energy * A * timestep

    if cell_right:
      cell_right._left_velocity = vface
      cell_right._mass += flux_mass * A * timestep
      cell_right._momentum += flux_momentum * A * timestep
      cell_right._energy += flux_energy * A * timestep

  # we need to do something special for the left boundary of the first cell
  # (we will just impose reflective boundary conditions)
  # note that this also means that the face does not move, so no need to do
  # anything different here
  densityL = cells[0]._density
  velocityL = -cells[0]._velocity
  pressureL = cells[0]._pressure
  xL = -cells[0]._centroid
  gradient_densityL = -cells[0]._gradient_density
  gradient_velocityL = cells[0]._gradient_velocity
  gradient_pressureL = -cells[0]._gradient_pressure
  densityR = cells[0]._density
  velocityR = cells[0]._velocity
  pressureR = cells[0]._pressure
  xR = cells[0]._centroid
  gradient_densityR = cells[0]._gradient_density
  gradient_velocityR = cells[0]._gradient_velocity
  gradient_pressureR = cells[0]._gradient_pressure
  # extrapolate the variables from the cell centroid position to the position
  # of the face
  dx = 0.5 * (xR - xL)
  densityL_ext = densityL + dx * gradient_densityL
  velocityL_ext = velocityL + dx * gradient_velocityL
  pressureL_ext = pressureL + dx * gradient_pressureL
  densityR_ext = densityR - dx * gradient_densityR
  velocityR_ext = velocityR - dx * gradient_velocityR
  pressureR_ext = pressureR - dx * gradient_pressureR
  # predict variables forward in time for half a time step
  densityL_ext -= 0.5 * timestep * (densityL * gradient_velocityL + \
                                    velocityL * gradient_densityL)
  velocityL_ext -= 0.5 * timestep * (velocityL * gradient_velocityL + \
                                     gradient_pressureL / densityL)
  pressureL_ext -= 0.5 * timestep * (velocityL * gradient_pressureL + \
                                     GAMMA * pressureL * gradient_velocityL)
  densityR_ext -= 0.5 * timestep * (densityR * gradient_velocityR + \
                                    velocityR * gradient_densityR)
  velocityR_ext -= 0.5 * timestep * (velocityR * gradient_velocityR + \
                                     gradient_pressureR / densityR)
  pressureR_ext -= 0.5 * timestep * (velocityR * gradient_pressureR + \
                                     GAMMA * pressureR * gradient_velocityR)
  # overwrite the left and right state with the extrapolated values
  densityL = densityL_ext
  velocityL = velocityL_ext
  pressureL = pressureL_ext
  densityR = densityR_ext
  velocityR = velocityR_ext
  pressureR = pressureR_ext
  # call the Riemann solver
  velocitysol, pressuresol = solver.solve_middle_state(
    densityL, velocityL, pressureL, densityR, velocityR, pressureR)
  densitysol = 0.
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

fig, ax = pl.subplots(1, 2)
# plot the reference solution and the actual solution
ax[0].plot(xref, rhoref, "r-")
ax[0].plot([cell._centroid for cell in cells],
           [cell._density for cell in cells],
           "k.")
ax[0].set_ylim(0., 1.1)
ax[0].set_xlabel("Position")
ax[0].set_ylabel("Density")

ax[1].plot([cell._centroid for cell in cells],
           [cell._mass for cell in cells],
           "k.")
ax[1].set_xlabel("Position")
ax[1].set_ylabel("Mass")
pl.tight_layout()
# save the plot as a PNG image
pl.savefig("sodshock_mmc.png")
