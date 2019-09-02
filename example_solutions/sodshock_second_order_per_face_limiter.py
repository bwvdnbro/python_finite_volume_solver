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
# @file sodshock_second_order_per_face_limiter.py
#
# @brief Second order finite volume solution for the 1D Sod shock problem, using
# a per face slope limiter with variable tolerance parameters.
#
# This script is only intended as an example; many improvements are possible,
# and you should definitely try to write your own before looking at this file.
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

# parameters that determine how diffuse the per face slope limiter is
# physically reasonable ranges are 0 <= PHI1 <= 1, 0 <= PHI2 <= 0.5
PHI1 = 0.5
PHI2 = 0.25

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

    # NEW: fields to store the cell-wide gradients of the primitive variables
    self._gradient_density = 0.
    self._gradient_velocity = 0.
    self._gradient_pressure = 0.

    self._right_ngb = None
    # Note: the surface area is not really necessary in the 1D case
    self._surface_area = 1.

##
# @brief Apply a per face slope limiter to the given extrapolated quantity.
#
# @param phi_mid_0 Unlimited value of the extrapolated quantity.
# @param phi_i Value for the quantity in the active (left) cell.
# @param phi_j Value for the quantity in the neighbouring (right) cell.
# @param x_ij Position of the interface where the quantity is used.
# @param x_i Center of the active (left) cell.
# @param x_j Center of the neighbouring (right) cell.
# @return Slope limited extrapolated value.
##
def slope_limit(phi_mid_0, phi_i, phi_j, x_ij, x_i, x_j):
  phi_min = min(phi_i, phi_j)
  phi_max = max(phi_i, phi_j)
  delta1 = PHI1 * abs(phi_i - phi_j)
  delta2 = PHI2 * abs(phi_i - phi_j)
  phi_ij_bar = phi_i + abs(x_ij - x_i) / abs(x_j - x_i) * (phi_j - phi_i)
  if (phi_max + delta1) * phi_max > 0.:
    phi_plus = phi_max + delta1
  else:
    if not phi_max == 0.:
      phi_plus = phi_max / (1. + delta1 / abs(phi_max))
    else:
      phi_plus = 0.
  if (phi_min - delta1) * phi_min > 0.:
    phi_minus = phi_min - delta1
  else:
    if not phi_min == 0.:
      phi_minus = phi_min / (1. + delta1 / abs(phi_min))
    else:
      phi_minus = 0.

  if phi_i < phi_j:
    return max(phi_minus, min(phi_ij_bar + delta2, phi_mid_0))
  else:
    if phi_i > phi_j:
      return min(phi_plus, max(phi_ij_bar - delta2, phi_mid_0))
    else:
      return phi_i

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

  # set the neighbour of the previous cell (only if there is a previous cell
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

  # NEW: compute gradients for the primitive variables
  for icell in range(numcell):
    # get the primitive variables and midpoint for the current cell
    xcell = cells[icell]._midpoint
    densitycell = cells[icell]._density
    velocitycell = cells[icell]._velocity
    pressurecell = cells[icell]._pressure

    # get the primitive variables and midpoint for the cell to the left, if it
    # exists. If it does not exist, apply reflective boundary conditions and
    # create a ghost cell on the other side of the left face.
    xmin = -cells[icell]._midpoint
    densitymin = cells[icell]._density
    velocitymin = -cells[icell]._velocity
    pressuremin = cells[icell]._pressure
    if icell > 0:
      xmin = cells[icell - 1]._midpoint
      densitymin = cells[icell - 1]._density
      velocitymin = cells[icell - 1]._velocity
      pressuremin = cells[icell - 1]._pressure

    # get the primitive variables and midpoint for the cell to the right, if it
    # exists. If it does not exist, apply reflective boundary conditions and
    # create a ghost cell on the other side of the right face.
    xplu = 2. - cells[icell]._midpoint
    densityplu = cells[icell]._density
    velocityplu = -cells[icell]._velocity
    pressureplu = cells[icell]._pressure
    if icell < numcell - 1:
      xplu = cells[icell + 1]._midpoint
      densityplu = cells[icell + 1]._density
      velocityplu = cells[icell + 1]._velocity
      pressureplu = cells[icell + 1]._pressure

    # the gradient is the difference between the left and the right cell,
    # divided by the distance between left and right cell
    dx = xplu - xmin
    gradient_density = (densityplu - densitymin) / dx
    gradient_velocity = (velocityplu - velocitymin) / dx
    gradient_pressure = (pressureplu - pressuremin) / dx

    cells[icell]._gradient_density = gradient_density
    cells[icell]._gradient_velocity = gradient_velocity
    cells[icell]._gradient_pressure = gradient_pressure

  # solve the Riemann problem and do the flux exchanges
  for cell in cells:
    densityL = cell._density
    velocityL = cell._velocity
    pressureL = cell._pressure

    # NEW: get the gradients and position of the current cell
    xL = cell._midpoint
    gradient_densityL = cell._gradient_density
    gradient_velocityL = cell._gradient_velocity
    gradient_pressureL = cell._gradient_pressure

    cell_right = cell._right_ngb
    if not cell_right:
      # the last cell does not have a right neigbhour: impose reflective
      # boundary conditions
      densityR = densityL
      velocityR = -velocityL
      pressureR = pressureL
      xR = 2. - cell._midpoint
      gradient_densityR = -cell._gradient_density
      gradient_velocityR = cell._gradient_velocity
      gradient_pressureR = -cell._gradient_pressure
    else:
      densityR = cell_right._density
      velocityR = cell_right._velocity
      pressureR = cell_right._pressure
      xR = cell_right._midpoint
      gradient_densityR = cell_right._gradient_density
      gradient_velocityR = cell_right._gradient_velocity
      gradient_pressureR = cell_right._gradient_pressure

    # NEW: extrapolate the variables from the cell midpoint position to the
    # position of the face
    dx = 0.5 * (xR - xL)
    densityL_ext = densityL + dx * gradient_densityL
    velocityL_ext = velocityL + dx * gradient_velocityL
    pressureL_ext = pressureL + dx * gradient_pressureL
    densityR_ext = densityR - dx * gradient_densityR
    velocityR_ext = velocityR - dx * gradient_velocityR
    pressureR_ext = pressureR - dx * gradient_pressureR

    # apply the per face slope limiter
    densityL_ext = slope_limit(densityL_ext, densityL, densityR, xL + dx,
                               xL, xR)
    velocityL_ext = slope_limit(velocityL_ext, velocityL, velocityR, xL + dx,
                                xL, xR)
    pressureL_ext = slope_limit(pressureL_ext, pressureL, pressureR, xL + dx,
                                xL, xR)
    densityR_ext = slope_limit(densityR_ext, densityR, densityL, xL + dx,
                               xR, xL)
    velocityR_ext = slope_limit(velocityR_ext, velocityR, velocityL, xL + dx,
                                xR, xL)
    pressureR_ext = slope_limit(pressureR_ext, pressureR, pressureL, xL + dx,
                                xR, xL)

    # NEW: predict variables forward in time for half a time step
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

    # NEW: overwrite the left and right state with the extrapolated values
    densityL = densityL_ext
    velocityL = velocityL_ext
    pressureL = pressureL_ext
    densityR = densityR_ext
    velocityR = velocityR_ext
    pressureR = pressureR_ext

    # NEW: some sanity checks, to make sure we do not overshoot the
    # extrapolation
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
  xL = -cells[0]._midpoint
  gradient_densityL = -cells[0]._gradient_density
  gradient_velocityL = cells[0]._gradient_velocity
  gradient_pressureL = -cells[0]._gradient_pressure
  densityR = cells[0]._density
  velocityR = cells[0]._velocity
  pressureR = cells[0]._pressure
  xR = cells[0]._midpoint
  gradient_densityR = cells[0]._gradient_density
  gradient_velocityR = cells[0]._gradient_velocity
  gradient_pressureR = cells[0]._gradient_pressure
  # extrapolate the variables from the cell midpoint position to the position
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
pl.savefig("sodshock_second_order_per_face_limiter.png")
