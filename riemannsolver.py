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
# @file riemannsolver.py
#
# @brief Standalone Riemann solver library.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
################################################################################

################################################################################
# This is a standalone Riemann solver, based on the exact Riemann solver that is
# part of Shadowfax (https://github.com/AstroUGent/shadowfax), CMacIonize
# (https://github.com/bwvdnbro/CMacIonize) and SWIFT
# (https://gitlab.cosma.dur.ac.uk/swift/swiftsim).
#
# It is based on the description given in Toro, E., Riemann Solvers and
# Numerical Methods for Fluid Dynamics, 3rd edition (Springer, 2009).
#
# This file should not be run directly (it just runs some unit tests if this is
# done). Instead, include it in another script as follows:
#  import riemannsolver
# (make sure the folder containing the file is part of the system PATH).
#
# Once included, you can create a RiemannSolver object for an ideal gas with a
# given polytropic index (e.g. 5/3):
#  solver = riemannsolver.RiemannSolver(5./3.)
# This object has a member function solve which can be used to solve the Riemann
# problem with a given left and right state:
#  rhosol, usol, psol, flag = solver.solver(rhoL, uL, pL, rhoR, uR, pR)
# rhosol, rhoL, and rhoR are the densities of respectively the solution, the
# left state, and the right state.
# usol, uL, and uR are the fluid velocities
# psol, pL, and pR are the pressures
# flag is an extra return variable that is set to -1, 1, or 0 if respectively
# the left state, the right state, or a vacuum state was sampled.
#
# By default, the Riemann solution is sampled for a reference velocity dxdt = 0.
# Alternatively, you can sample the solution for another reference velocity
# dxdt = x / t by providing dxdt as an extra argument to solve().
################################################################################

# Import the Python numerical libraries, which we need for sqrt
import numpy as np

##
# @brief Exact Riemann solver.
##
class RiemannSolver:

  ##############################################################################
  # @brief Constructor.
  #
  # @param gamma Adiabatic index \f$\gamma{}\f$.
  ##############################################################################
  def __init__(self, gamma):

    if gamma <= 1.:
      print "The adiabatic index needs to be larger than 1!"
      exit()

    self._gamma = gamma

    ## related quantities:
    # gamma plus 1 divided by 2 gamma
    self._gp1d2g = 0.5 * (gamma + 1.) / gamma
    # gamma minus 1 divided by 2 gamma
    self._gm1d2g = 0.5 * (gamma - 1.) / gamma
    # gamma minus 1 divided by gamma plus 1
    self._gm1dgp1 = (gamma - 1.) / (gamma + 1.)
    # two divided by gamma plus 1
    self._tdgp1 = 2. / (gamma + 1.)
    # two divided by gamma minus 1
    self._tdgm1 = 2. / (gamma - 1.)
    # gamma minus 1 divided by 2
    self._gm1d2 = 0.5 * (gamma - 1.)
    # two times gamma divided by gamma minus 1
    self._tgdgm1 = 2. * gamma / (gamma - 1.)
    # gamma inverse
    self._ginv = 1. / gamma

  ##############################################################################
  # @brief Get the soundspeed corresponding to the given density and pressure.
  #
  # @param rho Density value.
  # @param P Pressure value.
  # @return Soundspeed.
  ##############################################################################
  def get_soundspeed(self, rho, P):
    return np.sqrt(self._gamma * P / rho)

  ##############################################################################
  # @brief Riemann fL or fR function.
  #
  # @param rho Density of the left or right state.
  # @param P Pressure of the left or right state.
  # @param a Soundspeed of the left or right state.
  # @param Pstar (Temporary) pressure of the middle state.
  # @return Value of the fL or fR function.
  ##############################################################################
  def fb(self, rho, P, a, Pstar):
    if Pstar > P:
      A = self._tdgp1 / rho
      B = self._gm1dgp1 * P;
      fval = (Pstar - P) * np.sqrt(A / (Pstar + B))
    else:
      fval = self._tdgm1 * a * ((Pstar / P)**(self._gm1d2g) - 1.)

    return fval

  ##############################################################################
  # @brief Riemann f function.
  #
  # @param rhoL Density of the left state.
  # @param uL Velocity of the left state.
  # @param PL Pressure of the left state.
  # @param aL Soundspeed of the left state.
  # @param rhoR Density of the right state.
  # @param uR Velocity of the right state.
  # @param PR Pressure of the right state.
  # @param aR Soundspeed of the right state.
  # @param Pstar (Temporary) pressure of the middle state.
  # @return Value of the Riemann f function.
  ##############################################################################
  def f(self, rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pstar):
    return self.fb(rhoL, PL, aL, Pstar) + self.fb(rhoR, PR, aR, Pstar) + \
           (uR - uL)

  ##############################################################################
  # @brief Derivative of the Riemann fL or fR function.
  #
  # @param rho Density of the left or right state.
  # @param P Pressure of the left or right state.
  # @param a Soundspeed of the left or right state.
  # @param Pstar (Temporary) pressure of the middle state.
  # @return Value of the derivative of the Riemann fL or fR function.
  ##############################################################################
  def fprimeb(self, rho, P, a, Pstar):
    if Pstar > P:
      A = self._tdgp1 / rho
      B = self._gm1dgp1 * P
      fval = (1. - 0.5 * (Pstar - P) / (B + Pstar)) * np.sqrt(A / (Pstar + B))
    else:
      fval = 1. / (rho * a) * (Pstar / P)**(-self._gp1d2g)

    return fval

  ##############################################################################
  # @brief Derivative of the Riemann f function.
  #
  # @param rhoL Density of the left state.
  # @param PL Pressure of the left state.
  # @param aL Soundspeed of the left state.
  # @param rhoR Density of the right state.
  # @param PR Pressure of the right state.
  # @param aR Soundspeed of the right state.
  # @param Pstar (Temporary) pressure of the middle state.
  # @return Value of the derivative of the Riemann f function.
  ##############################################################################
  def fprime(self, rhoL, PL, aL, rhoR, PR, aR, Pstar):
    return self.fprimeb(rhoL, PL, aL, Pstar) + self.fprimeb(rhoR, PR, aR, Pstar)

  ##############################################################################
  # @brief Riemann gL or gR function.
  #
  # @param rho Density of the left or right state.
  # @param P Pressure of the left or right state.
  # @param Pstar (Temporary) pressure in the middle state.
  # @return Value of the gL or gR function.
  ##############################################################################
  def gb(self, rho, P, Pstar):
    A = self._tdgp1 / rho
    B = self._gm1dgp1 * P
    return np.sqrt(A / (Pstar + B))

  ##############################################################################
  # @brief Get an initial guess for the pressure in the middle state.
  #
  # @param rhoL Left state density.
  # @param uL Left state velocity.
  # @param PL Left state pressure.
  # @param aL Left state soundspeed.
  # @param rhoR Right state density.
  # @param uR Right state velocity.
  # @param PR Right state pressure.
  # @param aR Right state soundspeed.
  # @return Initial guess for the pressure in the middle state.
  ##############################################################################
  def guess_P(self, rhoL, uL, PL, aL, rhoR, uR, PR, aR):
    Pmin = min(PL, PR)
    Pmax = max(PL, PR)
    qmax = Pmax / Pmin
    Ppv = 0.5 * (PL + PR) - 0.125 * (uR - uL) * (PL + PR) * (aL + aR)
    Ppv = max(5.e-9 * (PL + PR), Ppv)
    if qmax <= 2 and Pmin <= Ppv and Ppv <= Pmax:
      Pguess = Ppv
    else:
      if Ppv < Pmin:
        # two rarefactions
        Pguess = ( (aL + aR - self._gm1d2 * (uR - uL)) / \
                   (aL / (PL**self._gm1d2g) + aR / (PR**self._gm1d2g) ) \
                 )**self._tgdgm1
      else:
        # two shocks
        gL = self.gb(rhoL, PL, Ppv)
        gR = self.gb(rhoR, PR, Ppv)
        Pguess = (gL * PL + gR * PR - uR + uL) / (gL + gR)

    # Toro: "Not that approximate solutions may predict, incorrectly, a
    # negative value for pressure (...). Thus in order to avoid negative guess
    # values we introduce the small positive constant _tolerance"
    # (tolerance is 1.e-8 in this case)
    Pguess = max(5.e-9 * (PL + PR), Pguess)
    return Pguess

  ##############################################################################
  # @brief Find the pressure of the middle state by using Brent's method.
  #
  # @param rhoL Density of the left state.
  # @param uL Velocity of the left state.
  # @param PL Pressure of the left state.
  # @param aL Soundspeed of the left state.
  # @param rhoR Density of the right state.
  # @param uR Velocity of the right state.
  # @param PR Pressure of the right state.
  # @param aR Soundspeed of the right state.
  # @param Plow Lower bound guess for the pressure of the middle state.
  # @param Phigh Higher bound guess for the pressure of the middle state.
  # @return Pressure of the middle state, with a 1.e-8 relative error precision.
  ##############################################################################
  def solve_brent(self, rhoL, uL, PL, aL, rhoR, uR, PR, aR, Plow, Phigh):
    a = Plow
    b = Phigh
    c = 0.
    d = 1.e230

    fa = self.f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, a)
    fb = self.f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, b)
    fc = 0.

    s = 0.
    fs = 0.

    if fa * fb > 0.:
      print "Equal sign function values provided to solve_brent!"
      exit()

    # if |f(a)| < |f(b)| then swap (a,b) end if
    if abs(fa) < abs(fb):
      tmp = a
      a = b
      b = tmp
      tmp = fa
      fa = fb
      fb = tmp

    c = a
    fc = fa
    mflag = True

    while (not fb == 0.) and (abs(a - b) > 5.e-9 * (a + b)):
      if (not fa == fc) and (not fb == fc):
        # Inverse quadratic interpolation
        s = a * fb * fc / (fa - fb) / (fa - fc) + \
            b * fa * fc / (fb - fa) / (fb - fc) + \
            c * fa * fb / (fc - fa) / (fc - fb)
      else:
        # Secant Rule
        s = b - fb * (b - a) / (fb - fa)

      tmp2 = 0.25 * (3. * a + b)
      if (not (((s > tmp2) and (s < b)) or ((s < tmp2) and (s > b)))) or \
         (mflag and (abs(s - b) >= 0.5 * abs(b - c))) or \
         ((not mflag) and (abs(s - b) >= 0.5 * abs(c - d))) or \
         (mflag and (abs(b - c) < 5.e-9 * (b + c))) or \
         ((not mflag) and (abs(c - d) < 5.e-9 * (c + d))):
        s = 0.5 * (a + b)
        mflag = True
      else:
        mflag = False
      fs = self.f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, s)
      d = c
      c = b
      fc = fb
      if fa * fs < 0.:
        b = s
        fb = fs
      else:
        a = s
        fa = fs

      # if |f(a)| < |f(b)| then swap (a,b) end if
      if abs(fa) < abs(fb):
        tmp = a
        a = b
        b = tmp
        tmp = fa
        fa = fb
        fb = tmp

    return b

  ##############################################################################
  # @brief Sample the Riemann problem solution for a position in the right
  # shock wave regime.
  #
  # @param rhoR Density of the right state.
  # @param uR Velocity of the right state.
  # @param PR Pressure of the right state.
  # @param aR Soundspeed of the right state.
  # @param ustar Velocity of the middle state.
  # @param Pstar Pressure of the middle state.
  # @param rhosol Density solution.
  # @param usol Velocity solution.
  # @param Psol Pressure solution.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution.
  ##############################################################################
  def sample_right_shock_wave(self, rhoR, uR, PR, aR, ustar, Pstar, dxdt = 0.):
    # variable used twice below
    PdPR = Pstar / PR
    # get the shock speed
    SR = uR + aR * np.sqrt(self._gp1d2g * PdPR + self._gm1d2g)
    if SR > dxdt:
      ## middle state (shock) regime
      rhosol = rhoR * (PdPR + self._gm1dgp1) / (self._gm1dgp1 * PdPR + 1.)
      usol = ustar
      Psol = Pstar
    else:
      ## right state regime
      rhosol = rhoR
      usol = uR
      Psol = PR

    return rhosol, usol, Psol

  ##############################################################################
  # @brief Sample the Riemann problem solution for a position in the right
  # rarefaction wave regime.
  #
  # @param rhoR Density of the right state.
  # @param uR Velocity of the right state.
  # @param PR Pressure of the right state.
  # @param aR Soundspeed of the right state.
  # @param ustar Velocity of the middle state.
  # @param Pstar Pressure of the middle state.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution.
  ##############################################################################
  def sample_right_rarefaction_wave(self, rhoR, uR, PR, aR, ustar, Pstar,
                                    dxdt = 0.):
    # get the velocity of the head of the rarefaction wave
    SHR = uR + aR
    if SHR > dxdt:
      ## rarefaction wave regime
      # variable used twice below
      PdPR = Pstar / PR
      # get the velocity of the tail of the rarefaction wave
      STR = ustar + aR * PdPR**self._gm1d2g
      if STR > dxdt:
        ## middle state regime
        rhosol = rhoR * PdPR**self._ginv
        usol = ustar
        Psol = Pstar
      else:
        ## rarefaction fan regime
        # variable used twice below
        base = self._tdgp1 - self._gm1dgp1 * (uR - dxdt) / aR
        rhosol = rhoR * base**self._tdgm1
        usol = self._tdgp1 * (-aR + self._gm1d2 * uR + dxdt)
        Psol = PR * base**self._tgdgm1
    else:
      ## right state regime
      rhosol = rhoR
      usol = uR
      Psol = PR

    return rhosol, usol, Psol


  ##############################################################################
  # @brief Sample the Riemann problem solution in the right state regime.
  #
  # @param rhoR Density of the right state.
  # @param uR Velocity of the right state.
  # @param PR Pressure of the right state.
  # @param aR Soundspeed of the right state.
  # @param ustar Velocity of the middle state.
  # @param Pstar Pressure of the middle state.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution.
  ##############################################################################
  def sample_right_state(self, rhoR, uR, PR, aR, ustar, Pstar, dxdt = 0.):
    if Pstar > PR:
      ## shock wave
      rhosol, usol, Psol = self.sample_right_shock_wave(rhoR, uR, PR, aR, ustar,
                                                        Pstar, dxdt)
    else:
      ## rarefaction wave
      rhosol, usol, Psol = self.sample_right_rarefaction_wave(rhoR, uR, PR, aR,
                                                              ustar, Pstar,
                                                              dxdt)

    return rhosol, usol, Psol

  ##############################################################################
  # @brief Sample the Riemann problem solution for a position in the left shock
  #  wave regime.
  #
  # @param rhoL Density of the left state.
  # @param uL Velocity of the left state.
  # @param PL Pressure of the left state.
  # @param aL Soundspeed of the left state.
  # @param ustar Velocity of the middle state.
  # @param Pstar Pressure of the middle state.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution.
  ##############################################################################
  def sample_left_shock_wave(self, rhoL, uL, PL, aL, ustar, Pstar, dxdt = 0.):
    # variable used twice below
    PdPL = Pstar / PL
    # get the shock speed
    SL = uL - aL * np.sqrt(self._gp1d2g * PdPL + self._gm1d2g)
    if SL < dxdt:
      ## middle state (shock) regime
      rhosol = rhoL * (PdPL + self._gm1dgp1) / (self._gm1dgp1 * PdPL + 1.)
      usol = ustar
      Psol = Pstar
    else:
      ## left state regime
      rhosol = rhoL
      usol = uL
      Psol = PL

    return rhosol, usol, Psol

  ##############################################################################
  # @brief Sample the Riemann problem solution for a position in the left
  # rarefaction wave regime.
  #
  # @param rhoL Density of the left state.
  # @param uL Velocity of the left state.
  # @param PL Pressure of the left state.
  # @param aL Soundspeed of the left state.
  # @param ustar Velocity of the middle state.
  # @param Pstar Pressure of the middle state.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution.
  ##############################################################################
  def sample_left_rarefaction_wave(self, rhoL, uL, PL, aL, ustar, Pstar,
                                   dxdt = 0.):
    # get the velocity of the head of the rarefaction wave
    SHL = uL - aL
    if SHL < dxdt:
      ## rarefaction wave regime
      # variable used twice below
      PdPL = Pstar / PL
      # get the velocity of the tail of the rarefaction wave
      STL = ustar - aL * PdPL**self._gm1d2g
      if STL > dxdt:
        ## rarefaction fan regime
        # variable used twice below
        base = self._tdgp1 + self._gm1dgp1 * (uL - dxdt) / aL
        rhosol = rhoL * base**self._tdgm1
        usol = self._tdgp1 * (aL + self._gm1d2 * uL + dxdt)
        Psol = PL * base**self._tgdgm1
      else:
        ## middle state regime
        rhosol = rhoL * PdPL**self._ginv
        usol = ustar
        Psol = Pstar
    else:
      ## left state regime
      rhosol = rhoL
      usol = uL
      Psol = PL

    return rhosol, usol, Psol

  ##############################################################################
  # @brief Sample the Riemann problem solution in the left state regime.
  #
  # @param rhoL Density of the left state.
  # @param uL Velocity of the left state.
  # @param PL Pressure of the left state.
  # @param aL Soundspeed of the left state.
  # @param ustar Velocity of the middle state.
  # @param Pstar Pressure of the middle state.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution.
  ##############################################################################
  def sample_left_state(self, rhoL, uL, PL, aL, ustar, Pstar, dxdt = 0.):
    if Pstar > PL:
      ## shock wave
      rhosol, usol, Psol = self.sample_left_shock_wave(rhoL, uL, PL, aL, ustar,
                                                       Pstar, dxdt)
    else:
      ## rarefaction wave
      rhosol, usol, Psol = self.sample_left_rarefaction_wave(rhoL, uL, PL, aL,
                                                             ustar, Pstar,
                                                             dxdt)

    return rhosol, usol, Psol

  ##############################################################################
  # @brief Sample the vacuum Riemann problem if the right state is a vacuum.
  #
  # @param rhoL Density of the left state.
  # @param uL Velocity of the left state.
  # @param PL Pressure of the left state.
  # @param aL Soundspeed of the left state.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution, and a flag indicating
  # wether the left state (-1), the right state (1), or a vacuum state (0) was
  # sampled.
  ##############################################################################
  def sample_right_vacuum(self, rhoL, uL, PL, aL, dxdt = 0.):
    if uL - aL < dxdt:
      ## vacuum regime
      # get the vacuum rarefaction wave speed
      SL = uL + self._tdgm1 * aL
      if SL > dxdt:
        ## rarefaction wave regime
        # variable used twice below
        base = self._tdgp1 + self._gm1dgp1 * (uL - dxdt) / aL
        rhosol = rhoL * base**self._tdgm1
        usol = self._tdgp1 * (aL + self._gm1d2 * uL + dxdt)
        Psol = PL * base**self._tgdgm1
        flag = -1
      else:
        ## vacuum
        rhosol = 0.
        usol = 0.
        Psol = 0.
        flag = 0
    else:
      ## left state regime
      rhosol = rhoL
      usol = uL
      Psol = PL
      flag = -1

    return rhosol, usol, Psol, flag

  ##############################################################################
  # @brief Sample the vacuum Riemann problem if the left state is a vacuum.
  #
  # @param rhoR Density of the right state.
  # @param uR Velocity of the right state.
  # @param PR Pressure of the right state.
  # @param aR Soundspeed of the right state.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution, and a flag indicating
  # wether the left state (-1), the right state (1), or a vacuum state (0) was
  # sampled.
  ##############################################################################
  def sample_left_vacuum(self, rhoR, uR, PR, aR, dxdt = 0.):
    if dxdt < uR + aR:
      ## vacuum regime
      # get the vacuum rarefaction wave speed
      SR = uR - self._tdgm1 * aR
      if SR < dxdt:
        ## rarefaction wave regime
        # variable used twice below
        base = self._tdgp1 - self._gm1dgp1 * (uR - dxdt) / aR
        rhosol = rhoR * base**self._tdgm1
        usol = self._tdgp1 * (-aR + self._tdgm1 * uR + dxdt)
        Psol = PR * base**self._tgdgm1
        flag = 1
      else:
        ## vacuum
        rhosol = 0.
        usol = 0.
        Psol = 0.
        flag = 0
    else:
      ## right state regime
      rhosol = rhoR
      usol = uR
      Psol = PR
      flag = 1

    return rhosol, usol, Psol, flag

  ##############################################################################
  # @brief Sample the vacuum Riemann problem in the case vacuum is generated in
  # between the left and right state.
  #
  # @param rhoL Density of the left state.
  # @param uL Velocity of the left state.
  # @param PL Pressure of the left state.
  # @param aL Soundspeed of the left state.
  # @param rhoR Density of the right state.
  # @param uR Velocity of the right state.
  # @param PR Pressure of the right state.
  # @param aR Soundspeed of the right state.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution, and a flag indicating
  # wether the left state (-1), the right state (1), or a vacuum state (0) was
  # sampled.
  ##############################################################################
  def sample_vacuum_generation(self, rhoL, uL, PL, aL, rhoR, uR, PR, aR, dxdt):
    # get the speeds of the left and right rarefaction waves
    SR = uR - self._tdgm1 * aR
    SL = uL + self._tdgm1 * aL
    if SR > dxdt and SL < dxdt:
      ## vacuum
      rhosol = 0.
      usol = 0.
      Psol = 0.
      flag = 0
    else:
      if SL < dxdt:
        ## right state
        if dxdt < uR + aR:
          ## right rarefaction wave regime
          # variable used twice below
          base = self._tdgp1 - self._gm1dgp1 * (uR - dxdt) / aR
          rhosol = rhoR * base**self._tdgm1
          usol = self._tdgp1 * (-aR + self._tdgm1 * uR + dxdt)
          Psol = PR * base**self._tgdgm1
        else:
          ## right state regime
          rhosol = rhoR
          usol = uR
          Psol = PR
        flag = 1
      else:
        ### left state
        if dxdt > uL - aL:
          ## left rarefaction wave regime
          # variable used twice below
          base = self._tdgp1 + self._gm1dgp1 * (uL - dxdt) / aL
          rhosol = rhoL * base**self._tdgm1
          usol = self._tdgp1 * (aL + self._tdgm1 * uL + dxdt)
          Psol = PL * base**self._tgdgm1
        else:
          ## left state regime
          rhosol = rhoL
          usol = uL
          Psol = PL
        flag = -1

    return rhosol, usol, Psol, flag

  ##############################################################################
  # @brief Vacuum Riemann solver.
  #
  # This solver is called when one or both states have a zero density, or when
  # the vacuum generation condition is satisfied (meaning vacuum is generated
  # in the middle state, although strictly speaking there is no "middle"
  # state if vacuum is involved).
  #
  # @param rhoL Density of the left state.
  # @param uL Velocity of the left state.
  # @param PL Pressure of the left state.
  # @param aL Soundspeed of the left state.
  # @param rhoR Density of the right state.
  # @param uR Velocity of the right state.
  # @param PR Pressure of the right state.
  # @param aR Soundspeed of the right state.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution, and a flag indicating
  # wether the left state (-1), the right state (1), or a vacuum state (0) was
  # sampled.
  ##############################################################################
  def solve_vacuum(self, rhoL, uL, PL, aL, rhoR, uR, PR, aR, dxdt = 0.):
    # if both states are vacuum, the solution is also vacuum
    if rhoL == 0. and rhoR == 0.:
      return 0., 0., 0., 0

    if rhoR == 0.:
      ## vacuum right state
      return self.sample_right_vacuum(rhoL, uL, PL, aL, dxdt)
    else:
      if rhoL == 0.:
        ## vacuum left state
        return self.sample_left_vacuum(rhoR, uR, PR, aR, dxdt)
      else:
        ## vacuum "middle" state
        return self.sample_vacuum_generation(rhoL, uL, PL, aL, rhoR, uR, PR, aR,
                                             dxdt)


  ##############################################################################
  # @brief Solve the Riemann problem with the given left and right state.
  #
  # @param rhoL Left state density.
  # @param uL Left state velocity.
  # @param PL Left state pressure.
  # @param rhoR Right state density.
  # @param uR Right state velocity.
  # @param PR Right state pressure.
  # @param dxdt Point in velocity space where we want to sample the solution.
  # @return Density, velocity and pressure solution, and a flag indicating
  # wether the left state (-1), the right state (1), or a vacuum state (0) was
  # sampled.
  ##############################################################################
  def solve(self, rhoL, uL, PL, rhoR, uR, PR, dxdt = 0.):

    # get the soundspeeds
    aL = self.get_soundspeed(rhoL, PL)
    aR = self.get_soundspeed(rhoR, PR)

    # handle vacuum
    if rhoL == 0. or rhoR == 0.:
      return self.solve_vacuum(rhoL, uL, PL, aL, rhoR, uR, PR, aR, dxdt)

    # handle vacuum generation
    if self._tdgm1 * (aL + aR) <= uR - uL:
      return self.solve_vacuum(rhoL, uL, PL, aL, rhoR, uR, PR, aR, dxdt)

    # find the pressure and velocity in the middle state
    # since this is an exact Riemann solver, this is an iterative process,
    # whereby we basically find the root of a function (the Riemann f function
    # defined above)
    # we start by using a Newton-Raphson method, since we do not have an
    # interval in which the function changes sign
    # however, as soon as we have such an interval, we switch to a much more
    # robust root finding method (Brent's method). We do this because the
    # Newton-Raphson method in some cases can overshoot and return a negative
    # pressure, for which the Riemann f function is not defined. Brent's method
    # will never stroll outside of the initial interval in which the function
    # changes sign.
    Pstar = 0.
    Pguess = self.guess_P(rhoL, uL, PL, aL, rhoR, uR, PR, aR)
    # we only store this variable to store the sign of the function for pressure
    # zero
    # we need to find a larger pressure for which this sign changes to have an
    # interval where we can use Brent's method
    fPstar = self.f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pstar)
    fPguess = self.f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pguess)
    if fPstar * fPguess >= 0.:
      # Newton-Raphson until convergence or until usable interval is found to
      # use Brent's method
      while abs(Pstar - Pguess) > 5.e-9 * (Pstar + Pguess) and fPguess < 0.:
        Pstar = Pguess
        Pguess = Pguess - fPguess / \
                          self.fprime(rhoL, PL, aL, rhoR, PR, aR, Pguess)
        fPguess = self.f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pguess)

    # As soon as there is a suitable interval: use Brent's method
    if abs(Pstar - Pguess) > 5.e-9 * (Pstar + Pguess) and fPguess > 0.:
      Pstar = self.solve_brent(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pstar,
                               Pguess)
    else:
      Pstar = Pguess

    # the middle state velocity is fixed once the middle state pressure is known
    ustar = 0.5 * (uL + uR) + \
            0.5 * (self.fb(rhoR, PR, aR, Pstar) - self.fb(rhoL, PL, aL, Pstar))

    # we now have solved the Riemann problem: we have the left, middle and
    # right state, and this completely fixes the solution
    # we just need to sample the solution for x/t = 0.
    if ustar < dxdt:
      ## right state
      rhosol, usol, Psol = self.sample_right_state(rhoR, uR, PR, aR, ustar,
                                                   Pstar, dxdt)
      flag = 1
    else:
      ## left state
      rhosol, usol, Psol = self.sample_left_state(rhoL, uL, PL, aL, ustar,
                                                  Pstar, dxdt)
      flag = -1

    return rhosol, usol, Psol, flag
    
  ##############################################################################
  # @brief Solve the Riemann problem with the given left and right state for
  # the velocity and pressure (but do not sample the solution).
  #
  # @param rhoL Left state density.
  # @param uL Left state velocity.
  # @param PL Left state pressure.
  # @param rhoR Right state density.
  # @param uR Right state velocity.
  # @param PR Right state pressure.
  # @return Velocity and pressure in the middle region.
  ##############################################################################
  def solve_middle_state(self, rhoL, uL, PL, rhoR, uR, PR):

    # get the soundspeeds
    aL = self.get_soundspeed(rhoL, PL)
    aR = self.get_soundspeed(rhoR, PR)

    # handle vacuum
    if rhoL == 0. or rhoR == 0.:
      print "Vacuum not handled yet!"
      exit()

    # handle vacuum generation
    if self._tdgm1 * (aL + aR) <= uR - uL:
      print "Vacuum not handled yet!"
      exit()

    # find the pressure and velocity in the middle state
    # since this is an exact Riemann solver, this is an iterative process,
    # whereby we basically find the root of a function (the Riemann f function
    # defined above)
    # we start by using a Newton-Raphson method, since we do not have an
    # interval in which the function changes sign
    # however, as soon as we have such an interval, we switch to a much more
    # robust root finding method (Brent's method). We do this because the
    # Newton-Raphson method in some cases can overshoot and return a negative
    # pressure, for which the Riemann f function is not defined. Brent's method
    # will never stroll outside of the initial interval in which the function
    # changes sign.
    Pstar = 0.
    Pguess = self.guess_P(rhoL, uL, PL, aL, rhoR, uR, PR, aR)
    # we only store this variable to store the sign of the function for pressure
    # zero
    # we need to find a larger pressure for which this sign changes to have an
    # interval where we can use Brent's method
    fPstar = self.f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pstar)
    fPguess = self.f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pguess)
    if fPstar * fPguess >= 0.:
      # Newton-Raphson until convergence or until usable interval is found to
      # use Brent's method
      while abs(Pstar - Pguess) > 5.e-9 * (Pstar + Pguess) and fPguess < 0.:
        Pstar = Pguess
        Pguess = Pguess - fPguess / \
                          self.fprime(rhoL, PL, aL, rhoR, PR, aR, Pguess)
        fPguess = self.f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pguess)

    # As soon as there is a suitable interval: use Brent's method
    if abs(Pstar - Pguess) > 5.e-9 * (Pstar + Pguess) and fPguess > 0.:
      Pstar = self.solve_brent(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pstar,
                               Pguess)
    else:
      Pstar = Pguess

    # the middle state velocity is fixed once the middle state pressure is known
    ustar = 0.5 * (uL + uR) + \
            0.5 * (self.fb(rhoR, PR, aR, Pstar) - self.fb(rhoL, PL, aL, Pstar))

    return ustar, Pstar

################################################################################
################################################################################

################################################################################
# @brief Check if the relative difference between the two given values is
# smaller than the given tolerance.
#
# @param A First value.
# @param B Second value.
# @param relative_error Tolerance relative difference value.
# @return True if the relative difference between the two values is smaller than
# the given tolerance value.
################################################################################
def relative_difference_smaller_than(A, B, relative_error):
  if A == B:
    return True
  else:
    return abs(A - B) < relative_error * abs(A + B)

################################################################################
# @brief Get the relative difference between the two given values.
#
# @param A First value.
# @param B Second value.
# @return Relative difference between the two values.
################################################################################
def relative_difference(A, B):
  return abs(A - B) / abs(A + B)

################################################################################
# @brief Run a basic Riemann solver test with given left and right state, and
# given reference pressure solution.
#
# @param rhoL Left state density.
# @param uL Left state velocity.
# @param PL Left state pressure.
# @param rhoR Right state density.
# @param uR Right state velocity.
# @param PR Right state pressure.
# @param Pref Reference solution pressure.
################################################################################
def run_riemannsolver_basic_test(solver, rhoL, uL, PL, rhoR, uR, PR, Pref):

  usol, Psol = solver.solve_middle_state(rhoL, uL, PL, rhoR, uR, PR)

  if not relative_difference_smaller_than(Psol, Pref, 1.e-4):
    print "Wrong pressure solution: {Psol}, should be {Pref}".format(
      Psol = Psol, Pref = Pref)
    print "(relative difference: {reldiff})!".format(
      reldiff = relative_difference(Psol, Pref))
    exit()

################################################################################
# @brief Run a Riemann solver test with given left and right state, and given
# reference solution.
#
# @param rhoL Left state density.
# @param uL Left state velocity.
# @param PL Left state pressure.
# @param rhoR Right state density.
# @param uR Right state velocity.
# @param PR Right state pressure.
# @param rhoref Reference solution density.
# @param uref Reference solution velocity.
# @param Pref Reference solution pressure.
# @param flagref Reference solution flag: 1 if the right state is sampled, -1 if
# the left state is sampled, and 0 if a vacuum state is sampled.
################################################################################
def run_riemannsolver_test(solver, rhoL, uL, PL, rhoR, uR, PR, rhoref, uref,
                           Pref, flagref):
  rhosol, usol, Psol, flagsol = solver.solve(rhoL, uL, PL, rhoR, uR, PR)

  if not relative_difference_smaller_than(rhosol, rhoref, 1.e-4):
    print "Wrong density solution: {rhosol}, should be {rhoref}".format(
      rhosol = rhosol, rhoref = rhoref)
    print "(relative difference: {reldiff})!".format(
      reldiff = relative_difference(rhosol, rhoref))
    exit()

  if not relative_difference_smaller_than(usol, uref, 1.e-4):
    print "Wrong velocity solution: {usol}, should be {uref}".format(
      usol = usol, uref = uref)
    print "(relative difference: {reldiff})!".format(
      reldiff = relative_difference(usol, uref))
    exit()

  if not relative_difference_smaller_than(Psol, Pref, 1.e-4):
    print "Wrong pressure solution: {Psol}, should be {Pref}".format(
      Psol = Psol, Pref = Pref)
    print "(relative difference: {reldiff})!".format(
      reldiff = relative_difference(Psol, Pref))
    exit()

  if not flagsol == flagref:
    print "Wrong solution sampled: {flagsol}, should be {flagref}".format(
      flagsol = flagsol, flagref = flagref)
    exit()

################################################################################
# @brief Default action when this file is run directly: run some unit tests.
################################################################################
if __name__ == "__main__":
  print "\nThis script should not be run directly. Instead, import it into " \
        "another script and use its functionality there.\n" \
        "Now that we're running anyway, we will quickly run some unit tests " \
        "to make sure everything still works...\n"

  # Toro tests
  solver = RiemannSolver(1.4)
  run_riemannsolver_basic_test(solver,
                               1., 0., 1.,
                               0.125, 0., 0.1,
                               0.30313)
  run_riemannsolver_basic_test(solver,
                               1., -2., 0.4,
                               1., 2., 0.4,
                               0.001894)
  run_riemannsolver_basic_test(solver,
                               1., 0., 1000.,
                               1., 0., 0.01,
                               460.894)
  run_riemannsolver_basic_test(solver,
                               1., 0., 0.01,
                               1., 0., 100.,
                               46.095)
  run_riemannsolver_basic_test(solver,
                               5.99924, 19.5975, 460.894,
                               5.99242, -6.19633, 46.0950,
                               1691.64)

  # Toro tests with sampling and different adiabatic index
  solver = RiemannSolver(5./3.)
  run_riemannsolver_test(solver,
                         1., 0., 1.,
                         0.125, 0., 0.1,
                         0.47969, 0.841194, 0.293945, -1)
  run_riemannsolver_test(solver,
                         1., -2., 0.4,
                         1., 2., 0.4,
                         0.00617903, 0., 8.32249e-05, -1)
  run_riemannsolver_test(solver,
                         1., 0., 1000.,
                         1., 0., 0.01,
                         0.615719, 18.2812, 445.626, -1)
  run_riemannsolver_test(solver,
                         1., 0., 0.01,
                         1., 0., 100.,
                         0.61577, -5.78011, 44.5687, 1)
  run_riemannsolver_test(solver,
                         5.99924, 19.5975, 460.894,
                         5.99242, -6.19633, 46.0950,
                         12.743, 8.56045, 1841.82, -1)
  # vacuum generation test
  run_riemannsolver_test(solver,
                         1., -1., 1.e-6,
                         1., 1., 1.0005e-6,
                         0., 0., 0., 0)

  print "Unit tests successfully finished. Everything still works!\n" \
        "Have a nice day!\n"
