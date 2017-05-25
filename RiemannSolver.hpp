/*******************************************************************************
 * This file is part of python_finite_volume_solver
 * Copyright (C) 2017 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 *
 * python_finite_volume_solver is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the License,
 * or (at your option) any later version.
 *
 * python_finite_volume_solver is distributed in the hope that it will be
 * useful, but WITOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero
 * General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with python_finite_volume_solver. If not, see
 * <http://www.gnu.org/licenses/>.
 ******************************************************************************/

/**
 * @file RiemannSolver.hpp
 *
 * @brief Exact Riemann solver.
 *
 * This Riemann solver is based on the ExactRiemannSolver class in the public
 * simulation code Shadowfax (Vandenbroucke & De Rijcke, 2016), and is almost
 * identical to the Riemann solver in the public simulation code CMacIonize
 * (https://github.com/bwvdnbro/CMacIonize).
 *
 * To use the Riemann solver, first create a RiemannSolver object with the
 * desired adiabatic index. Actual Riemann problem solutions are then obtained
 * by calling RiemannSolver::solve(), see the documentation for that function
 * for more information.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef RIEMANNSOLVER_HPP
#define RIEMANNSOLVER_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>

/**
 * @brief Macro that prints the given formatted string (with arguments) to the
 * given stream as a 5 space indented block of 65 characters with proper limits.
 */
#define print_indent(stream, s, ...)                                           \
  {                                                                            \
    char buffer[10000];                                                        \
    sprintf(buffer, s, ##__VA_ARGS__);                                         \
    int pos = 0;                                                               \
    int linepos = 0;                                                           \
    /* we scan the string char by char. If a tab is encountered, it is */      \
    /* replaced with four spaces. If a newline is found, we print */           \
    /* immediately. If a space is found, we need to figure out the position */ \
    /* of the next space and check if the next word fits on the line. */       \
    char line[65];                                                             \
    while (buffer[pos] != '\0') {                                              \
      if (buffer[pos] == '\n') {                                               \
        fprintf(stream, "     %-65s\n", line);                                 \
        ++pos;                                                                 \
        linepos = 0;                                                           \
      } else {                                                                 \
        if (buffer[pos] == ' ' || buffer[pos] == '\t') {                       \
          int old_linepos = linepos;                                           \
          if (buffer[pos] == '\t') {                                           \
            for (unsigned int j = 0; j < 4; ++j) {                             \
              line[linepos] = ' ';                                             \
              ++linepos;                                                       \
            }                                                                  \
          } else {                                                             \
            line[linepos] = ' ';                                               \
            ++linepos;                                                         \
          }                                                                    \
          /* find the end of the next word */                                  \
          int nextpos = 1;                                                     \
          while (buffer[pos + nextpos] != '\t' &&                              \
                 buffer[pos + nextpos] != ' ' &&                               \
                 buffer[pos + nextpos] != '\n' &&                              \
                 buffer[pos + nextpos] != '\0') {                              \
            ++nextpos;                                                         \
          }                                                                    \
          if (linepos + nextpos > 65) {                                        \
            /* print the line and reset */                                     \
            line[old_linepos] = '\0';                                          \
            linepos = 65;                                                      \
          }                                                                    \
        } else {                                                               \
          line[linepos] = buffer[pos];                                         \
          ++linepos;                                                           \
        }                                                                      \
        if (linepos == 65) {                                                   \
          fprintf(stream, "     %-65s\n", line);                               \
          linepos = 0;                                                         \
        }                                                                      \
        ++pos;                                                                 \
      }                                                                        \
    }                                                                          \
    if (linepos) {                                                             \
      line[linepos] = '\0';                                                    \
      fprintf(stream, "     %-65s\n", line);                                   \
    }                                                                          \
  }

/**
 * @brief Error macro. Prints the given error message (with C style formatting)
 * and aborts the code.
 */
#define cmac_error(s, ...)                                                     \
  {                                                                            \
    fprintf(stderr, "%s:%s():%i: Error:\n", __FILE__, __FUNCTION__, __LINE__); \
    print_indent(stderr, s, ##__VA_ARGS__);                                    \
    abort();                                                                   \
  }

/**
 * @brief Exact Riemann solver.
 */
class RiemannSolver {
private:
  /*! @brief Adiabatic index @f$\gamma{}@f$. */
  double _gamma;

  /*! @brief @f$\frac{\gamma+1}{2\gamma}@f$ */
  double _gp1d2g;

  /*! @brief @f$\frac{\gamma-1}{2\gamma}@f$ */
  double _gm1d2g;

  /*! @brief @f$\frac{\gamma-1}{\gamma+1}@f$ */
  double _gm1dgp1;

  /*! @brief @f$\frac{2}{\gamma+1}@f$ */
  double _tdgp1;

  /*! @brief @f$\frac{2}{\gamma-1}@f$ */
  double _tdgm1;

  /*! @brief @f$\frac{\gamma-1}{2}@f$ */
  double _gm1d2;

  /*! @brief @f$\frac{2\gamma}{\gamma-1}@f$ */
  double _tgdgm1;

  /*! @brief @f$\frac{1}{\gamma}@f$ */
  double _ginv;

  /**
   * @brief Get the soundspeed corresponding to the given density and pressure.
   *
   * @param rho Density value.
   * @param P Pressure value.
   * @return Soundspeed.
   */
  inline double get_soundspeed(double rho, double P) const {
    return std::sqrt(_gamma * P / rho);
  }

  /**
   * @brief Riemann fL or fR function.
   *
   * @param rho Density of the left or right state.
   * @param P Pressure of the left or right state.
   * @param a Soundspeed of the left or right state.
   * @param Pstar (Temporary) pressure of the middle state.
   * @return Value of the fL or fR function.
   */
  inline double fb(double rho, double P, double a, double Pstar) const {
    double fval = 0.;
    if (Pstar > P) {
      double A = _tdgp1 / rho;
      double B = _gm1dgp1 * P;
      fval = (Pstar - P) * std::sqrt(A / (Pstar + B));
    } else {
      fval = _tdgm1 * a * (std::pow(Pstar / P, _gm1d2g) - 1.);
    }
    return fval;
  }

  /**
   * @brief Riemann f function.
   *
   * @param rhoL Density of the left state.
   * @param uL Velocity of the left state.
   * @param PL Pressure of the left state.
   * @param aL Soundspeed of the left state.
   * @param rhoR Density of the right state.
   * @param uR Velocity of the right state.
   * @param PR Pressure of the right state.
   * @param aR Soundspeed of the right state.
   * @param Pstar (Temporary) pressure of the middle state.
   * @return Value of the Riemann f function.
   */
  inline double f(double rhoL, double uL, double PL, double aL, double rhoR,
                  double uR, double PR, double aR, double Pstar) const {
    return fb(rhoL, PL, aL, Pstar) + fb(rhoR, PR, aR, Pstar) + (uR - uL);
  }

  /**
   * @brief Derivative of the Riemann fL or fR function.
   *
   * @param rho Density of the left or right state.
   * @param P Pressure of the left or right state.
   * @param a Soundspeed of the left or right state.
   * @param Pstar (Temporary) pressure of the middle state.
   * @return Value of the derivative of the Riemann fL or fR function.
   */
  inline double fprimeb(double rho, double P, double a, double Pstar) const {
    double fval = 0.;
    if (Pstar > P) {
      double A = _tdgp1 / rho;
      double B = _gm1dgp1 * P;
      fval =
          (1. - 0.5 * (Pstar - P) / (B + Pstar)) * std::sqrt(A / (Pstar + B));
    } else {
      fval = 1. / (rho * a) * std::pow(Pstar / P, -_gp1d2g);
    }
    return fval;
  }

  /**
   * @brief Derivative of the Riemann f function.
   *
   * @param rhoL Density of the left state.
   * @param PL Pressure of the left state.
   * @param aL Soundspeed of the left state.
   * @param rhoR Density of the right state.
   * @param PR Pressure of the right state.
   * @param aR Soundspeed of the right state.
   * @param Pstar (Temporary) pressure of the middle state.
   * @return Value of the derivative of the Riemann f function.
   */
  inline double fprime(double rhoL, double PL, double aL, double rhoR,
                       double PR, double aR, double Pstar) const {
    return fprimeb(rhoL, PL, aL, Pstar) + fprimeb(rhoR, PR, aR, Pstar);
  }

  /**
   * @brief Riemann gL or gR function.
   *
   * @param rho Density of the left or right state.
   * @param P Pressure of the left or right state.
   * @param Pstar (Temporary) pressure in the middle state.
   * @return Value of the gL or gR function.
   */
  inline double gb(double rho, double P, double Pstar) const {
    double A = _tdgp1 / rho;
    double B = _gm1dgp1 * P;
    return std::sqrt(A / (Pstar + B));
  }

  /**
   * @brief Get an initial guess for the pressure in the middle state.
   *
   * @param rhoL Left state density.
   * @param uL Left state velocity.
   * @param PL Left state pressure.
   * @param aL Left state soundspeed.
   * @param rhoR Right state density.
   * @param uR Right state velocity.
   * @param PR Right state pressure.
   * @param aR Right state soundspeed.
   * @return Initial guess for the pressure in the middle state.
   */
  inline double guess_P(double rhoL, double uL, double PL, double aL,
                        double rhoR, double uR, double PR, double aR) const {
    double Pguess;
    double Pmin = std::min(PL, PR);
    double Pmax = std::max(PL, PR);
    double qmax = Pmax / Pmin;
    double Ppv = 0.5 * (PL + PR) - 0.125 * (uR - uL) * (PL + PR) * (aL + aR);
    Ppv = std::max(5.e-9 * (PL + PR), Ppv);
    if (qmax <= 2. && Pmin <= Ppv && Ppv <= Pmax) {
      Pguess = Ppv;
    } else {
      if (Ppv < Pmin) {
        // two rarefactions
        Pguess = std::pow(
            (aL + aR - _gm1d2 * (uR - uL)) /
                (aL / std::pow(PL, _gm1d2g) + aR / std::pow(PR, _gm1d2g)),
            _tgdgm1);
      } else {
        // two shocks
        double gL = gb(rhoL, PL, Ppv);
        double gR = gb(rhoR, PR, Ppv);
        Pguess = (gL * PL + gR * PR - uR + uL) / (gL + gR);
      }
    }
    // Toro: "Not that approximate solutions may predict, incorrectly, a
    // negative value for pressure (...). Thus in order to avoid negative guess
    // values we introduce the small positive constant _tolerance"
    // (tolerance is 1.e-8 in this case)
    Pguess = std::max(5.e-9 * (PL + PR), Pguess);
    return Pguess;
  }

  /**
   * @brief Find the pressure of the middle state by using Brent's method.
   *
   * @param rhoL Density of the left state.
   * @param uL Velocity of the left state.
   * @param PL Pressure of the left state.
   * @param aL Soundspeed of the left state.
   * @param rhoR Density of the right state.
   * @param uR Velocity of the right state.
   * @param PR Pressure of the right state.
   * @param aR Soundspeed of the right state.
   * @param Plow Lower bound guess for the pressure of the middle state.
   * @param Phigh Higher bound guess for the pressure of the middle state.
   * @param fPlow Value of the pressure function for the lower bound guess.
   * @param fPhigh Value of the pressure function for the upper bound guess.
   * @return Pressure of the middle state, with a 1.e-8 relative error
   * precision.
   */
  inline double solve_brent(double rhoL, double uL, double PL, double aL,
                            double rhoR, double uR, double PR, double aR,
                            double Plow, double Phigh, double fPlow,
                            double fPhigh) const {
    double a = Plow;
    double b = Phigh;
    double c = 0.;
    double d = 1e230;

    double fa = fPlow;
    double fb = fPhigh;
    double fc = 0.;

    double s = 0.;
    double fs = 0.;

    if (fa * fb > 0.) {
      cmac_error("Equal sign function values provided to solve_brent (%g %g)!",
                 fa, fb);
    }

    // if |f(a)| < |f(b)| then swap (a,b) end if
    if (std::abs(fa) < std::abs(fb)) {
      double tmp = a;
      a = b;
      b = tmp;
      tmp = fa;
      fa = fb;
      fb = tmp;
    }

    c = a;
    fc = fa;
    bool mflag = true;

    while (!(fb == 0.) && (std::abs(a - b) > 5.e-9 * (a + b))) {
      if ((fa != fc) && (fb != fc)) {
        // Inverse quadratic interpolation
        s = a * fb * fc / (fa - fb) / (fa - fc) +
            b * fa * fc / (fb - fa) / (fb - fc) +
            c * fa * fb / (fc - fa) / (fc - fb);
      } else {
        // Secant Rule
        s = b - fb * (b - a) / (fb - fa);
      }

      double tmp2 = 0.25 * (3. * a + b);
      if (!(((s > tmp2) && (s < b)) || ((s < tmp2) && (s > b))) ||
          (mflag && (std::abs(s - b) >= 0.5 * std::abs(b - c))) ||
          (!mflag && (std::abs(s - b) >= 0.5 * std::abs(c - d))) ||
          (mflag && (std::abs(b - c) < 5.e-9 * (b + c))) ||
          (!mflag && (std::abs(c - d) < 5.e-9 * (c + d)))) {
        s = 0.5 * (a + b);
        mflag = true;
      } else {
        mflag = false;
      }
      fs = f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, s);
      d = c;
      c = b;
      fc = fb;
      if (fa * fs < 0.) {
        b = s;
        fb = fs;
      } else {
        a = s;
        fa = fs;
      }

      // if |f(a)| < |f(b)| then swap (a,b) end if
      if (std::abs(fa) < std::abs(fb)) {
        double tmp = a;
        a = b;
        b = tmp;
        tmp = fa;
        fa = fb;
        fb = tmp;
      }
    }
    return b;
  }

  /**
   * @brief Sample the Riemann problem solution for a position in the right
   * shock wave regime.
   *
   * @param rhoR Density of the right state.
   * @param uR Velocity of the right state.
   * @param PR Pressure of the right state.
   * @param aR Soundspeed of the right state.
   * @param ustar Velocity of the middle state.
   * @param Pstar Pressure of the middle state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   */
  inline void sample_right_shock_wave(double rhoR, double uR, double PR,
                                      double aR, double ustar, double Pstar,
                                      double &rhosol, double &usol,
                                      double &Psol, double dxdt = 0.) const {
    // variable used twice below
    double PdPR = Pstar / PR;
    // get the shock speed
    double SR = uR + aR * std::sqrt(_gp1d2g * PdPR + _gm1d2g);
    if (SR > dxdt) {
      /// middle state (shock) regime
      rhosol = rhoR * (PdPR + _gm1dgp1) / (_gm1dgp1 * PdPR + 1.);
      usol = ustar;
      Psol = Pstar;
    } else {
      /// right state regime
      rhosol = rhoR;
      usol = uR;
      Psol = PR;
    }
  }

  /**
   * @brief Sample the Riemann problem solution for a position in the right
   * rarefaction wave regime.
   *
   * @param rhoR Density of the right state.
   * @param uR Velocity of the right state.
   * @param PR Pressure of the right state.
   * @param aR Soundspeed of the right state.
   * @param ustar Velocity of the middle state.
   * @param Pstar Pressure of the middle state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   */
  inline void sample_right_rarefaction_wave(double rhoR, double uR, double PR,
                                            double aR, double ustar,
                                            double Pstar, double &rhosol,
                                            double &usol, double &Psol,
                                            double dxdt = 0.) const {
    // get the velocity of the head of the rarefaction wave
    double SHR = uR + aR;
    if (SHR > dxdt) {
      /// rarefaction wave regime
      // variable used twice below
      double PdPR = Pstar / PR;
      // get the velocity of the tail of the rarefaction wave
      double STR = ustar + aR * std::pow(PdPR, _gm1d2g);
      if (STR > dxdt) {
        /// middle state regime
        rhosol = rhoR * std::pow(PdPR, _ginv);
        usol = ustar;
        Psol = Pstar;
      } else {
        /// rarefaction fan regime
        // variable used twice below
        double base = _tdgp1 - _gm1dgp1 * (uR - dxdt) / aR;
        rhosol = rhoR * std::pow(base, _tdgm1);
        usol = _tdgp1 * (-aR + _gm1d2 * uR + dxdt);
        Psol = PR * std::pow(base, _tgdgm1);
      }
    } else {
      /// right state regime
      rhosol = rhoR;
      usol = uR;
      Psol = PR;
    }
  }

  /**
   * @brief Sample the Riemann problem solution in the right state regime.
   *
   * @param rhoR Density of the right state.
   * @param uR Velocity of the right state.
   * @param PR Pressure of the right state.
   * @param aR Soundspeed of the right state.
   * @param ustar Velocity of the middle state.
   * @param Pstar Pressure of the middle state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   */
  inline void sample_right_state(double rhoR, double uR, double PR, double aR,
                                 double ustar, double Pstar, double &rhosol,
                                 double &usol, double &Psol,
                                 double dxdt = 0.) const {
    if (Pstar > PR) {
      /// shock wave
      sample_right_shock_wave(rhoR, uR, PR, aR, ustar, Pstar, rhosol, usol,
                              Psol, dxdt);
    } else {
      /// rarefaction wave
      sample_right_rarefaction_wave(rhoR, uR, PR, aR, ustar, Pstar, rhosol,
                                    usol, Psol, dxdt);
    }
  }

  /**
   * @brief Sample the Riemann problem solution for a position in the left shock
   *  wave regime.
   *
   * @param rhoL Density of the left state.
   * @param uL Velocity of the left state.
   * @param PL Pressure of the left state.
   * @param aL Soundspeed of the left state.
   * @param ustar Velocity of the middle state.
   * @param Pstar Pressure of the middle state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   */
  inline void sample_left_shock_wave(double rhoL, double uL, double PL,
                                     double aL, double ustar, double Pstar,
                                     double &rhosol, double &usol, double &Psol,
                                     double dxdt = 0.) const {
    // variable used twice below
    double PdPL = Pstar / PL;
    // get the shock speed
    double SL = uL - aL * std::sqrt(_gp1d2g * PdPL + _gm1d2g);
    if (SL < dxdt) {
      /// middle state (shock) regime
      rhosol = rhoL * (PdPL + _gm1dgp1) / (_gm1dgp1 * PdPL + 1.);
      usol = ustar;
      Psol = Pstar;
    } else {
      /// left state regime
      rhosol = rhoL;
      usol = uL;
      Psol = PL;
    }
  }

  /**
   * @brief Sample the Riemann problem solution for a position in the left
   * rarefaction wave regime.
   *
   * @param rhoL Density of the left state.
   * @param uL Velocity of the left state.
   * @param PL Pressure of the left state.
   * @param aL Soundspeed of the left state.
   * @param ustar Velocity of the middle state.
   * @param Pstar Pressure of the middle state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   */
  inline void sample_left_rarefaction_wave(double rhoL, double uL, double PL,
                                           double aL, double ustar,
                                           double Pstar, double &rhosol,
                                           double &usol, double &Psol,
                                           double dxdt = 0.) const {
    // get the velocity of the head of the rarefaction wave
    double SHL = uL - aL;
    if (SHL < dxdt) {
      /// rarefaction wave regime
      // variable used twice below
      double PdPL = Pstar / PL;
      // get the velocity of the tail of the rarefaction wave
      double STL = ustar - aL * std::pow(PdPL, _gm1d2g);
      if (STL > dxdt) {
        /// rarefaction fan regime
        // variable used twice below
        double base = _tdgp1 + _gm1dgp1 * (uL - dxdt) / aL;
        rhosol = rhoL * std::pow(base, _tdgm1);
        usol = _tdgp1 * (aL + _gm1d2 * uL + dxdt);
        Psol = PL * std::pow(base, _tgdgm1);
      } else {
        /// middle state regime
        rhosol = rhoL * std::pow(PdPL, _ginv);
        usol = ustar;
        Psol = Pstar;
      }
    } else {
      /// left state regime
      rhosol = rhoL;
      usol = uL;
      Psol = PL;
    }
  }

  /**
   * @brief Sample the Riemann problem solution in the left state regime.
   *
   * @param rhoL Density of the left state.
   * @param uL Velocity of the left state.
   * @param PL Pressure of the left state.
   * @param aL Soundspeed of the left state.
   * @param ustar Velocity of the middle state.
   * @param Pstar Pressure of the middle state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   */
  inline void sample_left_state(double rhoL, double uL, double PL, double aL,
                                double ustar, double Pstar, double &rhosol,
                                double &usol, double &Psol,
                                double dxdt = 0.) const {
    if (Pstar > PL) {
      /// shock wave
      sample_left_shock_wave(rhoL, uL, PL, aL, ustar, Pstar, rhosol, usol, Psol,
                             dxdt);
    } else {
      /// rarefaction wave
      sample_left_rarefaction_wave(rhoL, uL, PL, aL, ustar, Pstar, rhosol, usol,
                                   Psol, dxdt);
    }
  }

  /**
   * @brief Sample the vacuum Riemann problem if the right state is a vacuum.
   *
   * @param rhoL Density of the left state.
   * @param uL Velocity of the left state.
   * @param PL Pressure of the left state.
   * @param aL Soundspeed of the left state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   * @return Flag indicating wether the left state (-1), the right state (1), or
   * a vacuum state (0) was sampled.
   */
  inline int sample_right_vacuum(double rhoL, double uL, double PL, double aL,
                                 double &rhosol, double &usol, double &Psol,
                                 double dxdt = 0.) const {
    if (uL - aL < dxdt) {
      /// vacuum regime
      // get the vacuum rarefaction wave speed
      double SL = uL + _tdgm1 * aL;
      if (SL > dxdt) {
        /// rarefaction wave regime
        // variable used twice below
        double base = _tdgp1 + _gm1dgp1 * (uL - dxdt) / aL;
        rhosol = rhoL * std::pow(base, _tdgm1);
        usol = _tdgp1 * (aL + _gm1d2 * uL + dxdt);
        Psol = PL * std::pow(base, _tgdgm1);
        return -1;
      } else {
        /// vacuum
        rhosol = 0.;
        usol = 0.;
        Psol = 0.;
        return 0;
      }
    } else {
      /// left state regime
      rhosol = rhoL;
      usol = uL;
      Psol = PL;
      return -1;
    }
  }

  /**
   * @brief Sample the vacuum Riemann problem if the left state is a vacuum.
   *
   * @param rhoR Density of the right state.
   * @param uR Velocity of the right state.
   * @param PR Pressure of the right state.
   * @param aR Soundspeed of the right state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   * @return Flag indicating wether the left state (-1), the right state (1), or
   * a vacuum state (0) was sampled.
   */
  inline int sample_left_vacuum(double rhoR, double uR, double PR, double aR,
                                double &rhosol, double &usol, double &Psol,
                                double dxdt = 0.) const {
    if (dxdt < uR + aR) {
      /// vacuum regime
      // get the vacuum rarefaction wave speed
      double SR = uR - _tdgm1 * aR;
      if (SR < dxdt) {
        /// rarefaction wave regime
        // variable used twice below
        double base = _tdgp1 - _gm1dgp1 * (uR - dxdt) / aR;
        rhosol = rhoR * std::pow(base, _tdgm1);
        usol = _tdgp1 * (-aR + _tdgm1 * uR + dxdt);
        Psol = PR * std::pow(base, _tgdgm1);
        return 1;
      } else {
        /// vacuum
        rhosol = 0.;
        usol = 0.;
        Psol = 0.;
        return 0;
      }
    } else {
      /// right state regime
      rhosol = rhoR;
      usol = uR;
      Psol = PR;
      return 1;
    }
  }

  /**
   * @brief Sample the vacuum Riemann problem in the case vacuum is generated in
   * between the left and right state.
   *
   * @param rhoL Density of the left state.
   * @param uL Velocity of the left state.
   * @param PL Pressure of the left state.
   * @param aL Soundspeed of the left state.
   * @param rhoR Density of the right state.
   * @param uR Velocity of the right state.
   * @param PR Pressure of the right state.
   * @param aR Soundspeed of the right state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   * @return Flag indicating wether the left state (-1), the right state (1), or
   * a vacuum state (0) was sampled.
   */
  inline int sample_vacuum_generation(double rhoL, double uL, double PL,
                                      double aL, double rhoR, double uR,
                                      double PR, double aR, double &rhosol,
                                      double &usol, double &Psol,
                                      double dxdt) const {
    // get the speeds of the left and right rarefaction waves
    double SR = uR - _tdgm1 * aR;
    double SL = uL + _tdgm1 * aL;
    if (SR > dxdt && SL < dxdt) {
      /// vacuum
      rhosol = 0.;
      usol = 0.;
      Psol = 0.;
      return 0;
    } else {
      if (SL < dxdt) {
        /// right state
        if (dxdt < uR + aR) {
          /// right rarefaction wave regime
          // variable used twice below
          double base = _tdgp1 - _gm1dgp1 * (uR - dxdt) / aR;
          rhosol = rhoR * std::pow(base, _tdgm1);
          usol = _tdgp1 * (-aR + _tdgm1 * uR + dxdt);
          Psol = PR * std::pow(base, _tgdgm1);
        } else {
          /// right state regime
          rhosol = rhoR;
          usol = uR;
          Psol = PR;
        }
        return 1;
      } else {
        /// left state
        if (dxdt > uL - aL) {
          /// left rarefaction wave regime
          // variable used twice below
          double base = _tdgp1 + _gm1dgp1 * (uL - dxdt) / aL;
          rhosol = rhoL * std::pow(base, _tdgm1);
          usol = _tdgp1 * (aL + _tdgm1 * uL + dxdt);
          Psol = PL * std::pow(base, _tgdgm1);
        } else {
          /// left state regime
          rhosol = rhoL;
          usol = uL;
          Psol = PL;
        }
        return -1;
      }
    }
  }

  /**
   * @brief Vacuum Riemann solver.
   *
   * This solver is called when one or both states have a zero density, or when
   * the vacuum generation condition is satisfied (meaning vacuum is generated
   * in the middle state, although strictly speaking there is no "middle"
   * state if vacuum is involved).
   *
   * @param rhoL Density of the left state.
   * @param uL Velocity of the left state.
   * @param PL Pressure of the left state.
   * @param aL Soundspeed of the left state.
   * @param rhoR Density of the right state.
   * @param uR Velocity of the right state.
   * @param PR Pressure of the right state.
   * @param aR Soundspeed of the right state.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   * @return Flag indicating wether the left state (-1), the right state (1), or
   * a vacuum state (0) was sampled.
   */
  inline int solve_vacuum(double rhoL, double uL, double PL, double aL,
                          double rhoR, double uR, double PR, double aR,
                          double &rhosol, double &usol, double &Psol,
                          double dxdt = 0.) const {
    // if both states are vacuum, the solution is also vacuum
    if (rhoL == 0. && rhoR == 0.) {
      rhosol = 0.;
      usol = 0.;
      Psol = 0.;
      return 0;
    }

    if (rhoR == 0.) {
      /// vacuum right state
      return sample_right_vacuum(rhoL, uL, PL, aL, rhosol, usol, Psol, dxdt);
    } else if (rhoL == 0.) {
      /// vacuum left state
      return sample_left_vacuum(rhoR, uR, PR, aR, rhosol, usol, Psol, dxdt);
    } else {
      /// vacuum "middle" state
      return sample_vacuum_generation(rhoL, uL, PL, aL, rhoR, uR, PR, aR,
                                      rhosol, usol, Psol, dxdt);
    }
  }

public:
  /**
   * @brief Constructor.
   *
   * @param gamma Adiabatic index @f$\gamma{}@f$.
   */
  RiemannSolver(double gamma) : _gamma(gamma) {
    if (_gamma <= 1.) {
      cmac_error("The adiabatic index needs to be larger than 1!")
    }

    // related quantities:
    _gp1d2g = 0.5 * (_gamma + 1.) / _gamma; // gamma plus 1 divided by 2 gamma
    _gm1d2g = 0.5 * (_gamma - 1.) / _gamma; // gamma minus 1 divided by 2 gamma
    _gm1dgp1 =
        (_gamma - 1.) / (_gamma + 1.); // gamma minus 1 divided by gamma plus 1
    _tdgp1 = 2. / (_gamma + 1.);       // two divided by gamma plus 1
    _tdgm1 = 2. / (_gamma - 1.);       // two divided by gamma minus 1
    _gm1d2 = 0.5 * (_gamma - 1.);      // gamma minus 1 divided by 2
    _tgdgm1 =
        2. * _gamma / (_gamma - 1.); // two times gamma divided by gamma minus 1
    _ginv = 1. / _gamma;             // gamma inverse
  }

  /**
   * @brief Solve the Riemann problem with the given left and right state.
   *
   * @param rhoL Left state density.
   * @param uL Left state velocity.
   * @param PL Left state pressure.
   * @param rhoR Right state density.
   * @param uR Right state velocity.
   * @param PR Right state pressure.
   * @param rhosol Density solution.
   * @param usol Velocity solution.
   * @param Psol Pressure solution.
   * @param dxdt Point in velocity space where we want to sample the solution.
   * @return Flag signaling whether the left state (-1), the right state (1), or
   * a vacuum state (0) was sampled.
   */
  inline int solve(double rhoL, double uL, double PL, double rhoR, double uR,
                   double PR, double &rhosol, double &usol, double &Psol,
                   double dxdt = 0.) const {

    // get the soundspeeds
    double aL = get_soundspeed(rhoL, PL);
    double aR = get_soundspeed(rhoR, PR);

    // handle vacuum
    if (rhoL == 0. || rhoR == 0.) {
      return solve_vacuum(rhoL, uL, PL, aL, rhoR, uR, PR, aR, rhosol, usol,
                          Psol, dxdt);
    }

    // handle vacuum generation
    if (2. * aL / (_gamma - 1.) + 2. * aR / (_gamma - 1.) <= uR - uL) {
      return solve_vacuum(rhoL, uL, PL, aL, rhoR, uR, PR, aR, rhosol, usol,
                          Psol, dxdt);
    }

    // find the pressure and velocity in the middle state
    // since this is an exact Riemann solver, this is an iterative process,
    // whereby we basically find the root of a function (the Riemann f function
    // defined above)
    // we start by using a Newton-Raphson method, since we do not have an
    // interval in which the function changes sign
    // however, as soon as we have such an interval, we switch to a much more
    // robust root finding method (Brent's method). We do this because the
    // Newton-Raphson method in some cases can overshoot and return a negative
    // pressure, for which the Riemann f function is not defined. Brent's method
    // will never stroll outside of the initial interval in which the function
    // changes sign.
    double Pstar = 0.;
    double Pguess = guess_P(rhoL, uL, PL, aL, rhoR, uR, PR, aR);
    // we only store this variable to store the sign of the function for
    // pressure zero
    // we need to find a larger pressure for which this sign changes to have an
    // interval where we can use Brent's method
    double fPstar = f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pstar);
    double fPguess = f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pguess);
    if (fPstar * fPguess >= 0.) {
      // Newton-Raphson until convergence or until usable interval is
      // found to use Brent's method
      while (std::abs(Pstar - Pguess) > 5.e-9 * (Pstar + Pguess) &&
             fPguess < 0.) {
        Pstar = Pguess;
        fPstar = fPguess;
        Pguess -= fPguess / fprime(rhoL, PL, aL, rhoR, PR, aR, Pguess);
        fPguess = f(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pguess);
      }
    }

    // As soon as there is a suitable interval: use Brent's method
    if (std::abs(Pstar - Pguess) > 5.e-9 * (Pstar + Pguess) && fPguess > 0.) {
      Pstar = solve_brent(rhoL, uL, PL, aL, rhoR, uR, PR, aR, Pstar, Pguess,
                          fPstar, fPguess);
    } else {
      Pstar = Pguess;
    }

    // the middle state velocity is fixed once the middle state pressure is
    // known
    double ustar = 0.5 * (uL + uR) +
                   0.5 * (fb(rhoR, PR, aR, Pstar) - fb(rhoL, PL, aL, Pstar));

    // we now have solved the Riemann problem: we have the left, middle and
    // right state, and this completely fixes the solution
    // we just need to sample the solution for x/t = 0.
    if (ustar < dxdt) {
      // right state
      sample_right_state(rhoR, uR, PR, aR, ustar, Pstar, rhosol, usol, Psol,
                         dxdt);
      return 1;
    } else {
      // left state
      sample_left_state(rhoL, uL, PL, aL, ustar, Pstar, rhosol, usol, Psol,
                        dxdt);
      return -1;
    }
  }
};

#endif // RIEMANNSOLVER_HPP
