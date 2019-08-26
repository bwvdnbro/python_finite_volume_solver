# Finite volume schemes: an introduction

This repository contains:
  - `README.pdf`: An instruction document for a tutorial in which you are guided
    through the development of a 1D finite volume solver in Python.
  - `finite_volume_schemes_part1/2.pdf`: The slides for two guest lectures about
    finite volume schemes that serve as an introduction to the tutorial.
  - `finite_volume_schemes_single_file.pdf`: More compact version of the slides
    above that does not refer to the slides from Moira's fluids course.
  - `multidimensional_finite_volume_schemes.pdf`: Extra slides that explain how
    to go from a 1D finite volume solver to a multidimensional one.
  - `second_order_finite_volume_schemes.pdf`: Extra slides that explain how to
    go from a first order to a second order scheme.
  - `riemannsolver.py`: A standalone exact Riemann solver that can be used as an
    external library for the tutorial.
  - `sodshock_solution.py`: A Python script that returns a reference solution
    for the 1D Sod shock problem. This script can also be used as an external
    library for the tutorial.
  - `LICENSE`: The free software license that allows redistribution of the
    Python scripts mentioned above.
  - `README.md`: This file.
  - `RiemannSolver.hpp`: A C++ version of the Riemann solver, which is identical
    to the Riemann solver in https://github.com/bwvdnbro/CMacIonize.
  - `riemann.f`: A Fortran (77) version of the Riemann solver, kindly provided
    by Kenneth Wood.
  - `example_solutions`: Example solutions for the practical exercise. These are
    only intended as a guidance; you should definitely try to write your own!
  - `Riemann_problem.pdf`: A 59 slide document laying out the detailed solution
    to the general Riemann problem.
