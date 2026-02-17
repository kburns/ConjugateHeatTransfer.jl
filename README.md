## ConjugateHeatTransfer.jl

This repository contains Julia scripts and example notebooks related to the paper ["Steady advection-diffusion in multiply connected potential flows"](https://doi.org/10.1017/jfm.2025.11106).

The paper studies steady heat transfer between multiple impermeable obstacles in a two-dimensional incompressible potential flow, with prescribed (possibly non-uniform) boundary temperature distributions.
The problem is solved in two stages.
First, the multiply connected potential flow problem is solved using rational approximations via the AAA-LS algorithm.
The potential flow solution is then used as a multiply connected conformal map, transforming the domain into a constant-coefficient problem in streamline coordinates where each body appears as a horizontal slit.
The heat transfer problem is then solved in this domain using a mixed single-plus-double-layer boundary integral formulation.
This procedures enables the computation of temperature fields and heat fluxes to very high precision for many configurations.

**Reference**: [K. I. McKee and K. J. Burns, "Steady advection-diffusion in multiply connected potential flows," Journal of Fluid Mechanics, vol. 1029, p. A10, 2026.](https://doi.org/10.1017/jfm.2025.11106)