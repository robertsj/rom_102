# Reduced-Order Model Mini-Series: ROM 102

Jeremy Roberts
Associate Professor
Alan Levin Department of Mechanical and Nuclear Engineering
Kansas State University


## Abstract

In this second half of a two-seminar series on reduced-order modeling, the basic ingredients for POD-Galerkin from ROM 101 will be reviewed and applied to a simple, full-order model (FOM) based on transient heat conduction.  We’ll investigate the computational cost of POD-Galerkin, including the offline costs to produce the POD basis and the online costs associated with solving the projected system.  For more challenging (and realistic!) problems exhibiting nonlinearities (e.g., a temperature-dependent conductivity),  the online cost can become prohibitively expensive due to repeated projections at each new state point.  The discrete empirical interpolation method (DEIM) will be illustrated as one effective way to mitigate the cost due to nonlinearities for FOMs with explicit (matrix) representations.  Effective treatment of nonlinearities in “black-box” FOMs in which operators are available only as functions remains an active area of research.  All materials for this talk will be available at [https://github.com/robertsj/rom_102](https://github.com/robertsj/rom_102).

## Bio

Dr. Jeremy Roberts is an Associate Professor and Steve Hsu Keystone Research Scholar in the Alan Levin Department of Mechanical and
Nuclear Engineering Department at Kansas State University. Dr. Roberts earned his PhD from MIT and BS/MS from the University of Wisconsin, all in nuclear engineering.  His research focuses on computational methods for reactor physics and neutron transport, including reduced-order modeling techniques to support uncertainty quantification, optimization, and acceleration.



