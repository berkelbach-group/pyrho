pyrho: a python package for reduced density matrix techniques
==============================================================

Authors
-------
Timothy Berkelbach

Instructions
------------
The parent directory of `pyrho` must be in `$PYTHONPATH`, enabling
    `import pyrho`.

Contents
--------
* README.md

* ehrenfest/

    Module for classically-sampled, trajectory-based Ehrenfest dynamics

* ham/

    Module for system-bath Hamiltonians

* heom/

    Module for Hierarchical Equations of Motion

* integrate/
    
    Module for generic RK4-style time-dependent integrators

* lib/
    
    Module for miscellaneous tools, including physical constants
        and utility functions

* redfield/
    
    Module for Redfield and related second-order dynamical perturbation
        theories in the system-bath coupling strength

* spec/

    Module for spectroscopies, based on any underlying Dynamics class
        with a Dynamics.propagate() method

* examples/
    
    Example driver scripts for setting up a Hamiltonian and calculating
        reduced system dynamics

To-do
-----
- [ ] Spectroscopy
  - [x] Linear 
  - [ ] Nonlinear
- [ ] NIBA and related polaron-based QMEs
- [ ] Exact Harmonic mode(s) in the system Hamiltonian
  - [ ] Reaction-coordinate mapping
- [ ] HEOM
  - [x] Ohmic-Lorentz (Debye)
  - [ ] Pade decomposition
  - [ ] Single oscillator
- [x] Ehrenfest
  - [ ] Optimization
  - [ ] Persistent bath variables in propagate()

