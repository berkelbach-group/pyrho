pyrho: a python package for reduced density matrix techniques
==============================================================
[![DOI](https://zenodo.org/badge/50130882.svg)](https://zenodo.org/badge/latestdoi/50130882)

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

* tcf/

    Module for equilibrium time correlation functions, currently only 
        works for HEOM (in v1.0)

* examples/
    
    Example driver scripts for setting up a Hamiltonian and calculating
        reduced system dynamics

