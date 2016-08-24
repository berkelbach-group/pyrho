import numpy as np

from pyrho import ham, hybrid, frozen, redfield

def main():
    nbath = 2

    kB = 0.69352    # in cm-1 / K
    kT = kB*300.0
    hbar = 5308.8   # in cm-1 * fs

    # System Hamiltonian
    ham_sys = np.array([[100.0,  100.0],
                        [100.0,  0.0]])

    # System part of the the system-bath interaction
    # - a list of length 'nbath'
    # - currently assumes that each term has uncorrelated bath operators
    ham_sysbath = []
    ham_sysbath.append(np.array([[1.0, 0.0], 
                                 [0.0, 0.0]]))
    ham_sysbath.append(np.array([[0.0, 0.0], 
                                 [0.0, 1.0]]))

    # Initial reduced density matrix of the system
    rho_0 = np.array([[1.0, 0.0],
                      [0.0, 0.0]])

    for lamda in [100./50, 100./5, 100., 500.]: 
        for tau_c in [100., 500.]:
            omega_c = 1.0/tau_c # in 1/fs

            # Spectral densities - a list of length 'nbath'
            spec_densities = [['ohmic-lorentz', lamda, omega_c]]*nbath

            #TODO(TCB): Make this cleaner. Write a Hamiltonian.copy() method?
            my_ham = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT, hbar=hbar)
            my_ham_slow = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT, hbar=hbar)
            my_ham_fast = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT, hbar=hbar)

            ntraj = int(1e3)
            my_frozen = frozen.FrozenModes(my_ham_slow, nmode=300, ntraj=ntraj)
            times, rhos_site, rhos_eig = my_frozen.propagate(rho_0, 0.0, 1000.0, 1.0)
            my_redfield = redfield.Redfield(my_ham_fast, method='Redfield')

            my_hybrid = hybrid.Hybrid(my_ham, my_frozen, my_redfield, omega_split=None)
            times, rhos_site, rhos_eig = my_hybrid.propagate(rho_0, 0.0, 1000.0, 1.0)

            with open('pop_site_tau-%0.1f_lam-%0.2f_ntraj-%d.dat'%(tau_c,lamda,ntraj), 'w') as f:
                for (time, rho_site, rho_eig) in zip(times, rhos_site, rhos_eig):
                    f.write('%0.8f %0.8f %0.8f\n'%(time, rho_site[0,0].real, 
                                                         rho_site[1,1].real))


if __name__ == '__main__':
    main()
