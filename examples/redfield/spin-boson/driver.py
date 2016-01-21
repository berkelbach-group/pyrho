import sys
import numpy as np

from pyrho import ham, redfield

def main():
    nsite = 2
    nbath = 1

    # System Hamiltonian
    ham_sys = np.array([[1.0,  1.0],
                        [1.0, -1.0]])

    # System part of the the system-bath interaction
    # - a list of length 'nbath' 
    # - currently assumes that each term has uncorrelated bath operators
    ham_sysbath = []
    ham_sysbath.append(np.array([[1.0,  0.0], 
                                 [0.0, -1.0]]))

    # Initial reduced density matrix of the system
    rho_0 = np.array([[1.0, 0.0],
                      [0.0, 0.0]])

    for alpha in [0.1, 0.2, 0.4]: 
        # Spectral densities - a list of length 'nbath'
        omega_c = 7.5
        lamda = alpha*omega_c/2.0
        spec_densities = [['ohmic-exp', lamda, omega_c]]*nbath
        kT = 0.2

        my_ham = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT)

        for method in ['Redfield', 'TCL2', 'TC2']:
            if method == 'Redfield':
                is_secular = True
            else:
                is_secular = False
            my_redfield = redfield.Redfield(my_ham, method=method, is_secular=is_secular)
            times, rhos_site, rhos_eig = my_redfield.propagate(rho_0, 0.0, 14.0, 0.05)

            with open('pop_site_%s_alpha-%0.2f.dat'%(method,alpha), 'w') as f:
                for (time, rho_site, rho_eig) in zip(times, rhos_site, rhos_eig):
                    f.write('%0.8f %0.8f %0.8f\n'%(time, rho_site[0,0].real, 
                                                         rho_site[1,1].real))

if __name__ == '__main__':
    main()
