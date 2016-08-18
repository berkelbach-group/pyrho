import numpy as np

from pyrho import ham, unitary

def main():
    hbar = 5308.8   # in cm-1 * fs

    # System Hamiltonian
    ham_sys = np.array([[100.0,  100.0],
                        [100.0,  0.0]])

    # Initial reduced density matrix of the system
    rho_0 = np.array([[1.0, 0.0],
                      [0.0, 0.0]])

    my_ham = ham.HamiltonianSystem(ham_sys, hbar=hbar)

    my_unitary = unitary.Unitary(my_ham)
    times, rhos_site, rhos_eig = my_unitary.propagate(rho_0, 0.0, 1000.0, 0.5)

    with open('pop_site.dat', 'w') as f:
        for (time, rho_site, rho_eig) in zip(times, rhos_site, rhos_eig):
            f.write('%0.8f %0.8f %0.8f\n'%(time, rho_site[0,0].real, 
                                                 rho_site[1,1].real))


if __name__ == '__main__':
    main()
