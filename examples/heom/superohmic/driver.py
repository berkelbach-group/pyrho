import numpy as np

from pyrho import ham, heom

def main():
    nsite = 2
    nbath = 1
    
    V = 0.5
    eps = 1 * V

    kT = V / 0.5 

    # System Hamiltonian
    ham_sys = np.array([[eps,   V ],
                        [ V,  -eps]])

    # System part of the the system-bath interaction
    # - a list of length 'nbath'
    # - currently assumes that each term has uncorrelated bath operators
    ham_sysbath = []
    ham_sysbath.append(np.array([[1.0, 0.0], 
                                 [0.0, -1.0]]))

    # Initial reduced density matrix of the system
    rho_0 = np.array([[1.0, 0.0],
                      [0.0, 0.0]])

    eta = 6 * V 
    omega_c = 2 * V
    lamda = eta * omega_c / (3.0 * 3.1416) * 2 
    # Spectral densities - a list of length 'nbath'
    spec_densities = [['cubic-exp', lamda, omega_c]]*nbath

    my_ham = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT)

    for K in [0]:
        for L in [1,2]:
            my_heom = heom.HEOM(my_ham, L=L, K=K, L_truncation='TL')
            times, rhos_site = my_heom.propagate(rho_0, 0.0, 10.0/V, 0.01/V)

            with open('pop_site_lam-%0.2f_L-%d_K-%d.dat'%(lamda,L,K), 'w') as f:
                for (time, rho_site) in zip(times, rhos_site):
                    f.write('%0.8f %0.8f %0.8f\n'%(V*time, rho_site[0,0].real, 
                                                         rho_site[1,1].real))

if __name__ == '__main__':
    main()
