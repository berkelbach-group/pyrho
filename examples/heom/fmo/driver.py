import sys
import numpy as np

sys.path.append('/home/tberk/utils/python_modules/QuantumDynamics/')
from quantumdynamics.hamiltonian import Hamiltonian
from quantumdynamics.heom import HEOM

def main():
    nsite = 7
    nbath = 7

    kB = 0.69352    # in cm-1 / K
    hbar = 5308.8   # in cm-1 * fs

    # System Hamiltonian in cm-1
    ham_sys = np.array([[12410, -87.7,   5.5,  -5.9,   6.7, -13.7,  -9.9],
                        [-87.7, 12530,  30.8,   8.2,   0.7,  11.8,   4.3],
                        [  5.5,  30.8, 12210, -53.5,  -2.2,  -9.6,   6.0],
                        [ -5.9,   8.2, -53.5, 12320, -70.7, -17.0, -63.3],
                        [  6.7,   0.7,  -2.2, -70.7, 12480,  81.1,  -1.3],
                        [-13.7,  11.8,  -9.6, -17.0,  81.1, 12630,  39.7],
                        [ -9.9,   4.3,   6.0, -63.3,  -1.3,  39.7, 12440]])

    ham_sysbath = []
    for n in range(nbath):
        ham_sysbath_n = np.zeros((nsite,nsite))
        ham_sysbath_n[n,n] = 1.0
        ham_sysbath.append( ham_sysbath_n )

    K = 0
    for L in [1,2,4]:
        lamda = 35.0
        #tau = 50. # in fs
        tau = 166. # in fs
        omega_c = 1.0/tau
        for T in [300.]:
            kT = kB*T
            spec_densities = [['ohmic-lorentz', lamda, omega_c]]*nbath

            my_ham = Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT, hbar=hbar)
            my_heom = HEOM(my_ham, L=L, K=K)

            for init in [6]:
                # Initial reduced density matrix of the system
                rho_0 = np.zeros((nsite, nsite))
                rho_0[init-1,init-1] = 1.0
                times, rhos_site, rhos_eig = my_heom.propagate(rho_0, 
                                                0.0, 1000.0, 1.0)

                with open('pop_site_T-%.0f_init-%d_tau-%.0f_L-%d_K-%d.dat'
                           %(T,init,tau,L,K), 'w') as f:
                    for (time, rho_site) in zip(times, rhos_site):
                        f.write('%0.8f '%(time))
                        for i in range(nsite):
                            f.write('%0.8f '%(rho_site[i,i].real))
                        f.write('\n')

if __name__ == '__main__':
    main()
