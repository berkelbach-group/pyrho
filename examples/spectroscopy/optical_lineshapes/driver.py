import numpy as np
from pyrho import ham, redfield, heom, spec

def main():
    nsite = 3
    nbath = 2

    eps = 0.

    # System Hamiltonian
    # n = 0 is ground state
    ham_sys = np.array([[ 0.,  0.,  0.],
                        [ 0., eps, -1.],
                        [ 0., -1., eps]])

    # System part of the the system-bath interaction
    # - a list of length 'nbath'
    # - currently assumes that each term has uncorrelated bath operators
    ham_sysbath = []
    for n in range(1,nsite):
        ham_sysbath_n = np.zeros((nsite,nsite))
        ham_sysbath_n[n,n] = 1.0
        ham_sysbath.append( ham_sysbath_n )

    rho_g = np.zeros((nsite,nsite))
    rho_g[0,0] = 1.0

    dipole = np.array([[ 0.,  1.,  1.],
                       [ 1.,  0.,  0.],
                       [ 1.,  0.,  0.]])

    lamda = 1./2
    for omega_c in [1.0]:
        for beta in [1.0, 3.0]:
            kT = 1./beta
            # Spectral densities - a list of length 'nbath'
            spec_densities = [['ohmic-lorentz', lamda, omega_c]]*nbath

            my_ham = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT)

            for method in ['HEOM', 'TL', 'TNL']:
                if method == 'HEOM':
                    t_final = 50.0
                    for L in [1,2,4,18]:
                        for K in [2,4]:
                            my_method = heom.HEOM(my_ham, L=L, K=K)
                            my_spec = spec.Spectroscopy(dipole, my_method)
                            omegas, intensities = my_spec.absorption(
                                        -4.+eps, 4.+eps, 0.02, 
                                        rho_g, 0., t_final, 0.05)

                            with open('abs_omegac-%0.1f_beta-%0.1f_HEOM_L-%d_K-%d.dat'%(omega_c,beta,L,K), 'w') as f:
                                for (omega, intensity) in zip(omegas, intensities):
                                    f.write('%0.8f %0.8f\n'%(omega-eps, intensity))
                else:                
                    if method == 'TL':
                        my_method = redfield.Redfield(my_ham, method='TCL2')
                        t_final = 50.0
                    else:
                        my_method = redfield.Redfield(my_ham, method='TC2')
                        t_final = 100.0
            
                    my_spec = spec.Spectroscopy(dipole, my_method)
                    omegas, intensities = my_spec.absorption(-4.+eps, 4.+eps, 0.02, 0., t_final, 0.05)

                    with open('abs_omegac-%0.1f_beta-%0.1f_%s.dat'%(omega_c,beta,method), 'w') as f:
                        for (omega, intensity) in zip(omegas, intensities):
                            f.write('%0.8f %0.8f\n'%(omega-eps, intensity))

if __name__ == '__main__':
    main()
