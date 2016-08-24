import numpy as np
from pyrho import ham, heom, spec

def main():
    nbath = 2

    kB = 0.69352    # in cm-1 / K
    hbar = 5308.8   # in cm-1 * fs

    # System Hamiltonian
    # n = 0 is ground state

    # One-exciton Hamiltonian
    ham_sys_x = np.array([[ -50., -100.],
                          [ -100.,  50.]])
    # One-exciton sys-bath coupling
    nx = ham_sys_x.shape[0]
    ham_sysbath_x = []
    for b in range(nbath):
        ham_sysbath_b = np.zeros((nx,nx))
        for m in range(nx):
            ham_sysbath_b[m,m] = (m==b)
        ham_sysbath_x.append(ham_sysbath_b)
    # One-exciton dipole moments
    dipole_x = np.array([ 1., -0.2])

    print "ham_sys_x ="
    print ham_sys_x
    print "ham_sysbath_x ="
    print ham_sysbath_x
    print "dipole_x ="
    print dipole_x

    ham_sys, ham_sysbath, dipole = spec.convert_to_xx(ham_sys_x, ham_sysbath_x, dipole_x)
    nsite = ham_sys.shape[0]

    print "ham_sys ="
    print ham_sys
    print "ham_sysbath ="
    print ham_sysbath
    print "dipole ="
    print dipole

    lamda = 60.
    omega_c = 1./100. # in 1/fs
    kT = kB*77.

    spec_densities = [['ohmic-lorentz', lamda, omega_c]]*nbath

    my_ham = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT, hbar=hbar)

    dt = 10.0
    t_final = 500.0

    rho_g = np.zeros((nsite,nsite))
    rho_g[0,0] = 1.0

    for K in [0,1]:
        for L in [1,2,3]:
            my_method = heom.HEOM(my_ham, L=L, K=K)
            my_spec = spec.Spectroscopy(dipole, my_method)

            omegas, intensities = my_spec.absorption(
                        -400., 400., 2., 
                        rho_g, 0., t_final, dt)

            with open('abs_HEOM_dt-%0.0f_tf-%0.0f_L-%d_K-%d.dat'%(dt,t_final,L,K), 'w') as f:
                for (omega, intensity) in zip(omegas, intensities):
                    f.write('%0.8f %0.8f\n'%(omega, intensity))

            omega1s, omega3s, t2s, spectra = my_spec.two_dimensional(
                        -400., 400., 10.,
                        -400., 400., 10.,
                        0., 700.0, 100.0,
                        rho_g, 0., t_final, dt)

            omega1s, omega3s, t2s, spectra = my_spec.two_dimensional(
                        120., 120, 10.,
                        120., 120., 10.,
                        0., 700.0, 100.0,
                        rho_g, 0., t_final, dt)

            for t2, spectrum in zip(t2s, spectra):
                with open('2d_t2-%0.1f_HEOM_dt-%0.0f_tf-%0.0f_L-%d_K-%d.dat'%(t2,dt,t_final,L,K), 'w') as f:
                    for w1 in range(len(omega1s)):
                        for w3 in range(len(omega3s)):
                            f.write('%0.8f %0.8f %0.8f\n'%(omega1s[w1], omega3s[w3], spectrum[w3,w1]))
                        f.write('\n')

if __name__ == '__main__':
    main()
