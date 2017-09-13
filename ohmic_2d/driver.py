import sys
import numpy as np
from pyrho import ham, heom, spec

def main():
    args = sys.argv[1:]
    if len(args) != 3:
        print 'usage: L K lioupath'
        sys.exit(1)
    L = int(args[0])
    K = int(args[1])
    lioupath = args[2]
    assert lioupath in ['allese','allesa','allgsb','total']

    nbath = 2

    kB = 0.69352    # in cm-1 / K
    hbar = 5308.8   # in cm-1 * fs

    # System Hamiltonian
    # n = 0 is ground state

    # One-exciton Hamiltonian
    ham_sys_x = np.array([[ -50.,  100.],
                          [  100.,  50.]])
    # One-exciton sys-bath coupling
    nx = ham_sys_x.shape[0]
    ham_sysbath_x = []
    for b in range(nbath):
        ham_sysbath_b = np.zeros((nx,nx))
        for m in range(nx):
            ham_sysbath_b[m,m] = (m==b)
        ham_sysbath_x.append(ham_sysbath_b)
    omega_c = 53.0/hbar # in 1/fs
    #lamda = 1.2*hbar*omega_c
    lamda = 1.2*hbar*omega_c
    kT = kB*77.

    spec_densities = [['ohmic-exp', lamda, omega_c]]*nbath

    # Important: expand the Hilbert space (convert to biexciton space)
    dipole_x = [99, 99] # going to overwrite
    ham_sys, ham_sysbath, dipole = spec.convert_to_xx(ham_sys_x, ham_sysbath_x, dipole_x)
    nsite = ham_sys.shape[0]

    my_ham = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT, hbar=hbar)

    dipole = my_ham.eig2site(
                np.array([[0., 1., 1., 0.],
                          [1., 0., 0., 1.],
                          [1., 0., 0., 1.],
                          [0., 1., 1., 0.]]))

    # Numerical propagation parameters
    t_final, dt = 1000., 10.

    # Waiting time parameters
    T_init, T_final, dT = 10., 626., 615.

    rho_g = np.zeros((nsite,nsite))
    rho_g[0,0] = 1.0

    my_method = heom.HEOM(my_ham, L=L, K=K)
    my_method.write_bath_corr_fn(np.arange(0.0,t_final,dt))
    f_abs = 'abs_HEOM_nopi_dt-%0.0f_tf-%0.0f_L-%d_K-%d.dat'%(dt,t_final,L,K)

    my_spec = spec.Spectroscopy(dipole, my_method)
    omegas, intensities = my_spec.absorption(
                -400., 400., 2., 
                rho_g, t_final, dt)

    with open(f_abs, 'w') as f:
        for (omega, intensity) in zip(omegas, intensities):
            f.write('%0.8f %0.8f\n'%(omega, intensity))

    omega1s, omega3s, t2s, spectra = my_spec.two_dimensional(
                -400., 410., 5.,
                -400., 410., 5.,
                T_init, T_final, dT,
                rho_g, t_final, dt,
                lioupath=lioupath, is_damped=False)

    for t2, spectrum in zip(t2s, spectra):
        f_2d = '2d_%s_HEOM_nopi_t2-%0.1f_dt-%0.0f_tf-%0.0f_L-%d_K-%d.dat'%(lioupath,t2,dt,t_final,L,K)
        with open(f_2d, 'w') as f:
            for w1 in range(len(omega1s)):
                for w3 in range(len(omega3s)):
                    f.write('%0.8f %0.8f %0.8f\n'%(omega1s[w1], omega3s[w3], spectrum[w3,w1]))
                f.write('\n')

if __name__ == '__main__':
    main()
