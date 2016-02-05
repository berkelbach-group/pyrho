""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
from pyrho.lib import const, utils

class Spectroscopy(object):

    def __init__(self, dipole_site, dynamics):
        # dynamics instance must have a dynamics.propagate() method
        self.dipole_site = dipole_site
        self.dynamics = dynamics

    def absorption(self, e_min, e_max, de, rho_0, t_init, t_final, dt):

        utils.print_banner("CALCULATING LINEAR ABSORPTION SPECTRUM")

        energies = np.arange(e_min, e_max, de)
        omegas = energies/const.hbar

        mu = self.dipole_site
        mu_rho_0 = np.dot(mu, rho_0)
        times, mu_rho, x = self.dynamics.propagate(mu_rho_0, t_init, t_final, dt)
        fourier_transform = np.zeros(len(omegas))
        dipole_file = open('dipole_dipole.dat','w')
        for t in range(len(times)):
            weight = dt - 0.5*dt*(t==0 or t==len(times)-1)
            time = times[t]
            Ct = np.trace(np.dot(mu,mu_rho[t]))
            dipole_file.write('%0.6f %0.6f %0.6f\n'%(time, Ct.real, Ct.imag))
            for w in range(len(omegas)):
                omega = omegas[w]
                fourier_transform[w] += (weight*np.exp(1j*omega*time)*Ct).real

        dipole_file.close()
        return energies, fourier_transform

    def absorption_fft(self, e_min, e_max, de):
        [w_min, w_max, dw] = np.array([e_min, e_max, de])/const.hbar
        n_requested = int((e_max-e_min)/de) + 1

        w_lim = max(abs(w_min),abs(w_max))
        freqs = np.arange(-w_lim, w_lim+dw, dw)
        n_fft = freqs.size
        dt = 2*np.pi/(dw*n_fft)
        t_init = 0.0 
        t_final = dt*n_fft

        ns = self.ham.nsite
        dipoles = np.zeros(freqs.size, dtype=np.complex_)
        intensities = np.zeros(freqs.size, dtype=np.complex_)
        for i in range(ns):
            for j in range(i):
                rho_0 = np.zeros((ns,ns))
                rho_0[i,j] = self.dipole_site[i,j]

                times, rhos_site, xxx = self.propagate(rho_0, t_init, t_final, dt)
                dipoles = [rhos_site[t][i,j] for t in range(n_fft)]
                intensities += self.dipole_site[i,j]*np.fft.fft(dipoles)*dt
                
        freqs_fft = np.fft.fftfreq(n_fft)*2*np.pi/dt

        # Sort and truncate into requested [e-min, e_max]
        idx = freqs_fft.argsort()
        energies_fft = const.hbar*freqs_fft[idx]
        intensities = np.real(intensities[idx])
        if energies_fft[-1] > e_max:
            return energies_fft[1:n_requested+1], intensities[1:n_requested+1]
        else:
            return energies_fft[n_fft-n_requested:], intensities[n_fft-n_requested:]

