""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
from pyrho.integrate import Integrator
from pyrho.lib import const, utils

class Ehrenfest(object):
    """An Ehrenfest (mean-field) class
    """

    def __init__(self, hamiltonian, nmode=300, ntraj=100):
        """Initialize the Ehrenfest class.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            An instance of the pyrho Hamiltonian class.

        nmode : int
            The number of explicit classical modes

        ntraj : int
            The number of trajectories over which to average

        """
        utils.print_banner("PERFORMING EHRENFEST DYNAMICS")

        self.ham = hamiltonian
        self.nmode = nmode
        self.ntraj = ntraj

    def deriv(self, t, yt):
        rho, qps = self.unpack(yt)

        ham_t = self.ham.sys.copy()
        dqps = np.zeros_like(qps)

        for n,modes_n in enumerate(self.modes):
            q, p = qps[n,:,:].T
            omegasq = self._omegasq[n]
            c = self._c[n]
            Fn = self.ham.sysbath[n]    # this is like |n><n|
            ham_t += Fn*np.dot(c, q)
            F_avg = np.trace(np.dot(Fn,rho)).real
            dq = p
            dp = -omegasq*q - c*F_avg
            dqps[n,:,0] = dq 
            dqps[n,:,1] = dp 

        drho = -1j/const.hbar * utils.commutator(ham_t,rho)

        return self.pack(drho, dqps)

    def pack(self, rho, qps):
        yt = np.concatenate((utils.to_liouville(rho), qps.flatten())) 
        return yt

    def unpack(self, yt):
        rho = utils.from_liouville(yt[:self.ham.nsite**2])
        qps = yt[self.ham.nsite**2:].reshape(self.ham.nbath,self.nmode,2).real
        return rho, qps

    def propagate(self, rho_0, t_init, t_final, dt, is_verbose=True):
        """Propagate the RDM according to Ehrenfest dynamics.

        Parameters
        ----------
        rho_0 : np.array
            The initial RDM.
        t_init : float
            The initial time.
        t_final : float
            The final time.
        dt : float
            The timestep.
        is_verbose : bool
            Flag to indicate verbose printing.

        Returns
        -------
        times : list of floats
            The times at which the RDM has been calculated.
        rhos_site : list of np.arrays
            The RDM at each time in the site basis.
        rhos_eig : list of np.arrays
            The RDM at each time in the system eigen-basis.

        """
        times = np.arange(t_init, t_final, dt)
        modes = self.modes = self.ham.init_classical_modes(self.nmode)
        self._omegasq = np.zeros((self.ham.nbath,self.nmode))
        self._c = np.zeros((self.ham.nbath,self.nmode))
        for n,modes_n in enumerate(self.modes):
            self._omegasq[n,:] = np.array([mode.omega**2 for mode in modes_n])
            self._c[n:] = np.array([mode.c for mode in modes_n])

        def deriv_fn(t,y):
            return self.deriv(t,y)

        rhos_site_avg = []
        for trajectory in range(self.ntraj):
            self.ham.sample_classical_modes(modes)
            qps = np.zeros((self.ham.nbath, self.nmode, 2))
            for n in range(self.ham.nbath):
                for k in range(self.nmode):
                    qps[n,k,:] = modes[n][k].Q, modes[n][k].P

            integrator = Integrator('ODE', dt, deriv_fn=deriv_fn)
            integrator.set_initial_value(self.pack(rho_0,qps), t_init)

            rhos_site = []
            while integrator.t < t_final+1e-8:
                # Retrieve data from integrator
                rho_site, qkpk = self.unpack(integrator.y)
                #print integrator.t, qkpk[0,0,0]

                # Collect results
                rhos_site.append(rho_site)

                # Propagate one timestep
                integrator.integrate()

            # Collect RDM for averaging
            # Remember: rhos_site is a Python list, not a numpy array
            if trajectory == 0:
                for rho_site in rhos_site:
                    rhos_site_avg.append(rho_site/self.ntraj)
            else:
                for rho_site_avg, rho_site in zip(rhos_site_avg, rhos_site):
                    rho_site_avg += rho_site/self.ntraj

        rhos_eig_avg = []
        for rho_site_avg in rhos_site_avg:
            rhos_eig_avg.append(self.ham.site2eig(rho_site_avg))

        if is_verbose:
            print "\n--- Finished performing RDM dynamics"
        
        return times, rhos_site_avg, rhos_eig_avg

