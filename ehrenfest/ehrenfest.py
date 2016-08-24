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
        rho, q, p = self.unpack(yt)

        cq = np.einsum('nk,nk->n',self._c,q)
        ham_t = self.ham.sys + np.einsum('nab,n->ab',self._hamsb,cq)
        F_avg = np.einsum('nab,ba->n',self._hamsb,rho).real

        dq = p.copy()
        dp = - self._omegasq*q \
             - np.einsum('nk,n->nk',self._c,F_avg) 

        drho = -1j/const.hbar * utils.commutator(ham_t,rho)

        return self.pack(drho, dq, dp)

    def pack(self, rho, q, p):
        yt = np.concatenate((utils.to_liouville(rho), 
                             q.flatten(), p.flatten())) 
        return yt

    def unpack(self, yt):
        rho = utils.from_liouville(yt[:self.ham.nsite**2], self.ham.nsite)
        qp = yt[self.ham.nsite**2:].real
        q, p = qp[:self.ham.nbath*self.nmode], qp[self.ham.nbath*self.nmode:]
        q = q.reshape(self.ham.nbath,self.nmode)
        p = p.reshape(self.ham.nbath,self.nmode)
        return rho, q, p

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
        self._hamsb = np.array(self.ham.sysbath)
        self._omegasq = np.zeros((self.ham.nbath,self.nmode))
        self._c = np.zeros((self.ham.nbath,self.nmode))
        for n,modes_n in enumerate(self.modes):
            self._omegasq[n,:] = np.array([mode.omega**2 for mode in modes_n])
            self._c[n:] = np.array([mode.c for mode in modes_n])

        def deriv_fn(t,y):
            return self.deriv(t,y)

        rhos_site_avg = np.zeros((len(times),self.ham.nsite,self.ham.nsite), dtype=np.complex) 
        rhos_eig_avg = np.zeros((len(times),self.ham.nsite,self.ham.nsite), dtype=np.complex) 
        for trajectory in range(self.ntraj):
            self.ham.sample_classical_modes(modes)
            q = np.zeros((self.ham.nbath, self.nmode))
            p = np.zeros((self.ham.nbath, self.nmode))
            for n in range(self.ham.nbath):
                for k in range(self.nmode):
                    q[n,k] = modes[n][k].Q
                    p[n,k] = modes[n][k].P

            integrator = Integrator('ODE', dt, deriv_fn=deriv_fn)
            integrator.set_initial_value(self.pack(rho_0,q,p), t_init)

            rhos_site = []
            rhos_eig = []
            for time in times:
                # Retrieve data from integrator
                rho_site, q, p = self.unpack(integrator.y)

                # Collect results
                rhos_site.append(rho_site)
                rhos_eig.append(self.ham.site2eig(rho_site))

                # Propagate one timestep
                integrator.integrate()

            # Remember: rhos_site is a Python list, not a numpy array
            rhos_site_avg += np.array(rhos_site)/self.ntraj
            rhos_eig_avg += np.array(rhos_eig)/self.ntraj

        if is_verbose:
            print "\n--- Finished performing RDM dynamics"
        
        # Return as a list of 2D ndarrays
        return times, [x for x in rhos_site_avg], [x for x in rhos_eig_avg]
