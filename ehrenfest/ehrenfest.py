""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
from pyrho.integrate import Integrator
from pyrho.lib import const, utils

from pyrho.unitary import Unitary
class Ehrenfest(Unitary):
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
        ham_t = np.einsum('nab,n->ab',self._hamsb,cq)
        ham_t = self.ham.to_interaction(self.ham.site2eig(ham_t), t)

        hamsb_int = list()
        for n, hamsb in enumerate(self._hamsb):
            hamsb_int_n = self.ham.to_interaction(self.ham.site2eig(hamsb), t)
            hamsb_int.append(hamsb_int_n)
        hamsb_int = np.array(hamsb_int)
        F_avg = np.einsum('nab,ba->n',hamsb_int,rho).real

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

    def act_from_left(self, op, rho_bath):
        rho, q, p = self.unpack(rho_bath)
        op_rho = np.dot(op, rho)
        return self.pack(op_rho, q, p)

    def act_from_right(self, op, rho_bath):
        rho, q, p = self.unpack(rho_bath)
        rho_op = np.dot(rho, op)
        return self.pack(rho_op, q, p)

    def initialize_from_rdm(self, rho):
        """Initialize a trajectory containing the RDM and bath modes.

        Returns
        -------
        rho_bath : (nsite*nsite + nmode + nmode) ndarray
            The RDM and each mode's Q and P.

        """
        modes = self.ham.init_classical_modes(self.nmode)
        self.ham.sample_classical_modes(modes)
        q = np.zeros((self.ham.nbath, self.nmode))
        p = np.zeros((self.ham.nbath, self.nmode))
        for n in range(self.ham.nbath):
            for k in range(self.nmode):
                q[n,k] = modes[n][k].Q
                p[n,k] = modes[n][k].P
        rho_bath = self.pack(rho.copy(), q, p)

        # Precompute some "hidden" variables for speed
        self._hamsb = np.array(self.ham.sysbath)
        self._omegasq = np.zeros((self.ham.nbath, self.nmode))
        self._c = np.zeros((self.ham.nbath, self.nmode))
        for n,modes_n in enumerate(modes):
            self._omegasq[n,:] = np.array([mode.omega**2 for mode in modes_n])
            self._c[n:] = np.array([mode.c for mode in modes_n])

        return rho_bath

    def reduce_to_rdm(self, rho_bath_traj):
        rho_traj = list()
        for rho_bath in rho_bath_traj:
            rho_t, q, p = self.unpack(rho_bath)
            rho_traj.append(rho_t)
        return np.array(rho_traj)

    def propagate_full(self, rho_bath, t_init, t_final, dt):
        times = np.arange(t_init, t_final, dt)
        def deriv_fn(t,y):
            return self.deriv(t,y)

        rho_0, q, p = self.unpack(rho_bath)
        rho_int_0 = self.ham.to_interaction(self.ham.site2eig(rho_0), t_init)
        integrator = Integrator('ODE', dt, deriv_fn=deriv_fn)
        integrator.set_initial_value(self.pack(rho_int_0,q,p), t_init)

        rho_bath_traj = list()
        for time in times:
            rho_int_t, qt, pt = self.unpack(np.array(integrator.y))
            rho_t = self.ham.eig2site(self.ham.from_interaction(rho_int_t, time))
            rho_bath_traj.append(self.pack(rho_t, qt, pt))
            integrator.integrate()
        
        return times, np.array(rho_bath_traj)

    def propagate(self, rho_0, t_init, t_final, dt):
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

        Returns
        -------
        times : list of floats
            The times at which the RDM has been calculated.
        rhos_site : 3D ndarray
            The RDM at each time in the site basis.

        """
        times = np.arange(t_init, t_final, dt)
        rhos_site_avg = np.zeros((len(times),self.ham.nsite,self.ham.nsite), dtype=np.complex) 
        for trajectory in range(self.ntraj):
            rho_bath = self.initialize_from_rdm(rho_0)
            times, rhos_bath = self.propagate_full(rho_bath, t_init, t_final, dt)
            rhos_site = self.reduce_to_rdm(rhos_bath)
            rhos_site_avg += np.array(rhos_site)

        return times, rhos_site_avg/self.ntraj
