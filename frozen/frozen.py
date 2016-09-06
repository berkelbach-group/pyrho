""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
from pyrho import ham, unitary
from pyrho.lib import const, utils

from pyrho.unitary import Unitary
class FrozenModes(Unitary):
    """A FrozenModes class
    """

    def __init__(self, hamiltonian, nmode=300, ntraj=100):
        """Initialize the FrozenModes class.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            An instance of the pyrho Hamiltonian class

        nmode : int
            The number of explicit classical modes

        ntraj : int
            The number of trajectories over which to average

        """
        utils.print_banner("PERFORMING FROZEN MODES DYNAMICS")

        self.ham = hamiltonian
        self.nmode = nmode
        self.ntraj = ntraj

    def propagate(self, rho_0, t_init, t_final, dt, is_verbose=True, dynamics=None):
        """Propagate the RDM according to Frozen Modes dynamics.

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

        rhos_site_avg = np.zeros((len(times),self.ham.nsite,self.ham.nsite), dtype=np.complex) 
        rhos_eig_avg = np.zeros((len(times),self.ham.nsite,self.ham.nsite), dtype=np.complex) 
        for trajectory in range(self.ntraj):
            self.ham.sample_classical_modes(modes)
            q = np.zeros((self.ham.nbath, self.nmode))
            for n in range(self.ham.nbath):
                for k in range(self.nmode):
                    q[n,k] = modes[n][k].Q

            cq = np.einsum('nk,nk->n',self._c,q)
            frozen_bias = np.einsum('nab,n->ab',self._hamsb,cq)
            ham_biased = self.ham.sys + frozen_bias

            #dynamics_is_verbose = True
            dynamics_is_verbose = False
            if dynamics is None:
                ham_traj = ham.HamiltonianSystem(ham_biased, const.hbar, is_verbose=dynamics_is_verbose)
                dynamics = unitary.Unitary(ham_traj, is_verbose=dynamics_is_verbose)
            else:
                dynamics.ham.init_system(ham_biased, is_verbose=dynamics_is_verbose)
            dynamics.is_verbose = dynamics_is_verbose
            dynamics.setup()
            xtimes, rhos_site, rhos_eig = dynamics.propagate(rho_0, t_init, t_final, dt, 
                                                             is_verbose=dynamics_is_verbose)

            rhos_site_avg += np.array(rhos_site)/self.ntraj
            rhos_eig_avg += np.array(rhos_eig)/self.ntraj

        if is_verbose:
            print "\n--- Finished performing RDM dynamics"
        
        return times, rhos_site_avg, rhos_eig_avg

