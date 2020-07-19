""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
from pyrho import ham, unitary
from pyrho.lib import const, utils


def switching(omega, omega_split):
    '''Switching function that switches smoothly from 1 (at omega=0)
    to 0 (at omega=omega_split)
    '''
    if abs(omega) < omega_split:
        return (1 - (omega/omega_split)**2)**2
    else:
        return 0.0


def partition_specdens(J, omega_split, use_PD=True):
    def Jfast(omega):
        if use_PD:
            pure_dephasing = J(omega)*(abs(omega) < 1e-4)
        else:
            pure_dephasing = 0.
        return (1-switching(omega,omega_split))*J(omega) + pure_dephasing
    def Jslow(omega):
        return switching(omega,omega_split)*J(omega)

    return Jslow, Jfast


def rabi_two_level(ham):
    return 2*np.sqrt( (ham[0,0]-ham[1,1])**2/4.0 + ham[0,1]**2 )/const.hbar


from pyrho.unitary import Unitary
class FrozenModes(Unitary):
    """A FrozenModes class
    """

    def __init__(self, hamiltonian, nmode=300, ntraj=100,
                 dynamics=None, omega_split=None, use_PD=True):
        """Initialize the FrozenModes class.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            An instance of the pyrho Hamiltonian class

        dynamics :
            An instance of any pyrho dynamics class (e.g. Redfield)

        nmode : int
            The number of explicit classical modes to sample

        ntraj : int
            The number of trajectories over which to average

        """
        utils.print_banner("PERFORMING FROZEN MODES DYNAMICS")

        # .copy() is important to have an instance that is not updated by FM sampling
        # FrozenModes.ham.sys must always be the original, dynamics.ham.sys will be
        # updated in each realization
        self.ham = hamiltonian.copy()
        self.nmode = nmode
        self.ntraj = ntraj

        if dynamics is None:
            # Pure frozen modes with unitary dynamics
            self.dynamics = unitary.Unitary(hamiltonian)
        else:
            # "Hybrid" frozen modes with dissipative dynamics
            self.dynamics = dynamics
            if omega_split is None:
                # Determine splitting frequency
                omega_split = list()
                if self.ham.nsite == 2:
                    omega_R = 2*np.sqrt( (ham[0,0]-ham[1,1])**2/4.0 + ham[0,1]**2 )/const.hbar
                else:
                    print("Automated splitting frequency for Nsys > 2 not implemented!")
                    raise SystemExit

                for n in range(self.ham.nbath):
                    omega_split.append( omega_R/4. )

            else:
                assert( len(omega_split) == self.ham.nbath )

            for n in range(self.ham.nbath):
                print("\n--- Splitting energy for bath %d = %0.2lf"%(
                            n, const.hbar*omega_split[n] ))

            for n in range(self.ham.nbath):
                Jslow, Jfast = partition_specdens(self.ham.sd[n].J, omega_split[n], use_PD)
                self.ham.sd[n].J, self.dynamics.ham.sd[n].J = Jslow, Jfast


    def initialize_from_rdm(self, rho):
        """Initialize a trajectory containing the RDM and update the system
        Hamiltonian to reflect sampled, static disorder.
        """
        modes = self.ham.init_classical_modes(self.nmode)
        self.ham.sample_classical_modes(modes)
        q = np.zeros((self.ham.nbath, self.nmode))
        for n in range(self.ham.nbath):
            for k in range(self.nmode):
                q[n,k] = modes[n][k].Q

        # Precompute some "hidden" variables for speed
        self._hamsb = np.array(self.ham.sysbath)
        self._omegasq = np.zeros((self.ham.nbath, self.nmode))
        self._c = np.zeros((self.ham.nbath, self.nmode))
        for n,modes_n in enumerate(modes):
            self._omegasq[n,:] = np.array([mode.omega**2 for mode in modes_n])
            self._c[n,:] = np.array([mode.c for mode in modes_n])

        cq = np.einsum('nk,nk->n',self._c,q)
        frozen_bias = np.einsum('nab,n->ab',self._hamsb,cq)
        # Careful here that self.ham and self.dynamics.ham are not the same object
        ham_biased = self.ham.sys + frozen_bias
        self.dynamics.ham.init_system(ham_biased, is_verbose=False)
        try:
            rho = self.dynamics.initialize_from_rdm(rho)
        except AttributeError:
            pass

        return rho

    def reduce_to_rdm(self, rho):
        try:
            return self.dynamics.reduce_to_rdm(rho)
        except AttributeError:
            return rho

    def propagate_full(self, rho, t_init, t_final, dt):
        return self.dynamics.propagate_full(rho, t_init, t_final, dt)

    def propagate(self, rho_0, t_init, t_final, dt):
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
