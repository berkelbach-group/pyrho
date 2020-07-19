""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
from pyrho.lib import utils

class Unitary(object):
    """A unitary evolution class
    """

    def __init__(self, hamiltonian, is_verbose=True):
        """Initialize the Unitary evolution class. 

        Parameters
        ----------
        hamiltonian : HamiltonianSystem
            An instance of the pyrho HamiltonianSystem class.
        """
        if is_verbose:
            utils.print_banner("PERFORMING UNITARY DYNAMICS")

        self.ham = hamiltonian

    def setup(self):
        pass

    def propagate_full(self, rho, t_init, t_final, dt, is_verbose=False):
        return self.propagate(rho, t_init, t_final, dt, is_verbose)

    def propagate(self, rho_0, t_init, t_final, dt, is_verbose=False):
        """Propagate the RDM according to Unitary dynamics.

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

        """
        times = np.arange(t_init, t_final, dt)
        rho_int_0 = self.ham.to_interaction(self.ham.site2eig(rho_0), t_init)

        rhos_site = list()
        for time in times:
            # Note: In the interaction picture, rho(t) = rho(0), so just "come out" of the
            # interaction picture to do Schrodinger time evolution.
            rho_t = self.ham.eig2site(self.ham.from_interaction(rho_int_0, time))
            rhos_site.append(rho_t)

        if is_verbose:
            print("\n--- Finished performing RDM dynamics")
        
        return times, rhos_site

