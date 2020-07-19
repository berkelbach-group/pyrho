""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
from pyrho.integrate import Integrator
from pyrho.lib import const, utils

from pyrho.unitary import Unitary
class Redfield(Unitary):
    """A weak system-bath coupling (Redfield-like) class
    
    Follows Pollard and Friesner, J. Chem. Phys. 100, 5054 (1994);
            Add references here for TCL2 and TC2.

    """

    def __init__(self, hamiltonian, method='Redfield', is_secular=False, is_verbose=False):
        """Initialize the Redfield class.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            An instance of the pyrho Hamiltonian class.
        method : str
            The weak-coupling method, either 'Redfield', 'TCL2', or 'TC2'.
        is_secular : bool
            Flag to indicate the secular approximation.

        """
        assert(method in ['Redfield', 'TCL2', 'TC2'])
        if is_verbose:
            utils.print_banner("PERFORMING RDM DYNAMICS WITH METHOD = %s"%(method))

        self.ham = hamiltonian
        self.method = method
        self.is_secular = is_secular
        self.is_verbose = is_verbose
        self.setup()

    def initialize_from_rdm(self, rho):
        rho = rho.copy()
        if self.is_secular:
            if self.is_verbose:
                print("\n--- Using the secular approximation")

        ident = np.identity(self.ham.nsite)
        self.Omega = -1j*np.einsum('ij,ik,jl->ijkl', 
                                   self.ham.omega_diff, ident, ident)
        self.Omega = utils.to_liouville(self.Omega)
        self.R, self.K = None, None

        if self.method == 'Redfield':
            self.diffeq_type = 'TCL'
            self.make_redfield_tensor(np.inf)
            self.R = self._redfield_wrapper
        elif self.method == 'TCL2': 
            self.diffeq_type = 'TCL'
            self.make_redfield_tensor(0.0)
            self.R = self._tcl_wrapper
            self.tensor_is_changing = True
        elif self.method == 'TC2':
            self.diffeq_type = 'TC'
            self.K = self._tc_wrapper
            self.make_redfield_tensor(0.0)
            self.tensor_is_changing = True

        return rho

    def _redfield_wrapper(self, n, i):
        """A wrapper that yields a function-like interface to the Redfield tensor"""
        return self.redfield_tensor

    def _tcl_wrapper(self, n, i):
        """A wrapper that yields a function-like interface to the TCL2 tensor"""
        if n > self.n_markov:
            n = self.n_markov
        # On coarse grid (throw away i-steps):
        return self.redfield_tensor_n[n]
        # On fine grid:
        #return self.redfield_tensor_ni[n][i]

    def _tc_wrapper(self, n, i, l, j):
        """A wrapper that yields a function-like interface to the TC2 tensor"""
        # On coarse grid (throw away i,j-steps): 
        if n-l > self.n_markov:
            return np.zeros_like(self.redfield_tensor_n[0])
        else:
            return self.redfield_tensor_n[n-l]


    def make_redfield_tensor(self, t):
        """Make and store the Redfield-like tensor, only at time t.

        Parameters
        ----------
        t : float
            The time at which the Redfield-like tensor is to be calculated.

        """
        ns = self.ham.nsite
        gamma_plus = np.zeros((ns,ns,ns,ns), dtype=np.complex_)
        gamma_minus = np.zeros((ns,ns,ns,ns), dtype=np.complex_)

        for a in range(self.ham.nbath):
            if self.method in ['TC2', 'TCL2']:
                bath_corr_t = self.ham.sd[a].bath_corr(t)

            Ga = self.ham.site2eig( self.ham.sysbath[a] )
            theta_plus_a = np.zeros((ns,ns), dtype=np.complex_)
            for i in range(ns):
                for j in range(ns):
                    omega_ij = self.ham.omega_diff[i,j]
                    if self.method in ['TC2', 'TCL2']:
                        theta_plus_a[i,j] = np.exp(-1j*omega_ij*t)*bath_corr_t
                    else:
                        # Note, our C(w) has \int e^{+iwt}.
                        theta_plus_a[i,j] = self.ham.sd[a].bath_corr_ft(-omega_ij,t)
            theta_minus_a = theta_plus_a.conj().transpose()

            gamma_plus += np.einsum('lj,ik,ik->ljik', Ga, Ga, theta_plus_a)
            gamma_minus += np.einsum('lj,ik,lj->ljik', Ga, Ga, theta_minus_a)
        gamma_plus  /= const.hbar**2
        gamma_minus /= const.hbar**2

        ident = np.identity(ns)
        self.redfield_tensor = (  gamma_plus.transpose(2,1,3,0) 
                                + gamma_minus.transpose(2,1,3,0)
                                - np.einsum('lj,irrk->ijkl', ident, gamma_plus)
                                - np.einsum('ik,lrrj->ijkl', ident, gamma_minus) )

        if self.method == 'TC2':
            for i in range(ns):
                for j in range(ns):
                    for k in range(ns):
                        for l in range(ns):
                            omega_kl = self.ham.omega_diff[k,l]
                            self.redfield_tensor[i,j,k,l] *= np.exp(-1j*omega_kl*t)

        if self.is_secular:
            for i in range(ns):
                for j in range(ns):
                    for k in range(ns):
                        for l in range(ns):
                            if abs(self.ham.omega_diff[i,j]
                                   -self.ham.omega_diff[k,l]) > 1e-6:
                                self.redfield_tensor[i,j,k,l] = 0.0

        if ns == 2 and self.method == "Redfield":
            if self.is_verbose:
                print("\n--- The Redfield tensor")
                print(self.redfield_tensor)

                print("\n--- Checking detailed balance")
                for i in range(ns):
                    for j in range(i+1,ns):
                        print("R[%d,%d,%d,%d]/R[%d,%d,%d,%d] = %0.10lf =? %0.10lf")%(
                                i,i,j,j,j,j,i,i,
                                np.real( self.redfield_tensor[i,i,j,j]
                                        / self.redfield_tensor[j,j,i,i] ),
                                np.exp(-(self.ham.evals[i]-self.ham.evals[j])
                                       / const.kT) )

        self.redfield_tensor = utils.to_liouville(self.redfield_tensor)

    def precompute_redfield_tensor(self, t_init, t_final, dt):
        """Precompute the time-dependent Redfield-like tensor, store in array.

        Parameters
        ----------
        t_init : float
            Initial time at which to start precomputing.
        f_final : float
            Final time at which to stop precomputing.
        dt : float
            Timestep for precomputing.
        integrator : Integrator
            An instance of the pyrho Integrator class.

        Notes
        -----
        The tensor is stored on a coarse grid which is ignorant to the 
        Integrator grid (e.g. RK4).

        """
        if self.is_verbose:
            print("\n--- Precomputing the Redfield tensor")
        n_timesteps = int( (t_final-t_init)/dt + 1 )
        self.redfield_tensor_n = []
        redfield_tensor_integral_n = []
        cost = []
        if self.method == 'TCL2':
            redfield_tensor_n = []
            for time in np.arange(0.0, t_init, dt):
                self.make_redfield_tensor(time)
                redfield_tensor_n.append(self.redfield_tensor)
            redfield_tensor_integral_0_t_init = np.trapz(redfield_tensor_n, dx=dt, axis=0)

        for n in range(n_timesteps):
            time = t_init + n*dt
            self.make_redfield_tensor(time)
            self.redfield_tensor_n.append(self.redfield_tensor)

            # Collect running integral for TCL2
            if self.method == 'TCL2':
                redfield_tensor_integral_n.append(redfield_tensor_integral_0_t_init
                    + np.trapz(self.redfield_tensor_n[:n+1], dx=dt, axis=0) )

            cost.append( utils.tensor_diff( self.redfield_tensor, 
                                            self.redfield_tensor_n[n-1] ) )
            if n > 0 and cost[n] < self.markov_tol*np.std(cost):
                if self.is_verbose:
                    print("\n--- Tensor has stopped changing at t ="), time
                self.n_markov = n
                break

            if n > 0 and time >= self.markov_time:
                if self.is_verbose:
                    print("\n--- Tensor calculation stopped at t ="), time
                self.n_markov = n
                break

        self.n_markov = n
        if self.method == 'TCL2':
            self.redfield_tensor_n = redfield_tensor_integral_n

        if self.is_verbose:
            print("\n--- Done precomputing the Redfield tensor")

        cost_file = open('cost_%s.dat'%(self.method),'w')
        rate_file = open('rate_%s.dat'%(self.method),'w')
        for n in range(self.n_markov):
            time = t_init + n*dt
            cost_file.write('%0.6f %0.6f %0.6f %0.6f\n'
                            %(time, cost[n], np.std(cost[:n+1]), self.markov_tol))
            rate_file.write('%0.6f '%(time))
            for row in self.redfield_tensor_n[n]:
                for R in row:
                    rate_file.write('%0.6f %0.6f '%(R.real, R.imag))
            rate_file.write('\n')

        cost_file.close()
        rate_file.close()

    def precompute_redfield_tensor_finegrid(self, t_init, t_final, dt, integrator):
        if self.is_verbose:
            print("\n--- Precomputing the Redfield tensor on a fine RK4 grid")
        int_order = integrator.order
        c = integrator.c
        n_timesteps = int( (t_final-t_init)/dt + 1 )
        self.redfield_tensor_ni = []
        cost = []
        rate_file = open('rate_%s.dat'%(self.method),'w')
        for n in range(n_timesteps):
            self.redfield_tensor_ni.append([])
            for i in range(int_order):
                t = t_init + n*dt + c[i]*dt
                self.make_redfield_tensor(t)
                self.redfield_tensor_ni[n].append(self.redfield_tensor)

            cost.append( utils.tensor_diff( self.redfield_tensor, 
                                            self.redfield_tensor_ni[n-1][i] ) )
            if n > 0 and cost[n] < self.markov_tol*np.std(cost):
                if self.is_verbose:
                    print("\n--- Tensor has stopped changing at t ="), t
                self.n_markov = n
                rate_file.close()
                break

            if n > 0 and t >= self.markov_time:
                if self.is_verbose:
                    print("\n--- Tensor calculation stopped at t ="), t
                self.n_markov = n
                rate_file.close()
                break

            rate_file.write('%0.6f '%(t))
            for row in self.redfield_tensor:
                for R in row:
                    rate_file.write('%0.6f %0.6f '%(R.real, R.imag))
            rate_file.write('\n')

        self.n_markov = n
        rate_file.close()
        if self.is_verbose:
            print("\n--- Done precomputing the Redfield tensor")

    def propagate_full(self, rho_0, t_init, t_final, dt,
                  markov_tol = 1e-3, markov_time = np.inf,
                  is_verbose=False):
        self.markov_tol = markov_tol
        self.markov_time = markov_time

        times = np.arange(t_init, t_final, dt)
        rho_eig = self.ham.site2eig(rho_0)

        integrator = Integrator(self.diffeq_type, dt, 
                                Omega=self.Omega, R=self.R, K=self.K)
        if self.method == 'TCL2':
            if not hasattr(self, 'redfield_tensor_n'):
                self.precompute_redfield_tensor(t_init, t_final, dt)
            elif len(self.redfield_tensor_n) < int((t_final-t_init)/dt):
                self.precompute_redfield_tensor(t_init, t_final, dt)
        if self.method == 'TC2':
                self.precompute_redfield_tensor(t_init, t_final, dt)
            #self.precompute_redfield_tensor_finegrid(t_init, t_final, dt, 
            #                                         integrator)

        integrator.set_initial_value(utils.to_liouville(rho_eig), t_init)

        rhos_site = []
        for time in times:
            # Retrieve data from integrator
            rho_eig = utils.from_liouville(integrator.y)

            # Collect results
            rho_site = self.ham.eig2site(rho_eig)
            rhos_site.append(rho_site)

            # Propagate one timestep
            integrator.integrate()

        if is_verbose:
            print("\n--- Finished performing RDM dynamics")
        
        return times, rhos_site


    def propagate(self, rho_0, t_init, t_final, dt,
                  markov_tol = 1e-3, markov_time = np.inf,
                  is_verbose=False):
        """Propagate the RDM according to Redfield-like dynamics.

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
        markov_tol : float
            The relative tolerance at which to decide that the memory tensor
            has stopped changing (and henceforth Markovian).
        markov_time : float
            The hard-coded time to stop re-calculating the memory tensor.
        is_verbose : bool
            Flag to indicate verbose printing.

        Returns
        -------
        times : list of floats
            The times at which the RDM has been calculated.
        rhos_site : list of np.arrays
            The RDM at each time in the site basis.

        """
        rho_site = self.initialize_from_rdm(rho_0)
        return self.propagate_full(rho_site, t_init, t_final, dt,
                              markov_tol, markov_time, is_verbose)
