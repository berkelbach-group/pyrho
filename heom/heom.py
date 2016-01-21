""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
import itertools
from pyrho.integrate import Integrator
from pyrho.lib import const, utils

class HEOM:
    """A hierarchical equations of motion class

    Follows:
    [Chen]  
        "Optical line shapes of molecular aggregates: 
        Hierarchical equations of motion method", 
        L. Chen, R. Zheng, Q. Shi, and Y. Yan, J. Chem. Phys. 131, 094502 (2009)
    and
    [Liu]
        "Reduced quantum dynamics with arbitrary bath spectral densities: 
        Hierarchical equations of motion based on several different bath 
        decomposition schemes", 
        H. Liu, L. Zhu, S. Bai, and Q. Shi, J. Chem. Phys. 140, 134106 (2014)

    """

    def __init__(self, hamiltonian, L=1, K=0,
                 truncation_type='Ishizaki-Fleming', is_scaled=True):
        """Initialize the HEOM class.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            An instance of the pyrho Hamiltonian class.
        L : int
            The "hierarchy depth."
        K : int
            The Matsubara truncation.
        truncation_type : str
            The hierarchy truncation type (or terminator).
        is_scaled : bool
            Flag to use scaled ADMs, which are useful for filtering.

        """
        assert(truncation_type in ['None','Ishizaki-Fleming','TL'])
        utils.print_banner("PERFORMING RDM DYNAMICS WITH "
                           "THE HIERARCHICAL EQUATIONS OF MOTION")
 
        self.ham = hamiltonian
        self.Lmax = L
        self.Kmax = K
        self.is_scaled = is_scaled
        self.truncation_type = truncation_type
        print "--- Hierarchy depth, L =", self.Lmax
        print "--- Maximum Matsubara frequency, K =", self.Kmax
        print "--- Using Matsubara truncation scheme:", self.truncation_type

        # TODO(TCB): Implement filtering.
        # Scaling makes no difference unless using tolerant filtering
        if self.is_scaled:
            print "--- Using Shi et al. scaled ADMs."
        else:
            print "--- Using original *unscaled* ADMs."

        self.initialize_hierarchy_nmats()
        self.precompute_matsubara()

    def precompute_matsubara(self):
        """Precompute the Matsubara expansion parameters c_k and gamma_k.

        Notes
        -----
        gamma[j,k] has units of frequency
        c[j,k] has units of frequency^2

        """

        print "--- Initilizing Matsubara expansion ...",

        beta = 1./const.kT
        hbar = const.hbar

        if self.ham.sd[0].sd_type == 'ohmic-lorentz': 
            self.c = np.zeros((self.ham.nbath, self.Kmax+1), dtype=np.complex_)
            self.gamma = np.zeros((self.ham.nbath, self.Kmax+1))
            for j in range(self.ham.nbath):
                lamdaj = self.ham.sd[j].lamda
                omega_cj = self.ham.sd[j].omega_c
                for k in range(self.Kmax+1):
                    if k == 0:
                        # Chen Eq. (11)
                        self.gamma[j,k] = omega_cj
                        self.c[j,k] = ( lamdaj*omega_cj
                                        *(1./np.tan(beta*hbar*omega_cj/2.) - 1j) )
                    else:
                        # Chen Eqs. (12)
                        self.gamma[j,k] = 2*np.pi*k/(beta*hbar)
                        self.c[j,k] = ( 4*lamdaj*omega_cj/(beta*hbar)
                                        *self.gamma[j,k]
                                        /((self.gamma[j,k])**2 - omega_cj**2) )
        #elif self.ham.sd[0].sd_type == 'oscillators':

        print "done."

    def write_bath_corr_fn(self, times):
        """Write the bath autocorrelation function to a file.

        Parameters
        ----------
        times : list of floats
            Times to calculate and print the correlation function.

        """
        corr_fn_file = open('Ct_heom_K-%d.dat'%(self.Kmax),'w')
        for time in times: 
            corr_fn_file.write('%0.6f '%(time))
            for j in range(self.ham.nbath):
                ct_j = 0+0j
                for k in range(self.Kmax+1):
                    ct_j += self.c[j,k]*np.exp(-self.gamma[j,k]*time)
                corr_fn_file.write('%0.6f %0.6f '%(ct_j.real, ct_j.imag))
            corr_fn_file.write('\n')
        corr_fn_file.close()

    def initialize_hierarchy_nmats(self):
        """Initialize the indices of the auxiliary density matrix hierarchy.
        
        The ADMs are "indexed" by a set of "n-matrices".
        The "n-matrices" are a set of nbath x (K+1) matrices, whose elements
        are integers 0, 1, 2... which sum to a number less than or equal to L. 
        
        This function creates a list (turned np.array) of those matrices
        "nmats", as well as a dictionary "nmat_hash" where each key is the
        n-matrix as a string and the value is the integer where that n-matrix
        is found in the "nmats" array.

        Elsewhere, rho_hierarchy will be a list-like np.array indexed by the
        integers 0, 1, 2, ... corresponding to the n-matrices.
        Thus,
            n an integer    : rho_hierarchy[n]
            nmat a np.array : rho_hierarchy[nmat_hash[nmat]]

        See also
        --------
        heom.matrix2string()

        """

        print "--- Initilizing auxiliary density matrix hierarchy ...",

        self.nmat_hash = {} 
        self.nmats = []

        n_index = 0
        for njklist in itertools.product(range(self.Lmax+1), 
                                         repeat=self.ham.nbath*(self.Kmax+1)):
            if sum(njklist) <= self.Lmax:
                nmat = np.array(njklist).reshape(self.ham.nbath,self.Kmax+1)
                self.nmats.append(nmat)
                self.nmat_hash[self.matrix2string(nmat)] = n_index 
                n_index += 1

                #print "nmat ="
                #print nmat
                #print "string =", self.matrix2string(nmat)
                #print "hashed nmat ="
                #print self.nmats[self.nmat_hash[self.matrix2string(nmat)]]

        #print self.nmat_hash

        # Precompute nmat_plus_hash[n,j,k] and nmat_minus_hash[n,j,k]
        self.nmat_plus_hash = np.zeros( (len(self.nmats),self.ham.nbath,self.Kmax+1),
                                        dtype=int )
        self.nmat_minus_hash = np.zeros( (len(self.nmats),self.ham.nbath,self.Kmax+1),
                                         dtype=int )
        for n, nmat in enumerate(self.nmats):
            for j in range(self.ham.nbath):
                for k in range(self.Kmax+1):
                    nmat_new = nmat.copy()
                    nmat_new[j,k] += 1
                    nmat_str = self.matrix2string(nmat_new)
                    if nmat_str in self.nmat_hash:
                        self.nmat_plus_hash[n,j,k] = self.nmat_hash[nmat_str]
                    else:
                        self.nmat_plus_hash[n,j,k] = -1
                    nmat_new = nmat.copy()
                    nmat_new[j,k] -= 1
                    nmat_str = self.matrix2string(nmat_new)
                    if nmat_str in self.nmat_hash:
                        self.nmat_minus_hash[n,j,k] = self.nmat_hash[nmat_str]
                    else:
                        self.nmat_minus_hash[n,j,k] = -1
                    
        print "done."

    def initialize_rho_hierarchy(self, rho_site):
        """Initialize the RDM to its initial value and the ADMs to zero.

        Parameters
        ----------
        rho_site : np.array
            The initial RDM in the site basis.

        Returns
        -------
        rho_hierarchy : np.array
            The RDM+ADM hierarchy as a 1D np.array of 2D np.arrays.

        """
        rho_hierarchy = []
        rho_hierarchy.append(rho_site)
        for n in range(1,len(self.nmats)): 
            rho_hierarchy.append(np.zeros_like(rho_site))
        return np.array(rho_hierarchy, dtype=np.complex_)

    def string2matrix(self, string, M, N):
        return np.array(list(string), dtype=np.uint8).reshape(M,N)

    def matrix2string(self, mat):
        return ''.join(mat.ravel().astype(str))

    def rho_plus(self, rho_hierarchy, n, j, k):
        nidx = self.nmat_plus_hash[n,j,k]
        if nidx == -1:
            return np.zeros((self.ham.nsite,self.ham.nsite))
        else:
            return rho_hierarchy[nidx]

    def rho_minus(self, rho_hierarchy, n, j, k):
        nidx = self.nmat_minus_hash[n,j,k]
        if nidx == -1:
            return np.zeros((self.ham.nsite,self.ham.nsite))
        else:
            return rho_hierarchy[nidx]

#    def rho_njk(self, rho_hierarchy, nmat_in, j, k, change):
#        # For Ohmic-Lorentz (Debye) J(w)
#        # Change should be +/- 1
#        nmat = nmat_in.copy()
#        nmat[j,k] += change
#        n = self.matrix2string(nmat)
#        # TODO(TCB): Is the if-else slow? Could use dict.get() on nmat_hash,
#        #            but then how to index into rho_hierarchy array?
#        if n in self.nmat_hash:
#            return rho_hierarchy[self.nmat_hash[n]]
#        else:
#            return np.zeros((self.ham.nsite,self.ham.nsite))
#
#    def rho_mjkn(self, rho_hierarchy, mmat_in, nmat_in, j, k, change):
#        # For undamped harmonic oscillators, Liu App. C
#        # Change should be +/- 1
#        mmat = mmat_in.copy()
#        mmat[j,k] += change
#        m = self.matrix2string(mmat)
#        nmat = nmat_in.copy()
#        n = self.matrix2string(nmat)
#        if m in self.nmat_hash and n in self.nmat_hash:
#            return rho_hierarchy[self.nmat_hash[m],self.nmat_hash[n]]
#        else:
#            return np.zeros((self.ham.nsite,self.ham.nsite))
#
#    def rho_mnjk(self, rho_hierarchy, mmat_in, nmat_in, j, k, change):
#        # For undamped harmonic oscillators, Liu App. C
#        # Change should be +/- 1
#        mmat = mmat_in.copy()
#        m = self.matrix2string(mmat)
#        nmat = nmat_in.copy()
#        nmat[j,k] += change
#        n = self.matrix2string(nmat)
#        if m in self.nmat_hash and n in self.nmat_hash:
#            return rho_hierarchy[self.nmat_hash[m],self.nmat_hash[n]]
#        else:
#            return np.zeros((self.ham.nsite,self.ham.nsite))

    def heom_deriv(self, t, rho):
        hbar = const.hbar
        drho = []
        if self.ham.sd[0].sd_type == 'ohmic-lorentz':
            # Chen Eq. (15)
            cjk_over_gammajk = np.einsum('jk,jk->j', self.c, 1./self.gamma)
            for n, nmat in enumerate(self.nmats): 
                L = np.sum(nmat)
                rho_n = rho[n]
                drho_n = -1j/hbar * utils.commutator(self.ham.sys,rho_n)
                njk_gammajk = np.sum(nmat*self.gamma)
                for j in range(self.ham.nbath):
                    Gj = self.ham.sysbath[j]    # this is like |j><j|
                    lamdaj = self.ham.sd[j].lamda
                    omega_cj = self.ham.sd[j].omega_c
                    for k in range(self.Kmax+1):
                        if self.is_scaled:
                            scale_minus = np.sqrt(nmat[j,k]/np.abs(self.c[j,k]))
                        else:
                            scale_minus = nmat[j,k] 

                        rho_njkminus = self.rho_minus(rho,n,j,k)
                        drho_n -= ( (1j/hbar)*scale_minus
                                   *(self.c[j,k]*np.dot(Gj,rho_njkminus)
                                   -self.c[j,k].conjugate()*np.dot(rho_njkminus,Gj)) )
                    # Note: The real part is taken because the Ishizaki-Tanimura
                    # truncation yields a sum over excluded Matsubara frequencies,
                    # which are purely real; the difference between the *real*
                    # finite K expansion and 2 lamda kT / omega_c gives the
                    # neglected contribution (in Markovian "TNL" approximation).
                    if self.truncation_type == 'Ishizaki-Fleming':
                        drho_n -= ( (2*lamdaj*const.kT/(hbar**2*omega_cj) 
                                     - cjk_over_gammajk[j]/hbar).real
                                   *utils.commutator(Gj, utils.commutator(Gj, rho_n)) )
                    if L < self.Lmax:
                        # If L == self.Lmax, then rho_nkplus = 0
                        rho_njkplus_sum = np.zeros_like(rho_n)
                        for k in range(self.Kmax+1):
                            if self.is_scaled:
                                scale_plus = np.sqrt((nmat[j,k]+1)*np.abs(self.c[j,k]))
                            else:
                                scale_plus = 1. 
                            rho_njkplus_sum += scale_plus*self.rho_plus(rho,n,j,k)
                        drho_n -= 1j*utils.commutator(Gj, rho_njkplus_sum)
                drho_n -= njk_gammajk*rho_n
                drho.append(drho_n)

#        TODO(TCB): Finish this, i.e. HEOM for single-oscillator(s) 
#        elif self.ham.sd[0].sd_type == 'oscillators':
#            # Liu App. C, Eq. (C1)
#            for m in range(len(self.nmats)):
#                mmat = self.nmats[m]
#                Lm = np.sum(mmat)
#                for n in range(len(self.nmats)): 
#                    nmat = self.nmats[n]
#                    Ln = np.sum(nmat)
#                    rho_mn = rho[m,n]
#                    drho_mn = np.zeros_like(rho_mn)
#                    drho_mn = -1j/hbar * utils.commutator(self.ham.sys,rho_mn)
#                    njk_gammajk = 0.
#
#                    # more here
#                    drho.append(drho_mn)

        return np.array(drho)

    def act_from_left(self, op_sys, rho_hierarchy):
        """Act on a hierarchical set of density operators with a system operator."""
        op_rho_hierarchy = []
        for ado in rho_hierarchy:
            op_rho_hierarchy.append(np.dot(op_sys, ado))
        return np.array(op_rho_hierarchy)

    def act_from_right(self, op_sys, rho_hierarchy):
        """Act on a hierarchical set of density operators with a system operator."""
        op_rho_hierarchy = []
        for ado in rho_hierarchy:
            op_rho_hierarchy.append(np.dot(ado, op_sys))
        return np.array(op_rho_hierarchy)


    def propagate(self, rho_0, t_init, t_final, dt,
                  input_has_bath=False, output_has_bath=False,
                  is_verbose=True):

        if is_verbose:
            print "--- Propagating RDM ...",

        times = np.arange(t_init, t_final, dt)
        self.write_bath_corr_fn(times)
        if input_has_bath:
            rho_hierarchy = rho_0.copy()
        else:
            rho_hierarchy = self.initialize_rho_hierarchy(rho_0)

        integrator = Integrator('ODE', dt, deriv_fn=self.heom_deriv)
        integrator.set_initial_value(rho_hierarchy, t_init)

        rhos_site = []
        rhos_eig = []
        while integrator.t < t_final+1e-8:
            #print " - Propagating at time t =", integrator.t

            # Retrieve data from integrator
            if output_has_bath:
                rho_site = integrator.y
                rho_eig = []
                for ado_site in rho_site:
                    rho_eig.append(self.ham.site2eig(ado_site))
                rho_eig = np.array(rho_eig)
                    
            else:
                rho_site = integrator.y[0]
                rho_eig = self.ham.site2eig(rho_site)

            # Collect results
            rhos_site.append(rho_site.copy())
            rhos_eig.append(rho_eig.copy())

            # Propagate one timestep
            integrator.integrate()

            # TODO(TCB): If filtering, check if we can remove any ADMs

        if is_verbose:
            print "done."
        
        return times, rhos_site, rhos_eig

