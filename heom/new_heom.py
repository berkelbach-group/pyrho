""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
import itertools
from pyrho.integrate import Integrator
from pyrho.lib import const, utils

from pyrho.unitary import Unitary
class HEOM(Unitary):
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
                 K_truncation='Ishizaki-Tanimura', L_truncation='TNL', is_scaled=True):
        """Initialize the HEOM class.

        Parameters
        ----------
        hamiltonian : Hamiltonian
            An instance of the pyrho Hamiltonian class.
        L : int
            The "hierarchy depth."
        L_truncation: str
            Hierarchy closure type (time-local or time-nonlocal)
        K : int
            The Matsubara truncation.
        K_truncation : str
            The Matsubara truncation type (or terminator).
        is_scaled : bool
            Flag to use scaled ADMs, which are useful for filtering.

        """
        assert(K_truncation in ['None','Ishizaki-Tanimura'])
        assert(L_truncation in ['TNL','TL'])
        utils.print_banner("PERFORMING RDM DYNAMICS WITH "
                           "THE HIERARCHICAL EQUATIONS OF MOTION")
 
        self.ham = hamiltonian
        self.Lmax = L
        self.Kmax = K
        self.is_scaled = is_scaled
        self.L_truncation = L_truncation
        self.K_truncation = K_truncation
        print "--- Hierarchy depth, L =", self.Lmax
        print "--- Using hierarchy closure:", self.L_truncation
        print "--- Maximum Matsubara frequency, K =", self.Kmax
        print "--- Using Matsubara truncation scheme:", self.K_truncation

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

        print "--- Initializing Matsubara expansion ...",

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
                ct_j = np.dot(self.c[j,:], np.exp(-self.gamma[j,:]*time))
                corr_fn_file.write('%0.6f %0.6f '%(ct_j.real, ct_j.imag))
            corr_fn_file.write('\n')
        corr_fn_file.close()

    def initialize_hierarchy_nmats(self):
        """Initialize the indices of the auxiliary density matrix hierarchy.
        
        The ADMs are "indexed" by a set of "n-matrices".
        The "n-matrices" are a set of nbath x (K+1) matrices, whose elements
        are integers 0, 1, 2... which sum to a number less than or equal to L. 
        
        This function creates a list (turned ndarray) of those matrices
        "nmats", as well as a dictionary "nmat_hash" where each key is the
        n-matrix as a string and the value is the integer where that n-matrix
        is found in the "nmats" array.

        Elsewhere, rho_hierarchy will be a list-like ndarray indexed by the
        integers 0, 1, 2, ... corresponding to the n-matrices.
        Thus,
            n an integer   : rho_hierarchy[n]
            nmat a ndarray : rho_hierarchy[nmat_hash[nmat]]

        See also
        --------
        heom.matrix2string()

        """

        print "--- Initializing auxiliary density matrix hierarchy ...",

        self.nmat_hash = dict()
        self.nmats = list()

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

    def initialize_from_rdm(self, rho_site):
        """Initialize the RDM to its initial value and the ADMs to zero.

        Parameters
        ----------
        rho_site : ndarray
            The initial RDM in the site basis.

        Returns
        -------
        rho_hierarchy : ndarray
            The RDM+ADM hierarchy as a 3d ndarray (1D ndarray of 2D ndarrays).

        """
        rho_hierarchy = list()
        rho_hierarchy.append(rho_site)
        for n in range(1,len(self.nmats)): 
            rho_hierarchy.append(np.zeros_like(rho_site))
        return np.array(rho_hierarchy, dtype=np.complex_)

    def reduce_to_rdm(self, rho_hierarchy):
        if rho_hierarchy.ndim == 3:
            return rho_hierarchy[0]
        else:
            return rho_hierarchy[:,0]

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
        drho = list() 
        if self.ham.sd[0].sd_type == 'ohmic-lorentz':
            # Chen Eq. (15)
            cjk_over_gammajk = np.einsum('jk,jk->j', self.c, 1./self.gamma)
            for n, nmat in enumerate(self.nmats): 
                L = np.sum(nmat)
                rho_n = rho[n]
                drho_n = -1j/hbar * utils.commutator(self.ham.sys,rho_n)
                njk_gammajk = np.sum(nmat*self.gamma)
                for j in range(self.ham.nbath):
                    Fj = self.ham.sysbath[j]    # this is like |j><j|
                    lamdaj = self.ham.sd[j].lamda
                    omega_cj = self.ham.sd[j].omega_c
                    for k in range(self.Kmax+1):
                        if self.is_scaled:
                            scale_minus = np.sqrt(nmat[j,k]/np.abs(self.c[j,k]))
                        else:
                            scale_minus = nmat[j,k] 

                        rho_njkminus = self.rho_minus(rho,n,j,k)
                        drho_n -= ( (1j/hbar)*scale_minus
                                   *(self.c[j,k]*np.dot(Fj,rho_njkminus)
                                   -self.c[j,k].conjugate()*np.dot(rho_njkminus,Fj)) )
                    # Note: The real part is taken because the Ishizaki-Tanimura
                    # truncation yields a sum over excluded Matsubara frequencies,
                    # which are purely real; the difference between the *real*
                    # finite K expansion and 2 lamda kT / omega_c gives the
                    # neglected contribution (in Markovian "TNL" approximation).
                    if self.K_truncation == 'Ishizaki-Tanimura':
                        drho_n -= ( (2*lamdaj*const.kT/(hbar**2*omega_cj) 
                                     - cjk_over_gammajk[j]/hbar).real
                                   *utils.commutator(Fj, utils.commutator(Fj, rho_n)) )
                    if L < self.Lmax:
                        # If L == self.Lmax, then rho_nkplus = 0
                        rho_njkplus_sum = np.zeros_like(rho_n)
                        for k in range(self.Kmax+1):
                            if self.is_scaled:
                                scale_plus = np.sqrt((nmat[j,k]+1)*np.abs(self.c[j,k]))
                            else:
                                scale_plus = 1. 
                            rho_njkplus_sum += scale_plus*self.rho_plus(rho,n,j,k)
                        drho_n -= 1j*utils.commutator(Fj, rho_njkplus_sum)
    
    
    
                    if self.L_truncation == 'TL' and L==self.Lmax:
                        E, evec = utils.diagonalize(self.ham.sys)
                        Q_j=np.zeros((self.ham.nsite,self.ham.nsite),dtype=np.complex)
                        for a in range(self.ham.nsite):
                            aouter = np.outer(evec[:,a],evec[:,a])
                            for b in range(self.ham.nsite):
                                bouter = np.outer(evec[:,b],evec[:,b])
                                if self.is_scaled:
                                    num = np.sqrt(nmat[j,k]+1)*self.c[j,:]*(1.-np.exp(-self.gamma[j,:]*t
                                            -1j*self.ham.omega_diff[a,b]*t))
                                else:
                                    num = self.c[j,:]*(1.-np.exp(-self.gamma[j,:]*t
                                            -1j*self.ham.omega_diff[a,b]*t))

                                denom = self.gamma[j,:]+1j*self.ham.omega_diff[a,b]  
                                Q_j += np.sum(num/denom)*np.dot(aouter,np.dot(self.ham.sysbath[j],bouter))

                        rho_njkplus_sum = -(1j/hbar)*(np.dot(Q_j,rho_n)-np.dot(rho_n,np.conj(Q_j).T))
                        drho_n -= 1j*utils.commutator(Fj, rho_njkplus_sum)
                 
                 #       for k in range(self.Kmax+1):
                 #           if self.is_scaled:
                 #               scale_plus = np.sqrt((nmat[j,k]+1)*np.abs(self.c[j,k]))
                 #           else:
                 #               scale_plus = 1. 
                 #           rho_njkplus_sum += scale_plus*self.rho_plus(rho,n,j,k)
                 #       
                 #                                   



                
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
        op_rho_hierarchy = list() 
        for ado in rho_hierarchy:
            op_rho_hierarchy.append(np.dot(op_sys, ado))
        return np.array(op_rho_hierarchy)

    def act_from_right(self, op_sys, rho_hierarchy):
        """Act on a hierarchical set of density operators with a system operator."""
        op_rho_hierarchy = list() 
        for ado in rho_hierarchy:
            op_rho_hierarchy.append(np.dot(ado, op_sys))
        return np.array(op_rho_hierarchy)

    def propagate_full(self, rho_hierarchy, t_init, t_final, dt):
        integrator = Integrator('ODE', dt, deriv_fn=self.heom_deriv)
        integrator.set_initial_value(rho_hierarchy, t_init)

        rhos_site = list() 
        times = np.arange(t_init, t_final, dt)
        for time in times:
            rhos_site.append(integrator.y.copy())
            integrator.integrate()
            # TODO(TCB): If filtering, check if we can remove any ADMs

        return times, np.array(rhos_site)

    def propagate(self, rho_0, t_init, t_final, dt):
        """Propagate the RDM according to HEOM dynamics.

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
        #self.write_bath_corr_fn(times)
        rho_hierarchy = self.initialize_from_rdm(rho_0)
        times, rhos_hierarchy = self.propagate_full(rho_hierarchy, t_init, t_final, dt)
        rhos_site = self.reduce_to_rdm(rhos_hierarchy)
        return times, rhos_site
