""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
import itertools
import iterhelper
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

        Notes
        -----
        In pyrho:
            J(w) has units of energy
            Cj(t) = hbar/pi \int dw J_j(w) ... has units of energy^2
                 = \sum_k c[j,k] exp(-gamma[j,k] t)
            c[j,k] has units of energy^2
            gamma[j,k] has units of frequency
            rdo(t) is dimensionless
            ado(t) at tier N has units of frequency^N
            p[j,l] has units of energy*frequency^3
            d[p,l] has units of energy^2 (same as c[j,k])
    """

    def __init__(self, hamiltonian, L=1, K=0,
                 truncation_type='Ishizaki-Tanimura'):
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

        """
        assert(truncation_type in ['None','Ishizaki-Tanimura','TL'])
        utils.print_banner("PERFORMING RDM DYNAMICS WITH "
                           "THE HIERARCHICAL EQUATIONS OF MOTION")
 
        self.ham = hamiltonian
        self.Lmax = L
        self.Kmax = K
        self.Nk = K+1
        self.truncation_type = truncation_type
        print "--- Hierarchy depth, L =", self.Lmax
        print "--- Maximum Matsubara frequency, K =", self.Kmax
        print "--- Using Matsubara truncation scheme:", self.truncation_type

        if self.ham.sd[0].sd_type == 'ohmic-exp':
            self.nlorentz = 3 # number of Lorenztians to fit
        else:
            self.nlorentz = 0

        self.initialize_hierarchy_nmats()
        self.precompute_matsubara()

    def precompute_matsubara(self):
        """Precompute the Matsubara expansion parameters c_k and gamma_k.
        """

        print "--- Initializing Matsubara expansion ...",

        beta = 1./const.kT
        hbar = const.hbar

        if self.ham.sd[0].sd_type == 'ohmic-lorentz': 
            self.c = np.zeros((self.ham.nbath, self.Nk), dtype=np.complex)
            self.gamma = np.zeros((self.ham.nbath, self.Nk))
            for j in range(self.ham.nbath):
                lamdaj = self.ham.sd[j].lamda
                omega_cj = self.ham.sd[j].omega_c
                for k in range(self.Nk):
                    if k == 0:
                        # Chen Eq. (11)
                        self.gamma[j,k] = omega_cj
                        self.c[j,k] = ( lamdaj*hbar*omega_cj
                                        *(1./np.tan(beta*hbar*omega_cj/2.) - 1j) )
                    else:
                        # Chen Eqs. (12)
                        self.gamma[j,k] = 2*np.pi*k/(beta*hbar)
                        self.c[j,k] = ( 4*lamdaj*omega_cj/beta
                                        *self.gamma[j,k]
                                        /((self.gamma[j,k])**2 - omega_cj**2) )
        elif self.ham.sd[0].sd_type == 'ohmic-exp':
            self.p = np.zeros((self.ham.nbath, self.nlorentz))
            self.dp = np.zeros((self.ham.nbath, self.nlorentz), dtype=np.complex)
            self.dm = np.zeros((self.ham.nbath, self.nlorentz), dtype=np.complex)
            self.Gamma = np.zeros((self.ham.nbath, self.nlorentz))
            self.Omega = np.zeros((self.ham.nbath, self.nlorentz))
            self.c = np.zeros((self.ham.nbath, self.Nk), dtype=np.complex)
            self.gamma = np.zeros((self.ham.nbath, self.Nk))
            self._coth_b_OminusG = np.zeros((self.ham.nbath, self.nlorentz), dtype=np.complex)
            self._coth_b_OplusG = np.zeros((self.ham.nbath, self.nlorentz), dtype=np.complex)
            for j in range(self.ham.nbath):
                lamdaj = self.ham.sd[j].lamda
                omega_cj = self.ham.sd[j].omega_c
                self.p[j,0] = np.pi*lamdaj*omega_cj**3 *( 12.0677)
                self.p[j,1] = np.pi*lamdaj*omega_cj**3 *(-19.9762)
                self.p[j,2] = np.pi*lamdaj*omega_cj**3 *(  0.1834)
                self.Omega[j,0] = omega_cj * 0.2378
                self.Omega[j,1] = omega_cj * 0.0888
                self.Omega[j,2] = omega_cj * 0.0482
                self.Gamma[j,0] = omega_cj * 2.2593
                self.Gamma[j,1] = omega_cj * 5.4377
                self.Gamma[j,2] = omega_cj * 0.8099
                for l in range(self.nlorentz):
                    self.dp[j,l] = ( hbar*self.p[j,l]/(8*self.Omega[j,l]*self.Gamma[j,l])
                                    *(1./np.tanh(beta*hbar*(self.Omega[j,l] - 1j*self.Gamma[j,l])/2.) + 1) )
                    self.dm[j,l] = ( hbar*self.p[j,l]/(8*self.Omega[j,l]*self.Gamma[j,l])
                                    *(1./np.tanh(beta*hbar*(self.Omega[j,l] + 1j*self.Gamma[j,l])/2.) - 1) )
                    self._coth_b_OplusG[j,l] = 1./np.tanh(beta*hbar*(self.Omega[j,l] + 1j*self.Gamma[j,l])/2.)
                    self._coth_b_OminusG[j,l] = 1./np.tanh(beta*hbar*(self.Omega[j,l] - 1j*self.Gamma[j,l])/2.)
                for k in range(self.Nk):
                        # Liu Eq. (A3)
                        self.gamma[j,k] = 2*np.pi*(k+1)/(beta*hbar)
                        self.c[j,k] = 0.0
                        for l in range(self.nlorentz):
                            self.c[j,k] += - ( 2*self.p[j,l]/beta
                                        *self.gamma[j,k]
                                        /((self.Gamma[j,l]**2 + self.Omega[j,l]**2 - self.gamma[j,k]**2)**2
                                          + 4*self.gamma[j,k]**2*self.Omega[j,l]**2) )

            # Write the approximate spectral density to a file.
            for j in range(self.ham.nbath):
                Jw_file = open('Jw%d_approx.dat'%(j),'w')
                lamda = 0.0
                omega_cj = self.ham.sd[j].omega_c
                dw = 10*omega_cj/1000.
                omegas = np.arange(-10*omega_cj,
                                    10*omega_cj, dw) + 1e-14
                for omega in omegas:
                    Jw = 0.0
                    for l in range(self.nlorentz):
                        Jw += ( self.p[j,l]*omega/
                               ( ((omega+self.Omega[j,l])**2+self.Gamma[j,l]**2)
                                *((omega-self.Omega[j,l])**2+self.Gamma[j,l]**2) ))
                    if omega > 0:
                        lamda += dw*(1./np.pi)*Jw/omega
                    Jw_file.write('%0.6f %0.6f %0.6f\n'
                                  %(const.hbar*omega, Jw, lamda))
                Jw_file.close()

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
                if self.ham.sd[j].sd_type == 'ohmic-lorentz':
                    ct_j = np.dot(self.c[j,:], np.exp(-self.gamma[j,:]*time))
                elif self.ham.sd[j].sd_type == 'ohmic-exp':
                    ct_j = (np.dot(self.dp[j,:], np.exp(-(self.Gamma[j,:]+1j*self.Omega[j,:])*time))
                           +np.dot(self.dm[j,:], np.exp(-(self.Gamma[j,:]-1j*self.Omega[j,:])*time))
                           +np.dot(self.c[j,:], np.exp(-self.gamma[j,:]*time)))
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

        if self.ham.sd[0].sd_type == 'ohmic-exp':
            nmodes = self.Nk + 2*self.nlorentz
        else:
            nmodes = self.Nk
        n_index = 0
        try:
            for njklist in iterhelper.product(self.Lmax, self.ham.nbath*nmodes):
                nmat = np.array(njklist).reshape(self.ham.nbath,nmodes)
                self.nmats.append(nmat)
                self.nmat_hash[self.matrix2string(nmat)] = n_index 
                n_index += 1
        except NotImplementedError:
            for njklist in itertools.product(range(self.Lmax+1), 
                                             repeat=self.ham.nbath*nmodes):
                if sum(njklist) <= self.Lmax:
                    nmat = np.array(njklist).reshape(self.ham.nbath,nmodes)
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
        self.nmat_plus_hash = np.zeros( (len(self.nmats),self.ham.nbath,nmodes),
                                        dtype=int )
        self.nmat_minus_hash = np.zeros( (len(self.nmats),self.ham.nbath,nmodes),
                                         dtype=int )
        for n, nmat in enumerate(self.nmats):
            for j in range(self.ham.nbath):
                for k in range(nmodes):
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
        return np.array(rho_hierarchy, dtype=np.complex)

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
            cjk_over_gammajk_re = np.einsum('jk,jk->j', self.c, 1./self.gamma).real
            for n, nmat in enumerate(self.nmats): 
                L = np.sum(nmat)
                rho_n = rho[n]
                drho_n = -1j/hbar * utils.commutator(self.ham.sys,rho_n)
                drho_n -= np.sum(nmat*self.gamma)*rho_n
                for j in range(self.ham.nbath):
                    Fj = self.ham.sysbath[j]    # this is like |j><j|
                    lamdaj = self.ham.sd[j].lamda
                    omega_cj = self.ham.sd[j].omega_c
                    if self.truncation_type == 'Ishizaki-Tanimura':
                        drho_n -= (1./hbar**2)*( 
                                (2*lamdaj*const.kT/omega_cj - cjk_over_gammajk_re[j])
                                   *utils.commutator(Fj, utils.commutator(Fj, rho_n)) )
                    if L < self.Lmax:
                        # If L == self.Lmax, then rho_nkplus = 0
                        rho_njkplus_sum = np.zeros_like(rho_n)
                        for k in range(self.Nk):
                            rho_njkplus_sum += self.rho_plus(rho,n,j,k)
                        drho_n -= 1j*utils.commutator(Fj, rho_njkplus_sum)
                    for k in range(self.Nk):
                        rho_njkminus = self.rho_minus(rho,n,j,k)
                        drho_n -= (1j/hbar**2)*( nmat[j,k]
                                   *(self.c[j,k]*np.dot(Fj,rho_njkminus)
                                    -self.c[j,k].conjugate()*np.dot(rho_njkminus,Fj)) )
                drho.append(drho_n)
        elif self.ham.sd[0].sd_type == 'ohmic-exp':
            # Chen Eq. (15)
            beta = 1./const.kT
            cjk_over_gammajk_re = np.einsum('jk,jk->j', self.c, 1./self.gamma).real
            for nab, nabmat in enumerate(self.nmats):
                L = np.sum(nabmat)
                nmat = nabmat[:,:self.Nk]
                amat = nabmat[:,self.Nk:self.Nk+self.nlorentz]
                bmat = nabmat[:,self.Nk+self.nlorentz:self.Nk+2*self.nlorentz]
                rho_nab = rho[nab]
                drho_nab = -1j/hbar * utils.commutator(self.ham.sys,rho_nab)
                drho_nab -=    np.sum((amat+bmat)*self.Gamma)*rho_nab
                drho_nab -= 1j*np.sum((amat-bmat)*self.Omega)*rho_nab
                drho_nab -= np.sum(nmat*self.gamma)*rho_nab
                for j in range(self.ham.nbath):
                    Fj = self.ham.sysbath[j]    # this is like |j><j|
                    lamdaj = self.ham.sd[j].lamda
                    omega_cj = self.ham.sd[j].omega_c
                    #if self.truncation_type == 'Ishizaki-Tanimura':
                    #    drho_nab -= (1./hbar**2)*(
                    #            (2*lamdaj*const.kT/omega_cj - cjk_over_gammajk_re[j])
                    #               *utils.commutator(Fj, utils.commutator(Fj, rho_nab)) )
                    if L < self.Lmax:
                        # If L == self.Lmax, then rho_nabkplus = 0
                        rho_nabjkplus_sum = np.zeros_like(rho_nab)
                        for k in range(self.Nk+2*self.nlorentz):
                            rho_nabjkplus_sum += self.rho_plus(rho,nab,j,k)
                        drho_nab -= 1j*utils.commutator(Fj, rho_nabjkplus_sum)
                    for k in range(self.Nk):
                        rho_nabjkminus = self.rho_minus(rho,nab,j,k)
                        drho_nab -= (1j/hbar**2)*( nmat[j,k]*self.c[j,k]
                                                  *utils.commutator(Fj, rho_nabjkminus) )
                    for l in range(self.nlorentz):
                        rho_nabjkminus = self.rho_minus(rho,nab,j,self.Nk+l)
                        drho_nab -= (1j/hbar)*( amat[j,l]
                                        * self.p[j,l]/(8*self.Omega[j,l]*self.Gamma[j,l])
                                        * (self._coth_b_OminusG[j,l] * utils.commutator(Fj, rho_nabjkminus)
                                           + utils.anticommutator(Fj, rho_nabjkminus)) )
                    for l in range(self.nlorentz):
                        rho_nabjkminus = self.rho_minus(rho,nab,j,self.Nk+self.nlorentz+l)
                        drho_nab -= (1j/hbar)*( bmat[j,l]
                                        * self.p[j,l]/(8*self.Omega[j,l]*self.Gamma[j,l])
                                        * (self._coth_b_OplusG[j,l] * utils.commutator(Fj, rho_nabjkminus)
                                           - utils.anticommutator(Fj, rho_nabjkminus)) )
                drho.append(drho_nab)

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
