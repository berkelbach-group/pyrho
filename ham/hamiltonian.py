""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
from scipy import integrate
from pyrho.lib import const, utils 

class HamiltonianSystem(object):
    """A system-only Hamiltonian class"""

    def __init__(self, ham_sys, hbar=1, is_verbose=True):
        """Initialize the Hamiltonian class.

        Parameters
        ----------
        ham_sys : np.array
            The system Hamiltonian in the site basis.
        hbar : float
            Reduced Planck constant, hbar, in chosen units.

        """
        const.hbar = hbar
        self.init_system(ham_sys, is_verbose)
        
    def init_system(self, ham_sys, is_verbose=True):
        self.sys = ham_sys
        self.nsite = ham_sys.shape[0]
        self.evals, self.Umat = utils.diagonalize(self.sys)
        self.omega_diff = np.zeros((self.nsite,self.nsite))
        for i in range(self.nsite):
            for j in range(self.nsite):
                self.omega_diff[i,j] = (self.evals[i] - self.evals[j])/const.hbar

        if is_verbose:
            print "\n--- System eigenvalues"
            print self.evals

    def site2eig(self, rho):
        """Transform rho from the site basis to the eigen basis."""
        return utils.matrix_dot(self.Umat.conj().T,rho,self.Umat)

    def eig2site(self, rho):
        """Transform rho from the eigen basis to the site basis."""
        return utils.matrix_dot(self.Umat,rho,self.Umat.conj().T)

    def to_interaction(self, rho, t):
        """Transform rho (in the eigen basis) into the interaction picture."""
        #Ut = np.diag(np.exp(-1j*self.evals*t/const.hbar))
        #return utils.matrix_dot(Ut,rho,Ut.conj().T)
        return np.exp(-1j*self.omega_diff*t)*rho 

    def from_interaction(self, rho, t):
        """Transform rho (in the eigen basis) out of the interaction picture."""
        #Ut = np.diag(np.exp(-1j*self.evals*t/const.hbar))
        #return utils.matrix_dot(Ut.conj().T,rho,Ut)
        return np.exp(1j*self.omega_diff*t)*rho 


class Hamiltonian(HamiltonianSystem):
    """A system-bath Hamiltonian class"""

    def __init__(self, ham_sys, ham_sysbath, spec_densities, kT, hbar=1, is_verbose=True):
        """Initialize the Hamiltonian class.

        Parameters
        ----------
        ham_sys : np.array
            The system Hamiltonian in the site basis.
        ham_sysbath : list of np.arrays
            The system part of the system-bath operator for each bath.
        spec_densities : list of functions
            The spectral density function for each bath.
        kT : float
            Thermal energy, kT, in chosen units.
        hbar : float
            Reduced Planck constant, hbar, in chosen units.

        """
        # Set these shared constants
        const.kT = kT
        const.hbar = hbar

        if is_verbose:
            utils.print_banner("INITIALIZING SYSTEM+BATH HAMILTONIAN")

        self.init_system(ham_sys, is_verbose)

        self.sysbath = ham_sysbath
        self.spec_densities = spec_densities
        assert len(ham_sysbath) == len(spec_densities)
        self.nbath = len(ham_sysbath)

        # 'sd' is a list of 'SpecDens' instances that have member 
        # variables lamda, omega_c and member function J(omega)
        self.sd = []
        n = 0
        for spec_dens in self.spec_densities:
            self.sd.append(SpecDens(spec_dens))
            self.sd[n].write_Jw('Jw%d.dat'%(n))
            n += 1

    def xxx_init_classical_modes(self,nmode=300):
        modes = []
        for n in range(self.nbath):
            # Initialize classical bath frequencies and couplings
            lamda_n = self.sd[n].lamda
            omega_c_n = self.sd[n].omega_c
            modes.append([])
            for k in range(nmode):
                if self.sd[n].sd_type == 'ohmic-exp':
                    omega_nk = omega_c_n*(-np.log((k+0.5)/nmode))
                    rho_wnk =  nmode/(np.pi*lamda_n*omega_nk)
                    #c_nk = omega_nk*np.sqrt(2*lamda_n/nmode)
                elif self.sd[n].sd_type == 'ohmic-lorentz':
                    omega_nk = omega_c_n*np.tan((np.pi/2.)*(k+0.5)/nmode)
                    rho_wnk = nmode/(np.pi*lamda_n*omega_nk)
                else:
                    print "SpecDens type", self.sd[n].sd_type, "not yet implemented for classical baths"
                    raise NotImplementedError
                c_nk = np.sqrt(2/np.pi * omega_nk / rho_wnk)
                modes[n].append(ClassicalHO(omega_nk, None, None, c_nk))
        return modes

    def init_classical_modes(self,nmode=300):
        modes = []
        for n in range(self.nbath):
            # Initialize classical bath frequencies and couplings
            #lamda_n = self.sd[n].lamda
            omega_c_n = self.sd[n].omega_c
            modes.append([])
            for k in range(nmode):
                if self.sd[n].sd_type == 'ohmic-exp':
                    omega_nk = omega_c_n*(-np.log((nmode-k-0.5)/nmode))
                    #rho_wnk = nmode/(np.pi*lamda_n*omega_nk)*self.sd[n].J(omega_nk)
                    rho_wnk = nmode/omega_c_n * np.exp(-omega_nk/omega_c_n) / omega_c_n
                elif self.sd[n].sd_type == 'ohmic-lorentz':
                    omega_nk = omega_c_n*np.tan((np.pi/2.)*(k+0.5)/nmode)
                    #rho_wnk = nmode/(np.pi*lamda_n*omega_nk)*self.sd[n].J(omega_nk)
                    rho_wnk = nmode/(np.pi) *2 / (1 + (omega_nk/omega_c_n)**2) / omega_c_n
                else:
                    print "Spectral density type", self.sd[n].sd_type, "not supported."
                    raise NotImplementedError
                c_nk = np.sqrt(2/np.pi * omega_nk * self.sd[n].J(omega_nk) / rho_wnk)
                modes[n].append(ClassicalHO(omega_nk, None, None, c_nk))
        return modes

    def sample_classical_modes(self, modes):
        for n in range(self.nbath):
            for mode in modes[n]:
                mode.sample_boltzmann(const.kT)


class SpecDens(object):
    """A spectral density class"""

    def __init__(self, spec_dens_list):
        """Initialize the spectral density class.

        Notes
        -----
        The spectral density has units of energy.

        Parameters
        ----------
        spec_dens_list : list
            The spectral density parameters, including the type (str), lambda,
            omega_c, etc.

        """
        sd_type = spec_dens_list[0]
        if sd_type == 'ohmic-exp':
            self.J = self.ohmic_exp
            self.lamda = spec_dens_list[1]
            self.omega_c = spec_dens_list[2]
            self.omega_inf = 10*self.omega_c
        elif sd_type == 'ohmic-lorentz':
            self.J = self.ohmic_lorentz
            self.lamda = spec_dens_list[1]
            self.omega_c = spec_dens_list[2]
            self.omega_inf = 20*self.omega_c
        elif sd_type == 'cubic-exp':
            self.J = self.cubic_exp
            self.lamda = spec_dens_list[1]
            self.omega_c = spec_dens_list[2]
            self.omega_inf = 10*self.omega_c
        elif sd_type == 'ohmic-lorentz-oscillators':
            # spec_dens_list should be of the form: 
            # ['ohmic-lorentz-oscillators', debye_list, oscillators_list]
            self.J = self.ohmic_lorentz_oscillators
            self.debye_list = spec_dens_list[1]
            self.oscillators_list = spec_dens_list[2]
            omega_large = 0.0
            for lamda, omega_c in self.debye_list:
                if omega_c > omega_large:
                    self.omega_inf = 20*omega_c
            for lamda, omega_osc, gamma in self.oscillators_list:
                if omega_osc + gamma > omega_large:
                    self.omega_inf = omega_osc + 10*gamma
        elif sd_type == 'oscillators':
            # spec_dens_list should be of the form: 
            # ['oscillators', [ [lam1,omega1], [lam2,omega2], ... ] ]
            self.J = self.oscillators
            self.lamdas = spec_dens_list[1]
            self.omegas = spec_dens_list[2]
        else:
            print "Spectral density type", sd_type, "not found!"
            raise SystemExit

        self.sd_type = sd_type

    def ohmic_exp(self, omega):
        """Evaluate an Ohmic spectral density with an exponential cutoff."""
        w = abs(omega)
        Jw = (np.pi*self.lamda/self.omega_c)*w*np.exp(-w/self.omega_c)
        if omega > 0:
            return Jw
        else:
            return -Jw

    def ohmic_lorentz(self, omega):
        """Evaluate an Ohmic spectral density with a Lorentzian cutoff 
        (aka Debye spectral density)."""
        w = abs(omega)
        Jw = 2*self.lamda*self.omega_c*w/(w**2 + self.omega_c**2)
        if omega > 0:
            return Jw
        else:
            return -Jw

    def cubic_exp(self, omega):
        """Evaluate a cubic (super-Ohmic) spectral density with an exponential
        cutoff.

        Notes
        -----
        \int_0^infty x^s/x exp(-x/omega_c) = (omega_c)^s (s-1)! 
            = (omega_c)^s Gamma(s)
        For s=3 :: \int ... = 2 (omega_c)^3

        """
        w = abs(omega)
        Jw = (np.pi*self.lamda/(2*self.omega_c**3))*w**3*np.exp(-w/self.omega_c)
        if omega > 0:
            return Jw
        else:
            return -Jw

    def ohmic_lorentz_oscillators(self, omega):
        w = abs(omega)
        Jw = 0.0
        for lamda, omega_c in self.debye_list:
            Jw += 2*lamda*omega_c*w/(w**2 + omega_c**2)
        for lamda, omega_osc, gamma in self.oscillators_list:
            Jw += ( lamda*(gamma/2.)*w
                    *( 1.0/( (w-omega_osc)**2 + (gamma/2.)**2 )
                      +1.0/( (w+omega_osc)**2 + (gamma/2.)**2 ) ) )
        if omega > 0:
            return Jw
        else:
            return -Jw

    #def oscillators(self, omega):
    #    # Should never actually call this function!
    #    # TCB 10/5/15 - Why not?
    #    gamma = 1e-3
    #    w = abs(omega)
    #    Jw = 0.0
    #    for lamda, omega in zip(self.lamdas, self.omegas):
    #        Jw += ( lamda*(gamma/2.)*w
    #                *( 1.0/( (w-omega_osc)**2 + (gamma/2.)**2 )
    #                  +1.0/( (w+omega_osc)**2 + (gamma/2.)**2 ) ) )
    #    if omega > 0:
    #        return Jw
    #    else:
    #        return -Jw

    def bath_corr(self, t):
        """Evaluate the real-time bath autocorrelation function at time t."""
        re_Ct = integrate.quad(self.re_bath_corr,
                                -self.omega_inf, self.omega_inf,
                                limit=1000, weight='cos', wvar=t)
        im_Ct = integrate.quad(self.re_bath_corr, 
                                -self.omega_inf, self.omega_inf,
                                limit=1000, weight='sin', wvar=t)
        re_Ct, im_Ct = re_Ct[0], -im_Ct[0]
        return (1.0/np.pi)*(re_Ct + 1j*im_Ct)

    def bath_corr_ft(self, omega, t):
        """Evaluate the pseudo-Fourier-Laplace transform of the bath corr fn.

        Parameters
        ----------
        omega : float
            Fourier-Laplace transform frequency.
        t : float
            Upper limit of the FT integral, can be finite or np.inf.
        
        Notes
        -----
        \int_0^t ds e^{i omega s} C(s)

        """
        if t is np.inf:
            ppv = integrate.quad(self.re_bath_corr, 
                                 -self.omega_inf, self.omega_inf,
                                 limit=200, weight='cauchy', wvar=omega)
            ppv = -ppv[0]
            return self.re_bath_corr(omega) + (1j/np.pi)*ppv
        else:
            def re_integrand(omega2):
                return np.sin((omega-omega2)*t)*self.re_bath_corr(omega2)

            def im_integrand(omega2):
                return (1-np.cos((omega-omega2)*t))*self.re_bath_corr(omega2)

            re_ppv = integrate.quad(re_integrand, 
                                    -self.omega_inf, self.omega_inf,
                                    epsrel=1e-8, limit=1000,
                                    weight='cauchy', wvar=omega)
            im_ppv = integrate.quad(im_integrand, 
                                    -self.omega_inf, self.omega_inf,
                                    epsrel=1e-8, limit=1000,
                                    weight='cauchy', wvar=omega)
            re_ppv, im_ppv = -re_ppv[0], -im_ppv[0]

            return (1./np.pi)*( re_ppv + 1j*im_ppv )

    def re_bath_corr(self, omega):
        """Real part of the FT of the bath autocorrelation function."""
        def coth(x):
            # Slow coth(x):
            return 1.0/np.tanh(x)
            # Faster coth(x):
            #numer = (((x*x+378)*x*x+17325)*x*x+135135)*x
            #denom = ((28*x*x+3150)*x*x+62370)*x*x+135135
            #approx = numer/denom
            #return min(approx,1.0)*(omega>0) + max(approx,-1)*(omega<0)

        beta = 1.0/const.kT
        hbar = const.hbar
        omega += 1e-14
        n_omega = 0.5*(coth(beta*hbar*omega/2) - 1.0)
        return hbar*self.J(omega)*(n_omega+1)

    def write_Jw(self, filename):
        """Write the spectral density to a file."""
        Jw_file = open(filename,'w')
        lamda = 0.0
        dw = self.omega_inf/1000.
        omegas = np.arange(-self.omega_inf, 
                            self.omega_inf, dw) + 1e-14
        for omega in omegas:
            if omega > 0:
                lamda += dw*(1./np.pi)*self.J(omega)/omega
            Jw_file.write('%0.6f %0.6f %0.6f\n'
                          %(const.hbar*omega, self.J(omega), lamda))
        Jw_file.close()
        

class ClassicalHO:
    """A classical Harmonic Oscillator class"""

    def __init__(self, omega, Q, P, coupling):
        self.omega = omega
        self.Q = Q
        self.P = P
        self.c = coupling

    def sample_boltzmann(self, kT):
        self.Q = np.random.normal(0.0, np.sqrt(kT)/self.omega)
        self.P = np.random.normal(0.0, np.sqrt(kT))

