""" pyrho -
    a python package for reduced density matrix techniques
"""

import numpy as np
from pyrho.lib import const, utils

class Spectroscopy(object):

    def __init__(self, dipole_site, dynamics):
        # dynamics instance must have a dynamics.propagate() method
        self.dipole_site = dipole_site
        self.dynamics = dynamics

    def absorption(self, e_min, e_max, de, rho_0, t_final, dt,
                   dipole_file='dipole_dipole.dat',
                   is_damped=True):

        utils.print_banner("CALCULATING LINEAR ABSORPTION SPECTRUM")

        energies = np.arange(e_min, e_max, de)
        omegas = energies/const.hbar

        mu = self.dipole_site
        mu_rho_0 = np.dot(mu, rho_0)
        times, mu_rho = self.dynamics.propagate(mu_rho_0, 0.0, t_final, dt)
        Cw = np.zeros(len(omegas))
        fdipole = open(dipole_file,'w')
        t_damp = 0.
        expi = np.exp(1j*np.outer(omegas,times))
        for t, time in enumerate(times):
            weight = dt - 0.5*dt*(t==0 or t==len(times)-1)
            Ct = np.trace(np.dot(mu,mu_rho[t]))
            if is_damped and t > t_damp:
                Ct *= switch_to_zero(time,t_damp,t_final)
            fdipole.write('%0.6f %0.6f %0.6f\n'%(
                time, Ct.real, Ct.imag))
            Cw += weight*(expi[:,t]*Ct).real

        fdipole.close()
        return energies, Cw

    def two_dimensional(self, e1_min, e1_max, de1, 
                              e3_min, e3_max, de3,
                              time2_min, time2_max, dt2,
                              rho_g, time_final, dt, 
                              lioupath='total', is_damped=True):

        utils.print_banner("CALCULATING TWO-DIMENSIONAL SPECTRUM")
        
        if lioupath == 'total':
            print '--- Including all Liouville pathways.'
        else:
            print '--- Including only selected Liouville pathways.'

        energy1s = np.arange(e1_min, e1_max, de1)
        energy3s = np.arange(e3_min, e3_max, de3)
        omega1s = energy1s/const.hbar
        omega3s = energy3s/const.hbar

        time2_max += 1e-8 # try to include time2_max
        time2s = np.arange(time2_min, time2_max, dt2)
        times = np.arange(0.0, time_final, dt)

        spectrum = np.zeros( (len(omega3s),len(time2s),len(omega1s)) )

        print "--- Spectrum will require O(%d) propagations."%(len(times)*len(time2s))

        print "--- Calculating third-order response function ...",

        try:
            Rsignal = []
            Rsignal.append(np.zeros( (len(times),len(time2s),len(times)), dtype=np.complex))
            Rsignal.append(np.zeros( (len(times),len(time2s),len(times)), dtype=np.complex))
            for traj in range(self.dynamics.ntraj):
                R_rp, R_nr = self.calculate_R3(rho_g, time2_min, time2_max, dt2, time_final, dt, is_damped=is_damped, lioupath=lioupath)
                Rsignal[0] += R_rp/self.dynamics.ntraj
                Rsignal[1] += R_nr/self.dynamics.ntraj
        except AttributeError:
            Rsignal = self.calculate_R3(rho_g, time2_min, time2_max, dt2, time_final, dt, is_damped=is_damped, lioupath=lioupath)
     # print time correlation R3
    #   for n in range(1,len(times)-1):
    #       print Rsignal[0][0,0,n].real

        print "done."
        print "--- Performing 2D Fourier transform ...",

        expi1 = np.exp(1j*np.outer(omega1s,times))
        expi1[:,0] *= 0.5*dt 
        expi1[:,1:] *= dt 
        expi3 = np.exp(1j*np.outer(omega3s,times))
        expi3[:,0] *= 0.5*dt 
        expi3[:,1:] *= dt 
        spectrum =  np.einsum('ws,xu,uts->xtw',expi1,expi3,Rsignal[1]).real
        spectrum += np.einsum('ws,xu,uts->xtw',expi1.conj(),expi3,Rsignal[0]).real

#        for w3, omega3 in enumerate(omega3s):
#            for w1, omega1 in enumerate(omega1s):
#                for t1, time1 in enumerate(times):
#                    weight1 = dt - 0.5*dt*(t1==0 or t1==len(times)-1)
#                    exp_iw1t1 = np.exp(1j*omega1*time1)
#                    exp_miw1t1 = exp_iw1t1.conj()
#                    for t3, time3 in enumerate(times):
#                        weight3 = dt - 0.5*dt*(t3==0 or t3==len(times)-1)
#                        exp_iw3t3 = np.exp(1j*omega3*time3)
#                        for t2 in range(len(time2s)):
#                            spectrum[w3,t2,w1] += weight1*weight3*(
#                                exp_iw1t1*exp_iw3t3*Rsignal[1][t3,t2,t1]
#                                + exp_miw1t1*exp_iw3t3*Rsignal[0][t3,t2,t1] ).real
#                            #np.exp(1j*(omega1*time1+omega3*time3))*Rsignal[1][t3,t2,t1]
#                            #+np.exp(1j*(-omega1*time1+omega3*time3))*Rsignal[0][t3,t2,t1] ).real

        print "done."

        spectra = []
        for t2 in range(len(time2s)):
            spectra.append( spectrum[:,t2,:] )

        return energy1s, energy3s, time2s, spectra

    def calculate_R3(self, rho_g, time2_min, time2_max, dt2, time_final, dt, lioupath='total', is_damped=True):
        time2s = np.arange(time2_min, time2_max, dt2)
        times = np.arange(0.0, time_final, dt)
        dt2_over_dt = int(dt2/dt)

        mu_p = np.tril(self.dipole_site)
        mu_m = np.triu(self.dipole_site)

        mu_rp_ese = [ [mu_m,'R'], [mu_p,'L'], [mu_p,'R'],  1 ]  # R2
        mu_rp_gsb = [ [mu_m,'R'], [mu_p,'R'], [mu_p,'L'],  1 ]  # R3
        mu_rp_esa = [ [mu_m,'R'], [mu_p,'L'], [mu_p,'L'], -1 ]  # R5

        mu_nr_ese = [ [mu_p,'L'], [mu_m,'R'], [mu_p,'R'],  1 ]  # R1
        mu_nr_gsb = [ [mu_p,'L'], [mu_m,'L'], [mu_p,'L'],  1 ]  # R4
        mu_nr_esa = [ [mu_p,'L'], [mu_m,'R'], [mu_p,'L'], -1 ]  # R6

        # This part changes based on the desired spectrum
        # e.g. just the ESA of the non-rephasing spectrum:
        # mu_rp = []
        # mu_nr = [mu_nr_esa]
        
        # single liouville pathway
        assert(lioupath in ['total','allrp','allnr','allese','allesa','allgsb','rpese','nrese','rpesa','nresa','rpgsb','nrgsb'])
        if lioupath =='total':
            mu_rp = [mu_rp_ese, mu_rp_gsb, mu_rp_esa]
            mu_nr = [mu_nr_ese, mu_nr_gsb, mu_nr_esa]

        elif lioupath =='allrp':
            mu_rp = [mu_rp_ese, mu_rp_gsb, mu_rp_esa]
            mu_nr = []

        elif lioupath =='allnr':
            mu_rp = []
            mu_nr = [mu_nr_ese, mu_nr_gsb, mu_nr_esa]

        elif lioupath =='allese':
            mu_rp = [mu_rp_ese]
            mu_nr = [mu_nr_ese]
        
        elif lioupath =='allesa':
            mu_rp = [mu_rp_esa]
            mu_nr = [mu_nr_esa]
        
        elif lioupath =='allgsb':
            mu_rp = [mu_rp_gsb]
            mu_nr = [mu_nr_gsb]

        elif lioupath =='rpese':
            mu_rp = [mu_rp_ese]
            mu_nr = []

        elif lioupath =='rpgsb':
            mu_rp = [mu_rp_gsb]
            mu_nr = []

        elif lioupath =='rpesa':
            mu_rp = [mu_rp_esa]
            mu_nr = []

        elif lioupath =='nrese':
            mu_rp = []
            mu_nr = [mu_nr_ese]

        elif lioupath =='nrgsb':
            mu_rp = []
            mu_nr = [mu_nr_gsb]

        elif lioupath =='nresa':
            mu_rp = []
            mu_nr = [mu_nr_esa]

        # Total 2D spectrum:
        # mu_rp = [mu_rp_ese, mu_rp_gsb, mu_rp_esa]
        # mu_nr = [mu_nr_ese, mu_nr_gsb, mu_nr_esa]

        # This part never changes
        mu_ops = [mu_rp, mu_nr]

        try:
            rho_g_bath = self.dynamics.initialize_from_rdm(rho_g)
        except AttributeError:
            rho_g_bath = rho_g.copy()

        Rsignal = []
        Rsignal.append(np.zeros( (len(times),len(time2s),len(times)), dtype=np.complex))
        Rsignal.append(np.zeros( (len(times),len(time2s),len(times)), dtype=np.complex))

        t_damp = 0.
        for ph in [0,1]:
            for mu_op in mu_ops[ph]:
                rho_0 = self.act(mu_op[0],rho_g_bath)
                time1s, rhos_t1 = self.dynamics.propagate_full(
                                    rho_0, 0.0, time_final+time2_max+time_final, dt)

                for t1, time1 in enumerate(times):
                    rho_t1 = rhos_t1[t1]
                    rho_t1_0 = self.act(mu_op[1],rho_t1)
                    _time2s, rhos_t2 = self.dynamics.propagate_full(
                                        rho_t1_0, time1+time2_min, time1+time2_max, dt)

                    for t2, time2 in enumerate(time2s):
                        rho_t2 = rhos_t2[t2*dt2_over_dt]
                        rho_t2_0 = self.act(mu_op[2],rho_t2)
                        time3s, rhos_t3 = self.dynamics.propagate_full(
                                            rho_t2_0, time1+time2, time1+time2+time_final, dt)
                        try:
                            rhos_t3 = self.dynamics.reduce_to_rdm(rhos_t3)
                        except AttributeError:
                            pass

                        for t3, time3 in enumerate(time3s):
                            rho_t3 = rhos_t3[t3]
                            sign = mu_op[3]
                            Rsignal[ph][t3,t2,t1] += sign*np.trace(np.dot(mu_m,rho_t3))
                            if is_damped and t1 >= t_damp and t3 >= t_damp:
                        
                                Rsignal[ph][t3,t2,t1] *= switch_to_zero(t3,t_damp,len(times)) 
                                Rsignal[ph][t3,t2,t1] *= switch_to_zero(t1,t_damp,len(times))
        return Rsignal


    def act(self, opside, rho):
        op, side = opside
        if side == 'L':
            try:
                return self.dynamics.act_from_left(op,rho)
            except AttributeError:
                return np.dot(op,rho)
        else:
            try:
                return self.dynamics.act_from_right(op,rho)
            except AttributeError:
                return np.dot(rho,op)
            


def switch_to_zero(x,x0,x1):
    '''Switching function to go from 1 at x0 to 0 at x1'''
    # Cubic spline with zero derivative on both ends
    y = (x-x0)/(x1-x0)
    return 1 - 3*y**2 + 2*y**3

def convert_to_xx(ham_sys_x, ham_sysbath_x, dipole_x):
    import scipy
    # Two-exciton Hamiltonian
    nx = ham_sys_x.shape[0]
    nxx = nx*(nx-1)/2
    ham_sys_xx = np.zeros((nxx,nxx))
    mn = 0
    for m in range(nx):
        for n in range(m):
            if m != n:
                ham_sys_xx[mn,mn] = ham_sys_x[m,m] + ham_sys_x[n,n]
                op = 0
                for o in range(nx):
                    for p in range(o):
                        if o != p:
                            ham_sys_xx[mn,op] = ham_sys_x[m,o]*(n==p) + ham_sys_x[n,p]*(m==o)
                        op += 1
            mn += 1

    ham_sys = scipy.linalg.block_diag([[0]], ham_sys_x, ham_sys_xx)
    nsite = ham_sys.shape[0]

    nbath = len(ham_sysbath_x)
    ham_sysbath = []
    for b in range(nbath):
        ham_sysbath_xx_b = np.zeros((nxx,nxx))
        mn = 0
        for m in range(nx):
            for n in range(m):
                ham_sysbath_xx_b[mn,mn] = ham_sysbath_x[b][m,m] + ham_sysbath_x[b][n,n]
                mn += 1
        ham_sysbath.append( 
            scipy.linalg.block_diag([[0]], ham_sysbath_x[b], ham_sysbath_xx_b) )

    if len(np.array(dipole_x).shape) == 1:
        dipole_x_vec = np.array(dipole_x)
        dipole_x = np.zeros((1+nx,1+nx))
        for i in range(nx):
            dipole_x[0,i+1] = dipole_x[i+1,0] = dipole_x_vec[i]

    dipole_xx = np.zeros((nx,nxx))
    for i in range(nx):
        mn = 0
        for m in range(nx):
            for n in range(m):
                dipole_xx[i,mn] = dipole_x[0,m+1]*(i==n) + dipole_x[0,n+1]*(i==m)
                mn += 1

    dipole = np.zeros((1+nx+nxx,1+nx+nxx))
    for i in range(nx):
        dipole[0,i+1] = dipole[i+1,0] = dipole_x[0,i+1]
        mn = 0
        for m in range(nx):
            for n in range(m):
                dipole[i+1,mn+1+nx] = dipole[mn+1+nx,i+1] = dipole_xx[i,mn]
                mn += 1

    for i in range(nx):
        dipole[i+1,i+1] = dipole_x[i+1,i+1]

    #mn = 0 
    #for m in range(nx):
    #    for n in range(m):
    #        dipole[mn+1+nx,mn+1+nx] = dipole_x[1+m,1+m] + dipole_x[1+n,1+n]

    return ham_sys, ham_sysbath, dipole
