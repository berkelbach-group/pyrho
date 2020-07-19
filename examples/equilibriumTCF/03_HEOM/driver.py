# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:59:36 2020

@author: petra
"""
import numpy as np
from pyrho import ham, heom
import pyrho.tcf.equilTCF as equilTCF
import re
import datetime

def read_2DMatrix(_filename, _len):
    _matrix = np.zeros((_len,_len))
    lineList = [line.rstrip('\n') for line in open(_filename + '.dat')]
    assert _len == len(lineList)
    for i in range(_len):
        for j in range(_len):
            _line = re.split(r' ', lineList[i])
            _matrix[i,j] = float(_line[j])
    return _matrix

def read_Jparameters(_filename, _N):
    _plist = []
    _wlist = []
    _glist = []
    lineList = [line.rstrip('\n') for line in open(_filename + '.dat')]
    for i in range(_N):
        _line = re.split(r' ', lineList[i+1])
        _plist.append(float(_line[0]))
        _wlist.append(float(_line[1]))
        _glist.append(float(_line[2]))
    return _plist, _wlist, _glist

def main():
    nsys = 3
    nbath = 1

    kT = 1.0
    beta = 1.0/kT
    hbar = 1.0 
    
    K_list = [0]
    L_list = [2]
    single_harmonic = False
    # Propagate until equilibrium
    t_equil, dt = 2000., 0.1
    # Propagate to retrieve TCF
    T_init, T_final, dT = 0., 2000.0, 0.1
    e_min, e_max, de = -2., +2., 0.01 # in unit of energy, cm-1

    # Spectral densities - a list of length 'nbath'
    sd_type = 'custom'
    file_Jw = '../02_spectral_density/Data_fitted_Jw_parameters_updated_iteration03'
    N_lorentz = 3
    p_list, w_list, g_list = read_Jparameters(file_Jw, N_lorentz)
    customPara = [p_list, w_list, g_list] # a 2d list
    spec_densities = [['custom', customPara]] * nbath

    File_Hs_DVR = '../01_DVR/Matrix_Hs'
    File_rho_0 = '../01_DVR/Matrix_rho_0'
    File_X_operator_DVR = '../01_DVR/Matrix_q_operator'

    # System Hamiltonian, should be in unit of cm-1
    ham_sys = read_2DMatrix(File_Hs_DVR, nsys)
    # position operator
    position = read_2DMatrix(File_X_operator_DVR, nsys)
    # System part of the the system-bath interaction
    # - a list of length 'nbath'
    # - currently assumes that each term has uncorrelated bath operators
    ham_sysbath = []
    ham_sysbath.append(position)

    # Initial reduced density matrix of the system 
    rho_0 = read_2DMatrix(File_rho_0, nsys)
    
    my_ham = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT, hbar=hbar)
    for K in K_list:
        for L in L_list:
            my_method = heom.HEOM(my_ham, L=L, K=K)
            my_equilTCF = equilTCF.EquilibriumTCF(position, my_method)

            # First step: Equilibrate Dynamics
            rho_hierarchy_equil, time1s, rhos_DVR = my_equilTCF.EquilibrateDynamics(rho_0, t_equil, dt)
            with open('pop_beta-%0.2f_dt-%0.2f_teqil-%0.0f_L-%d_K-%d_'%(beta, dt, t_equil, L, K)+sd_type+'_500.dat', 'w') as f:
                for (time, rho_DVR) in zip(time1s, rhos_DVR):
                    f.write('%0.8f '%(time))
                    for i in range(nsys):
                        f.write('%0.8f '%(rho_DVR[i,i].real))
                    f.write('\n')

            # Second step: Evaluate TCF
            if single_harmonic == True:
                spec_densities = [[sd_type, 0.0, omega_c]]*nbath
                my_ham = ham.Hamiltonian(ham_sys, ham_sysbath, spec_densities, kT, hbar=hbar)
                my_method = heom.HEOM(my_ham, L=L, K=K)
                my_equilTCF = equilTCF.EquilibriumTCF(position, my_method)                    
            time2s, Ct, Ct_damp, omegas, Cw, Cw_damp, Dw = my_equilTCF.EvaluateTCF(rho_hierarchy_equil, T_init, T_final, dT, e_min, e_max, de)


            with open('Ct_beta-%0.2f_teqil-%0.0f_dT-%0.3f_Tfinal-%0.0f_L-%d_K-%d.dat'%(beta, t_equil, dT, T_final, L, K), 'w') as f1:
                for (t, intensity) in zip(time2s, Ct):
                    f1.write('%0.8f %0.8f %0.8f\n'%(t, intensity.real, intensity.imag))
            with open('CtDamp_beta-%0.2f_teqil-%0.0f_dT-%0.3f_Tfinal-%0.0f_L-%d_K-%d.dat'%(beta, t_equil, dT, T_final, L, K), 'w') as f2:
                for (t, intensity) in zip(time2s, Ct_damp):
                    f2.write('%0.8f %0.8f %0.8f\n'%(t, intensity.real, intensity.imag))
            with open('Cw_beta-%0.2f_teqil-%0.0f_dT-%0.3f_Tfinal-%0.0f_L-%d_K-%d.dat'%(beta, t_equil, dT, T_final, L, K), 'w') as f3:
                for (omega, intensity) in zip(omegas, Cw):
                    f3.write('%0.8f %0.8f\n'%(omega, intensity)) 
            with open('CwDamp_beta-%0.2f_teqil-%0.0f_dT-%0.3f_Tfinal-%0.0f_L-%d_K-%d.dat'%(beta, t_equil, dT, T_final, L, K), 'w') as f3:
                for (omega, intensity) in zip(omegas, Cw_damp):
                    f3.write('%0.8f %0.8f\n'%(omega, intensity))                    
            with open('Dw_beta-%0.2f_teqil-%0.0f_dT-%0.3f_Tfinal-%0.0f_L-%d_K-%d.dat'%(beta, t_equil, dT, T_final, L, K), 'w') as f5:
                for (omega, intensity) in zip(omegas, Dw):
                    f5.write('%0.8f %0.8f %0.8f\n'%(omega, intensity.real, intensity.imag))                    


print(datetime.datetime.now())
if __name__ == '__main__':
    main()
print(datetime.datetime.now())
