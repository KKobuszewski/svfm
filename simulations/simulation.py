#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function,division

import os, sys, glob, mmap, struct

import math
import numpy as np
from scipy.integrate import ode
import scipy.fftpack
from scipy.interpolate import splev, splrep

import timeit

#import mayavi
#from mayavi import mlab



# simulation - here need to set path to directory with module cfunctions 
#sys.path.append('/home/kobuszewski/mgr/svfm/python')
sys.path.append('/home/konrad/Pulpit/Praca Magisterska/svfm/python')
import cfunctions
from svfm_datautil import SVFMDataWriter

if __name__ == '__main__':
    
    
    """
    usage: python simulation.py <prefix> <eta> <vext> <a_lattice> <N> 
    
    """
    xmlinfo = SVFMDataWriter()
    argc = len(sys.argv)
    
    prefix = 'vor8A' # 'vor8A'
    if argc > 1:
        prefix = sys.argv[1]
    xmlinfo.add_parameter('prefix',prefix)
    
    # ============================================ SIMULATION ===============================================================
    
    # Global data
    
    HBARC=197.327053
    Np=50.0 
    
    
    
    # external flow / neutron liquid parameters
    eta    = 0.00        # this is reduced eta: eta/rho/kappa
    if argc > 2:
        eta = float(sys.argv[2])
    xmlinfo.add_parameter('eta',eta)
    
    
    vext = 0.0
    if argc > 3:
        vext = float(sys.argv[3])
    vext_x = vext
    vext_y = 0.0
    xmlinfo.add_parameter('vext_x',vext_x)
    xmlinfo.add_parameter('vext_y',vext_y)
    
    
    # superfluid parameters
    mn     = 939.56563 # MeV 
    kappa  = 0.6548
    xmlinfo.add_parameter('neutron_mass_MeV',mn); xmlinfo.add_parameter('kappa',kappa);
    kF     = None
    delta  = None
    rho    = None
    if   prefix == 'vor5A':
        rho    = 0.014*mn  # background neutron density * mass of neutron
        kF     = 0.75 # 1/fm
        delta  = 2.00 # MeV
        fVN_params = np.ascontiguousarray([-4549.83,-4525.79,-505.60,6455.46,6299.41,23440.34,-5640.24,341.73], 
                                           dtype=np.float64) # ctypes.c_double
        
        T_estimated = 1.4 / ( rho*kappa*kappa / 4.0 / np.pi )
    elif prefix == 'vor8A':
        rho    = 0.031*mn  # background neutron density * mass of neutron
        kF     = 0.97 # 1/fm
        delta  = 1.50 # MeV 
        fVN_params = np.ascontiguousarray([-3735.28,-988.42,-4257.99,6738.74,8430.92,35498.57,-7190.64,397.51], 
                                           dtype=np.float64) # ctypes.c_double
        
        T_estimated = 7.3 / ( rho*kappa*kappa / 4.0 / np.pi ) # from article, here assuming   d2udz2 \approx 1
    
    xi     = (HBARC**2 * kF) / (np.pi * delta * mn)
    a      = 1.0*xi         # a = 0.1
    Tv     = 3.0            # B. Link tension parameter 3 / 4 \pi \rho \kappa^2
    
    xmlinfo.add_parameter('kF_n',kF)
    xmlinfo.add_parameter('delta',delta)
    xmlinfo.add_parameter('rho_n',rho)
    xmlinfo.add_parameter('T_V',T_estimated)
    xmlinfo.add_parameter('xi_BCS',xi)
    
    
    # ======================== LATTICE =======================================
    
    a_lattice = 50
    if prefix == 'vor8A':
        a_lattice = 30
    if argc > 4:
        a_lattice = float(sys.argv[4])
    xmlinfo.add_parameter('lattice_const',a_lattice)
    
    #RCM       = np.zeros(3*nnuclei)                           # center of mass of nuclei
    #RCM[2::3]  = np.linspace(-1,nnuclei-2,nnuclei)*a_lattice + a_lattice//2
    #for it,rcm in enumerate(zip(RCM[ ::3],RCM[1::3],RCM[2::3])):
    #    print('{:d}. nucleus:'.format(it),rcm)
    
    lattice_type = 'bcc'; xmlinfo.add_parameter('lattice_type',lattice_type);
    lattice_points = cfunctions.create_lattice(a_lattice, -1,2,-1,2,-1,2, theta=0., phi=0., primitive_cell=lattice_type)
    lattice_points = lattice_points[np.where(lattice_points[:,0] <= 2.0*a_lattice)[0],:] # truncate in x direction to get periodic bonduary conditions
    lattice_points = lattice_points[np.where(lattice_points[:,1] <= 1.5*a_lattice)[0],:] # truncate in y direction to get periodic bonduary conditions
    lattice_points = lattice_points[np.where(lattice_points[:,1] >=-0.5*a_lattice)[0],:] # truncate in y direction to get periodic bonduary conditions
    lattice_points = lattice_points[np.where(lattice_points[:,2] <= 2.0*a_lattice)[0],:] # truncate in z direction to get periodic bonduary conditions
    nnuclei = lattice_points.shape[0]
    print('Number of nuclei:',nnuclei); xmlinfo.add_parameter('num_nuclei',nnuclei);
    
    #mlab.points3d(lattice_points[:,0],lattice_points[:,1],lattice_points[:,2],np.ones(lattice_points.shape[0])*8.0, scale_factor=1.0)
    RCM = lattice_points.flatten()
    
    #mlab.outline(extent=[-a_lattice,2*a_lattice,-a_lattice,2*a_lattice,-0.5*a_lattice,1.5*a_lattice]) # outline to present bonduaries
    #mlab.axes()
    #mlab.orientation_axes()
    
    
    # ====================================== VORTEX PARAMETERS ===================================================
    
    N = 128
    if argc > 5:
        N = int(sys.argv[5])
    zmin = -0.5*a_lattice
    zmax =  1.5*a_lattice     #-zmin
    xv   = np.zeros(N)
    yv   = (0.5*a_lattice)*np.ones(N)
    zv   = np.linspace(zmin,zmax,N,endpoint=False)
    dz   = zv[1] - zv[0]
    print('Periodic box z:',[zmin,zmax])
    
    xmlinfo.add_parameter('N',N)
    xmlinfo.add_parameter('zmin',zmin)
    xmlinfo.add_parameter('zmax',zmax)
    xmlinfo.add_parameter('dz',dz)
    
    
    # ==================== SIMULATION PARAMETERS - TIME =============================
    
    dt   = 1.0
    if prefix == 'vor5A':
        dt   = 5.0
    
    # total time of simulation in fm/c should be equal to time corresponding to moving free nuclei by one lattice constant
    tmax = 2.5*1e6
    if (vext > 1e-6):
        tmax = a_lattice / vext
    
    nt  = int(tmax/dt)+1
    
    nom = int(tmax/500.)+1
    time = np.linspace(0.,tmax,nom,endpoint=True)
    
    xmlinfo.add_parameter('dt',dt)
    xmlinfo.add_parameter('tmax',tmax)
    xmlinfo.add_parameter('nom',nom)
    
    
    
    # ============================================= EVOLUTION ===================================================================
    
    # initial condition to k-space
    y0     = np.empty(2*N,dtype = np.complex128)
    y0[:N] = scipy.fftpack.fft(xv)
    y0[N:] = scipy.fftpack.fft(yv)
    kz     = np.fft.fftshift(2*np.pi* scipy.fftpack.fftfreq(N, d=dz))
    
    # forces
    fx = np.zeros(N,dtype = np.complex128)
    fy = np.zeros(N,dtype = np.complex128)
    fz = np.zeros(N,dtype = np.float64)
    
    print()
    cfunctions.initialize(N)
    print('eta:            ',eta)
    print('v:             [',vext_x,',',vext_y,']')
    print('lattice const:  ',a_lattice)
    print('N:              ',N)
    print('dt:             ',dt)
    print('tmax:           ',tmax)
    print('nom:            ',nom)
    print()
    
    # equation solving
    #filename = '/home/kobuszewski/mgr/simulations/BLink/'+ \
    filename = os.getcwd() + '/BLink/' + \
               '{0}_sim_data_Tv={1:.2f}_eta={2:.2e}'.format(prefix, T_estimated*kappa*kappa*rho/(4.0*np.pi), eta) + \
               '_N={0:d}_{1:s}_alatt={2:.2f}_vext={3:.2e}'.format(N,lattice_type,a_lattice,vext_x)
    print('saving data to files:',filename+'.*')
    print()
    sim_data = np.memmap( filename+'.vort.bin', dtype=np.float64, mode='w+', shape=(nom,2*N) )
    nucl_data = np.memmap( filename+'.nucl.bin', dtype=np.float64, mode='w+', shape=(nom,RCM.size) )
    sim_data[0,:N] = xv
    sim_data[0,N:] = yv
    
    func = lambda t,x,rVN : cfunctions.multiple_impurities( t,x, 
                                np.empty(2*N,dtype = np.complex128), 
                                np.empty(N), np.empty(N), zv, 
                                np.empty(N), np.empty(N), 
                                2*np.pi*scipy.fftpack.fftfreq(N, d=dz), 
                                fx, fy, fVN_params, rVN,
                                vext_x, vext_y, eta, T_estimated, kappa, rho )
    solver = ode(func).set_integrator('zvode', method='bdf')
    #solver = ode(func).set_integrator('dopri5')
    
    solver.set_initial_value(y0, time[0])
    solver.set_f_params( RCM )
    
    
    for it in range(nom-1):
        e_t = timeit.default_timer()
        #print(e_t)
        time1 = time[it]
        time2 = time[it+1]
            
        while ( solver.successful() and (solver.t < time2) ):
            # vortex filament model solution
            # TODO: 
            sol = solver.integrate(solver.t+dt)
            N   = sol.size // 2
            
            #print(solver.t)
        sim_data[it+1,:N] = scipy.fftpack.ifft(sol[:N]).real
        sim_data[it+1,N:] = scipy.fftpack.ifft(sol[N:]).real
        #mlab.plot3d(sim_data[it+1,:N], sim_data[it+1,N:], zv,line_width=2.5,tube_radius=0.5)
        #sim_data.flush()
        
        nucl_data[it+1,:] = RCM
        #nucl_data.flush()
        print( '# saving vortex and nuclei positions to file at time: {0:8.2f} [fm/c]        computing time: {1:.3f} [s]'.format(time2,timeit.default_timer()-e_t) )
    sim_data.flush()
    nucl_data.flush()
    
    xmlinfo.save_info(filename=filename)
    # end evolution
    cfunctions.finalize()
    #mlab.show()
