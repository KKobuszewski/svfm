#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function,division

import os, sys, glob, mmap, struct

import math
import numpy as np
from scipy.integrate import ode
import scipy.fftpack
from scipy.interpolate import splev, splrep

#import mayavi
#from mayavi import mlab



# simulation
sys.path.append('/home/kobuszewski/mgr/svfm/python')
import cfunctions
from svfm_datautil import SVFMDataWriter

if __name__ == '__main__':
    
    
    """
    usage: python Comparison2.py <prefix> <eta> <N>
    
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
    vext_x = 0.0
    vext_y = 0.0
    eta    = 0.05        # this is reduced eta: eta/rho/kappa
    if argc > 2:
        eta = float(sys.argv[2])
    xmlinfo.add_parameter('eta',eta)
    
    mn     = 939.56563 # MeV 
    kappa  = 0.6548
    xmlinfo.add_parameter('neutron mass MeV',mn); xmlinfo.add_parameter('kappa',kappa);
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
    
    xmlinfo.add_parameter('kF,n',kF)
    xmlinfo.add_parameter('delta',delta)
    xmlinfo.add_parameter('rho_n',rho)
    xmlinfo.add_parameter('T_V',T_estimated)
    
    
    # ==================== SIMULATION PARAMETERS - TIME =============================
    
    
    dt   = 1.0
    tmax = 100000.
    nt  = int(tmax/dt)+1
    
    nom = int(tmax/1000.)+1
    time = np.linspace(0.,tmax,nom,endpoint=True)
    
    xmlinfo.add_parameter('dt',dt)
    xmlinfo.add_parameter('tmax',tmax)
    xmlinfo.add_parameter('nom',nom)
    
    
    # ======================== LATTICE =======================================
    
    a_lattice = 50
    if prefix == 'vor8A':
        a_lattice = 30
    xmlinfo.add_parameter('lattice const.',a_lattice)
    
    #RCM       = np.zeros(3*nnuclei)                           # center of mass of nuclei
    #RCM[2::3]  = np.linspace(-1,nnuclei-2,nnuclei)*a_lattice + a_lattice//2
    #for it,rcm in enumerate(zip(RCM[ ::3],RCM[1::3],RCM[2::3])):
    #    print('{:d}. nucleus:'.format(it),rcm)
    
    lattice_type = 'bcc'; xmlinfo.add_parameter('lattice type',lattice_type);
    
    lattice_points = cfunctions.create_lattice(a_lattice, -1,1,-1,1,-1,2, theta=0., phi=0., primitive_cell=lattice_type)
    lattice_points = lattice_points[np.where(lattice_points[:,2] <= 2.0*a_lattice)[0],:] # truncate in z direction to get periodic bonduary conditions
    nnuclei = lattice_points.shape[0]
    print('Number of nuclei:',nnuclei); xmlinfo.add_parameter('number of nuclei',nnuclei);
    
    #mlab.points3d(lattice_points[:,0],lattice_points[:,1],lattice_points[:,2],np.ones(lattice_points.shape[0])*8.0, scale_factor=1.0)
    RCM = lattice_points.flatten()
    
    #mlab.outline(extent=[-a_lattice,2*a_lattice,-a_lattice,2*a_lattice,-0.5*a_lattice,1.5*a_lattice]) # outline to present bonduaries
    #mlab.axes()
    #mlab.orientation_axes()
    
    
    # ====================================== VORTEX PARAMETERS ===================================================
    
    N = 128
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
    
    cfunctions.initialize(N)
    for _eta in [0.00,0.05]: #[0.00,0.01,0.02,0.05,0.10,0.15,0.20,0.25,0.5]:
        for vext in [0.0000,0.0001]: #[0.0000,0.0001,0.0002,0.0005,0.0010,0.0020,0.0050,0.0100,0.0200]:
            eta = _eta
            vext_x = vext
            print('eta:',eta)
            print('v: ',eta)
            
            # equation solving
            filename = 'BLink/'+ \
                        '{0}_sim_data_Tv={1:.2f}_eta={2:.2f}'.format(prefix,T_estimated*kappa/(4.0*np.pi),eta)+ \
                        '_N={0:d}_alatt={1:.2f}_vext={2:.1e}'.format(N,a_lattice,vext_x)
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
            #r = ode(func).set_integrator('dopri5')
            
            solver.set_initial_value(y0, time[0])
            solver.set_f_params( RCM )
            
            
            for it in range(nom-1):
                time1 = time[it]
                time2 = time[it+1]
            
            
                while (solver.t < time2):
                    # vortex filament model solution
                    # TODO: 
                    sol = solver.integrate(solver.t+dt)
                    N   = sol.size // 2
                    
                    print(solver.t)
                sim_data[it+1,:N] = scipy.fftpack.ifft(sol[:N]).real
                sim_data[it+1,N:] = scipy.fftpack.ifft(sol[N:]).real
                #mlab.plot3d(sim_data[it+1,:N], sim_data[it+1,N:], zv,line_width=2.5,tube_radius=0.5)
                sim_data.flush()
                
                nucl_data[it+1,:] = RCM
                nucl_data.flush()
                print('# saving vortex position to file')
            sim_data.flush()
            nucl_data.flush()
            
            xmlinfo.save_info(filename=filename)
            # end evolution
    cfunctions.finalize()
    #mlab.show()
