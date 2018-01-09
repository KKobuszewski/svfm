#cython: boundscheck=False, nonecheck=False, cdivision=True
from __future__ import print_function, division

import math

import numpy as np
cimport numpy as np
cimport cython

from cython.parallel import parallel, prange


# ========================================================  C FUNCTIONS  ============================================================

cdef extern from "cfunctions.h" nogil:
    void _initialize(int nelems);
    void _finalize();
    void get_forces(double* xV, double* yV, double* zV, double* dx, double* dy, double* rN, double complex* fx, double complex* fy, double* fVN_params);
    void get_forces_multiple_impurities(
                    double* xV, double* yV, double* zV, double* dx, double* dy, double* rN,
                    double complex* fx, double complex* fy, double* fVN_params, const int M );


def initialize(nelems):
    _initialize(int(nelems))


def finalize():
    _finalize()


def create_lattice(a, nx_beg,nx_end,ny_beg,ny_end,nz_beg,nz_end, theta=0., phi=0.,primitive_cell='bcc'):
    """
    @param a                  - lattice constant
    @param nx_beg             - number of lattice consants before (0,0,0) point in x direction
    @param nx_end             - number of lattice consants following (0,0,0) point in x direction
    @param ny_beg             - number of lattice consants before (0,0,0) point in x direction
    @param ny_end             - number of lattice consants following (0,0,0) point in x direction
    @param nz_beg             - number of lattice consants before (0,0,0) point in x direction
    @param nz_end             - number of lattice consants following (0,0,0) point in x direction
    @param theta              - elevation of TODO: should it be in radians???
    @param phi                - 
    
    @return                   - positions of lattice nodes, each row contains [x,y,z] of every 
    """
    
    
    NX,NY,NZ = np.meshgrid(np.arange(nx_beg,nx_end+1),
                           np.arange(ny_beg,ny_end+1),
                           np.arange(nz_beg,nz_end+1),
                           indexing='ij')
    
    print(NX.shape,NX.size)
    #print(NX)
    
    # lattice poits
    lattice_points = None
    if   primitive_cell == 'bcc':
        lattice_points = np.empty( [2*NX.size,3], order='C', dtype=np.float64 )
        
        for it,(nx,ny,nz) in enumerate(zip(NX.flatten(),NY.flatten(),NZ.flatten())):
            print(nx,ny,nz)
            lattice_points[it,0] = a*nx
            lattice_points[it,1] = a*ny
            lattice_points[it,2] = a*nz
            lattice_points[it+NX.size,0] = a*(nx+0.5)
            lattice_points[it+NX.size,1] = a*(ny+0.5)
            lattice_points[it+NX.size,2] = a*(nz+0.5)
    elif primitive_cell == 'fcc':
        lattice_points = np.empty( [4*NX.size,3], order='C', dtype=np.float64 )
        
        for it,(nx,ny,nz) in enumerate(zip(NX.flatten(),NY.flatten(),NZ.flatten())):
            print(nx,ny,nz)
            lattice_points[it,0] = a*nx
            lattice_points[it,1] = a*ny
            lattice_points[it,2] = a*nz
            
            lattice_points[it+  NX.size,0] = a*(nx+0.5)
            lattice_points[it+  NX.size,1] = a*(ny+0.5)
            lattice_points[it+  NX.size,2] = a*(nz+0.0)
            
            lattice_points[it+2*NX.size,0] = a*(nx+0.5)
            lattice_points[it+2*NX.size,1] = a*(ny+0.0)
            lattice_points[it+2*NX.size,2] = a*(nz+0.5)
            
            lattice_points[it+3*NX.size,0] = a*(nx+0.0)
            lattice_points[it+3*NX.size,1] = a*(ny+0.5)
            lattice_points[it+3*NX.size,2] = a*(nz+0.5)
    
    # apply rotation
    rot_theta = np.array( [[ math.cos(theta),              0., math.sin(theta)],
                           [              0.,              1.,              0.],
                           [-math.sin(theta),              0., math.cos(theta)]] )
    rot_phi   = np.array( [[ math.cos(phi), math.sin(phi),              0.],
                           [-math.sin(phi), math.cos(phi),              0.],
                           [            0.,              0.,            1.]] )
    rot = np.dot(rot_phi,rot_theta)
    for it in range(lattice_points.shape[0]):
        lattice_points[it,:] = rot.dot(lattice_points[it,:])
    
    
    return lattice_points










cdef omega1(k,double a=1.0):
    return k**2

cdef omega2_numpy(k,double a=1.0, kappa=4.0*np.pi):
    w = np.empty(k.shape)
    with np.errstate(divide='ignore'):
        w = k**2 * np.log(1.0/np.abs(a*k)) * kappa / (4.0*np.pi)
        w[ np.where(np.abs(k) < 1e-15) ] = 0.0
        w[ np.where(np.abs(k) > 1.0/a)   ] = 0.0
    return w


cdef omega2(double k,double a=1.0, kappa=4.0*np.pi):
    if ( (math.abs(k) < 1e-15) or (math.abs(k) > 1.0/a) ):
        return 0.0
    else :
        return k**2 * math.log(1.0/math.abs(a*k)) * kappa / (4.0*math.pi)




cpdef f1(double t,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] xk,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] dxdt,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] x,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] y,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] z,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dx,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz,
         double vext_x, double vext_y, double eta, double Tv, double kappa, double rho):
    """
    Bennet Link model without external force.
    """
    cdef int N = xk.size//2
    
    for it in range(N):
    #for it in prange(N, schedule='static', nogil=True): # TODO
        dxdt[  it] = ( Tv*kappa*kz[it]*kz[it]/(4.0*np.pi) * ( -eta*xk[it] + 1.0*xk[N+it])  ) / (1.0 + eta*eta)  # - eta*fx + 1.0*fy
        dxdt[N+it] = ( Tv*kappa*kz[it]*kz[it]/(4.0*np.pi) * ( -1.0*xk[it] - eta*xk[N+it])  ) / (1.0 + eta*eta)  # - 1.0*fy - eta*fy
    
    #dxdt[:N] = ( kz*kz * ( -eta*x[:N] + 1.0*x[N:]) ) / (1.0 + eta*eta) + fy  # - eta*fx + 1.0*fy
    #dxdt[N:] = ( kz*kz * ( -1.0*x[:N] - eta*x[N:]) ) / (1.0 + eta*eta) - fx  # - 1.0*fy - eta*fy
    dxdt[0]  += vext_x*N
    dxdt[N]  += vext_y*N
    
    
    return dxdt


cpdef f2(double t,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] xk,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] dxdt,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] x,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] y,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] z,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dx,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz,
         double vext_x, double vext_y, double eta, double a, double kappa, double rho):
    """
    Kelvin dispersion model without external force. Tv changes to effective vortex core radius a, that corresponds to UV cutoff of theory.
    """
    cdef int N = xk.size//2
    
    for it in range(N):
    #for it in prange(N, schedule='static', nogil=True): # TODO
        dxdt[  it] = ( omega2(kz[it],a,kappa) * ( -eta*xk[it] + 1.0*xk[N+it])  ) / (1.0 + eta*eta)  # - eta*fx + 1.0*fy
        dxdt[N+it] = ( omega2(kz[it],a,kappa) * ( -1.0*xk[it] - eta*xk[N+it])  ) / (1.0 + eta*eta)  # - 1.0*fy - eta*fy
    
    dxdt[0]  += vext_x*N
    dxdt[N]  += vext_y*N
    
    
    return dxdt


cpdef f3(double t,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] xk,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] dxdt,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] x,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] y,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] z,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dx,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] fx,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] fy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] fVN_params,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] rN,
         double vext_x, double vext_y, double eta, double a, double kappa, double rho):
    """
    Kelvin dispersion model with presence of external force generated by single impurity.
    """
    cdef int N = xk.size//2
    #cdef np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz = 2*np.pi* scipy.fftpack.fftfreq(N, d=(z[1]-z[0]))
    #cdef np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dxdt = np.empty(2*N)
    
    # get positions of vortex filaments
    x = np.ascontiguousarray( np.real( np.fft.ifft(xk[:N]) ) )
    y = np.ascontiguousarray( np.real( np.fft.ifft(xk[N:]) ) )
    
    
    # get first derivatives of positions
    dxdt[:N] = 1j*kz*xk[:N]
    dxdt[N:] = 1j*kz*xk[N:]
    dxdt[N//2] = 0
    dx = np.ascontiguousarray( np.fft.ifft(dxdt[:N]).real )
    dxdt[N+N//2] = 0
    dy = np.ascontiguousarray( np.fft.ifft(dxdt[N:]).real )
    
    # evaluate forces acting on a nucleus
    get_forces(&x[0], &y[0], &z[0], &dx[0], &dy[0], &rN[0], &fx[0], &fy[0], &fVN_params[0])
    
    # transform forces to Fourier space
    fx = np.ascontiguousarray( np.fft.fft(fx) )
    fy = np.ascontiguousarray( np.fft.fft(fy) )
    
    
    for it in range(N):
    #for it in prange(N, schedule='static', nogil=True): # TODO
        dxdt[  it] = ( omega2(kz[it],a,kappa) * (-eta*xk[it] + 1.0*xk[N+it]) + (-eta*fx[it] + 1.0*fy[it])/(rho*kappa) ) / (1.0 + eta*eta) #+ fy[it]   
        dxdt[N+it] = ( omega2(kz[it],a,kappa) * (-1.0*xk[it] - eta*xk[N+it]) + (-1.0*fx[it] - eta*fy[it])/(rho*kappa) ) / (1.0 + eta*eta) #- fx[it]   
    
    #dxdt[:N] = ( kz*kz * ( -eta*x[:N] + 1.0*x[N:]) ) / (1.0 + eta*eta) + fy  # - eta*fx + 1.0*fy
    #dxdt[N:] = ( kz*kz * ( -1.0*x[:N] - eta*x[N:]) ) / (1.0 + eta*eta) - fx  # - 1.0*fy - eta*fy
    dxdt[0]  += vext_x*N
    dxdt[N]  += vext_y*N
    
    
    return dxdt


cpdef f4(double t,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] xk,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] dxdt,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] x,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] y,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] z,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dx,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] fx,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] fy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] fVN_params,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] rN,
         double vext_x, double vext_y, double eta, double Tv, double kappa, double rho):
    cdef int N = xk.size//2
    #cdef np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz = 2*np.pi* scipy.fftpack.fftfreq(N, d=(z[1]-z[0]))
    #cdef np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dxdt = np.empty(2*N)
    
    # get positions of vortex filaments
    x = np.ascontiguousarray( np.real( np.fft.ifft(xk[:N]) ) )
    y = np.ascontiguousarray( np.real( np.fft.ifft(xk[N:]) ) )
    
    
    # get first derivatives of positions
    dxdt[:N] = 1j*kz*xk[:N]
    dxdt[N:] = 1j*kz*xk[N:]
    dxdt[N//2] = 0
    dx = np.ascontiguousarray( np.fft.ifft(dxdt[:N]).real )
    dxdt[N+N//2] = 0
    dy = np.ascontiguousarray( np.fft.ifft(dxdt[N:]).real )
    
    # evaluate forces acting on a nucleus
    get_forces(&x[0], &y[0], &z[0], &dx[0], &dy[0], &rN[0], &fx[0], &fy[0], &fVN_params[0])
    
    # transform forces to Fourier space
    fx = np.ascontiguousarray( np.fft.fft(fx) )
    fy = np.ascontiguousarray( np.fft.fft(fy) )
    
    #eta = eta #/(rho*kappa)
    
    for it in range(N):
    #for it in prange(N, schedule='static', nogil=True): # TODO
        dxdt[  it] = ( Tv*kappa*kz[it]*kz[it]/(4.0*np.pi) * ( -eta*xk[it] + 1.0*xk[N+it]) + (-eta*fx[it] + 1.0*fy[it])/(rho*kappa) ) / (1.0 + eta*eta)  # - eta*fx + 1.0*fy
        dxdt[N+it] = ( Tv*kappa*kz[it]*kz[it]/(4.0*np.pi) * ( -1.0*xk[it] - eta*xk[N+it]) + (-1.0*fx[it] - eta*fy[it])/(rho*kappa) ) / (1.0 + eta*eta)  # - 1.0*fy - eta*fy
    
    #dxdt[:N] = ( kz*kz * ( -eta*x[:N] + 1.0*x[N:]) ) / (1.0 + eta*eta) + fy  # - eta*fx + 1.0*fy
    #dxdt[N:] = ( kz*kz * ( -1.0*x[:N] - eta*x[N:]) ) / (1.0 + eta*eta) - fx  # - 1.0*fy - eta*fy
    dxdt[0]  += vext_x*N
    dxdt[N]  += vext_y*N
    
    
    return dxdt


cpdef f5(double t,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] xk,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] dxdt,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] x,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] y,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] z,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dx,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] fx,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] fy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] fVN_params,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] rN,
         double vext_x, double vext_y, double eta, double Tv, double kappa, double rho):
    cdef int N = xk.size//2
    #cdef np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz = 2*np.pi* scipy.fftpack.fftfreq(N, d=(z[1]-z[0]))
    #cdef np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dxdt = np.empty(2*N)
    
    # get positions of vortex filaments
    x = np.ascontiguousarray( np.real( np.fft.ifft(xk[:N]) ) )
    y = np.ascontiguousarray( np.real( np.fft.ifft(xk[N:]) ) )
    
    
    # get first derivatives of positions
    dxdt[:N] = 1j*kz*xk[:N]
    dxdt[N:] = 1j*kz*xk[N:]
    dxdt[N//2] = 0
    dx = np.ascontiguousarray( np.fft.ifft(dxdt[:N]).real )
    dxdt[N+N//2] = 0
    dy = np.ascontiguousarray( np.fft.ifft(dxdt[N:]).real )
    
    # evaluate forces acting on a nucleus
    get_forces(&x[0], &y[0], &z[0], &dx[0], &dy[0], &rN[0], &fx[0], &fy[0], &fVN_params[0])
    
    # transform forces to Fourier space
    fx = np.ascontiguousarray( np.fft.fft(fx) )
    fy = np.ascontiguousarray( np.fft.fft(fy) )
    #cdef complex[:] fx = np.ascontiguousarray( np.fft.fft(fx) )
    #cdef complex[:] fy = np.ascontiguousarray( np.fft.fft(fy) )
    
    #eta = eta #/(rho*kappa)
    
    for it in range(N):
    #for it in prange(N, schedule='static', nogil=True): # TODO
        dxdt[  it] = ( Tv*kappa*kz[it]*kz[it]/(4.0*np.pi) * ( -eta*xk[it] + 1.0*xk[N+it]) + (-eta*fx[it] + 1.0*fy[it])/(rho*kappa) ) / (1.0 + eta*eta)  # - eta*fx + 1.0*fy
        dxdt[N+it] = ( Tv*kappa*kz[it]*kz[it]/(4.0*np.pi) * ( -1.0*xk[it] - eta*xk[N+it]) + (-1.0*fx[it] - eta*fy[it])/(rho*kappa) ) / (1.0 + eta*eta)  # - 1.0*fy - eta*fy
    
    #dxdt[:N] = ( kz*kz * ( -eta*x[:N] + 1.0*x[N:]) ) / (1.0 + eta*eta) + fy  # - eta*fx + 1.0*fy
    #dxdt[N:] = ( kz*kz * ( -1.0*x[:N] - eta*x[N:]) ) / (1.0 + eta*eta) - fx  # - 1.0*fy - eta*fy
    dxdt[0]  += vext_x*N
    dxdt[N]  += vext_y*N
    
    
    return dxdt


cpdef multiple_impurities(double t,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] xk,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] dxdt,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] x,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] y,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] z,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dx,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] fx,
         np.ndarray[np.complex128_t,ndim=1,negative_indices=False,mode='c'] fy,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] fVN_params,
         np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] rN,
         double vext_x, double vext_y, double eta, double Tv, double kappa, double rho):
    cdef int N = xk.size//2
    #cdef np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] kz = 2*np.pi* scipy.fftpack.fftfreq(N, d=(z[1]-z[0]))
    #cdef np.ndarray[np.float64_t,ndim=1,negative_indices=False,mode='c'] dxdt = np.empty(2*N)
    
    
    # get positions of vortex filaments
    x[:] = np.ascontiguousarray( np.real( np.fft.ifft(xk[:N]) ) )
    y[:] = np.ascontiguousarray( np.real( np.fft.ifft(xk[N:]) ) )
    
    
    # get first derivatives of positions
    dxdt[:N] = 1j*kz*xk[:N]
    dxdt[N:] = 1j*kz*xk[N:]
    dxdt[N//2] = 0
    dx = np.ascontiguousarray( np.fft.ifft(dxdt[:N]).real )
    dxdt[N+N//2] = 0
    dy = np.ascontiguousarray( np.fft.ifft(dxdt[N:]).real )
    
    # evaluate forces acting on a nucleus
    get_forces_multiple_impurities(&x[0], &y[0], &z[0], &dx[0], &dy[0], &rN[0], &fx[0], &fy[0], &fVN_params[0],rN.size)
    
    # transform forces to Fourier space
    fx = np.ascontiguousarray( np.fft.fft(fx) )
    fy = np.ascontiguousarray( np.fft.fft(fy) )
    #cdef complex[:] fx = np.ascontiguousarray( np.fft.fft(fx) )
    #cdef complex[:] fy = np.ascontiguousarray( np.fft.fft(fy) )
    
    #eta = eta #/(rho*kappa)
    
    for it in range(N):
    #for it in prange(N, schedule='static', nogil=True): # TODO
        dxdt[  it] = ( Tv*kappa*kz[it]*kz[it]/(4.0*np.pi) * ( -eta*xk[it] + 1.0*xk[N+it]) + (-eta*fx[it] + 1.0*fy[it])/(rho*kappa) ) / (1.0 + eta*eta)  # - eta*fx + 1.0*fy
        dxdt[N+it] = ( Tv*kappa*kz[it]*kz[it]/(4.0*np.pi) * ( -1.0*xk[it] - eta*xk[N+it]) + (-1.0*fx[it] - eta*fy[it])/(rho*kappa) ) / (1.0 + eta*eta)  # - 1.0*fy - eta*fy
    
    #dxdt[:N] = ( kz*kz * ( -eta*x[:N] + 1.0*x[N:]) ) / (1.0 + eta*eta) + fy  # - eta*fx + 1.0*fy
    #dxdt[N:] = ( kz*kz * ( -1.0*x[:N] - eta*x[N:]) ) / (1.0 + eta*eta) - fx  # - 1.0*fy - eta*fy
    dxdt[0]  += vext_x*N
    dxdt[N]  += vext_y*N
    
    
    return dxdt
