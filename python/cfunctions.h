#ifndef __CFUNCTIONS_H__
#define __CFUNCTIONS_H__

//#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>


#include <omp.h>

#include <fftwdiff.h>


#include <complex>
#include <cmath>

//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#ifndef M_PI
#define M_PI 3.141592653589793
#endif





//double tension_params[NUM_TENSION_PARAMS];      // vortex tension parameter T_V
//double fVN_params[NUM_FVN_PARAMS];              // vortex-nucleus force Pade approximant coeficients    TODO: It should be tunable 


double* sina;
double* _dx;
double* _dy;
double *xVN, *yVN, *zVN;

int N;


inline void _initialize(int nelems)
{
    N = nelems;
    sina = (double*) malloc( sizeof(double) * nelems );
    _dx = (double*) malloc( sizeof(double) * nelems );
    _dy = (double*) malloc( sizeof(double) * nelems );
    xVN = (double*) malloc( sizeof(double) * nelems );
    yVN = (double*) malloc( sizeof(double) * nelems );
    zVN = (double*) malloc( sizeof(double) * nelems );
    //sina = (double*) malloc( sizeof(double) * nelems );
    
    create_plan(N);
}

inline void _finalize()
{
    free(sina);
    free(_dx);
    free(_dy);
    free(xVN);
    free(yVN);
    free(zVN);
}




inline void count_sina(double* sina, double* dx, double* dy, double* xVN, double* yVN, double* zVN)
{
    /** *************************************************************************************************************** **
    Return sin of angle between rVN vector and dl vetor.
    
    @param sina          numpy 1D array     - sin of angle between rVN vector and dl vetor.
    @param dx            numpy 1D array     - derivative of vortex displacement in x direction
    @param dy            numpy 1D array     - derivative of vortex displacement in y direction
    @param xVN           numpy 1D array     - component of rVN vector in one of direction x
    @param yVN           numpy 1D array     - component of rVN vector in one of direction y
    @param zVN           numpy 1D array     - component of rVN vector in one of direction z
    
    @return
    
    ** *************************************************************************************************************** **/
    
    #pragma omp simd
    for (uint16_t ix=0; ix < N; ix++)
    {
        const double dl   = sqrt(dx[ix]*dx[ix] + dy[ix]*dy[ix] + 1);
        const double _rVN = sqrt( xVN[ix]*xVN[ix] + yVN[ix]*yVN[ix] + zVN[ix]*zVN[ix] );
        sina[ix]  = sin(  acos( (dx[ix]*xVN[ix] + dy[ix]*yVN[ix] + 1.*zVN[ix])/(dl*_rVN) ) );
    }
}


inline void VN_force(double* forces, double* rVN, double* uVN, double* sina, double* fVN_params)
{
   /** *************************************************************************************************************** **
    Return Vortex-Nucleus force in direction of given component uVN of rVN vector.
    
    @param rVN          numpy 1D array     - lenght of rVN vector.
    @param uVN          numpy 1D array     - component of rVN vector in one of direction x,y,z.
    @param sina         numpy 1D array     - sin of angle between rVN vector and dl vetor.
    @param fVN_params   tuple              - tuple containing parametrization of force.
    
    @return
    
    ** *************************************************************************************************************** **/
    
    #pragma omp simd
    for(uint16_t ix = 0; ix < N; ix++)
    {
        double _rVN = rVN[ix];
        double f = fVN_params[0] + fVN_params[1]*_rVN + fVN_params[2]*_rVN*_rVN;
                        f = f/(1 + fVN_params[3]*_rVN + fVN_params[4]*_rVN*_rVN + fVN_params[5]*_rVN*_rVN*_rVN + 
                            fVN_params[6]*_rVN*_rVN*_rVN*_rVN + fVN_params[7]*_rVN*_rVN*_rVN*_rVN*_rVN);
        
        forces[2*ix] += sina[ix] * f * uVN[ix]/_rVN;
    }
}

inline double fVN_sum(double* rVN, double* uVN, double* sina, double* fVN_params)
{
    /** *************************************************************************************************************** **
    Return Sum of Vortex-Nucleus forces from each  in direction of given component uVN of rVN vector.
    
    @param rVN          numpy 1D array     - lenght of rVN vector.
    @param uVN          numpy 1D array     - component of rVN vector in one of direction x,y,z.
    @param sina         numpy 1D array     - sin of angle between rVN vector and dl vetor.
    @param fVN_params   tuple              - tuple containing parametrization of force.
    
    @return sum         double             - sum of elements
    
    ** *************************************************************************************************************** **/
    
    
    double sum = 0.;
    #pragma omp simd reduction(+:sum)
    for (uint16_t ix = 0; ix < N; ix++)
    {
        double _rVN = rVN[ix];
        double f = fVN_params[0] + fVN_params[1]*_rVN + fVN_params[2]*_rVN*_rVN;
                        f = f/(1 + fVN_params[3]*_rVN + fVN_params[4]*_rVN*_rVN + fVN_params[5]*_rVN*_rVN*_rVN + 
                            fVN_params[6]*_rVN*_rVN*_rVN*_rVN + fVN_params[7]*_rVN*_rVN*_rVN*_rVN*_rVN);
        sum += sina[ix] * f * uVN[ix]/_rVN;
    }
    
    return sum;
}


inline void get_forces(double* xV, double* yV, double* zV, double* dx, double* dy, double* rN, double *fx, double* fy, double* fVN_params)
{
    for (int ix=0; ix<N; ix++)
    {
        xVN[ix] = xV[ix] - rN[0];
        yVN[ix] = yV[ix] - rN[1];
        zVN[ix] = zV[ix] - rN[2];
    }
    
    
    count_sina(sina, dx, dy, xVN, yVN, zVN);
    
    #pragma omp simd
    for(uint16_t ix = 0; ix < N; ix++)
    {
        double _rVN = sqrt( xVN[ix]*xVN[ix] + yVN[ix]*yVN[ix] + zVN[ix]*zVN[ix] );
        double f = fVN_params[0] + fVN_params[1]*_rVN + fVN_params[2]*_rVN*_rVN;
                        f = f/(1 + fVN_params[3]*_rVN + fVN_params[4]*_rVN*_rVN + fVN_params[5]*_rVN*_rVN*_rVN + 
                            fVN_params[6]*_rVN*_rVN*_rVN*_rVN + fVN_params[7]*_rVN*_rVN*_rVN*_rVN*_rVN);
        
        fx[ix] = -1.0 * sina[ix] * f * xVN[ix]/_rVN;
        fy[ix] = -1.0 * sina[ix] * f * yVN[ix]/_rVN;
    }
    
}

// //using namespace std::literals::complex_literals;

inline void get_forces(double* xV, double* yV, double* zV, double* dx, double* dy, double* rN, std::complex<double> *fx, std::complex<double>* fy, double* fVN_params)
{
    for (int ix=0; ix<N; ix++)
    {
        xVN[ix] = xV[ix] - rN[0];
        yVN[ix] = yV[ix] - rN[1];
        zVN[ix] = zV[ix] - rN[2];
    }
    
    
    count_sina(sina, dx, dy, xVN, yVN, zVN);
    
    #pragma omp simd
    for(uint16_t ix = 0; ix < N; ix++)
    {
        double _rVN = sqrt( xVN[ix]*xVN[ix] + yVN[ix]*yVN[ix] + zVN[ix]*zVN[ix] );
        double f = fVN_params[0] + fVN_params[1]*_rVN + fVN_params[2]*_rVN*_rVN;
        f        = f/(1 + fVN_params[3]*_rVN + fVN_params[4]*_rVN*_rVN + fVN_params[5]*_rVN*_rVN*_rVN + 
                          fVN_params[6]*_rVN*_rVN*_rVN*_rVN + fVN_params[7]*_rVN*_rVN*_rVN*_rVN*_rVN);
        
        fx[ix] = std::complex<double>( sina[ix] * f * xVN[ix]/_rVN, 0.);
        fy[ix] = std::complex<double>( sina[ix] * f * yVN[ix]/_rVN, 0.);
    }
    
}


inline void get_forces_multiple_impurities(double* xV, double* yV, double* zV, double* dx, double* dy, double* rN,
                                           double *fx, double* fy, double* fVN_params, const int M)
{
    
    for (int ix=0; ix<N; ix++)
    {
        xVN[ix] = xV[ix] - rN[0];
        yVN[ix] = yV[ix] - rN[1];
        zVN[ix] = zV[ix] - rN[2];
    }
    
    
    count_sina(sina, dx, dy, xVN, yVN, zVN);
    
    #pragma omp simd
    for(uint16_t ix = 0; ix < N; ix++)
    {
        double _rVN = sqrt( xVN[ix]*xVN[ix] + yVN[ix]*yVN[ix] + zVN[ix]*zVN[ix] );
        double f = fVN_params[0] + fVN_params[1]*_rVN + fVN_params[2]*_rVN*_rVN;
                        f = f/(1 + fVN_params[3]*_rVN + fVN_params[4]*_rVN*_rVN + fVN_params[5]*_rVN*_rVN*_rVN + 
                            fVN_params[6]*_rVN*_rVN*_rVN*_rVN + fVN_params[7]*_rVN*_rVN*_rVN*_rVN*_rVN);
        
        fx[ix] = -1.0 * sina[ix] * f * xVN[ix]/_rVN;
        fy[ix] = -1.0 * sina[ix] * f * yVN[ix]/_rVN;
    }
    
}

// //using namespace std::literals::complex_literals;

inline void get_forces_multiple_impurities(double* xV, double* yV, double* zV, double* dx, double* dy, double* rN, 
                                           std::complex<double> *fx, std::complex<double>* fy, double* fVN_params, const int M)
{
    for (int ix=0; ix<N; ix++)
    {
        xVN[ix] = xV[ix] - rN[0];
        yVN[ix] = yV[ix] - rN[1];
        zVN[ix] = zV[ix] - rN[2];
    }
    
    
    count_sina(sina, dx, dy, xVN, yVN, zVN);
    
    #pragma omp simd
    for(uint16_t ix = 0; ix < N; ix++)
    {
        double _rVN = sqrt( xVN[ix]*xVN[ix] + yVN[ix]*yVN[ix] + zVN[ix]*zVN[ix] );
        double f = fVN_params[0] + fVN_params[1]*_rVN + fVN_params[2]*_rVN*_rVN;
        f        = f/(1 + fVN_params[3]*_rVN + fVN_params[4]*_rVN*_rVN + fVN_params[5]*_rVN*_rVN*_rVN + 
                          fVN_params[6]*_rVN*_rVN*_rVN*_rVN + fVN_params[7]*_rVN*_rVN*_rVN*_rVN*_rVN);
        
        fx[ix] = std::complex<double>( sina[ix] * f * xVN[ix]/_rVN, 0. );
        fy[ix] = std::complex<double>( sina[ix] * f * yVN[ix]/_rVN, 0. );
    }
    
    for (int in=1; in < M/3; in++)
    {
        #pragma omp simd
        for (int ix=0; ix<N; ix++)
        {
            xVN[ix] = xV[ix] - rN[3*in+0];
            yVN[ix] = yV[ix] - rN[3*in+1];
            zVN[ix] = zV[ix] - rN[3*in+2];
        }
        
        
        count_sina(sina, dx, dy, xVN, yVN, zVN);
        
        #pragma omp simd
        for(uint16_t ix = 0; ix < N; ix++)
        {
            double _rVN = sqrt( xVN[ix]*xVN[ix] + yVN[ix]*yVN[ix] + zVN[ix]*zVN[ix] );
            double f = fVN_params[0] + fVN_params[1]*_rVN + fVN_params[2]*_rVN*_rVN;
            f        = f/(1 + fVN_params[3]*_rVN + fVN_params[4]*_rVN*_rVN + fVN_params[5]*_rVN*_rVN*_rVN + 
                            fVN_params[6]*_rVN*_rVN*_rVN*_rVN + fVN_params[7]*_rVN*_rVN*_rVN*_rVN*_rVN);
            
            fx[ix] += std::complex<double>( sina[ix] * f * xVN[ix]/_rVN, 0. );
            fy[ix] += std::complex<double>( sina[ix] * f * yVN[ix]/_rVN, 0. );
        }
    }
}



#endif