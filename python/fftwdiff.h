#ifndef __FFTWDIFF_H__
#define __FFTWDIFF_H__

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include <omp.h>

//#ifdef __cplusplus
//extern "C" {
//#endif

//#include <complex.h>
#include <ccomplex>
#include <fftw3.h>



/* *********************************************************************************************************************** *
 *                                                                                                                         *
 *                                        FFTW INTERFACE AND DIFFERENTIATION                                               *
 *                                                                                                                         *
 * *********************************************************************************************************************** */

fftw_complex *fftw_input = NULL;
fftw_complex *fftw_input2 = NULL;
fftw_complex *fftw_output = NULL;
fftw_complex *fftw_output2 = NULL;
fftw_plan pf, pi, pi2;


void create_plan(size_t xdim)
{
    fftw_input   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xdim);
    fftw_input2  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xdim);
    fftw_output  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xdim);
    fftw_output2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * xdim);
    
    pf  = fftw_plan_dft_1d(xdim, fftw_input, fftw_output, FFTW_FORWARD,  FFTW_PATIENT);
    pi  = fftw_plan_dft_1d(xdim, fftw_output, fftw_input, FFTW_BACKWARD, FFTW_PATIENT);
    pi2 = fftw_plan_dft_1d(xdim, fftw_output2, fftw_input2, FFTW_BACKWARD, FFTW_PATIENT);
    
#ifdef VERBOSE
    printf("FFTW initialized.");
#endif
}


inline void fftwdiff_once(double *in_arr, double *dfdx, double *d2fd2x, const size_t xdim, double period)
{
    /**
     * Assuming in_arr is array of double of length 2N and in_arr[0::2] are x coordinates and in_arr[1::2] are y coordinates.
     * 
     * 
     * 
     * */
    register size_t ix;
    register int jx;
    
    // copy input array to fftw input (double to complex)
    #pragma omp simd
    for (ix=0; ix < xdim; ix++) fftw_input[ix] = in_arr[ix] + 0.*I; // NOTE: Add incx variable to get every incx-th element in in_arr
    
    
    // execute fftw forward
    fftw_execute(pf);
    
    // copy fourier transform
    memcpy(fftw_output2, fftw_output, xdim * sizeof(double __complex__));
    
    #pragma omp parallel sections default(shared) private(ix,jx) num_threads(2)
    {
    #pragma omp section
    {
        /* *** first derivative *** */
        // multiply by i*k_x / xdim (division by number of samples in case to normalize)
        #pragma omp simd
        for ( ix = 0; ix < xdim / 2; ix++ )
        {
            fftw_output[ix] = (2. * ( double ) M_PI / period * ( double ) ix)* I * fftw_output[ix]/ ((double) xdim);
        }
        jx = - ix+1;
        
        fftw_output[xdim / 2] = 0.; // NOTE: http://math.mit.edu/~stevenj/fft-deriv.pdf
        
        #pragma omp simd
        for ( ix = xdim / 2 + 1; ix < xdim; ix++ ) 
        {
            fftw_output[ix] = (2. * ( double ) M_PI / period * ( double ) jx)* I * fftw_output[ix]/ ((double) xdim);
            jx++;
        }
        
        // execute fftw inverse
        fftw_execute(pi);
        
        // copy result to double* array
        #pragma omp simd
        for (ix=0; ix < xdim; ix++) dfdx[ix] = creal(fftw_input[ix]);
    }
    #pragma omp section
    {
        /* *** second derivative *** */
        // multiply by -k_x*k_x / xdim (division by number of samples in case to normalize)
        #pragma omp simd
        for ( ix = 0; ix < xdim / 2; ix++ )
        {
            fftw_output2[ix] = -1. * (2. * ( double ) M_PI / period * ( double ) ix)
                                   * (2. * ( double ) M_PI / period * ( double ) ix) * fftw_output2[ix] / ((double) xdim);
        }
        jx = - ix;
        #pragma omp simd
        for ( ix = xdim / 2; ix < xdim; ix++ )
        {
            fftw_output2[ix] = -1. * (2. * ( double ) M_PI / period * ( double ) jx)
                                   * (2. * ( double ) M_PI / period * ( double ) jx) * fftw_output2[ix] / ((double) xdim);
            jx++;
        }
        
        // execute fftw inverse
        fftw_execute(pi2);
        
        // copy result to double* array
        #pragma omp simd
        for (ix=0; ix < xdim; ix++) d2fd2x[ix] = creal(fftw_input2[ix]);
    }
    }
}



inline void fftwdiff(double *in_arr, double *dfdx, double *d2fd2x, const size_t xdim, double period)
{
    /**
     * Assuming in_arr is array of double of length 2N and in_arr[0::2] are x coordinates and in_arr[1::2] are y coordinates.
     * 
     * 
     * 
     * */
    register size_t ix;
    register int jx;
    
    // copy input array to fftw input (double to complex)
    #pragma omp simd
    for (ix=0; ix < xdim; ix++) fftw_input[ix] = in_arr[2*ix] + 0.*I; // NOTE: Every second element in in_arr
    
    
    // execute fftw forward
    fftw_execute(pf);
    
    // copy fourier transform
    memcpy(fftw_output2, fftw_output, xdim * sizeof(double __complex__));
    
    #pragma omp parallel sections default(shared) private(ix,jx) num_threads(2)
    {
    #pragma omp section
    {
        /* *** first derivative *** */
        // multiply by i*k_x / xdim (division by number of samples in case to normalize)
        #pragma omp simd
        for ( ix = 0; ix < xdim / 2; ix++ )
        {
            fftw_output[ix] = (2. * ( double ) M_PI / period * ( double ) ix)* I * fftw_output[ix]/ ((double) xdim);
        }
        jx = - ix;
        #pragma omp simd
        for ( ix = xdim / 2; ix < xdim; ix++ ) 
        {
            fftw_output[ix] = (2. * ( double ) M_PI / period * ( double ) jx)* I * fftw_output[ix]/ ((double) xdim);
            jx++;
        }
        
        // execute fftw inverse
        fftw_execute(pi);
        
        // copy result to double* array
        #pragma omp simd
        for (ix=0; ix < xdim; ix++) dfdx[ix] = creal(fftw_input[ix]);
    }
    #pragma omp section
    {
        /* *** second derivative *** */
        // multiply by -k_x*k_x / xdim (division by number of samples in case to normalize)
        #pragma omp simd
        for ( ix = 0; ix < xdim / 2; ix++ )
        {
            fftw_output2[ix] = -1. * (2. * ( double ) M_PI / period * ( double ) ix)
                                   * (2. * ( double ) M_PI / period * ( double ) ix) * fftw_output2[ix] / ((double) xdim);
        }
        jx = - ix;
        #pragma omp simd
        for ( ix = xdim / 2; ix < xdim; ix++ )
        {
            fftw_output2[ix] = -1. * (2. * ( double ) M_PI / period * ( double ) jx)
                                   * (2. * ( double ) M_PI / period * ( double ) jx) * fftw_output2[ix] / ((double) xdim);
            jx++;
        }
        
        // execute fftw inverse
        fftw_execute(pi2);
        
        // copy result to double* array
        #pragma omp simd
        for (ix=0; ix < xdim; ix++) d2fd2x[ix] = creal(fftw_input2[ix]);
    }
    }
}


inline void fftwdiff_simple(double *in_arr, double *dfdx, double *d2fd2x, const size_t xdim)
{
    fftwdiff(in_arr, dfdx, d2fd2x, xdim, (double) xdim);
}


void destroy_plan(void)
{
    #pragma omp parallel num_threads(4) shared(pf,pi,pi2,fftw_input,fftw_output,fftw_output2)
    {
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                fftw_destroy_plan(pf);
                if (fftw_input)   free(fftw_input);
            }
            #pragma omp section
            {
                fftw_destroy_plan(pi);
                if (fftw_output)  free(fftw_output);
            }
            #pragma omp section
            {
                if (fftw_output2) {  fftw_destroy_plan(pi2); free(fftw_output2);  }
            }
            #pragma omp section
            {
                if (fftw_input2)  free(fftw_input2);
            }
        }
    }
}


//#ifdef __cplusplus
//} // extern "C"
//#endif



#endif /* __FFTWDIFF_H__ */