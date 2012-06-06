"""
cython detection_efficiency_worker.pyx
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -L/usr/include/python2.7 -o delta_chi_squared.so delta_chi_squared.c
"""
from __future__ import division

import numpy as np
cimport numpy as np
cimport cython

import scipy.optimize as so

cdef extern from "math.h":
    double exp(double)
    double pow(double, double)

@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef double linear_model(np.ndarray[double, ndim=1] a, double t): 
    return a[0] + a[1]*t

@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef linear_error_function(np.ndarray[double, ndim=1] p, np.ndarray[double, ndim=1] t, np.ndarray[double, ndim=1] mag, np.ndarray[double, ndim=1] err):
    cdef int ii
    cdef long size
    #cdef double err_f_val
    cdef np.ndarray[double, ndim=1] err_f_vals

    size = len(mag)
    err_f_vals = np.zeros(size, dtype=float)
    
    for ii in range(size):
        err_f_vals[ii] = (mag[ii] - linear_model(p, t[ii])) / err[ii]
    
    return err_f_vals

@cython.boundscheck(False) # turn of bounds-checking for entire function
cpdef double linear_chi_squared(np.ndarray[double, ndim=1] p, np.ndarray[double, ndim=1] t, np.ndarray[double, ndim=1] mag, np.ndarray[double, ndim=1] err):
    cdef int ii
    cdef long size
    cdef double chi_sq
    
    size = len(mag)
    
    chi_sq = 0.
    for ii in range(size):
        chi_sq += pow((mag[ii] - linear_model(p, t[ii])) / err[ii], 2.)
    
    return chi_sq

@cython.boundscheck(False) # turn of bounds-checking for entire function    
cpdef double microlensing_model(np.ndarray[double, ndim=1] b, double t): 
    return b[0] + 1./(b[2]*2*np.pi) * exp(-(t-b[1])**2 / (2*b[2]**2))*b[3]

@cython.boundscheck(False) # turn of bounds-checking for entire function    
cpdef microlensing_error_function(np.ndarray[double, ndim=1] p, np.ndarray[double, ndim=1] t, np.ndarray[double, ndim=1] mag, np.ndarray[double, ndim=1] err): 
    cdef int ii
    cdef long size
    #cdef double err_f_val
    cdef np.ndarray[double, ndim=1] err_f_vals
    
    size = len(mag)
    err_f_vals = np.zeros(size, dtype=float)
    
    for ii in range(size):
        err_f_vals[ii] = (mag[ii] - microlensing_model(p, t[ii])) / err[ii]
    
    return err_f_vals

@cython.boundscheck(False) # turn of bounds-checking for entire function      
cpdef double microlensing_chi_squared(np.ndarray[double, ndim=1] p, np.ndarray[double, ndim=1] t, np.ndarray[double, ndim=1] mag, np.ndarray[double, ndim=1] err):
    cdef int ii
    cdef long size
    cdef double chi_sq
    
    size = len(mag)
    
    chi_sq = 0.
    for ii in range(size):
        chi_sq += pow((mag[ii] - microlensing_model(p, t[ii])) / err[ii], 2.)
    
    return chi_sq

def compute_delta_chi_squared(np.ndarray[double, ndim=1] mjd, np.ndarray[double, ndim=1] mag, np.ndarray[double, ndim=1] error):
    """ """
    
    cdef int ii
    cdef long size, lin_num, ml_num
    cdef double linear_chisq, microlensing_chisq
    cdef np.ndarray[double, ndim=1] linear_fit_params
    cdef np.ndarray[double, ndim=1] microlensing_fit_params
    
    num = len(mjd)
    
    linear_fit_params, lin_num = so.leastsq(linear_error_function, x0=(np.median(mag), 0.), args=(mjd, mag, error))
    microlensing_fit_params, ml_num = so.leastsq(microlensing_error_function, x0=(np.median(mag), mjd[mag.argmin()], 10., -25.), args=(mjd, mag, error), maxfev=1000)
    
    linear_chisq = linear_chi_squared(linear_fit_params, mjd, mag, error)
    microlensing_chisq = microlensing_chi_squared(microlensing_fit_params, mjd, mag, error)
    
    return linear_chisq - microlensing_chisq