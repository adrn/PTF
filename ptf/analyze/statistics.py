# coding: utf-8
from __future__ import division

""" This module contains code for computing the variability statistics """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np

from .aov import aov_periodogram_asczerny, testperiod_asczerny, findPeaks_aov

__all__ = ["estimate_continuum", "gaussian_constant_delta_chi_squared", "stetson_j", "stetson_k", \
           "eta", "consecutive_brighter", "corr", "compute_variability_indices"]

def estimate_continuum(light_curve, sigma_clip=True, clip_sigma=2.5):
    """ Estimate the continuum magnitude of a light curve.
        
        Parameters
        ----------
        light_curve : LightCurve
            A LightCurve object -- must have mjd, mag, and error attributes.
        sigma_clip : bool
            Use sigma-clipping to fit the constant.
        clip_sigma : float
            The sigma cutoff in the sigma-clipping routine.
    """
    
    if sigma_clip:
        std = np.std(light_curve.mag)
        lc_len = len(light_curve.mag)
        
        # Start sigma-clipping the light curve
        w = np.ones(lc_len, dtype=bool)
        mags = light_curve.mag[w]
        while True:
            w = np.fabs(mags - np.mean(mags)) < (clip_sigma*np.std(mags))
            mags = light_curve.mag[w]
            
            # Convergence condition
            if len(mags) == lc_len:
                break
            
            lc_len = len(mags)
            
            # If we've thrown away 50% of the data points, break the loop and assume it won't converge
            #if (new_len / lc_len) <= 0.5:
            #    break
        
        mjd = light_curve.mjd[w]
        mag = light_curve.mag[w]
        error = light_curve.error[w]
    else:
        mjd = light_curve.mjd
        mag = light_curve.mag
        error = light_curve.error
    
    constant_params = Parameters()
    constant_params.add('b', value=np.random.normal(np.median(mag), 0.5))
    constant_result = minimize(constant_error_func, constant_params, args=(mjd, mag, error))
    sig = np.std(mag)
        
    return constant_params["b"].value, sig
        
def gaussian_constant_delta_chi_squared(light_curve, num_attempts=5):
    """ Compute the difference in chi-squared between a Gaussian and a straight (constant) line. """
    
    gaussian_chisqr = 1E6
    for ii in range(num_attempts):
        gaussian_params = Parameters()
        
        t0 = np.random.normal(light_curve.mjd[np.argmin(light_curve.mag)], 1.)
        if t0 > light_curve.mjd.max() or t0 < light_curve.mjd.min():
            t0 = light_curve.mjd[np.argmin(light_curve.mag)]
        gaussian_params.add('A', value=np.random.uniform(-1., -20.), min=-1E4, max=0.)
        gaussian_params.add('mu', value=t0, min=light_curve.mjd.min(), max=light_curve.mjd.max())
        gaussian_params.add('sigma', value=abs(np.random.normal(10., 2.)), min=1.)
        gaussian_params.add('B', value=np.random.normal(np.median(light_curve.mag), 0.5))

        gaussian_result = minimize(gaussian_error_func, gaussian_params, args=(light_curve.mjd, light_curve.mag, light_curve.error))
        
        if gaussian_result.chisqr < gaussian_chisqr:
            gaussian_chisqr = gaussian_result.chisqr
    
    constant_chisqr = 1E6
    for ii in range(num_attempts):
        constant_params = Parameters()
        constant_params.add('b', value=np.random.normal(np.median(light_curve.mag), 0.5))
        constant_result = minimize(constant_error_func, constant_params, args=(light_curve.mjd, light_curve.mag, light_curve.error))
        
        if constant_result.chisqr < constant_chisqr:
            constant_chisqr = constant_result.chisqr
    
    return constant_chisqr - gaussian_chisqr

def stetson_j(light_curve, weights=None):
    """ Compute the statistic J from Stetson 1996 """
    mag = light_curve.mag
    err = light_curve.error
    N = len(mag)
    mu = np.mean(mag)
    
    # Compute the mean square successive difference
    bias = math.sqrt(float(N)/(N-1.))
    delta_n = bias * (mag[:-1] - mu) / err[:-1]
    delta_n_plus_1 = bias * (mag[1:] - mu) / err[1:]
    
    if weights != None:
        J = np.sum(weights*np.sign(delta_n*delta_n_plus_1)*np.sqrt(np.fabs(delta_n*delta_n_plus_1))) / np.sum(weights)
    else:
        J = np.sum(np.sign(delta_n*delta_n_plus_1)*np.sqrt(np.fabs(delta_n*delta_n_plus_1)))
    
    return J

def stetson_k(light_curve):
    """ Compute the statistic K from Stetson 1996 """
    mag = light_curve.mag
    err = light_curve.error
    N = len(mag)
    mu = np.mean(mag)
    
    # Compute the mean square successive difference
    bias = math.sqrt(float(N)/(N-1.))
    delta_n = bias * (mag[:-1] - mu) / err[:-1]
    
    K = math.sqrt(1./N) * np.sum(np.fabs(delta_n)) / math.sqrt(np.sum(delta_n**2))
    return K

def eta(light_curve):
    """ Compute the von Neumann (1941) ratio """
    
    mag = light_curve.mag
    err = light_curve.error
    N = len(mag)
    
    delta_squared = np.sum((mag[1:] - mag[:-1])**2 / (N - 1.))
    eta = delta_squared / np.var(mag)
    
    return eta

def consecutive_brighter(light_curve, Nsigma=2.):
    """ Compute the number of groups of consecutive points brighter than 3sigma from the baseline """
    
    continuum_mag, noise_sigma = estimate_continuum(light_curve)
    where_condition = np.where(light_curve.mag < (continuum_mag - (Nsigma*noise_sigma)))[0]
    
    n_groups = 0
    for group in np.array_split(where_condition, np.where(np.diff(where_condition)!=1)[0]+1):
        if len(group) > 3:
            n_groups += 1
    
    return n_groups

def corr(light_curve):
    """ Compute the time-reversed cross-correlation for this light curve. """
    return np.correlate(light_curve.mag, light_curve.mag[::-1])[0]

def aovm(light_curve):
    """ Compute the analysis of variance maximum period and power in that period """
    fp = findPeaks_aov(light_curve.mjd.copy(), light_curve.mag.copy(), light_curve.error.copy(), 3, 1., 2.*light_curve.baseline, 1., 0.1, 20)
    return {"period" : fp["peak_period"][0], \
            "power" : fp["peak_power"][0]}

def compute_variability_indices(light_curve, indices=[], return_tuple=False):
    """ Computes variability statistics, such as those explained in M.-S. Shin et al. 2009.
        Valid indices:
            j : Stetson J parameter
            k : Stetson K parameter, kurtosis
            delta_chi_squared : Difference in chi-squared values for a linear fit vs. a Gaussian
            eta : Ratio of mean square successive difference to the sample variance, von Neumann 1941
            continuum : the continuum magnitude
            sigma_mu : the ratio of the root-sample variance to the continuum magnitude
            con : the number of clusters of 3 or more points brighter than 3-sigma from the baseline
            corr : the time-reversed cross-correlation of a light curve with itself
        
        Parameters
        ----------
        light_curve : LightCurve
            A LightCurve object -- must have mjd, mag, and error attributes.
        indices : iterable
            See valid indices above.
    """
    if len(indices) == 0:
        indices = ["j", "k", "sigma_mu", "eta", "delta_chi_squared"]
        
    idx_dict = dict()
    
    if "j" in indices:
        idx_dict["j"] = stetson_j(light_curve)
    
    if "k" in indices:
        idx_dict["k"] = stetson_k(light_curve)
    
    if "eta" in indices:
        idx_dict["eta"] = eta(light_curve)
    
    if "delta_chi_squared" in indices:
        idx_dict["delta_chi_squared"] = gaussian_constant_delta_chi_squared(light_curve)
        
    if "continuum" in indices or "sigma_mu" in indices:
        continuum_mag, noise_sigma = estimate_continuum(light_curve, sigma_clip=True)
    
    if "continuum" in indices:
        idx_dict["continuum"] = continuum_mag
    
    if "sigma_mu" in indices:
        idx_dict["sigma_mu"] = np.std(light_curve.mag) / continuum_mag
    
    if "con" in indices:
        idx_dict["con"] = consecutive_brighter(light_curve, 2.) / (len(light_curve.mjd)-2)
    
    if "corr" in indices:
        idx_dict["corr"] = corr(light_curve)
    
    if return_tuple:
        return tuple([idx_dict[idx] for idx in indices])
    else:
        return idx_dict
