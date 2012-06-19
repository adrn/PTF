# -*- coding: utf-8 -*-

"""
    Provides utility functions for the various PTF microlensing
    simulations we will run.
    
    I imagine the simulation will go like this:
        - Read in light curve
        - Throw away data points so left with 15, 25, 50, 75, 100, 125, 150, 175, 200
        - Add microlensing event
        - Try to detect event
        - Repeat N times
        
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard Library
import copy
import sys, os
import logging

# Third-party dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import scipy.optimize as so

# Project
#from ..aov import findPeaks_aov

def straight(t, b):
    return np.ones((len(t),), dtype=float)*b

def fit_line(x, y, sigma_y):
    popt, pcov = curve_fit(straight, x, y, sigma=sigma_y, p0=(np.median(y),))
    return popt[0]

def RMagToFlux(R):
    # Returns a flux in Janskys
    return 2875.*10**(R/-2.5)

def FluxToRMag(f):
    # Accepts a flux in Janskys
    return -2.5*np.log10(f/2875.)

def u_t(t, u_0, t_0, t_E):
    return np.sqrt(u_0**2 + ((t - t_0)/t_E)**2)

def A_u(u):
    return (u**2 + 2) / (u*np.sqrt(u**2 + 4))

def fluxModel(t, **p):
    return p["F0"]*A_u(u_t(t, p["u0"], p["t0"], p["tE"]))

# These models are for computing delta_chi_squared
def linear_model(a, t): return a[0] + a[1]*t
def linear_error_function(p, t, mag, err): return (mag - linear_model(p, t)) / err
def microlensing_model(b, t): return b[0] + 1./(b[2]*2*np.pi) * np.exp(-(t-b[1])**2 / (2*b[2]**2))*b[3]
def microlensing_error_function(p, t, mag, err): return (mag - microlensing_model(p, t)) / err

def estimateContinuum(mjd, mag, error, clipSigma=2.5):
    """ Estimate the continuum of the light curve using sigma clipping """
    
    rootVariance = np.std(mag)
    b = np.median(mag)
    
    mags = mag
    mjds = mjd
    sigmas = error
    
    while True:
        w = np.fabs(mags - np.median(mags)) < clipSigma*rootVariance
        
        new_mags = mags[w]
        new_mjds = mjds[w]
        new_sigmas = sigmas[w]
        
        if (len(mags) - len(new_mags)) <= (0.02*len(mags)):
            break
        else:
            mags = new_mags
            mjds = new_mjds
            sigmas = new_sigmas
            rootVariance = np.std(mags)
    
    try:
        continuumMag = fit_line(mjds, mags, sigmas)
        continuumSigma = rootVariance
    except TypeError:
        logging.debug("Sigma-clipping fit failed")
        continuumMag = fit_line(mjd, mag, error)
        continuumSigma = rootVariance
    
    return continuumMag, continuumSigma

def findClusters(w, mag, continuumMag, continuumSigma, num_points_per_cluster=4, num_sigma=3.):
    # Determine which points are outside of continuumMag +/- 2 or 3 continuumSigma, and see if they are clustered    
    allGroups = []
    
    group = []
    in_group = False
    group_finished = False
    for idx, pt in enumerate(np.logical_not(w)):
        if len(group) > 0:
            in_group = True
        
        if pt:
            group.append(idx)
        else:
            if in_group and len(group) >= num_points_per_cluster:
                allGroups.append(np.array(group))
                
            in_group = False
            group = []
    
    if in_group and len(group) >= num_points_per_cluster:
        allGroups.append(np.array(group))
    
    return allGroups

def findClustersBrighter(mag, continuumMag, continuumSigma, num_points_per_cluster=4, num_sigma=3.):
    # Determine which points are outside of continuumMag +/- 2 or 3 continuumSigma, and see if they are clustered
    w = (mag > (continuumMag - num_sigma*continuumSigma))
    
    return findClusters(w, mag, continuumMag, continuumSigma, num_points_per_cluster, num_sigma)

def findClustersFainter(mag, continuumMag, continuumSigma, num_points_per_cluster=4, num_sigma=3.):
    # Determine which points are outside of continuumMag +/- 2 or 3 continuumSigma, and see if they are clustered
    w = (mag < (continuumMag - num_sigma*continuumSigma))
    
    return findClusters(w, mag, continuumMag, continuumSigma, num_points_per_cluster, num_sigma)

def compute_delta_chi_squared(light_curve, force_fit=False):
    """ """
    if isinstance(light_curve, tuple):
        mjd, mag, error = light_curve
    else:
        mjd = light_curve.amjd
        mag = light_curve.amag
        error = light_curve.error
    
    if force_fit:
        lin_ier = 0
        ml_ier = 0
        
        tries = 0
        while lin_ier not in [1,2,3,4] and tries <= 10:
            linear_fit_params, covx, infodict, mesg, lin_ier = so.leastsq(linear_error_function, x0=(np.random.normal(np.median(mag), 0.5), 0.), args=(mjd, mag, error), full_output=1)
            tries += 1
        
        tries = 0
        while ml_ier not in [1,2,3,4] and tries <= 10:
            microlensing_fit_params, covx, infodict, mesg, ml_ier = so.leastsq(microlensing_error_function, x0=(np.random.normal(np.median(mag), 0.5), np.random.normal(mjd[mag.argmin()], 10), np.random.normal(10., 2), np.random.normal(-25., 5)), args=(mjd, mag, error), maxfev=1000, full_output=1)
            tries += 1
            
    else:
        linear_fit_params, lin_ier = so.leastsq(linear_error_function, x0=(np.random.normal(np.median(mag), 0.5), 0.), args=(mjd, mag, error))
        microlensing_fit_params, ml_ier = so.leastsq(microlensing_error_function, x0=(np.median(mag), mjd[mag.argmin()], 10., -25.), args=(mjd, mag, error), maxfev=1000)
        
    
    linear_chisq = np.sum(linear_error_function(linear_fit_params, \
                                                mjd, \
                                                mag, \
                                                error)**2)# / len(linear_fit_params)
    
    microlensing_chisq = np.sum(microlensing_error_function(microlensing_fit_params, \
                                                            mjd, \
                                                            mag, \
                                                            error)**2)# / len(microlensing_fit_params)
    
    return linear_chisq - microlensing_chisq

def compute_variability_indices(lightCurve, indices=[]):
    """ Computes the 6 (5) variability indices as explained in M.-S. Shin et al. 2009
        
        Parameters
        ----------
        lightCurve : SimulatedLightCurve
            a SimulatedLightCurve object to compute the indices from
    """
    N = len(lightCurve.mjd)
    contMag, contSig = estimateContinuum(lightCurve.mjd, lightCurve.mag, lightCurve.error)
    
    # ===========================
    # Compute variability indices
    # ===========================
    
    idx_dict = {} 
    
    # sigma/mu : root-variance / mean
    #if "sigma_mu" in indices:
    mu = contMag #np.mean(lightCurve.mag)
    idx_dict["mu"] = mu
    
    #sigma = np.sqrt(np.sum(lightCurve.mag - mu)**2 / (N-1.))
    sigma = np.std(lightCurve.mag)
    sigma_to_mu = sigma / mu
    idx_dict["sigma_mu"] = sigma_to_mu
    
    if "con" in indices:
        # Con : number of consecutive series of 3 points BRIGHTER than the light curve
        num_sigma = 2.
        clusters = findClustersBrighter(lightCurve.mag, contMag, contSig, 3, num_sigma=num_sigma)
        Con = len(clusters) / (N - 2.)
        idx_dict["con"] = Con
    
    if "b" in indices:
        clusters = findClustersBrighter(lightCurve.mag, contMag, contSig, 3, num_sigma=3)
        B = len(clusters) # Number of clusters of >3 points BRIGHTER than 3-sigma over the baseline
        idx_dict["b"] = B
    
    if "f" in indices:
        clusters = findClustersFainter(lightCurve.mag, contMag, contSig, 3, num_sigma=3)
        F = len(clusters) # Number of clusters of >3 points FAINTER than 3-sigma over the baseline
        idx_dict["f"] = F
    
    # eta : ratio of mean square successive difference to the sample variance
    if "eta" in indices:
        delta_squared = np.sum((lightCurve.mag[1:] - lightCurve.mag[:-1])**2 / (N - 1.))
        variance = sigma**2
        eta = delta_squared / variance
        idx_dict["eta"] = eta
    
    if "j" in indices or "k" in indices:
        delta_n = np.sqrt(float(N)/(N-1.)) * (lightCurve.mag[:-1] - mu) / lightCurve.error[:-1] 
        delta_n_plus_1 = np.sqrt(float(N)/(N-1.)) * (lightCurve.mag[1:] - mu) / lightCurve.error[1:]
    
    if "j" in indices:
        # J : eqn. 3 in M.-S. Shin et al. 2009
        #   Modified 2012-05-24 to use eq. 1 in Fruth et al.
        #weight = np.exp(-(lightCurve.mjd[1:]-lightCurve.mjd[:-1]) / np.mean(lightCurve.mjd[1:]-lightCurve.mjd[:-1]))
        #J = np.sum(np.sign(delta_n*delta_n_plus_1)*np.sqrt(np.fabs(delta_n*delta_n_plus_1)) / weight)
        J = np.sum(np.sign(delta_n*delta_n_plus_1)*np.sqrt(np.fabs(delta_n*delta_n_plus_1)))
        idx_dict["j"] = J
    
    if "k" in indices:
        # K : eqn. 3 in M.-S. Shin et al. 2009
        delta_n = np.sqrt(float(N)/(N-1.)) * (lightCurve.mag - mu) / lightCurve.error
        K = np.sum(np.fabs(delta_n)) / (float(N)*np.sqrt((1./N)*np.sum(delta_n**2)))
        idx_dict["k"] = K
    
    if "delta_chi_squared" in indices:
        # delta_chi_squared : matched filter approach, fit a line, then a gaussian; compare.
        delta_chi_squared = compute_delta_chi_squared(lightCurve)
        idx_dict["delta_chi_squared"] = delta_chi_squared
    
    # Analysis of Variance maximum
    if "aovm" in indices:
        peaks = findPeaks_aov(lightCurve.mjd, lightCurve.mag, lightCurve.error, 2, 0.1, 100., 0.1, 0.01, 20)
        idx_dict["aovm"] = peaks["peak_power"][0]
    
    if indices:
        return tuple([idx_dict[x] for x in indices])
    else:
        return idx_dict
