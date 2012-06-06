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
from ..ptflightcurve import PTFLightCurve
from ..aov import findPeaks_aov

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
    if "sigma_mu" in indices:
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
    
class SimulatedLightCurve(PTFLightCurve):
    
    def __init__(self, mjd, mag=None, error=None, outliers=False):
        """ Creates a simulated PTF light curve
        
            Parameters
            ----------
            mjd : numpy.array
                An array of mjd values. If none, creates one internally.
            error : numpy.array
                An array of error values (sigmas). If none, creates one internally.
            mag : numpy.array
                An optional array of magnitude values
            outliers : bool, optional
                This controls whether to sample from an outlier distribution
                when creating magnitude values for the light curve
            
        """
        
        self.amjd = self.mjd = np.array(mjd)
        
        if error != None:
            self.error = np.array(error)
        else:
            # TODO: Implement this
            raise NotImplementedError("TODO!")
        
        if mag != None:
            self.amag = self.mag = np.array(mag)
        else:
            if outliers:
                # Add ~1% outliers
                outlier_points = (np.random.uniform(0.0, 1.0, size=len(self.mjd)) < 0.01).astype(float) * np.random.normal(0.0, 2.0, size=len(self.mjd))
            else:
                outlier_points = np.zeros(len(self.mjd))
            
            self.F0 = np.random.uniform(0.1, 1.) / 100.
            self.mag = np.ones(len(self.mjd), dtype=float)*FluxToRMag(self.F0) + outlier_points
    
            self._addNoise()
        
        self._original_mag = copy.copy(self.mag)
    
    def reset(self):
        self.amag = self.mag = copy.copy(self._original_mag)
        self.u0 = None
        self.t0 = None
        self.tE = None
    
    def addMicrolensingEvent(self, u0=None, t0=None, tE=None):
        """ Adds a simulated microlensing event to the light curve
            
            u0 : float, optional
                The impact parameter for the microlensing event. If not specified,
                the value will be drawn from the measured u0 distribution 
                [TODO: REFERENCE]
            t0 : float, optional
                The peak time of the event (shouldn't really be specified)
                This is just drawn from a uniform distribution between mjd_min
                and mjd_max
            tE : float, optional
                The length of the microlensing event. If not specified,
                the value will be drawn from the measured tE distribution 
                [TODO: REFERENCE]        
        """
        
        # If u0 is not specified, draw from u0 distribution
        #   - see for example Popowski & Alcock 
        #   - u0 maximum defined by detection limit of survey, but in our
        #       case assum the amplifcation should be >1.4. Using eq. 1 from
        #       Popowski & Alcock, this corresponds to a maximum u0 of ~0.9261
        if u0 == None: self.u0 = np.random.uniform()*0.9261
        else: self.u0 = float(u0)
        
        # If t0 is not specified, draw from uniform distribution between days
        if t0 == None: self.t0 = np.random.uniform(min(self.mjd), max(self.mjd))
        else: self.t0 = float(t0)
        
        if (self.t0 > max(self.mjd)) or (self.t0 < min(self.mjd)):
            logging.warn("t0 is outside of the mjd range for this light curve!")
        
        # If tE is not specified, draw from tE distribution
        #   I use an estimate of Wood's "observed" distribution for now:
        #   http://onlinelibrary.wiley.com/store/10.1111/j.1365-2966.2005.09357.x/asset/j.1365-2966.2005.09357.x.pdf?v=1&t=h1whtf1h&s=7b4d93a69aa684387a49ece5fc33c32fa5037052
        if tE == None: self.tE = 10**np.random.normal(1.3, 0.5)
        else: self.tE = float(tE)
        
        flux = fluxModel(self.mjd, u0=self.u0, t0=self.t0, tE=self.tE, F0=1.)#self.F0)
        self.mag = FluxToRMag(flux*RMagToFlux(self.mag))
        
        self.amag = self.mag
    
    def _addNoise(self):
        """ Add scatter to the light curve """
        self.mag += np.random.normal(0.0, self.error)
