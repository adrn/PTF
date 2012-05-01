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

import sys, os
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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
        
def computeVariabilityIndices(lightCurve, tuple=False):
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
    
    # sigma/mu : root-variance / mean
    mu = contMag #np.mean(lightCurve.mag)
    sigma = np.sqrt(np.sum(lightCurve.mag - mu)**2 / (N-1.))
    sigma_to_mu = sigma / mu

    # Con : number of consecutive series of 3 points BRIGHTER than the light curve
    num_sigma = 2.
    clusters = findClustersBrighter(lightCurve.mag, contMag, contSig, 3, num_sigma=num_sigma)
    Con = len(clusters) / (N - 2.)
    
    clusters = findClustersBrighter(lightCurve.mag, contMag, contSig, 3, num_sigma=2.5)
    B = len(clusters) # Number of clusters of >3 points BRIGHTER than 2.5-sigma over the baseline
    
    clusters = findClustersFainter(lightCurve.mag, contMag, contSig, 3, num_sigma=2.5)
    F = len(clusters) # Number of clusters of >3 points FAINTER than 2.5-sigma over the baseline
    
    # eta : ratio of mean square successive difference to the sample variance
    delta_squared = np.sum((lightCurve.mag[1:] - lightCurve.mag[:-1])**2 / (N - 1.))
    variance = sigma**2
    eta = delta_squared / variance
    
    delta_n = np.sqrt(float(N)/(N-1.)) * (lightCurve.mag[:-1] - mu) / lightCurve.error[:-1] 
    delta_n_plus_1 = np.sqrt(float(N)/(N-1.)) * (lightCurve.mag[1:] - mu) / lightCurve.error[1:]
    # J : eqn. 3 in M.-S. Shin et al. 2009
    J = np.sum(np.sign(delta_n*delta_n_plus_1)*np.sqrt(np.fabs(delta_n*delta_n_plus_1)))
    
    # K : eqn. 3 in M.-S. Shin et al. 2009
    delta_n = np.sqrt(float(N)/(N-1.)) * (lightCurve.mag - mu) / lightCurve.error
    K = np.sum(np.fabs(delta_n)) / (float(N)*np.sqrt((1./N)*np.sum(delta_n**2)))
    
    if tuple:
        return (sigma_to_mu, Con, eta, J, K)
    else:
        return {"sigma_mu" : sigma_to_mu,\
                "con" : Con,\
                "eta" : eta,\
                "j" : J,\
                "k" : K,\
                "b" : B,\
                "f" : F}

class PTFLightCurve:
    
    def __init__(self, mjd, mag, error):
        self.mjd = np.array(mjd)
        self.mag = np.array(mag)
        self.error = np.array(error)
    
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
        #       case assum the amplifcation should be >1.5. Using eq. 1 from
        #       Popowski & Alcock, this corresponds to a maximum u0 of ~0.8
        if u0 == None: self.u0 = np.random.uniform()*0.8
        else: self.u0 = float(u0)
        
        # If t0 is not specified, draw from uniform distribution between days
        if t0 == None: self.t0 = np.random.uniform(min(self.mjd), max(self.mjd))
        else: self.t0 = float(t0)
        
        if (self.t0 > max(self.mjd)) or (self.t0 < min(self.mjd)):
            logging.warn("t0 is outside of the mjd range for this light curve!")
        
        # If tE is not specified, draw from tE distribution
        #   I used Alcock's suggestion of a log uniform sampling from 1-2000 days
        if tE == None: self.tE = 10**np.random.uniform(0., 3.3012)
        else: self.tE = float(tE)
        
        flux = fluxModel(self.mjd, u0=self.u0, t0=self.t0, tE=self.tE, F0=1.)#self.F0)
        self.mag = FluxToRMag(flux*RMagToFlux(self.mag))
    
    def addNoise(self):
        """ Add scatter to the light curve """
        self.mag += np.random.normal(0.0, self.error)
    
    def plot(self, ax=None):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.mjd, self.mag, self.error, ls='none', marker='o', c='k', ecolor='0.7', capsize=0)
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set_xlim(min(self.mjd), max(self.mjd))
            plt.show()
        
        ax.errorbar(self.mjd, self.mag, self.error, ls='none', marker='o', c='k', ecolor='0.7', capsize=0)
        ax.set_ylim(ax.get_ylim()[::-1])

class SimulatedLightCurve(PTFLightCurve):
    
    def __init__(self, mjd, error, outliers=False):
        """ Creates a simulated PTF light curve
        
            Parameters
            ----------
            mjd : numpy.array
                An array of mjd values. If none, creates one internally.
            error : numpy.array
                An array of error values (sigmas). If none, creates one internally.
            outliers : bool, optional
                This controls whether to sample from an outlier distribution
                when creating magnitude values for the light curve
            
            Notes
            -----
            
        """
        
        self.mjd = np.array(mjd)
        self.error = np.array(error)
        
        if outliers:
            # Add ~1% outliers
            outlier_points = (np.random.uniform(0.0, 1.0, size=len(self.mjd)) < 0.01).astype(float) * np.random.normal(0.0, 2.0, size=len(self.mjd))
        else:
            outlier_points = np.zeros(len(self.mjd))
        
        self.F0 = np.random.uniform(0.1, 1.) / 100.
        self.mag = np.ones(len(self.mjd), dtype=float)*FluxToRMag(self.F0) + outlier_points

        self.addNoise()