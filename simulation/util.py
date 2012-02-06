# -*- coding: utf-8 -*-

"""
    Provides utility functions for the various PTF microlensing
    simulations we will run.
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

import sys, os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def straight(t, b):
    return np.ones((len(t),), dtype=float)*b

def fit_line(x, y, sigma_y):
    popt, pcov = curve_fit(straight, x, y, sigma=sigma_y)
    return popt[0]

def RMagToFlux(R):
    # Returns a flux in Janskys
    return 2875.*10**(R/-2.5)

def FluxToRMag(f):
    # Accepts a flux in Janskys
    return -2.5*np.log10(f/2875.)

def model(t, *p):
    p = dict(zip(["u0", "t0", "tE", "F0"], p))
    return p["F0"]*MicrolensingEvent.A_u(MicrolensingEvent.u_t(t, p["u0"], p["t0"], p["tE"]))

class MicrolensingEvent:
    """ You can create event with your own parameters like so:
            mlEvent = MicrolensingEvent(t0=5., u0=0.1, tE=100.)
        
        or you can let the class do the draws for you, if you supply 
        an observed mjd array:
            mlEvent = MicrolensingEvent.AndParameters(mjd)
    """
            
    @staticmethod
    def u_t(t, u_0, t_0, t_E):
        return np.sqrt(u_0**2 + ((t - t_0)/t_E)**2)
    
    @staticmethod
    def A_u(u):
        return (u**2 + 2) / (u*np.sqrt(u**2 + 4))
    
    @staticmethod
    def AndParameters(mjd):
        eventParameters = dict()
        
        # THIS IS WRONG AND SHOULD BE A DRAW FROM THE u0 DISTRIBUTION -- uniform!
        eventParameters["u0"] = np.exp(-10.0*np.random.uniform())
        
        # Draw from uniform distribution between min(mjd), max(mjd)
        eventParameters["t0"] = np.random.uniform(min(mjd), max(mjd))
        
        # THIS IS WRONG AND SHOULD BE A DRAW FROM THE OBSERVED 
        #   TIMESCALE DISTRIBUTION from literature (MACHO)
        eventParameters["tE"] = np.random.uniform(5., 100.)
        
        return MicrolensingEvent(**eventParameters)
    
    def __init__(self, **kwargs):
        self.u0 = kwargs["u0"]
        self.t0 = kwargs["t0"]
        self.tE = kwargs["tE"]
    
    def fluxModel(self, t):
        return self.A_u(self.u_t(t, self.u0, self.t0, self.tE))
    
    def plot(self, t, ax=None):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        ax.plot(t, self.fluxModel(t), 'g-', label="Model")
        return ax

class SimLightCurve:
    
    def __init__(self, dbLightCurve):
        w = (np.array(dbLightCurve.flags) < 8) & (np.array(dbLightCurve.sys_error) > 0) & (np.array(dbLightCurve.imaflags, dtype=int) & 3797 == 0)
        self.mjd = np.array(dbLightCurve.mjd)[w]
        self.db_mag = np.array(dbLightCurve.mag)[w]
        self.mag = self.db_mag
        self.magError = np.array(dbLightCurve.mag_error)[w]
        self.sysError = np.array(dbLightCurve.sys_error)[w]
        self.error = np.sqrt(self.magError**2 + self.sysError**2)
        self.objid = dbLightCurve.objid
        
    def addMicrolensingEvent(self, event):
        self.event = event
        self.mag = FluxToRMag(RMagToFlux(self.mag) * self.event.fluxModel(self.mjd))
    
    def estimateContinuum(self, clipSigma=2.):
        """ Estimate the continuum of the light curve using sigma clipping """
        
        rootVariance = np.std(self.mag)
        b = np.median(self.mag)
        
        mags = self.mag
        mjds = self.mjd
        sigmas = self.error
        
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
        
        self.continuumMag = fit_line(mjds, mags, sigmas)
        self.continuumSigma = rootVariance
        
        return self.continuumMag, self.continuumSigma
    
    def findClusters(self, num_points_per_cluster=4):
        # Determine which points are outside of continuumMag +/- 2 or 3 continuumSigma, and see if they are clustered
        w = (self.mag > (self.continuumMag - 3*self.continuumSigma))
        
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
        
        self.clusterIndices = allGroups
        
        return allGroups
        
    def fitCluster(self, indices):
        
        initial_u0 = np.exp(-10.*np.random.uniform())
        initial_t0 = np.median(lc.mjd[clusterIdx])
        initial_tE = (max(lc.mjd[clusterIdx]) - min(lc.mjd[clusterIdx])) / 2.
        initial_F0 = RMagToFlux(lc.continuumMag)
        
        objective = lambda p, lc: (RMagToFlux(lc.mag) - model(lc.mjd, *p))**2 / (2.*lc.error**2)
        
        p0 = (initial_u0, initial_t0, initial_tE, initial_F0)
        full_out = so.fmin_powell(objective, p0, args=(self,), full_output=True, disp=0)
        print full_out