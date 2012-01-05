# Outlier detection

# Standard Library
import sys, os
import cPickle as pickle

# External Packages
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def straight(t, b):
    return np.ones((len(t),), dtype=float)*b
    
def fit_line(x, y, sigma_y):
    popt, pcov = curve_fit(straight, x, y, sigma=sigma_y)
    return popt[0]

def get_continuum(data):
    sigma = np.std(data.mag)
    #b = fit_line(data.t, data.mag, data.sigma)
    b = np.median(data.mag)
    
    # This is a stupid sigma-clipping way to find the continuum magnitude 
    CLIPSIG = 3
    
    sig = sigma
    mag = data.mag
    t = data.t
    sigmas = data.sigma
    while True:
        w = np.fabs(mag - np.median(mag)) < CLIPSIG*sig
        new_mag = mag[w]
        new_t = t[w]
        new_sigmas = sigmas[w]

        if (len(mag) - len(new_mag)) <= (0.02*len(mag)):
            break
        else:
            mag = new_mag
            t = new_t
            sigmas = new_sigmas
            sig = np.std(mag)
    
    contMag = fit_line(t, mag, sigmas)
    contStd = np.std(mag)
    
    return contMag, contStd

def find_clusters(data, contMag, contStd):
    # Next, determine which points are outside of contMag +/- 2 or 3 contStd, and see if they are clustered
    w = (data.mag > (contMag - 3*contStd)) & (data.mag < (contMag + 3*contStd))
    
    group = []
    in_group = False
    group_finished = False
    for idx, pt in enumerate(np.logical_not(w)):
        if len(group) > 0:
            in_group = True
        
        if pt:
            group.append(idx)
        else:
            if in_group and len(group) >= 3: 
                print "cluster found!"
                return np.array(group)
                
            in_group = False
            group = []
    
    if in_group and len(group) >= 3: 
        print "cluster found!"
        return np.array(group)
    else:
        return np.array([False]*len(data.t))