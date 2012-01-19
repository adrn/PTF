# Try to fit ML curve to the data

# Standard Library
import sys, os
import cPickle as pickle

# External Packages
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Project Scripts
import simulated_event as SE

#SE.FLUXMODEL(modelT, *true_params)

def fit_lightcurve(data, p0):
    popt, pcov = curve_fit(SE.FLUXMODEL, data.t, SE.RMagToFlux(data.mag), p0=p0, sigma=data.sigma, maxfev=10000)
    goodness = np.sum((SE.RMagToFlux(data.mag) - SE.FLUXMODEL(data.t, *popt)) / data.sigma)**2
    return popt, goodness

    