# coding: utf-8
from __future__ import division

""" This module contains classes and functions used to analyze PTF data. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import math

# Third-party
import numpy as np
import scipy.optimize as so

try:
    from apwlib.globals import greenText, yellowText, redText
except ImportError:
    raise ImportError("apwlib not found! \nDo: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install' to install.")

# ------
# Models
# ------
def error_function(p, x, y, sigma_y, model):
    return (y - model(p,x)) / sigma_y
    
constant_model = lambda p, x: p[0] + np.zeros(len(x))
constant_error_func = lambda p, x, mag, sigma: (mag - constant_model(p, x)) / sigma

linear_model = lambda p, x: p[0]*x + p[1]
linear_error_func = lambda p, x, mag, sigma: (mag - linear_model(p, x)) / sigma

gaussian_model = lambda p, x: p[0]*np.exp(-(x - p[1])**2 / (2*p[2]**2)) + p[3]
gaussian_error_func = lambda p, x, mag, sigma: (mag - gaussian_model(p, x)) / sigma

def u_t(p, t):
    return np.sqrt(p[0]**2 + ((t - p[1])/p[2])**2)

def A_u(u):
    return (u**2 + 2) / (u*np.sqrt(u**2 + 4))

microlensing_flux_model = lambda t, p: p[0]*A_u(u_t(p[1:], t))

# -------------
# Analysis code
# -------------

#def estimate_continuum(light_curve, continuum_model, initial_parameters, sigma_clip=True, clip_sigma=2.5):
def estimate_continuum(light_curve, sigma_clip=True, clip_sigma=2.5):
    """ Estimate the continuum level of a light curve.
        
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
    
    # Try to fit the continuum with a constant function
    popt, covx, infodict, mesg, lin_ier = so.leastsq(constant_error_func, \
                                                     x0=(np.median(mag),),
                                                     args=(mjd, mag, error), \
                                                     full_output=1)
    sig = np.std(mag)
        
    return popt, sig
        
def test_estimate_continuum():
    import matplotlib.pyplot as plt
    
    np.random.seed(1)
    # Generate a flat light curve and test continuum fit on flat
    #   light curve
    for truth in np.random.random(100)*100:
        mjds = np.linspace(0., 100., 100)
        sigmas = np.zeros(len(mjds)) + 0.1
        mags = truth*np.ones(len(mjds)) + np.random.normal(0., sigmas)
        mags += np.exp(-(mjds-50)**2/(2.*5**2))
     
        from ptf.ptflightcurve import PTFLightCurve
        lc = PTFLightCurve(mjds, mags, sigmas)
        
        popt, sigma = estimate_continuum(lc, sigma_clip=False)
        popt_clipped, sigma_clipped = estimate_continuum(lc, sigma_clip=True)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(np.linspace(0., 100., 100.), [popt[0]]*100, 'r--', label="not clipped")
        ax.plot(np.linspace(0., 100., 100.), [popt_clipped[0]]*100, 'b--', label="clipped")
        ax.plot(np.linspace(0., 100., 100.), [truth]*100, 'g-')
        ax = lc.plot(ax)
        ax.legend()
        plt.show()

def compute_delta_chi_squared(light_curve, model1, model1_initial, model2, model2_initial, force_fit=False, num_attempts=10):
    """ Compute the difference in chi-squared between two different model 
        fits to the light curve.
        
        Parameters
        ----------
        light_curve : LightCurve
            A LightCurve object -- must have mjd, mag, and error attributes.
        model1 : function
        model1_initial : tuple
            The initial parameter guess for model1.
        model2 : function
        model2_initial : tuple
            The initial parameter guess for model2.
        force_fit : bool
            Force the model fits to converge. Will try 'num_attempts' times.
        num_attempts : int
            Number of times to try iterating the fit with new initial conditions. Only relevant
            if force_fit=True.
    """
    
    error_func_1 = lambda p, x, mag, sigma: (mag - model1(p, x)) / sigma
    error_func_2 = lambda p, x, mag, sigma: (mag - model2(p, x)) / sigma
    
    # If we need to force the fits to converge, we have to iterate the fits
    #   until the fit succeeds.
    if force_fit:
        model1_ier = 0
        model2_ier = 0
        
        # Loop breaks when we reach 'num_attempts' tries or when the convergence value
        #   model1_ier is one of the accepted, e.g. 1, 2, 3, or 4.
        tries = 0
        initial_params = model1_initial
        while model1_ier not in [1,2,3,4] and tries <= num_attempts:
            model1_params, covx, infodict, mesg, model1_ier = so.leastsq(error_func_1, \
                x0=initial_params, \
                args=(light_curve.mjd, light_curve.mag, light_curve.error), \
                full_output=1)
            tries += 1
            
            for ii,p in enumerate(initial_params):
                if p == 0: initial_params[ii] = np.random.normal(p, 0.5*tries)
                else: initial_params[ii] = np.random.normal(p, p/10.*tries)
        
        # Loop breaks when we reach 'num_attempts' tries or when the convergence value
        #   model2_ier is one of the accepted, e.g. 1, 2, 3, or 4.
        tries = 0
        initial_params = model2_initial
        while model2_ier not in [1,2,3,4] and tries <= num_attempts:
            model2_params, covx, infodict, mesg, model2_ier = so.leastsq(error_func_2, \
                x0=initial_params, \
                args=(light_curve.mjd, light_curve.mag, light_curve.error), \
                full_output=1)
            tries += 1
            
            for ii,p in enumerate(initial_params):
                if p == 0: initial_params[ii] = np.random.normal(p, 0.5*tries)
                else: initial_params[ii] = np.random.normal(p, p/10.*tries)
        
    else:
        # Otherwise, only try to fit once
        model1_params, covx, infodict, mesg, model1_ier = so.leastsq(error_func_1, \
                x0=model1_initial, \
                args=(light_curve.mjd, light_curve.mag, light_curve.error), \
                full_output=1)
        
        model2_params, covx, infodict, mesg, model2_ier = so.leastsq(error_func_2, \
                x0=model2_initial, \
                args=(light_curve.mjd, light_curve.mag, light_curve.error), \
                full_output=1)
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(light_curve.mjd, model1(model1_params, light_curve.mjd), 'r--')
    ax.plot(light_curve.mjd, model2(model2_params, light_curve.mjd), 'g--')
    ax = light_curve.plot(ax)
    plt.show()
    
    model1_chisq = np.sum(error_func_1(model1_params, \
                                       light_curve.mjd, \
                                       light_curve.mag, \
                                       light_curve.error)**2)# / len(model1_params)
    
    model2_chisq = np.sum(error_func_2(model2_params, \
                                       light_curve.mjd, \
                                       light_curve.mag, \
                                       light_curve.error)**2)# / len(model2_params)
    
    return model1_chisq - model2_chisq

def test_compute_delta_chi_squared():
    
    from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
    mjd = np.linspace(0., 500., 150)
    sigmas = 10.**np.random.uniform(-2, -3, size=150)
    light_curve = SimulatedLightCurve(mjd, error=sigmas)
    
    # Make a flat light curve, and show that the difference in chi-squared is
    #   negligible between the two fits.
    dcs = compute_delta_chi_squared(light_curve, \
                                    model1=linear_model, model1_initial=(0.0, np.median(light_curve.mag)),\
                                    model2=gaussian_model, model2_initial=(-1.0, np.median(light_curve.mjd), 5.0, np.median(light_curve.mag))\
                                    )
    
    print dcs
    
    # Make a light curve with a gaussian peak, and show that the gaussian model
    #   chi-squared is better.
    light_curve.addMicrolensingEvent(t0=250.)
    dcs = compute_delta_chi_squared(light_curve, \
                                    model1=linear_model, model1_initial=(0.0, np.median(light_curve.mag)),\
                                    model2=gaussian_model, model2_initial=(-5.0, np.median(light_curve.mjd), 5.0, np.median(light_curve.mag))\
                                    )
    
    print dcs

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

def test_stetson_j():
    import copy
    from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
    mjd = np.linspace(0., 500., 150)
    sigmas = 10.**np.random.uniform(-2, -3, size=150)
    light_curve = SimulatedLightCurve(mjd, error=sigmas)
    lc2 = copy.copy(light_curve)
    
    # Make a flat light curve + microlensing event, compute J
    light_curve.addMicrolensingEvent(t0=250.)
    
    assert stetson_j(lc2) < stetson_j(light_curve)

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

def test_stetson_k():
    import copy
    from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
    mjd = np.linspace(0., 500., 150)
    sigmas = 10.**np.random.uniform(-2, -3, size=150)
    light_curve = SimulatedLightCurve(mjd, error=sigmas)
    
    # Make a flat light curve + microlensing event, compute J
    light_curve.addMicrolensingEvent(t0=250.)
    
    assert 0.7 < stetson_k(light_curve) < 0.9

def eta(light_curve):
    """ Compute the von Neumann (1941) ratio """
    mag = light_curve.mag
    err = light_curve.error
    N = len(mag)
    
    delta_squared = np.sum((mag[1:] - mag[:-1])**2 / (N - 1.))
    eta = delta_squared / np.var(mag)
    
    return eta

def test_eta():
    from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
    mjd = np.linspace(0., 500., 150)
    sigmas = 10.**np.random.uniform(-2, -3, size=150)
    light_curve = SimulatedLightCurve(mjd, error=sigmas)
    
    print eta(light_curve)
    
    # Make a flat light curve + microlensing event, compute J
    light_curve.addMicrolensingEvent(t0=250.)
    
    print eta(light_curve)

def compute_variability_indices(light_curve, indices=[]):
    """ Computes variability statistics, such as those explained in M.-S. Shin et al. 2009.
        Valid indices:
            j : Stetson J parameter
            k : Stetson K parameter, kurtosis
            delta_chi_squared : Difference in chi-squared values for a linear fit vs. a Gaussian
            eta : Ratio of mean square successive difference to the sample variance, von Neumann 1941
            continuum : the continuum magnitude
        
        Parameters
        ----------
        light_curve : LightCurve
            A LightCurve object -- must have mjd, mag, and error attributes.
        indices : iterable
            See valid indices above.
    """
    idx_dict = dict()
    
    if "j" in indices:
        idx_dict["j"] = stetson_j(light_curve)
    
    if "k" in indices:
        idx_dict["k"] = stetson_k(light_curve)
    
    if "eta" in indices:
        idx_dict["eta"] = eta(light_curve)
    
    if "delta_chi_squared" in indices:
        idx_dict["delta_chi_squared"] = compute_delta_chi_squared(light_curve,\
                                                                  model1=linear_model, \
                                                                  model1_initial=(0.0, np.median(light_curve.mag)),\
                                                                  model2=gaussian_model, \
                                                                  model2_initial=(-5.0, np.median(light_curve.mjd), 5.0, np.median(light_curve.mag))\
                                                                 )
        
    if "continuum" in indices:
        continuum_mag, noise_sigma = estimate_continuum(light_curve, sigma_clip=True)
        idx_dict["continuum"] = continuum_mag
    
    