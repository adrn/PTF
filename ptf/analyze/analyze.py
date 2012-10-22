# coding: utf-8
from __future__ import division

""" This module contains classes and functions used to analyze PTF data. """

# TODO: look in to lmfit's stderr and whether I should look at this when assessing fits

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import copy

# Third-party
import numpy as np
from lmfit import minimize, Parameters

# PTF
from .models import *

__all__ = ["fit_subtract_microlensing", "fit_microlensing_event", "fit_constant_line"]

def _parameters_to_dict(parameters):
    """ Convert an lmfit Parameters object to a Python dictionary """
    dict_params = {}
    for key,param in parameters.items():
        dict_params[key] = param.value
    return dict_params

def fit_subtract_microlensing(light_curve, fit_data=None):
    """ Fit and subtract a microlensing event to the light curve """
    
    if fit_data == None:
        fit_data = fit_microlensing_event(light_curve)
    
    light_curve_new = copy.copy(light_curve)
    light_curve_new.mag = light_curve.mag - microlensing_model(fit_data, light_curve_new.mjd)
    
    light_curve.tE = fit_data["tE"].value
    light_curve.t0 = fit_data["t0"].value
    light_curve.u0 = fit_data["u0"].value
    light_curve.m0 = fit_data["m0"].value
    light_curve.chisqr = float(fit_data["result"].chisqr)
    
    return light_curve_new

def fit_microlensing_event(light_curve, initial_params={}):
    """ Fit a microlensing event to the light curve """
    
    t0 = np.random.normal(light_curve.mjd[np.argmin(light_curve.mag)], 1.)
    if t0 > light_curve.mjd.max() or t0 < light_curve.mjd.min():
        t0 = light_curve.mjd[np.argmin(light_curve.mag)]
        
    initial_tE = initial_params.get("tE", 10**np.random.uniform(1., 2.5))
    initial_t0 = initial_params.get("t0", t0)
    initial_u0 = initial_params.get("u0", np.random.uniform(1E-6, 1.33))
    initial_m0 = initial_params.get("m0", np.random.normal(np.median(light_curve.mag), 0.5))
    
    params = Parameters()
    params.add('tE', value=initial_tE, min=2., max=1000.)
    params.add('t0', value=initial_t0, min=light_curve.mjd.min(), max=light_curve.mjd.max())
    params.add('u0', value=initial_u0, min=0.0, max=1.34)
    params.add('m0', value=initial_m0)
    
    result = minimize(microlensing_error_func, params, args=(light_curve.mjd, light_curve.mag, light_curve.error))
    
    return {"tE" : params["tE"], \
            "t0" : params["t0"], \
            "u0" : params["u0"], \
            "m0" : params["m0"], \
            "result" : result}

def fit_constant_line(light_curve, initial_params={}):
    """ Fit a line of constant brightness to the light curve """
    
    initial_m0 = initial_params.get("m0", np.random.normal(np.median(light_curve.mag), 0.5))
    
    params = Parameters()
    params.add('m0', value=initial_m0)
    
    result = minimize(constant_error_func, params, args=(light_curve.mjd, light_curve.mag, light_curve.error))
    
    return {"m0" : params["m0"], \
            "result" : result}