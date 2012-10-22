# coding: utf-8
from __future__ import division

""" This module contains model and error functions for fitting to light curve data """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os

# Third-party
import numpy as np

try:
    import error_functions
except ImportError, RuntimeError:
    print "Error: C extension error_functions.so not found or unable to import it! Make sure to do 'python setup.py build_ext' before running."
    raise ImportError

_constant_error_func = error_functions.constant_error_func
_linear_error_func = error_functions.linear_error_func
_gaussian_error_func = error_functions.gaussian_error_func
_microlensing_error_func = error_functions.microlensing_error_func

__all__ = ["constant_model", "linear_model", "gaussian_model", "microlensing_model", \
           "constant_error_func", "linear_error_func", "gaussian_error_func", "microlensing_error_func"]

# ------
# Models
# ------

def A(p, t):
    """ Microlensing amplifiction factor """
    u = np.sqrt(p["u0"]*p["u0"] + ((t-p["t0"])/p["tE"])**2)
    return (u**2 + 2) / (u*np.sqrt(u**2 + 4))
    
constant_model = lambda p, x: p["b"] + np.zeros(len(x))
linear_model = lambda p, x: p["m"]*x + p["b"]
gaussian_model = lambda p, x: p["A"]*np.exp(-(x - p["mu"])**2 / (2*p["sigma"]**2)) + p["B"]
microlensing_model = lambda p, x: p["m0"] - 2.5*np.log10(A(p, t))

# ---------------------------------------------------------
# Python error functions, wrappers around C error functions
# ---------------------------------------------------------
def microlensing_error_func(p, t, mag, sigma):
    """ Helper for C-based microlensing error function """
    try:
        u0, t0, tE, m0 = p["u0"].value, p["t0"].value, p["tE"].value, p["m0"].value
    except AttributeError:
        u0, t0, tE, m0 = p["u0"], p["t0"], p["tE"], p["m0"]
    return _microlensing_error_func((u0, t0, tE, m0), t, mag, sigma)

def gaussian_error_func(p, t, mag, sigma):
    """ Helper for C-based Gaussian error function """
    try:
        A, mu, sig, B = p["A"].value, p["mu"].value, p["sigma"].value, p["B"].value
    except AttributeError:
        A, mu, sig, B = p["A"], p["mu"], p["sigma"], p["B"]
    return _gaussian_error_func((A, mu, sig, B), t, mag, sigma)

def linear_error_func(p, t, mag, sigma):
    """ Helper for C-based linear error function """
    try:
        m, b = p["m"].value, p["b"].value
    except AttributeError:
        m, b = p["m"], p["b"]
    return _linear_error_func((m, b), t, mag, sigma)

def constant_error_func(p, t, mag, sigma):
    """ Helper for C-based constant error function """
    try:
        b = p["b"].value
    except AttributeError:
        b = p["b"]
    return _constant_error_func((b, ), t, mag, sigma)