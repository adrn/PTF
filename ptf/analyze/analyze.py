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
from .statistics import *
from ..util import get_logger
logger = get_logger(__name__)

__all__ = ["fit_subtract_microlensing", "fit_microlensing_event", "fit_constant_line", "iscandidate"]

def _parameters_to_dict(parameters):
    """ Convert an lmfit Parameters object to a Python dictionary """
    dict_params = {}
    for key,param in parameters.items():
        try:
            dict_params[key] = param.value
        except AttributeError:
            pass
    return dict_params

def fit_subtract_microlensing(light_curve, fit_data=None):
    """ Fit and subtract a microlensing event to the light curve """

    if fit_data == None:
        fit_data = fit_microlensing_event(light_curve)

    light_curve_new = copy.copy(light_curve)
    light_curve_new.mag = light_curve.mag - microlensing_model(_parameters_to_dict(fit_data), light_curve_new.mjd)

    light_curve.tE = fit_data["tE"].value
    light_curve.t0 = fit_data["t0"].value
    light_curve.u0 = fit_data["u0"].value
    light_curve.m0 = fit_data["m0"].value
    light_curve.chisqr = float(fit_data["result"].chisqr)

    return light_curve_new

def fit_microlensing_event(light_curve, initial_params={}):
    """ Fit a microlensing event to the light curve """

    t0 = np.random.normal(light_curve.mjd[np.argmin(light_curve.mag)], 2.)
    if t0 > light_curve.mjd.max() or t0 < light_curve.mjd.min():
        t0 = light_curve.mjd[np.argmin(light_curve.mag)]

    #initial_tE = initial_params.get("tE", 10**np.random.uniform(1., 2.5))
    initial_tE = initial_params.get("tE", np.random.uniform(2, 500))
    initial_t0 = initial_params.get("t0", t0)
    #initial_u0 = initial_params.get("u0", np.random.uniform(1E-6, 1.33))
    initial_u0 = initial_params.get("u0", 10**np.random.uniform(-3, 0.12))
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

def iscandidate(light_curve, lower_eta_cut, num_fit_attempts=10):
    ''' Given a LightCurve object, determine whether or not it is considered a
        microlensing event candidate. We assume the light curve data has already
        been cleaned of any bad data.

        Parameters
        ----------
        light_curve : LightCurve instance
            A PTFLightCurve instance.
        lower_eta_cut : float
            The value of the lower cut on eta as determined by the FPR simulation.
        num_fit_attempts : int
            Number of times to fit a microlensing event to the data with random
            initial conditions.

    '''

    # Make sure that eta still passes the cut, since the light curve may have been 'cleaned'
    if light_curve.indices["eta"] > lower_eta_cut:
        logger.debug("Didn't pass initial eta cut")
        return False

    ml_chisq = 1E6
    ml_params = None
    for ii in range(num_fit_attempts):
        # DEBUG
        params = fit_microlensing_event(light_curve)
        new_chisq = params["result"].chisqr

        if new_chisq < ml_chisq:
            if abs(new_chisq-ml_chisq)/ml_chisq < 0.05: # less than a 5% change
                ml_chisq = new_chisq
                ml_params = params
                break
            else:
                ml_chisq = new_chisq
                ml_params = params

    try:
        light_curve.features
    except AttributeError:
        light_curve.features = dict()

    if ml_params == None:
        logger.debug("Fit didn't converge")
        return False

    # Add all indices as features for the light curve
    for key,val in light_curve.indices.items():
        light_curve.features[key] = val

    light_curve.features["N"] = len(light_curve)
    light_curve.features["median_error"] = np.median(light_curve.error)
    light_curve.features["chisqr"] = ml_params["result"].chisqr
    light_curve.features["t0"] = ml_params["t0"].value
    light_curve.features["u0"] = ml_params["u0"].value
    light_curve.features["tE"] = ml_params["tE"].value
    light_curve.features["m0"] = ml_params["m0"].value

    # Try to fit a microlensing model, then subtract it, then recompute eta and see
    #   if it is still an outlier
    new_light_curve = fit_subtract_microlensing(light_curve, fit_data=ml_params)
    new_eta = eta(new_light_curve)

    # Make sure the fit microlensing event parameters are reasonable:
    #     - if u0 is too big, it's probably just a flat light curve
    if round(light_curve.u0, 2) >= 1.34:
        logger.debug("fit u0 > 1.34")
        return False

    #     - if tE is too small, it's probably bad data
    if round(light_curve.tE) <= 2.:
        logger.debug("fit tE < 2 days")
        return False

    #     - if tE is too large, it's probably a long-period variable or the baseline is too short
    if light_curve.tE > 1.5*light_curve.baseline:
        logger.debug("fit tE > baseline")
        return False

    #    - if t0 is too close to either end of the light curve
    if light_curve.t0 < (light_curve.tE/2. + light_curve.mjd.min()):
        logger.debug("fit t0 too close to min mjd")
        return "subcandidate"
    elif light_curve.t0 > (light_curve.mjd.max() - light_curve.tE/2.):
        logger.debug("fit t0 too close to min mjd")
        return "subcandidate"

    # If light curve still passes the eta cut after subtracting the microlensing event, it
    #    is probably periodic or has some other variability
    if new_eta <= lower_eta_cut:
        logger.debug("new_eta cut failed: new eta {0}, cut {1}".format(new_eta, lower_eta_cut))
        return "subcandidate"

    # Slice out only data around microlensing event
    sliced_lc = light_curve.slice_mjd(ml_params["t0"].value-ml_params["tE"].value, ml_params["t0"].value+ml_params["tE"].value)
    if len(sliced_lc) < 5:
        return False

    # Count number of data points between t0-tE and t0+tE, make sure we have more than 5 brighter than 3sigma
    if sum(sliced_lc.mag < (np.median(light_curve.mag) - 3.*np.mean(sliced_lc.error))) < 5:
        logger.debug("not enough points in sliced_lc: {0}, {1}".format(len(sliced_lc), sum(sliced_lc.mag < (np.median(light_curve.mag) - 3.*np.std(light_curve.mag)))))
        return False

    # Fit another microlensing model to just the data around the event
    sliced_ml_params = fit_microlensing_event(sliced_lc)
    new_sliced_light_curve = fit_subtract_microlensing(sliced_lc, fit_data=sliced_ml_params)

    # If the scatter of the event subtracted, sliced light curve is > 2*median(sigma) for the errors in
    #    the sliced light curve, it is probably just bad data
    if np.std(new_sliced_light_curve.mag) > 2.5*np.mean(sliced_lc.error):
        logger.debug("scatter in sliced-subtracted light curve too high")
        return "subcandidate"

    return "candidate"

# If this returns subcandidate, do varstar search, if periodic remove subcandidate tag?