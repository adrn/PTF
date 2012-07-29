# coding: utf-8
""" General utilities for the PTF project """

# Standard library
import os, sys

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Project
from ptf.db.DatabaseConnection import *
import ptf.simulation.util as simu
from ptf import PTFLightCurve

parameter_to_label = {"j" : "J", "k" : "K", "sigma_mu" : r"$\sigma/\mu$", "eta" : r"$\eta$", "delta_chi_squared" : r"$\Delta \chi^2$"}

def txt_file_light_curve_to_recarray(filename):
    """ Load Marcel / Nick's text file light curves into a Numpy recarray.
    
        All columns in this file are:
            MJD
            R1	<-- ignore (I think; doesn't matter much
            R2	which of these two you use for this, I say)
            R2 err
            RA
            DEC
            X
            Y
            flag
            file
            field
            chip
            
            MJD's are relative to 2009-1-1 (54832)
    """
    
    names = ["mjd", "mag", "mag_err", "ra", "dec", "flag", "fieldid", "ccd"]
    usecols = [0, 2, 3, 4, 5, 8, 10, 11]
    dtypes = [float, float, float, float, float, float, int, int]
    
    return np.genfromtxt(filename, usecols=usecols, dtype=zip(names, dtypes)).view(np.recarray)

def compute_indices(light_curve_generator):
    """ Given a light curve generator, compute the variability indices for all light curves
        returned by the generator and update the database.
        
        A light curve generator is a generator object that returns chunks of light curves from
        the PTF database. For an example, see PraesepeLightCurves.py -> 
        
        This function also checks to see that more than 50% of the data points in a given 
        light curve are "good". If not, ignore that light curve.
        
        2012-05-21 TODO: Implement Nick's suggestions for how to select out "good" data points!
        
    """
    
    while True:
        num_bad = 0.0
        try:
            lightCurves = light_curve_generator.next()
        except StopIteration:
            break
            
        for lightCurve in lightCurves:
            try:
                # Remove old variability indices
                var = session.query(VariabilityIndices).join(LightCurve).filter(LightCurve.objid == lightCurve.objid).one()
                session.delete(var)
            except sqlalchemy.orm.exc.NoResultFound:
                pass
            
            # Only select points where the error is less than the acceptable percent error
            #   based on a visual fit to the plot Nick sent on 4/30/12
            if 15 <= np.median(lightCurve.amag) <= 19:
                percent_acceptable = 0.01 * 10**( (np.median(lightCurve.amag) - 15.) / 3.5 )
            elif np.median(lightCurve.amag) < 15:
                percent_acceptable = 0.01
            else:
                percent_acceptable = 0.5
            
            idx = (lightCurve.error / lightCurve.amag) < percent_acceptable
            
            # Check that less than half of the data points are bad
            if float(sum(idx)) / len(lightCurve.error) <= 0.5:
                logging.debug("Bad light curve -- ignore=True")
                num_bad += 1
                lightCurve.ignore = True
                continue
            
            lc = PTFLightCurve(lightCurve.amjd[idx], lightCurve.amag[idx], lightCurve.error[idx])
            try:
                # Returns a dictionary with keys = the names of the indices
                var_indices = simu.compute_variability_indices(lc)
            except NameError:
                continue
            
            logging.debug("Good light curve -- ignore=False")
            lightCurve.ignore = False
            variabilityIndices = VariabilityIndices()
            
            for key in var_indices.keys():
                setattr(variabilityIndices, key, var_indices[key])

            variabilityIndices.light_curve = lightCurve
            session.add(variabilityIndices)
            
        logging.info("Fraction of good light curves: {}".format(1-num_bad/1000))
        session.flush()
        
    session.flush()

# THIS FUNCTION NEEDS AN OVERHAUL
def load_and_match_txt_coordinates(file, vi_names):
    """ Load a text file with a list of coordinates and match them to
        my local database of light curves
    """
    raise NotImplementedError("This implementation is broken. -apw")
    raDecs = np.genfromtxt(file, delimiter=",")
    
    matchedTargets = []
    for raDec in raDecs:
        ra = raDec[0]
        dec = raDec[1]
        try:
            varIdx = session.query(VariabilityIndices)\
                        .join(LightCurve)\
                        .filter(LightCurve.objid < 100000)\
                        .filter(func.q3c_radial_query(LightCurve.ra, LightCurve.dec, ra, dec, 5./3600))\
                        .one()
                        
            matchedTargets.append(tuple([getattr(varIdx, idx) for idx in vi_names]))
        except sqlalchemy.orm.exc.NoResultFound:
            pass
            
    return np.array(matchedTargets, dtype=zip(vi_names, [float]*len(vi_names))).view(np.recarray)

def some_name(light_curves, indices, figure=None, figsize=(25,25)):
    """ Given a list of light curve objects and a list of indices, make a variability 
        indices figure
    """
    num_indices = len(indices)
    
    if figure == None:
        figure, subplots = plt.subplots(num_indices, num_indices, figsize=figsize)
    else:
        subplots = np.array(figure.axes).reshape(num_indices, num_indices)
    
    # Double loop over each index to produce an N by N plot
    for ii, yParameter in enumerate(indices):
        for jj, xParameter in enumerate(indices):
            pass
        