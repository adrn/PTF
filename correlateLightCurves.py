""" This script computes the correlation of nearby light curves as a 
    function of distance in RA/Dec.
    
    TODO: Load in the entire field with >6000 observations, and run this on THAT field!
"""

# Standard library
import os, sys
import logging
logging.basicConfig(level=logging.DEBUG)

# Third party
import numpy as np
from sqlalchemy import func
import apwlib.geometry as g
import mlpy

# Project
from ptf.db.DatabaseConnection import *

__author__ = "adrn@astro.columbia.edu"

def correlation_function(number_of_points_range, number_of_light_curves=100):
    """ Given the minimum and maximum number of observations,
        compute the correlation function for how similar nearby
        light curves are as a function of angular separation
    """
    data = []
    for ii in range(number_of_light_curves):
        light_curve = session.query(LightCurve).filter(func.array_length(LightCurve.mjd, 1) > number_of_points_range[0])\
                                                .filter(func.array_length(LightCurve.mjd, 1) < number_of_points_range[1])\
                                                .order_by(func.random()).limit(1).one()
        
        nearby_light_curves = session.query(LightCurve).filter(func.array_length(LightCurve.mjd, 1) > number_of_points_range[0])\
                                                .filter(func.array_length(LightCurve.mjd, 1) < number_of_points_range[1])\
                                                .filter(func.q3c_radial_query(LightCurve.ra, LightCurve.dec, light_curve.ra, light_curve.dec, 1.0))\
                                                .all()
        
        for lc in nearby_light_curves:
            distance = g.subtends(light_curve.ra, light_curve.dec, lc.ra, lc.dec, units="degrees")
            correlation = mlpy.dtw_std(light_curve.Rmag, lc.Rmag)
            data.append((distance, correlation))
        data = np.array(data, dtype=[("distance", float), ("correlation", float)]).view(np.recarray)
        print max(data.distance), min(data.distance)
        break
        for bin in np.arange(0., 3600., 50.): #arcseconds
            thisData = data[(data.distance > bin/3600.) & (data.distance < (bin+50)/3600.)]
            #print np.sum(thisData.correlation) / float(len(thisData))
            print np.median(thisData.correlation), np.mean(thisData.correlation)
            
if __name__ == "__main__":
    correlation_function((100,150), 1)