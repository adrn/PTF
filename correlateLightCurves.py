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
import pyfits as pf
import matplotlib.pyplot as plt
from sqlalchemy import func
import apwlib.geometry as g
import mlpy

# Project
from ptf.db.DatabaseConnection import *

__author__ = "adrn@astro.columbia.edu"

def correlation_function(light_curve):
    """ Compute the 'correlation function' for a given light curve with 
        its neighboring light curves
    """
    
    # Get all nearby light curves with +/- 10% of the number of points as the input
    num_points = len(light_curve.Rmjd)
    nearby_light_curves = session.query(LightCurve).filter(func.array_length(LightCurve.mjd, 1) > (num_points-0.1*num_points))\
                                            .filter(func.array_length(LightCurve.mjd, 1) < (num_points+0.1*num_points))\
                                            .filter(func.q3c_radial_query(LightCurve.ra, LightCurve.dec, light_curve.ra, light_curve.dec, 1.0))\
                                            .all()
    
    mjd1 = light_curve.Rmjd
    sorted_mjd1 = mjd1[np.argsort(mjd1)]
    Rmag1 = light_curve.Rmag
    sorted_Rmag1 = Rmag1[np.argsort(Rmag1)]
    
    data = []
    for lc in nearby_light_curves:
        sorted_mjd2 = lc.Rmjd[np.argsort(lc.Rmjd)]
        Rmag2 = lc.Rmag
        sorted_Rmag2 = Rmag2[np.argsort(Rmag2)]
        
        if len(sorted_mjd1) < len(sorted_mjd2):
            sorted_filtered_Rmag2 = sorted_Rmag2[np.in1d(sorted_mjd2, sorted_mjd1)]
            sorted_filtered_Rmag1 = sorted_Rmag1
        elif len(sorted_mjd1) > len(sorted_mjd2):
            sorted_filtered_Rmag1 = sorted_Rmag1[np.in1d(sorted_mjd1, sorted_mjd2)]
            sorted_filtered_Rmag2 = sorted_Rmag2
        else:
            sorted_filtered_Rmag1 = sorted_Rmag1
            sorted_filtered_Rmag2 = sorted_Rmag2
        
        distance = g.subtends(light_curve.ra, light_curve.dec, lc.ra, lc.dec, units="degrees")
        correlation = mlpy.dtw_std(sorted_filtered_Rmag1, sorted_filtered_Rmag2)
        data.append((distance, correlation))
        
        if correlation > 30000 or correlation < 3:
            fig = plt.figure()
            ax1 = fig.add_subplot(211)
            ax1.plot(sorted_mjd1, sorted_Rmag1, 'k.')
            
            ax2 = fig.add_subplot(212)
            ax2.plot(sorted_mjd2, sorted_Rmag2, 'k.')
            
            fig.savefig("{0}_{1}.png".format(light_curve.objid, lc.objid))

    data = np.array(data, dtype=[("distance", float), ("correlation", float)]).view(np.recarray)
    print max(data.distance), min(data.distance)
    
    fig2 = plt.figure()
    ax = fig.add_subplot(111)
    ax.semilogy(data.distance, data.correlation, 'ko')
    fig.savefig("plots/correlation.pdf")
    
    return
    
    for bin in np.arange(0., 3600., 50.): #arcseconds
        thisData = data[(data.distance > bin/3600.) & (data.distance < (bin+50)/3600.)]
        #print np.sum(thisData.correlation) / float(len(thisData))
        print np.median(thisData.correlation), np.mean(thisData.correlation)
            
if __name__ == "__main__":
    lc = session.query(LightCurve).filter(func.array_length(LightCurve.mjd, 1) > 1000).limit(1).one()
    correlation_function(lc)