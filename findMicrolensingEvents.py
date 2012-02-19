"""
    Search the light curves in the local PTF database for 
    microlensing event candidates.
"""

# Standard library
import os, sys, glob
from argparse import ArgumentParser
import logging
import cPickle as pickle

# Third-party
import sqlalchemy
import numpy as np
import apwlib.geometry as g
import apwlib.convert as c
import matplotlib.pyplot as plt

# Project
import ptf.simulation.util as simu

from ptf.db.DatabaseConnection import *

def plot_lightcurve(lightCurve, m, save=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax = lightCurve.plot(ax)
    ax.axhline(m)
    ax.set_ylim(ax.get_ylim()[::-1])
    
    if save:
        filename = os.path.join("plots", "{0}.png".format(lightCurve.objid))
        plt.savefig(filename)
    else:
        plt.show()

def estimateContinuum(dbLightCurve, clipSigma=2.5):
    """ Estimate the continuum of the light curve using sigma clipping """
    
    rootVariance = np.std(dbLightCurve.mag)
    b = np.median(dbLightCurve.mag)
    
    mags = dbLightCurve.amag
    mjds = dbLightCurve.amjd
    sigmas = np.sqrt(dbLightCurve.amag_error**2 + dbLightCurve.asys_error**2)
    
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
    
    continuumMag = simu.fit_line(mjds, mags, sigmas)
    continuumSigma = rootVariance
    
    return continuumMag, continuumSigma

def findClusters(dbLightCurve, continuumMag, continuumSigma, num_points_per_cluster=4, sigma_multiplier=3.):
    # Determine which points are outside of continuumMag +/- 2 or 3 continuumSigma, and see if they are clustered
    #w = (dbLightCurve.mag > (continuumMag - 3*continuumSigma))
    w = (dbLightCurve.goodMag > (continuumMag - sigma_multiplier*continuumSigma))
    
    # This allows for decreases in brightness as well -- e.g. from an eclipse!
    #w = np.logical_or((dbLightCurve.goodMag > (continuumMag - 3*continuumSigma)), (dbLightCurve.goodMag < (continuumMag + 3*continuumSigma)))
    
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
    
    return allGroups
    
def fitCluster(dbLightCurve, indices):
    
    initial_u0 = np.exp(-10.*np.random.uniform())
    initial_t0 = np.median(lc.mjd[clusterIdx])
    initial_tE = (max(lc.mjd[clusterIdx]) - min(lc.mjd[clusterIdx])) / 2.
    initial_F0 = RMagToFlux(lc.continuumMag)
    
    objective = lambda p, lc: (RMagToFlux(lc.mag) - model(lc.mjd, *p))**2 / (2.*lc.error**2)
    
    p0 = (initial_u0, initial_t0, initial_tE, initial_F0)
    full_out = so.fmin_powell(objective, p0, args=(dbLightCurve,), full_output=True, disp=0)
    print full_out

def main():
    while True:
        lightCurves = session.query(LightCurve).filter(LightCurve.candidate == None).limit(100).all()
        if len(lightCurves) == 0: break
        
        logging.info("Light curves loaded!")

        for lightCurve in lightCurves:
            #if len(lightCurve.mjd) < 50: continue
            
            if len(lightCurve.goodMJD) < 10:
                logging.debug("Skipping light curve, not enough good data points...{0}".format(len(lightCurve.goodMJD)))
                lightCurve.candidate = 999
                continue
            
            if len(lightCurve.goodMJD) < 25:
                mjd = np.sort(lightCurve.goodMJD)
                medCadence = np.median(mjd[1:]-mjd[:-1])
                
                if medCadence > 5:
                    lightCurve.candidate = 999
                    logging.debug("Skipping light curve due to bad cadence...{0}".format(medCadence))
                    continue
            
            m, s = estimateContinuum(lightCurve)
            clusterIndices = findClusters(lightCurve, m, s, 4, sigma_multiplier=2.5)
            
            if len(clusterIndices) == 0: 
                logging.debug("No clusters found for this light curve")
                lightCurve.candidate = 0
                continue
            
            logging.info("-------------------------------------------------\nLight curve candidate!\n {0},{1}\nObjid: {2}\n-------------------------------------------------".format(lightCurve.ra, lightCurve.dec, lightCurve.objid))
            
            lightCurve.candidate = 1
            
            plot_lightcurve(lightCurve, m, save=True)
        
        session.flush()
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
        
    main()