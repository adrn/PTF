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
import db.util as dbu
import simulation.util as simu

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

def findClusters(dbLightCurve, continuumMag, continuumSigma, num_points_per_cluster=4):
    # Determine which points are outside of continuumMag +/- 2 or 3 continuumSigma, and see if they are clustered
    #w = (dbLightCurve.mag > (continuumMag - 3*continuumSigma))
    w = (dbLightCurve.goodMag > (continuumMag - 3*continuumSigma))
    
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
    lightCurves = dbu.session.query(dbu.LightCurve).all()
    print "light curves loaded"
    
    for lightCurve in lightCurves:
        #if len(lightCurve.mjd) < 50: continue
        if len(lightCurve.goodMJD) < 50: continue
        
        m, s = estimateContinuum(lightCurve)
        clusterIndices = findClusters(lightCurve, m, s, 4)
        
        if len(clusterIndices) == 0: continue
        
        print lightCurve.ra, lightCurve.dec
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax = lightCurve.plot(ax)
        ax.axhline(m)
        #for clusterIdx in clusterIndices:
        #    ax.axvline(np.median(lightCurve.mjd[clusterIdx]))
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.show()
        del fig
    
if __name__ == "__main__":
    main()