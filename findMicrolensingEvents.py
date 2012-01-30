# -*- coding: utf-8 -*-

""" Search the ptf_microlensing database to find events. """

__author__ = "adrn <adrn@astro.columbia.edu>"

import sys
from optparse import OptionParser
import logging
import multiprocessing

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.optimize import curve_fit
import scipy.optimize as so
import matplotlib.pyplot as plt
from sqlalchemy.sql.expression import func

from DatabaseConnection import *
session = Session

class PTFLightCurve:

    def __init__(self, dbLightCurve):
        """ Accepts a sqlalchemy LightCurve object, and
            creates a new object with bonus features and
            no ties to the psycopg2 engine (e.g. you can
            use multiprocessing with these objects!)
        """
        self.fieldid = dbLightCurve.fieldid
        
        reOrder = np.argsort(dbLightCurve.mjd)
        filterid = np.array(dbLightCurve.filterid)[reOrder]
        self.mjd = np.array(dbLightCurve.mjd)[reOrder][filterid == 2]
        self.mag = np.array(dbLightCurve.mag)[reOrder][filterid == 2]
        self.mag_error = np.array(dbLightCurve.mag_error)[reOrder][filterid == 2]
        self.sys_error = np.array(dbLightCurve.sys_error)[reOrder][filterid == 2]
        self.objid = str(dbLightCurve.objid)

        # Get rid of messy data points

        """
        MAXERR = 0.1
        w = (np.sqrt(self.mag_error**2 + self.sys_error**2) < MAXERR)
        self.mjd = self.mjd[w]
        self.mag = self.mag[w]
        self.mag_error = self.mag_error[w]
        """
        #if ((len(dbLightCurve.mjd) - len(self.mjd)) / len(dbLightCurve.mjd)) > 0.05:
        #    raise ValueError("Too many bad data points!")
        
    def measureContinuum(self, clipSigma=2.):
        """ This is a stupid sigma-clipping way to find 
            the continuum magnitude. This often fails when
            the microlensing event occurs over a significant 
            number of data points, but I'm not sure what a
            better method would be?
        """
        
        sigma = np.std(self.mag)
        b = np.median(self.mag)
        
        mags = self.mag
        mjds = self.mjd
        sigmas = self.mag_error
        
        while True:
            w = np.fabs(mags - np.median(mags)) < clipSigma*sigma
            new_mags = mags[w]
            new_mjds = mjds[w]
            new_sigmas = sigmas[w]
    
            if (len(mags) - len(new_mags)) <= (0.02*len(mags)):
                break
            else:
                mags = new_mags
                mjds = new_mjds
                sigmas = new_sigmas
                sigma = np.std(mags)
        
        self.continuumMag = fit_line(mjds, mags, sigmas)
        self.continuumSigma = np.std(mags)
        
        return self.continuumMag, self.continuumSigma
    
    def findClusters(self, num_points_per_cluster=5):
        # Determine which points are outside of continuumMag +/- 2 or 3 continuumSigma, and see if they are clustered
        w = (self.mag > (self.continuumMag - 3*self.continuumSigma))
        
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
                    #print "cluster found!"
                    # return np.array(group)
                    allGroups.append(np.array(group))
                    
                in_group = False
                group = []
        
        if in_group and len(group) >= num_points_per_cluster: 
            #print "cluster found!"
            # return np.array(group)
            allGroups.append(np.array(group))
        
        self.clusterIndices = allGroups
        
        return allGroups
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        try:
            ax.axhline(self.continuumMag, c='c', lw=3.)
            ax.axhline(self.continuumMag+3*self.continuumSigma, ls="--", c='c')
            ax.axhline(self.continuumMag-3*self.continuumSigma, ls="--", c='c')
        except:
            pass
            
        ax.errorbar(self.mjd, self.mag, self.mag_error, c='k', ls='None', marker='.')
        
        try:
            for clusterIdx in self.clusterIndices:
                ax.plot(self.mjd[clusterIdx], self.mag[clusterIdx], 'r+', ms=15, lw=3)
            ax.axvline(self.params[1], c='b', ls='--')
        except:
            pass
        
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlim(min(self.mjd)-10., max(self.mjd)+10.)
        
        return ax
        
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#  Utility Functions
#
def RMagToFlux(R):
    # Returns a flux in Janskys
    return 2875.*10**(R/-2.5)

def FluxToRMag(f):
    # Accepts a flux in Janskys
    return -2.5*np.log10(f/2875.)
    
def u_t(t, u_0, t_0 , t_E):
    return np.sqrt(u_0**2 + ((t - t_0)/t_E)**2)

def A_u(u):
    return (u**2 + 2) / (u*np.sqrt(u**2 + 4))

def fluxModel(t, u0, t0, tE, F0):
    return F0*A_u(u_t(t, u0, t0, tE))

def straight(t, b):
    return np.ones((len(t),), dtype=float)*b

def fit_line(x, y, sigma_y):
    popt, pcov = curve_fit(straight, x, y, sigma=sigma_y)
    return popt[0]

# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#  Plotting Functions
#
def plot_event(lightCurveData, clusterIndices, continuumMag, continuumSigma, eventParameters, fitParameters):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axhline(continuumMag, c='c', lw=3.)
    ax.axhline(continuumMag+3*continuumSigma, ls="--", c='c')
    ax.axhline(continuumMag-3*continuumSigma, ls="--", c='c')
    ax.errorbar(lightCurveData.amjd, lightCurveData.amag, lightCurveData.amag_error, c='k', ls='None', marker='.')
    for clusterIdx in clusterIndices:
        ax.plot(lightCurveData.amjd[clusterIdx], lightCurveData.amag[clusterIdx], 'r+', ms=15, lw=3)
    ax.axvline(eventParameters[1], c='b', ls='--')
    
    modelT = np.arange(min(lightCurveData.amjd), max(lightCurveData.amjd), 0.1)
    ax.plot(modelT, FluxToRMag(fluxModel(modelT, *fitParameters)), 'g-', label='Fit')
    
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlim(min(lightCurveData.amjd)-10., max(lightCurveData.amjd)+10.)
    

# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#  Procedural functions
#
def sample_microlensing_parameters(mjdMin, mjdMax):
    # THIS DRAW IS TOTALLY WRONG
    u0 = np.exp(-10.*np.random.uniform())
    
    # Draw from uniform distribution between min(mjd), max(mjd)
    t0 = np.random.uniform(mjdMin, mjdMax)
    
    # Draw from timescale distribution
    # THIS IS TOTALLY WRONG!!
    tE = np.random.uniform(5., 50.)
    
    # This is now predetermined by the data fed in!
    # F0 = SE.RMagToFlux(contMag)
    
    return (u0, t0, tE)

def lightcurve_chisq(p, lc):
    if (p[0] < 0) or (p[0] > 1):
        return 9999999.0
    ch = np.sum((RMagToFlux(lc.mag) - fluxModel(lc.mjd, *p))**2. / (2.*lc.mag_error**2.))
    #print ch
    return ch

def fit_lightcurve_cluster(lc, clusterIdx):
    initial_u0 = np.exp(-10.*np.random.uniform())
    initial_t0 = np.median(lc.mjd[clusterIdx])
    initial_tE = (max(lc.mjd[clusterIdx]) - min(lc.mjd[clusterIdx])) / 2.
    initial_F0 = RMagToFlux(lc.continuumMag)
    
    p0 = (initial_u0, initial_t0, initial_tE, initial_F0)
    
    full_out = so.fmin_powell(lightcurve_chisq, p0, args=(lc,), full_output=True, disp=0)
    
    return full_out[0], full_out[1]
    
    
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#  Main routine
#
#def work(lightCurve):
def work(dbLightCurve):
    
    ####
    lightCurve = PTFLightCurve(dbLightCurve)
    logging.debug("objid: {0}, Number of points: {1}".format(lightCurve.objid, len(lightCurve.mjd)))
    if len(lightCurve.mjd) < 25: return
    ###
    
    # Fit for continuum level, F0, by sigma clipping away outliers
    continuumMag, continuumSigma = lightCurve.measureContinuum(clipSigma=2.)
    
    # Find clusters of points **brighter** than the continuumMag
    clusterIndices = lightCurve.findClusters(num_points_per_cluster=8)
    
    dbLightCurve.candidate = 1
    session.flush()
    
    if len(clusterIndices) == 0:
        logging.debug("Found no clusters!")
    
    # Try to fit the light curve with a point-lens, point-source event shape
    fractionalErrors = []
    for clusterIdx in clusterIndices:
        fitParameters, goodness = fit_lightcurve_cluster(lightCurve, clusterIdx)
        logging.debug("Fit Parameters:\n\t- u0 = {0}\n\t- t0 = {1}\n\t- tE = {2}".format(*fitParameters))
        
        if (fitParameters[0] > 0.1) or (np.fabs(fitParameters[2]) < 5) or (np.fabs(fitParameters[2]) > 100): continue
        
        logging.info("Cluster found! objid: {0}".format(lightCurve.objid))
        
        lightCurve.params = fitParameters
        #ax = lightCurve.plot()
        #modelT = np.arange(min(lightCurve.mjd), max(lightCurve.mjd), 0.1)
        #ax.plot(modelT, FluxToRMag(fluxModel(modelT, *fitParameters)), 'r-')
        #plt.savefig("candidates/{0}_{1}.pdf".format(lightCurve.objid, clusterIdx[0]))
        
        print "Candidate!"
        dbLightCurve.candidate = 2
        session.flush()
    
    logging.debug("-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~")

def run(lightCurves):
    
    #pool = multiprocessing.Pool()
    #pool.map_async(work, lightCurves)
    
    for lc in lightCurves:
        work(lc)
    
    return
    
if __name__ == "__main__":
    np.random.seed(101)
    
    parser = OptionParser(description="")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_option("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    
    (options, args) = parser.parse_args()
    if options.verbose: logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    elif options.quiet: logging.basicConfig(level=logging.ERROR, format='%(message)s')
    else: logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    lightCurves = []
    for ii in range(55):
        lcs = session.query(LightCurve).filter(LightCurve.candidate == 0).limit(10000).all()
        logging.info("Light curves loaded...")
        s = run(lcs)
        
    """for lc in session.query(LightCurve).order_by(LightCurve.objid).offset(ii*1000).limit(1000).all():
        ptfLightCurve = PTFLightCurve(lc)
        if len(ptfLightCurve.mjd) < 25: continue
        lightCurves.append(ptfLightCurve)
    
    logging.debug("Light curves loaded...")
    
    s = run(lightCurves)
    """
