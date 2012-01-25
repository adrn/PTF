# -*- coding: utf-8 -*-

"""
    - Read in baseline PTF light curve [from a pickle?]
    - Sample parameters for microlensing event model
        -> Use u0 and tE distributions based on RA, Dec
    - Add microlensing event to the light curve
    - Get continuum (F0)
        -> Sigma clip, then fit a straight line
    - Find clusters of points outside ~3σ (?)
        -> Needs to be able to find multiple clusters
        -> Use median of cluster as t0
    - Fit event to light curve
        -> Possible extract region around cluster, and
            only fit to that?
        -> This could be parallel (e.g. try fitting with 
            a bunch of different starting positions and
            use the best fit), or maybe we should just 
            use MCMC, or maybe we could use simulated
            annealing?
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

import sys
from optparse import OptionParser
import logging
import multiprocessing

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
        self.mjd = np.array(dbLightCurve.mjd)
        self.original_mag = np.array(dbLightCurve.mag)
        self.mag_error = np.array(dbLightCurve.mag_error)
        self.objid = str(dbLightCurve.objid)

        # Get rid of messy data points
        MAXERR = 0.2
        self.mjd = self.mjd[self.mag_error < MAXERR]
        self.original_mag = self.original_mag[self.mag_error < MAXERR]
        self.mag = self.original_mag
        self.mag_error = self.mag_error[self.mag_error < MAXERR]
        
        if ((len(dbLightCurve.mjd) - len(self.mjd)) / len(dbLightCurve.mjd)) > 0.05:
            raise ValueError("Too many bad data points!")
    
    def addMicrolensingEvent(self, params):
        self.params = params
        newFlux = RMagToFlux(self.mag) * A_u(u_t(self.mjd, *params))
        self.mag = FluxToRMag(newFlux)
        
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
    
    def findClusters(self, num_points_per_cluster=4):
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
        ax.axhline(self.continuumMag, c='c', lw=3.)
        ax.axhline(self.continuumMag+3*self.continuumSigma, ls="--", c='c')
        ax.axhline(self.continuumMag-3*self.continuumSigma, ls="--", c='c')
        ax.errorbar(self.mjd, self.mag, self.mag_error, c='k', ls='None', marker='.')
        for clusterIdx in self.clusterIndices:
            ax.plot(self.mjd[clusterIdx], self.mag[clusterIdx], 'r+', ms=15, lw=3)
        ax.axvline(self.params[1], c='b', ls='--')
        
        #modelT = np.arange(min(self.mjd), max(self.mjd), 0.1)
        #ax.plot(modelT, FluxToRMag(fluxModel(modelT, *fitParameters)), 'g-', label='Fit')
        
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
    
    full_out = so.fmin_powell(lightcurve_chisq, p0, args=(lc,), full_output=True)
    
    return full_out[0], full_out[1]
    
    """
    popt, pcov = curve_fit(fluxModel, lc.mjd, RMagToFlux(lc.mag), p0=p0, sigma=lc.mag_error, maxfev=10000)
    goodness = np.sum((RMagToFlux(lc.mag) - fluxModel(lc.mjd, *popt)) / lc.mag_error)**2
    """
    Nwalkers = 10
    Ndim = 4
    Nsteps = 100
    
    print "initial", np.exp(-10.*np.random.uniform()), initial_t0, initial_tE, FluxToRMag(initial_F0)
    initial_params = [[np.exp(-10.*np.random.uniform()), initial_t0, initial_tE, initial_F0] for ii in xrange(Nwalkers)] #?? DAMNIT, step sizes suck..
    sampler = pyest.EnsembleSampler(Nwalkers, Ndim, lightcurve_chisq, postargs=[lc], threads=2, a=4.)
    
    pos,prob,state = sampler.run_mcmc(initial_params, None, Nsteps/10)
    sampler.clear_chain()
    sampler.run_mcmc(pos, state, Nsteps)
    
    for jj in xrange(Nwalkers):
        ii = np.argmax(sampler.lnprobability[jj])
        u0 = sampler.chain[jj][0] #[chain][param][link]
        t0 = sampler.chain[jj][1]
        tE = sampler.chain[jj][2]
        F0 = sampler.chain[jj][3]
        
        #print "Derped:", u0[ii], t0[ii], tE[ii], FluxToRMag(F0[ii])
        
    plt.clf()
    plt.subplot(221)
    plt.plot(u0)
    plt.subplot(222)
    plt.plot(t0)
    plt.subplot(223)
    plt.plot(tE)
    plt.subplot(224)
    plt.plot(F0)
    plt.show()
    return (u0[ii], t0[ii], tE[ii], F0[ii]), 1.0
    
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#  Main routine
#
def work(lightCurve):
    global counter, numEvents, falsePositives
    
    logging.info("objid: {0}".format(lightCurve.objid))
    
    # Sample microlensing event parameters
    eventParameters = sample_microlensing_parameters(min(lightCurve.mjd), max(lightCurve.mjd))
    logging.debug("Real Parameters:\n\t- u0 = {0}\n\t- t0 = {1}\n\t- tE = {2}".format(*eventParameters))
    
    # Add a microlensing event to the light curve 60% of the time
    addEvent = np.random.random() >= 0.4
    if addEvent:
        lightCurve.addMicrolensingEvent(eventParameters)
        numEvents.value += 1
    
    # Fit for continuum level, F0, by sigma clipping away outliers
    continuumMag, continuumSigma = lightCurve.measureContinuum(clipSigma=2.)
    
    # Find clusters of points **brighter** than the continuumMag
    clusterIndices = lightCurve.findClusters(num_points_per_cluster=4)
    
    #ax = lightCurve.plot()
    #plt.show()
    
    if len(clusterIndices) == 0:
        logging.debug("Found no clusters!")
        return
    
    # Try to fit the light curve with a point-lens, point-source event shape
    for clusterIdx in clusterIndices:
        fitParameters, goodness = fit_lightcurve_cluster(lightCurve, clusterIdx)
        logging.info("Fit Parameters:\n\t- u0 = {0}\n\t- t0 = {1}\n\t- tE = {2}".format(*fitParameters))
        
        ax = lightCurve.plot()
        modelT = np.arange(min(lightCurve.mjd), max(lightCurve.mjd), 0.1)
        ax.plot(modelT, FluxToRMag(fluxModel(modelT, *fitParameters)), 'r-')
        plt.show()
        
    logging.info("-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~")    
    #successes += 1
    
    if not addEvent:
        falsePositives.value += 1
    else:
        counter.value += 1

def run(number_of_simulations=1000):
    logging.debug("Number of simulations to run: {0}".format(number_of_simulations))
    
    lightCurves = []
    for lc in session.query(LightCurve).filter(func.array_upper(LightCurve.mjd, 1) >= 150).all():
        try:
            lightCurves.append(PTFLightCurve(lc))
        except ValueError:
            logging.debug("Light curve rejected due to bad points")
            logging.debug(lc.mag_error)
            
    logging.debug("Light curves loaded...")
    
    lightCurveQueue = []
    for idx in np.random.randint(len(lightCurves), size=number_of_simulations):
        lightCurveQueue.append(lightCurves[idx])
    
    """
    # Non-multiprocessing way to do it
    for ptfLightCurve in lightCurveQueue:
        work(ptfLightCurve)
    """
        
    # Multiprocessing way to do it!
    # Multiprocessing Method!
    counter = None
    chisq = None

    def init(cnt, numEv, fPos):
        ''' store the counter for later use '''
        global counter, numEvents, falsePositives
        counter = cnt
        numEvents = numEv
        falsePositives = fPos
    
    counter = multiprocessing.Value('i', 0)
    numEvents = multiprocessing.Value('j', 0)
    falsePositives = multiprocessing.Value('k', 0)
    pool = multiprocessing.Pool(initializer = init, initargs = (counter, numEvents, falsePositives))
    p = pool.map_async(work, lightCurveQueue)
    p.wait()
    
    print "Fraction:", counter.value/float(numEvents.value)
    print "False positives:", falsePositives.value
    
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
    elif options.quiet: logging.basicConfig(level=logging.WARN, format='%(message)s')
    else: logging.basicConfig(level=logging.INFO, format='%(message)s')
        
    s = run(10)
