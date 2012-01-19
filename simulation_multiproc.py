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
import sys
from optparse import OptionParser
import logging
import multiprocessing

import numpy as np
from scipy.optimize import curve_fit, anneal
import sqlalchemy
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, deferred
from sqlalchemy.schema import Column
from sqlalchemy.types import Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql.expression import func
import esutil as eu
import matplotlib.pyplot as plt

# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#  Database Connection and caching
#
# Postgresql Database Connection
database_connection_string = 'postgresql://%s:%s@%s:%s/%s' \
    % ('adrian','','localhost','5432','ptf_microlensing')

engine = create_engine(database_connection_string, echo=False)
metadata = MetaData()
metadata.bind = engine
Base = declarative_base(bind=engine)
session = scoped_session(sessionmaker(bind=engine, autocommit=True, autoflush=False))

# Model Class for light curves
class LightCurve(Base):
    __tablename__ = 'light_curve'
    __table_args__ = {'autoload' : True}
    
    mjd = deferred(Column(ARRAY(Float)))
    mag = deferred(Column(ARRAY(Float)))
    mag_error = deferred(Column(ARRAY(Float)))
    
    def __repr__(self):
        return self.__class__.__name__
    
    @property
    def np_mjd(self):
        return np.array(self.mjd)
    
    @property
    def np_mag(self):
        return np.array(self.mag)
    
    @property
    def np_mag_error(self):
        return np.array(self.mag_error)

class LC:
    def __init__(self, lc):
        self.mjd = np.array(lc.mjd)
        self.mag = np.array(lc.mag)
        self.mag_error = np.array(lc.mag)

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
    ax.errorbar(lightCurveData.mjd, lightCurveData.mag, lightCurveData.mag_error, c='k', ls='None', marker='.')
    for clusterIdx in clusterIndices:
        ax.plot(lightCurveData.mjd[clusterIdx], lightCurveData.mag[clusterIdx], 'r+', ms=15, lw=3)
    ax.axvline(eventParameters[1], c='b', ls='--')
    
    modelT = np.arange(min(lightCurveData.mjd), max(lightCurveData.mjd), 0.1)
    ax.plot(modelT, FluxToRMag(fluxModel(modelT, *fitParameters)), 'g-', label='Fit')
    
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.set_xlim(min(lightCurveData.mjd)-10., max(lightCurveData.mjd)+10.)
    

# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#  Procedural functions
#
def get_light_curve(objid=None):
    if objid == None:
        objid = objids[np.random.randint(len(objids))]
        logging.debug("Current objid: {0}".format(objid))
        
    try:
        #lc = session.query(LightCurve).filter(LightCurve.objid == objid).one()
        LIGHTCURVES
        #except sqlalchemy.orm.exc.NoResultFound:
    except IndexError:
        raise ValueError("objid not found in database!")
    
    return lc

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

def add_microlensing_event(lcData, params):
    flux = RMagToFlux(lcData.mag) * A_u(u_t(lcData.mjd, *params))
    lcData.mag = FluxToRMag(flux)
    
def get_continuum(lcData):
    sigma = np.std(lcData.mag)
    b = np.median(lcData.mag)
    
    # This is a stupid sigma-clipping way to find the continuum magnitude 
    CLIPSIG = 2
    
    mags = lcData.mag
    mjds = lcData.mjd
    sigmas = lcData.mag_error
    
    while True:
        w = np.fabs(mags - np.median(mags)) < CLIPSIG*sigma
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
    
    continuumMag = fit_line(mjds, mags, sigmas)
    continuumSigma = np.std(mags)
    
    return continuumMag, continuumSigma

def find_clusters(lcData, continuumMag, continuumSigma, num_points_per_cluster=4):
    # Determine which points are outside of continuumMag +/- 2 or 3 continuumSigma, and see if they are clustered
    w = (lcData.mag > (continuumMag - 3*continuumSigma)) #& (lcData.mag < (continuumMag + 3*continuumSigma))
    
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
    
    return allGroups

def fit_lightcurve(lcData, p0):
    popt, pcov = curve_fit(fluxModel, lcData.mjd, RMagToFlux(lcData.mag), p0=p0, sigma=lcData.mag_error, maxfev=10000)
    goodness = np.sum((RMagToFlux(lcData.mag) - fluxModel(lcData.mjd, *popt)) / lcData.mag_error)**2
    
    return popt, goodness
    
# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
#  Main routine
# 

def work(lightCurveData):
    # Read in light curve data from PTF
    #lightCurveData = get_light_curve()

    # Sample microlensing event parameters
    eventParameters = sample_microlensing_parameters(min(lightCurveData.mjd), max(lightCurveData.mjd))
    logging.info("Real Parameters:\n\t- u0 = {0}\n\t- t0 = {1}\n\t- tE = {2}".format(*eventParameters))
    
    # Add a microlensing event to the light curve
    add_microlensing_event(lightCurveData, eventParameters)
    
    # Fit for continuum level, F0, by sigma clipping away outliers
    continuumMag, continuumSigma = get_continuum(lightCurveData)
    
    # Find clusters of points **brighter** than the continuumMag
    clusterIndices = find_clusters(lightCurveData, continuumMag, continuumSigma)
    if len(clusterIndices) == 0:
        logging.debug("Found no clusters!")
        return
    
    # Try to fit the light curve with a point-lens, point-source event shape
    for clusterIdx in clusterIndices:
        old_goodness = 1E10
        old_popt = None
        #for jj in range(10):
        for jj in range(1):
            initial_u0 = np.exp(-10.*np.random.uniform())
            initial_t0 = np.median(lightCurveData.mjd[clusterIdx])
            initial_tE = (max(lightCurveData.mjd[clusterIdx]) - min(lightCurveData.mjd[clusterIdx])) / 2.
            initial_F0 = RMagToFlux(continuumMag)
            
            p0 = (initial_u0,\
                  initial_t0, \
                  initial_tE, \
                  initial_F0)
                  
            logging.debug("-> Trying: u0 = {0}, t0 = {1}, tE = {2}, F0 = {2} (M0 = {3})".format(p0[0], p0[1], p0[2], p0[3], FluxToRMag(p0[3])))
            
            try:
                popt, goodness = fit_lightcurve(lightCurveData, p0)
                if goodness < old_goodness:
                    old_goodness = goodness
                    old_popt = popt
                else:
                    continue
                    
            except RuntimeError:
                logging.debug("maxfev reached!")
                continue
        
        if old_popt == None:
            logging.debug("Fit failed!")
            continue
    
    if old_popt == None:
        return
    
    fitParameters = old_popt
    
    logging.info("Fit Parameters:\n\t- u0 = {0}\n\t- t0 = {1}\n\t- tE = {2}".format(*fitParameters))
    logging.info("-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~")
    
    global counter
    counter.value += 1
    
    return True

def run(lightCurves, number_of_simulations=1000):
    logging.debug("Number of simulations to run: {0}".format(number_of_simulations))
    
    randomLightCurves = []
    for ii in range(number_of_simulations):
        randomLightCurves.append(lightCurves[np.random.randint(len(lightCurves))])
    
    # Multiprocessing Method!
    counter = None
    chisq = None

    def init(cnt):
        ''' store the counter for later use '''
        global counter
        counter = cnt
    
    counter = multiprocessing.Value('i', 0)
    pool = multiprocessing.Pool(initializer = init, initargs = (counter, ))
    p = pool.map_async(work, randomLightCurves)
    p.wait()
    
    print "Value:", counter.value
    print "Fraction:", counter.value/float(number_of_simulations)
    
if __name__ == "__main__":
    np.random.seed(10)
        
    parser = OptionParser(description="")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_option("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    
    (options, args) = parser.parse_args()
    if options.verbose: logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    elif options.quiet: logging.basicConfig(level=logging.WARN, format='%(message)s')
    else: logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # Load all light curves into memory
    lightCurves = session.query(LightCurve).filter(func.array_upper(LightCurve.mjd, 1) >= 100).all()
    lightCurves = [LC(x) for x in lightCurves]
    logging.warn("Light curves converted.")
    
    run(lightCurves, 10000)

    