# Standard library
import os, sys
import argparse
import re
import logging

# Third party
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import scoreatpercentile

# Project
import ptf.simulation.util as simu
#import ptf.db.util as dbu
from ptf.db.DatabaseConnection import *

""" TODO:
    - Take an entire field, add ML events to ~25% of light curves and save indices, make plots
        for parameter projections
"""

class PTFLightCurve:
    
    def __init__(self, mjd, mag, error):
        self.mjd = np.array(mjd)
        self.mag = np.array(mag)
        self.error = np.array(error)
    
    def addMicrolensingEvent(self, u0=None, t0=None, tE=None):
        """ Adds a simulated microlensing event to the light curve
            
            u0 : float, optional
                The impact parameter for the microlensing event. If not specified,
                the value will be drawn from the measured u0 distribution 
                [TODO: REFERENCE]
            t0 : float, optional
                The peak time of the event (shouldn't really be specified)
                This is just drawn from a uniform distribution between mjd_min
                and mjd_max
            tE : float, optional
                The length of the microlensing event. If not specified,
                the value will be drawn from the measured tE distribution 
                [TODO: REFERENCE]        
        """
        
        # If u0 is not specified, draw from u0 distribution
        #   - see for example Popowski & Alcock 
        #   - u0 maximum defined by detection limit of survey, but in our
        #       case assum the amplifcation should be >1.5. Using eq. 1 from
        #       Popowski & Alcock, this corresponds to a maximum u0 of ~0.8
        if u0 == None: self.u0 = np.random.uniform()*0.8
        else: self.u0 = float(u0)
        
        # If t0 is not specified, draw from uniform distribution between days
        if t0 == None: self.t0 = np.random.uniform(min(self.mjd), max(self.mjd))
        else: self.t0 = float(t0)
        
        if (self.t0 > max(self.mjd)) or (self.t0 < min(self.mjd)):
            logging.warn("t0 is outside of the mjd range for this light curve!")
        
        # If tE is not specified, draw from tE distribution
        # [TODO: right now this just added by hand!]
        if tE == None: self.tE = np.random.uniform(5., 50.)
        else: self.tE = float(tE)
        
        flux = simu.fluxModel(self.mjd, u0=self.u0, t0=self.t0, tE=self.tE, F0=1.)#self.F0)
        self.mag = simu.FluxToRMag(flux*simu.RMagToFlux(self.mag))
    
    def addNoise(self):
        """ Add scatter to the light curve """
        self.mag += np.random.normal(0.0, self.error)
    
    def plot(self, ax=None):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.mjd, self.mag, self.error, ls='none', marker='o', c='k', ecolor='0.7', capsize=0)
            ax.set_ylim(ax.get_ylim()[::-1])
            ax.set_xlim(min(self.mjd), max(self.mjd))
            plt.show()
        
        ax.errorbar(self.mjd, self.mag, self.error, ls='none', marker='o', c='k', ecolor='0.7', capsize=0)
        ax.set_ylim(ax.get_ylim()[::-1])

class SimulatedLightCurve(PTFLightCurve):
    
    def __init__(self, mjd=None, cadence=None, sigma=0.05, F0=None, add_event=False, \
                       randomize_cadence=False, outliers=False,  number_of_observations=None, clumpy=False):
        """ Creates a simulated PTF light curve
        
            Parameters
            ----------
            mjd : numpy.array, optional
                An array of mjd values. If none, creates one internally.
            cadence : float, optional
                The separation between observations
            sigma : float, optional
                The 1-sigma scatter in the light curve in units of R-band 
                magnitudes           
            F0 : float, optional
                The flux of the source.
            add_event : bool, optional
                If true, will add a microlensing event to the light curve
            outliers : bool, optional
                This controls whether to sample from an outlier distribution
                when creating magnitude values for the light curve
                [TODO: implement this!]
            randomize_cadence : bool, optional
                This controls whether to have a uniform cadence or a randomized
                cadence. [TODO: Add ability to control clumpyness, to make more
                similar to PTF light curves]
            number_of_observations : int, optional
                If randomize_cadence is True, specify how many times to 'observe'
                the object since cadence doesn't really matter.
            clumpy : bool, optional
                Make randomized observation clumpy.
            
            Notes
            -----
            
        """
        
        if cadence == None and randomize_cadence == False and mjd == None:
            raise ValueError("You must set either the cadence, or randomize_cadence and number_of_observations")
        
        if cadence != None:
            self.cadence = float(cadence)
        
        self.sigma = sigma
        
        if F0 == None: self.F0 = np.random.uniform(0.1, 1.) / 100.
        else: self.F0 = float(F0)
        
        if mjd == None:
            mjd_min = 0.
            mjd_max = 365.
            if randomize_cadence:
                if number_of_observations == None: raise ValueError("If randomize_cadence==True, you must set 'number_of_observations'!")
                if clumpy:
                    self.mjd = np.sort(np.random.uniform(mjd_min, mjd_max-(mjd_max-mjd_min)/np.random.uniform(1.1,2.5), size=number_of_observations))
                else:
                    self.mjd = np.sort(np.random.uniform(mjd_min, mjd_max, size=number_of_observations))
            else:
                self.mjd = np.arange(mjd_min, mjd_max, self.cadence)
        else:
            self.mjd = np.array(mjd)
        
        self.mag = np.ones(len(self.mjd), dtype=float)*simu.FluxToRMag(self.F0) #np.random.normal(0.0, sigma, size=len(self.mjd))
        self.error = np.array([self.sigma]*len(self.mjd))
        
        self.mjd_min = min(self.mjd)
        self.mjd_max = max(self.mjd)
        
        if add_event:
            self.addMicrolensingEvent()
            self.addNoise()
        else:
            self.addNoise()
        
def computeVariabilityIndices(lightCurve):
    """ Computes the 6 (5) variability indices as explained in M.-S. Shin et al. 2009
        
        Parameters
        ----------
        lightCurve : SimulatedLightCurve
            a SimulatedLightCurve object to compute the indices from
    """
    N = len(lightCurve.mjd)
    contMag, contSig = simu.estimateContinuum(lightCurve.mjd, lightCurve.mag, lightCurve.error)
    
    # ===========================
    # Compute variability indices
    # ===========================
    
    # sigma/mu : root-variance / mean
    mu = contMag #np.mean(lightCurve.mag)
    sigma = np.sqrt(np.sum(lightCurve.mag - mu)**2 / (N-1.))
    sigma_to_mu = sigma / mu

    # Con : number of consecutive series of 3 points BRIGHTER than the light curve
    num_sigma = 2.
    clusters = simu.findClusters(lightCurve.mag, contMag, contSig, 3, num_sigma=num_sigma)
    Con = len(clusters) / (N - 2.)
    
    # eta : ratio of mean square successive difference to the sample variance
    delta_squared = np.sum((lightCurve.mag[1:] - lightCurve.mag[:-1])**2 / (N - 1.))
    variance = sigma**2
    eta = delta_squared / variance
    
    delta_n = np.sqrt(float(N)/(N-1.)) * (lightCurve.mag[:-1] - mu) / lightCurve.error[:-1] 
    delta_n_plus_1 = np.sqrt(float(N)/(N-1.)) * (lightCurve.mag[1:] - mu) / lightCurve.error[1:]
    # J : eqn. 3 in M.-S. Shin et al. 2009
    J = np.sum(np.sign(delta_n*delta_n_plus_1)*np.sqrt(np.fabs(delta_n*delta_n_plus_1)))
    
    # K : eqn. 3 in M.-S. Shin et al. 2009
    delta_n = np.sqrt(float(N)/(N-1.)) * (lightCurve.mag - mu) / lightCurve.error
    K = np.sum(np.fabs(delta_n)) / (float(N)*np.sqrt((1./N)*np.sum(delta_n**2)))
    
    return sigma_to_mu, Con, eta, J, K

def plot_five_by_five(varIndicesNoEvent, varIndicesWithEvent, plot_prefix="plots"):
    """ Generate a plot of each variabilty index vs. each other on a 5x5 grid """
    
    if not os.path.exists(plot_prefix): os.mkdir(plot_prefix)
    
    kk = 0
    params = ["sigma_to_mu", "Con", "eta", "J", "K"]
    for ii in range(len(params)):
        if params[ii] == "J":
            bins = np.arange(0, 2500, 100)
        else:
            bins = 50
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(varIndicesNoEvent[params[ii]], color='b', alpha=0.4, bins=bins, normed=True)
        ax.hist(varIndicesWithEvent[params[ii]], color='r', alpha=0.4, bins=bins, normed=True)
        fig.savefig("plots/{0}_hist.png".format(params[ii]))
        
        # Plot-fu to make pretty figures
        for jj in range(len(params)):
            if ii == jj: continue
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            #ax1.plot(varIndicesNoEvent[params[ii]], varIndicesNoEvent[params[jj]], 'b.', alpha=0.4)
            ax1.loglog(varIndicesNoEvent[params[ii]], varIndicesNoEvent[params[jj]], 'b.', alpha=0.2)
            
            ax2 = fig.add_subplot(122)
            #ax2.plot(varIndicesWithEvent[params[ii]], varIndicesWithEvent[params[jj]], 'r.', alpha=0.4)
            ax2.loglog(varIndicesWithEvent[params[ii]], varIndicesWithEvent[params[jj]], 'r.', alpha=0.2)
            
            xmin1,xmax1 = ax1.get_xlim()
            xmin2,xmax2 = ax2.get_xlim()
            ax1.set_xlim(min(xmin1, xmin2), max(xmax1, xmax2))
            ax2.set_xlim(min(xmin1, xmin2), max(xmax1, xmax2))
            
            ymin1,ymax1 = ax1.get_ylim()
            ymin2,ymax2 = ax2.get_ylim()
            ax1.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
            ax2.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
            
            ax1.set_xlabel(params[ii])
            ax2.set_xlabel(params[ii])
            ax1.set_ylabel(params[jj])
            fig.savefig(os.path.join(plot_prefix, "{0}_vs_{1}.png".format(params[ii], params[jj])))

# ===========================================================================================

def simulation1(number_of_light_curves, number_per_light_curve, number_of_points_range=(50,60)):
    """ Default values should be something like
            number_of_light_curves = 10
            number_per_light_curve = 1000
    """
    
    # For each light curve, compute the variability indices WITHOUT adding a microlensing event
    variabilityIndices = []
    for jj in range(number_of_light_curves):
        mjd = None
        for ii in range(number_per_light_curve):
            if mjd == None:
                lightCurve = SimulatedLightCurve(randomize_cadence=True, number_of_observations=100)
                mjd = lightCurve.mjd
            else:
                #lightCurve = SimulatedLightCurve(mjd=mjd)
                # [TODO: possible hack..]
                lightCurve = SimulatedLightCurve(mjd=mjd)
            
            variabilityIndices.append(computeVariabilityIndices(lightCurve))
    
    varIndicesNoEvent = np.array(variabilityIndices, dtype=[("sigma_to_mu",float), ("Con",float), ("eta",float), ("J",float), ("K",float)]).view(np.recarray)

def simulation2(number_of_light_curves, number_per_light_curve, number_of_points_range=(50,60)):
    """ Default values should be something like
            number_of_light_curves = 10
            number_per_light_curve = 1000
    """
    
    offsetNumber = 0
    q = 100
    light_curves = []
    
    logging.debug("Selecting light curves from database...")
    
    # Select light curves with enough data points (as specified with number_of_points_range)
    while len(light_curves) < number_of_light_curves:
        pre_light_curves = session.query(LightCurve).group_by(LightCurve.pk).\
                                                     offset(offsetNumber).\
                                                     limit(number_of_light_curves*q).all()
        offsetNumber += q
        
        # For each of the "pre-selected" light curves, count how many data points it has
        for lc in pre_light_curves:
            if number_of_points_range[0] < len(lc.goodMJD) < number_of_points_range[1]:
                light_curves.append(lc)
            
            if len(light_curves) >= number_of_light_curves: break
    
    if len(light_curves) < number_of_light_curves: 
        logging.warn("Only able to select {0} light curves with more than {1} but less than {2} observations.".format(len(light_curves), *number_of_points_range))
        yesOrNo = raw_input("Is that ok? [y]/n:")
        if yesOrNo == "y":
            number_of_light_curves = len(light_curves)
        else:
            sys.exit(0)
    else: 
        logging.info("Selected {0} light curves with more than {1} but less than {2} observations.".format(len(light_curves), *number_of_points_range))
    
    # For each light curve, compute the variability indices WITHOUT adding a microlensing event
    variabilityIndices = []
    for jj in range(number_of_light_curves):
        lightCurve = PTFLightCurve(light_curves[jj].mjd, light_curves[jj].mag, light_curves[jj].error)
        try:
            variabilityIndices.append(computeVariabilityIndices(lightCurve))
        except TypeError:
            logging.debug("Continuum fit failed!")
            continue
            
    varIndicesNoEvent = np.array(variabilityIndices, dtype=[("sigma_to_mu",float), ("Con",float), ("eta",float), ("J",float), ("K",float)]).view(np.recarray)
    logging.debug("Done with light curves with no events...")
    
    # For each light curve, compute the variability indices AFTER adding a microlensing event
    variabilityIndices = []
    for jj in range(number_of_light_curves):
        logging.debug("Computing indices for light curve {0}".format(light_curves[jj].objid))
        for ii in range(number_per_light_curve):
            lightCurve = PTFLightCurve(light_curves[jj].mjd, light_curves[jj].mag, light_curves[jj].error)
            lightCurve.addMicrolensingEvent()
            try:
                variabilityIndices.append(computeVariabilityIndices(lightCurve))
            except TypeError:
                logging.debug("Continuum fit failed!")
                continue
    
    varIndicesWithEvent = np.array(variabilityIndices, dtype=[("sigma_to_mu",float), ("Con",float), ("eta",float), ("J",float), ("K",float)]).view(np.recarray)
    logging.debug("Done with light curves with events...")
    
    logging.debug("Plotting...")
    plot_five_by_five(varIndicesNoEvent, varIndicesWithEvent, plot_prefix="plots/{0}-{1}".format(*number_of_points_range))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                        help="Be chatty!")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                        help="Be quiet!")
    parser.add_argument("-r", "--range", type=str, dest="range", default="(45,55)",
        				help="Accepted range of number of observations")
    parser.add_argument("-l", "--number-of-light-curves", type=int, dest="num_light_curves", default=1000,
        				help="Number of light curves to select from the databse")
    parser.add_argument("-s", "--number-of-simulations", type=int, dest="num_simulations", default=100,
        				help="Number of simulations per light curve")
    
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.WARN)
    else:
        logging.basicConfig(level=logging.INFO)
    
    pattr = re.compile("^[\(|\s]{0,1}([0-9]+)[\,|\-|\s]([0-9]+)[\)|\s]{0,1}")
    try:
        num_observations_range = map(int, pattr.match(args.range.strip()).groups())
    except ValueError:
        raise ValueError("Invalid --range input, must be of the form 10-20 or (10,20)")
            
    simulation2(args.num_light_curves, args.num_simulations, num_observations_range)