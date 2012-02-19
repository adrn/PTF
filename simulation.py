# Standard library
import os, sys
from argparse import ArgumentParser
import logging

# Third party
import matplotlib.pyplot as plt
import numpy as np

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
        # [TODO: right now this just added by hand!]
        if u0 == None: self.u0 = np.exp(-10.0*np.random.uniform())
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
    contMag, contSig = simu.estimateContinuum(lightCurve.mjd, lightCurve.mag, lightCurve.error)\
    
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

def plot_five_by_five(varIndices, varIndicesNoEvent):
    kk = 0
    fig = plt.figure()
    params = ["sigma_to_mu", "Con", "eta", "J", "K"]
    for ii in range(5):
        for jj in range(5):
            ax = fig.add_subplot(5,5,jj+5*ii+1)
            if ii == jj:
                ax.hist(varIndicesNoEvent[params[ii]], color='b', alpha=0.4)
                ax.hist(varIndices[params[ii]], color='r', alpha=0.4)
            else:
                ax.plot(varIndicesNoEvent[params[ii]], varIndicesNoEvent[params[jj]], 'b.', alpha=0.2)
                ax.plot(varIndices[params[ii]], varIndices[params[jj]], 'r.', alpha=0.2)
            
            if ii == 4:
                ax.set_xlabel(params[kk])
                kk += 1
            
            if jj in [0, 5, 10, 15, 20]:
                ax.set_ylabel(params[ii])
    
    plt.show()

# ===========================================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    
    number_of_light_curves = 10
    number_per_light_curve = 1000
    plot = False
    
    light_curves = session.query(LightCurve).limit(number_of_light_curves).all()
    
    sigma_to_mus = []
    Cons = []
    etas = []
    Js = []
    Ks = []
    
    for jj in range(number_of_light_curves):
        mjd = None
        for ii in range(number_per_light_curve):
            """if mjd == None:
                lightCurve = SimulatedLightCurve(randomize_cadence=True, number_of_observations=100)
                mjd = lightCurve.mjd
            else:
                #lightCurve = SimulatedLightCurve(mjd=mjd)
                # [TODO: possible hack..]
                lightCurve = SimulatedLightCurve(mjd=mjd)
            """
            lightCurve = PTFLightCurve(light_curves[jj].mjd, light_curves[jj].mag, light_curves[jj].error)
            lightCurve.addMicrolensingEvent()
            
            sigma_to_mu, Con, eta, J, K = computeVariabilityIndices(lightCurve)
            sigma_to_mus.append(sigma_to_mu)
            Cons.append(Con)
            etas.append(eta)
            Js.append(J)
            Ks.append(K)
    
    varIndices = np.array(zip(sigma_to_mus, Cons, etas, Js, Ks), dtype=[("sigma_to_mu",float), ("Con",float), ("eta",float), ("J",float), ("K",float)]).view(np.recarray)
    
    sigma_to_mus = []
    Cons = []
    etas = []
    Js = []
    Ks = []
    for jj in range(number_of_light_curves):
        mjd = None
        for ii in range(number_per_light_curve):
            """if mjd == None:
                lightCurve = SimulatedLightCurve(randomize_cadence=True, number_of_observations=100)
                mjd = lightCurve.mjd
            else:
                #lightCurve = SimulatedLightCurve(mjd=mjd)
                # [TODO: possible hack..]
                lightCurve = SimulatedLightCurve(mjd=mjd)
            """
            lightCurve = PTFLightCurve(light_curves[jj].mjd, light_curves[jj].mag, light_curves[jj].error)
            
            sigma_to_mu, Con, eta, J, K = computeVariabilityIndices(lightCurve)
            sigma_to_mus.append(sigma_to_mu)
            Cons.append(Con)
            etas.append(eta)
            Js.append(J)
            Ks.append(K)
    
    varIndicesNoEvent = np.array(zip(sigma_to_mus, Cons, etas, Js, Ks), dtype=[("sigma_to_mu",float), ("Con",float), ("eta",float), ("J",float), ("K",float)]).view(np.recarray)
    
    kk = 0
    params = ["sigma_to_mu", "Con", "eta", "J", "K"]
    for ii in range(5):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(varIndicesNoEvent[params[ii]], color='b', alpha=0.4)
        ax.hist(varIndices[params[ii]], color='r', alpha=0.4)
        fig.savefig("plots/{0}_hist.png".format(params[ii]))
        
        for jj in range(5):
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax1.plot(varIndicesNoEvent[params[ii]], varIndicesNoEvent[params[jj]], 'b.', alpha=0.4)
            
            ax2 = fig.add_subplot(122)
            ax2.plot(varIndices[params[ii]], varIndices[params[jj]], 'r.', alpha=0.4)
            
            xmin1,xmax1 = ax1.get_xlim()
            xmin2,xmax2 = ax2.get_xlim()
            newXMin = min([xmin1,xmin2])
            newXMax = max([xmax1,xmax2])
            ax1.set_xlim(newXMin-(newXMax-newXMin)/10., newXMax+(newXMax-newXMin)/10.)
            ax2.set_xlim(newXMin-(newXMax-newXMin)/10., newXMax+(newXMax-newXMin)/10.)
            
            ymin1,ymax1 = ax1.get_ylim()
            ymin2,ymax2 = ax2.get_ylim()
            newYMin = min([ymin1,ymin2])
            newYMax = max([ymax1,ymax2])
            ax1.set_ylim(newYMin-(newYMax-newYMin)/10., newYMax+(newYMax-newYMin)/10.)
            ax2.set_ylim(newYMin-(newYMax-newYMin)/10., newYMax+(newYMax-newYMin)/10.)
            
            ax1.set_xlabel(params[jj])
            ax2.set_xlabel(params[jj])
            ax1.set_ylabel(params[ii])
            fig.savefig("plots/{0}_vs_{1}.png".format(params[ii], params[jj]))


#ax.axhline(contMag, c='r', alpha=0.5)
#ax.axhline(contMag+num_sigma*contSig, c='r', ls="--", alpha=0.5)
#ax.axhline(contMag-num_sigma*contSig, c='r', ls="--", alpha=0.5)
#for cluster in clusters:
#    ax.plot(lightCurve.mjd[cluster], lightCurve.mag[cluster], 'go', ls='none')
