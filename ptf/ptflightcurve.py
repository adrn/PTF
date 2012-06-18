""" Contains the PTFLightCurve class """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard Library
import sys, os
import copy
import logging

# Third-party dependencies
import numpy as np
import matplotlib.pyplot as plt

# Package dependences
import aov

# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=

class PTFLightCurve:
    """ Represents a PTF Light Curve """
    
    def __init__(self, mjd, mag, error):
        self.amjd = self.mjd = np.array(mjd)
        self.amag = self.mag = np.array(mag)
        self.error = np.array(error)
    
    @classmethod
    def fromDBLightCurve(cls, db_light_curve):
        """ From a sqlalchemy LightCurve object (see model classes), create a 
            PTFLightCurve object
        """
        return cls(db_light_curve.amjd, db_light_curve.amag, db_light_curve.error)
    
    def plot(self, ax=None, **kwargs):
        """ Either plots the light curve and show()'s it to the display, or plots it on 
            the given matplotlib Axes object
        """
        
        if not kwargs.has_key("ls") and not kwargs.has_key("linestyle"):
            kwargs["ls"] = 'none'
        
        if not kwargs.has_key("marker"):
            kwargs["marker"] = 'o'
        
        if not kwargs.has_key("c") and not kwargs.has_key("color"):
            kwargs["c"] = 'k'
        
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.mjd, self.mag, self.error, ecolor='0.7', capsize=0, **kwargs)
            ax.set_xlim(min(self.mjd), max(self.mjd))
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.show()
            return
        
        ax.errorbar(self.mjd, self.mag, self.error, ecolor='0.7', capsize=0, **kwargs)
        ax.set_ylim(ax.get_ylim()[::-1])
        return ax
    
    def lombScargle(self, ws):
        """ Use scipy's lomgscargle function to create a periodogram of the
            light curve 
        """
        try:
            from scipy.signal import lombscargle
        except ImportError:
            raise ImportError("You must have Scipy >0.11dev installed for the Lomb-Scargle algorithm!")
        
        return lombscargle(self.mjd, self.mag, ws)
    
    def aovPeriodogram(self, periods=None, nbins=None):
        """ Create a periodogram using the Analysis of Variance method presented
            in Schwarzenberg-Czerny 1996.
            
            Parameters
            ----------
            periods : numpy.array
                
            nbins : int
                
        """
        if not periods:
            T = 2.*(self.mjd[-1] - self.mjd[0])
            min_period = 0.1 #days
            max_period = 200 #days
            subsample = 0.1
            
            freqstep = subsample / T
            freq = 1. / min_period
            min_freq = 1. / max_period
            
            periods = []
            while freq >= min_freq:
                periods.append(1./freq)
                freq -= freqstep
            
            periods = np.array(periods)
        
        if not nbins:
            nbins = 20
            
        return (periods, aov.aov_periodogram_asczerny(self.mjd, self.mag, len(periods), periods, nbins))
    
    def aovFindPeaks(self, min_period, max_period, subsample=0.1, finetune=0.01, Npeaks=5):
        #raise NotImplementedError("TODO!")

        T = 2.*(self.mjd[-1] - self.mjd[0])
        
        #min_period = 0.1 #days
        #max_period = T/2. #days
        
        freqstep = subsample / T
        freq = 1. / min_period
        min_freq = 1. / max_period
        
        periods = []
        while freq >= min_freq:
            periods.append(1./freq)
            freq -= freqstep
        
        periods = np.array(periods)
        nbins = 20
        
        lc = copy.deepcopy(self)
        return aov.findPeaks_aov(lc.mjd, lc.mag, lc.error, Npeaks, min_period, max_period, subsample, finetune, nbins)
    
    def phase_fold(self, T0=None):
        raise NotImplementedError("aovFindPeaks takes FOREVER")
        
        if T0 == None:
            course_period_data = self.aovFindPeaks(0.1, 200.0, subsample=0.1, finetune=0.01, Npeaks=5)
            for peak_period in course_period_data["peak_period"]:
                fine_period_data = self.aovFindPeaks(peak_period-0.1*peak_period, peak_period+0.1*peak_period, subsample=0.01, finetune=0.001, Npeaks=1)
                print fine_period_data
                
    
    def aovBestPeriod(self):
        raise NotImplementedError("TODO!")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from ptf.db.DatabaseConnection import *
    
    #lc = session.query(LightCurve).filter(LightCurve.objid < 100000).limit(1).one()
    #lc = session.query(LightCurve).filter(LightCurve.objid == 71725).one()
    #lcs = session.query(LightCurve).filter(LightCurve.objid < 100000).limit(10).all()
    #lcs = session.query(LightCurve).filter(LightCurve.objid == 67538).limit(1).all()
    lcs = session.query(LightCurve).filter(LightCurve.objid == 70366).limit(1).all()
    
    ptf_lc = PTFLightCurve.fromDBLightCurve(lcs[0])
    ptf_lc.aovFindPeaks()
    sys.exit(0)
    for lc in lcs:
        ptf_lc = PTFLightCurve.fromDBLightCurve(lc)
        import time
        a = time.time()
        periods, periodogram = ptf_lc.aovPeriodogram()
        print time.time() - a 
        
        plt.clf()
        ax = plt.subplot(211)
        lc.plot(ax)
        
        plt.subplot(212)
        plt.plot(periods, -periodogram, 'k-')
        plt.show()