""" 
    Functions to help compute a detection efficiency
"""
from __future__ import division

# Standard library
import logging
import os
import sys

# Third-party
import apwlib.geometry as g
import apwlib.convert as c
import esutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import scoreatpercentile

# Project
from ptf.db.DatabaseConnection import *
import ptf.simulation.util as simu









# FUCK THE CODE BELOW HERE!

def plot_timescales(times):
    """ Given a list of event timescales, plot a log histogram """
    
    bins = np.logspace(-1, 3, 100)
    plt.hist(times, bins=bins)
    plt.xscale("log")
    plt.show()

def timescale_generator(file="data/randomTimeScales.npy"):
    """ Return an esutil.random.Generator object for the timescale
        distribution
    """
    
    timescales = np.load(file)
    #plot_timescales(timescales)
    bins = np.logspace(np.log10(min(timescales)), np.log10(max(timescales)), 100)
    
    probs, bin_edges = np.histogram(timescales, bins=bins, normed=True, density=True)
    
    return esutil.random.Generator(probs, (bin_edges[1:]+bin_edges[:-1])/2)

def magnitude_generator():
    """ Based on the mean magnitudes from Praesepe, determine the magnitude of the
        simulated light curve, which then effects the scatter of the points
    """
    # Select out all mean magnitude's (mu) from Praesepe (objid < 100000)
    mags = [x[0] for x in session.query(VariabilityIndices.mu).join(LightCurve).filter(LightCurve.objid < 100000).all()]
    probs, bin_edges = np.histogram(mags, bins=50, normed=True, density=True)
    
    return esutil.random.Generator(probs, (bin_edges[1:]+bin_edges[:-1])/2)

def magnitude_to_errors(mag):
    """ Using Nick Law's RMS Error per mean magnitude plot, given a magnitude,
        spit out a scatter
    """
    #0.006 +/- 0.004-> 0.3 +/- 0.2
    
    if mag <= 14: return 0.006
    elif mag >= 20: return 0.3
    else:
        a,b,c = [0.0091250470823083189, -0.27435645731284952, 2.0686597969927254]
        return a*mag**2 + b*mag + c
    
def test_convolve():
    mjds = np.array([56046.42378, 56046.45529, 56046.49608, 56047.4067, 56047.40863, 56047.43839, 56047.46486, 56048.40791, 56048.43994, 56048.47319, 56049.39991, 56049.42953, 56049.46463, 56050.40626, 56050.44962, 56050.48109, 56060.38538, 56060.41486, 56060.44488, 56061.40934, 56061.44043, 56061.47189, 56062.39826, 56062.43013, 56062.46013, 56063.36508, 56063.39524, 56063.42932, 56065.3419, 56065.40567, 56066.37235, 56066.40718, 56066.44103, 56067.38681, 56067.41907, 56067.4513, 56068.36751, 56068.4034, 56068.43101, 56069.41498, 56069.44475, 56069.47485, 56070.39932, 56070.43436, 56070.467])
    
    new_mjds = []
    boost = 0.
    for ii in range(6): # 6 monthsish
        boost = (mjds[-1] - mjds[0])*(ii + 1.) + 1.
        new_mjds += list(mjds + boost)
    
    np.random.seed(1)
    mean_mag = 16.5
    mean_sigma = magnitude_to_errors(mean_mag)
    error = np.sqrt(np.random.normal(mean_sigma, mean_sigma/3, len(new_mjds))**2)
    
    # Also add 1% outliers
    outliers = np.random.normal(0, 1., len(new_mjds))
    outliers[np.random.uniform(size=len(new_mjds)) < 0.99] = 0.
    mags = np.random.normal(mean_mag, error) + outliers
    
    light_curve = simu.SimulatedLightCurve(mjd=new_mjds, mag=mags, error=error)
    #light_curve.addMicrolensingEvent(tE=15, u0=0.7)
    
    filter_mjds = np.arange(min(new_mjds), max(new_mjds), 0.1)
    filter_lc1 = simu.SimulatedLightCurve(mjd=filter_mjds, mag=[1.]*len(filter_mjds), error=[0.]*len(filter_mjds))
    filter_lc1.addMicrolensingEvent(tE=7., u0=0.556, t0=np.mean(filter_mjds))
    
    filter_lc2 = simu.SimulatedLightCurve(mjd=filter_mjds, mag=[1.]*len(filter_mjds), error=[0.]*len(filter_mjds))
    filter_lc2.addMicrolensingEvent(tE=15., u0=0.556, t0=np.mean(filter_mjds))
    
    filter_lc3 = simu.SimulatedLightCurve(mjd=filter_mjds, mag=[1.]*len(filter_mjds), error=[0.]*len(filter_mjds))
    filter_lc3.addMicrolensingEvent(tE=45., u0=0.556, t0=np.mean(filter_mjds))
    
    plt.subplot(211)
    plt.plot(np.convolve(light_curve.mag / np.median(light_curve.mag), filter_lc1.mag, "full"), 'r-')
    plt.plot(np.convolve(light_curve.mag / np.median(light_curve.mag), filter_lc2.mag, "full"), 'g-')
    plt.plot(np.convolve(light_curve.mag / np.median(light_curve.mag), filter_lc3.mag, "full"), 'b-')
    
    light_curve.addMicrolensingEvent(tE=15., u0=0.01)
    plt.plot(np.convolve(light_curve.mag / np.median(light_curve.mag), filter_lc1.mag, "full"), 'c--')
    plt.plot(np.convolve(light_curve.mag / np.median(light_curve.mag), filter_lc2.mag, "full"), 'm--')
    plt.plot(np.convolve(light_curve.mag / np.median(light_curve.mag), filter_lc3.mag, "full"), 'y--')
    
    plt.subplot(212)
    plt.plot(light_curve.mjd, light_curve.mag / np.median(light_curve.mag), 'k.')
    plt.plot(filter_lc1.mjd, filter_lc1.mag, 'r-')
    plt.plot(filter_lc2.mjd, filter_lc2.mag, 'g-')
    plt.plot(filter_lc3.mjd, filter_lc3.mag, 'b-')
    
    plt.show()

def test_simulated_light_curve():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ["r", "g", "b", "c", "m", "y"]
    
    for ii, pts in enumerate([20, 50, 100, 500]):
        mjd = np.linspace(0., 75., pts)
        mag = [2.*(ii+1)]*len(mjd)
        error = [0.1]*len(mjd)
        lc = simu.SimulatedLightCurve(mjd=mjd, mag=mag, error=error)
        lc.addMicrolensingEvent(tE=20, t0=30, u0=0.5/(ii+1))
        ax.plot(lc.mjd, lc.mag/np.mean(lc.mag), c=cs[ii])
    
    plt.show()

def rup147(fraction_with_events=0.5, N=10000):
    """ Compute the detection efficiency of our Ruprecht 147 fields
        using the MATCHED FILTER approach and by extrapolating the
        time sampling as of 2012-05-29
        
        Note: Should I compute the detection efficiency in bins of magnitude?
    """
    
    # Look up time sampling so far, e.g. from Kepler LSD query, use this as starting point
    #   for mjd's
    mjds = np.array([56046.42378, 56046.45529, 56046.49608, 56047.4067, 56047.40863, 56047.43839, 56047.46486, 56048.40791, 56048.43994, 56048.47319, 56049.39991, 56049.42953, 56049.46463, 56050.40626, 56050.44962, 56050.48109, 56060.38538, 56060.41486, 56060.44488, 56061.40934, 56061.44043, 56061.47189, 56062.39826, 56062.43013, 56062.46013, 56063.36508, 56063.39524, 56063.42932, 56065.3419, 56065.40567, 56066.37235, 56066.40718, 56066.44103, 56067.38681, 56067.41907, 56067.4513, 56068.36751, 56068.4034, 56068.43101, 56069.41498, 56069.44475, 56069.47485, 56070.39932, 56070.43436, 56070.467])
    
    new_mjds = []
    boost = 0.
    for ii in range(6): # 6 monthsish
        boost = (mjds[-1] - mjds[0])*(ii + 1.) + 1.
        new_mjds += list(mjds + boost)
    
    timescale_gen = timescale_generator("data/randomTimeScales.npy")
    magnitude_gen = magnitude_generator() 
    
    # First we need to figure out the typical scatter for plain simulated light
    #   curves with no microlensing events
    backbone_delta_chi_squareds = []
    backbone_median_mags = []
    for ii in range(N//10):
        # Generate simulated microlensing event with some magnitude + scatter that is a function
        #   of the magnitude (e.g. Nick's RMS error plot)
        mean_mag = magnitude_gen.genrand(1)[0]
        mean_sigma = magnitude_to_errors(mean_mag)
        error = np.sqrt(np.random.normal(mean_sigma, mean_sigma/3, len(new_mjds))**2)
        
        # Also add 1% outliers
        outliers = np.random.normal(0, 1., len(new_mjds))
        outliers[np.random.uniform(size=len(new_mjds)) < 0.99] = 0.
        mags = np.random.normal(mean_mag, error) + outliers
        
        light_curve = simu.SimulatedLightCurve(mjd=new_mjds, mag=mags, error=error)
        backbone_delta_chi_squareds.append(simu.compute_delta_chi_squared(light_curve))
        backbone_median_mags.append(np.median(light_curve.mag))
    
    # Next we compute the delta chi-squared for a bunch of light curves with microlensing events
    #   for some fraction of the time (based on fraction_with_events)
    microlensing_delta_chi_squareds = []
    microlensing_median_mags = []
    event_added = []
    for ii in range(N):
        # Generate simulated microlensing event with some magnitude + scatter that is a function
        #   of the magnitude (e.g. Nick's RMS error plot)
        mean_mag = magnitude_gen.genrand(1)[0]
        mean_sigma = magnitude_to_errors(mean_mag)
        error = np.sqrt(np.random.normal(mean_sigma, mean_sigma/3, len(new_mjds))**2)
        
        # Also add 1% outliers
        outliers = np.random.normal(0, 1., len(new_mjds))
        outliers[np.random.uniform(size=len(new_mjds)) < 0.99] = 0.
        mags = np.random.normal(mean_mag, error) + outliers
        
        light_curve = simu.SimulatedLightCurve(mjd=new_mjds, mag=mags, error=error)
        
        # Load in timescale distribution
        #   - Make draws from this distribution when generating microlensing events to add
        #   - Only add events with fraction_with_events probability
        if np.random.uniform() < fraction_with_events:
            timescale = timescale_gen.genrand(1)
            light_curve.addMicrolensingEvent(tE=timescale)
            event_added.append(True)
        else:
            event_added.append(False)
            
        # Run matched filter detection efficiency code
        microlensing_delta_chi_squareds.append(simu.compute_delta_chi_squared(light_curve))
        microlensing_median_mags.append(np.median(light_curve.mag))
    
    microlensing_delta_chi_squareds = np.array(microlensing_delta_chi_squareds)
    backbone_delta_chi_squareds = np.array(backbone_delta_chi_squareds)
    event_added = np.array(event_added)
    
    backbone_rms = np.sqrt(np.sum(backbone_delta_chi_squareds**2.) / len(backbone_delta_chi_squareds))
    #cut = 2.*backbone_rms
    cut = 200.
    false_detections = np.sum(np.logical_and(microlensing_delta_chi_squareds > cut, np.logical_not(event_added)))
    
    # TODO: Use event_added somehow to only count the light curves with actual events!
    frac_detection_efficiency = (np.sum(microlensing_delta_chi_squareds > cut) - false_detections) / len(microlensing_delta_chi_squareds)
    absolute_detection_efficiency = frac_detection_efficiency / fraction_with_events
    
    print "False detections:", false_detections
    
    plt.title("Absolute Detection Efficiency: {}".format(absolute_detection_efficiency))
    plt.semilogy(backbone_median_mags, backbone_delta_chi_squareds, 'k.', alpha=0.3)
    plt.semilogy(microlensing_median_mags, microlensing_delta_chi_squareds, 'r.', alpha=0.3)
    plt.axhline(cut, color="g", ls="--")
    plt.savefig("plots/rup147_detection_efficiency.png")
    
def test_rup147():
    rup147(fraction_with_events=0.5, N=10000)

if __name__ == "__main__":
    test_rup147()