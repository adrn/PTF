""" 
    Fit a microlensing model to a PTF light curve.
    
    Notes:
    ------
    Our data have near absolute timestamps (x values with marginal 
    uncertainties), and magnitude values with some scatter. The uncertainties
    we have for the magnitudes are mostly Gaussian (well, Poisson), in the 
    sense that in most cases the posterior distributions for the data are 
    well-represented by a Gaussian with mean=0, and variance=error^2. In some
    cases, however, there are extreme nonlinearities in the error terms when 
    the errors are massively underreported. One example is when the source is
    near the edge of a CCD, and falls off the CCD during one exposure. Here the
    magnitude could drop by 5-6 magnitudes, or even slowly fall off the CCD over
    the course of many exposures during a night. Another example is when there is
    reflected light within the telescope body. In certain instances, a bright star
    or the Moon will reflect and cause a bright ghost over a large (>10 arcmin^2) 
    section of a CCD. This will cause a potentially huge increase in the brightness
    of the source, but the errors will be underrepresented. The errors are estimated 
    based on a combination of the photometric (Poisson?) errors and systematic errors
    estimated over a whole CCD.
    
"""
from __future__ import division

# Standard library
import logging
import multiprocessing
import os
import sys
import cPickle as pickle

# Create logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Third-party
from apwlib.globals import greenText, yellowText, redText
import numpy as np
import emcee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project
from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
from ptf.ptflightcurve import PTFLightCurve
from ptf.globals import index_to_label
import ptf.photometricdatabase as pdb
#import ptf.analyze.analyze as analyze

def A(t, u0, t0, tE):
    """ Microlensing amplifiction factor """
    u = np.sqrt(u0*u0 + ((t-t0)/tE)**2)
    
    return (u**2 + 2) / (u*np.sqrt(u**2 + 4))

def magnitude_model(p, t):
    """ """
    m0, u0, t0, tE = p
    return m0 - 2.5*np.log10(A(t, u0, t0, tE))

def test_magnitude_model():
    mjd = np.arange(0., 100., 0.2)
    sigma = np.zeros_like(mjd) + 0.01
    
    u0 = 0.1
    t0 = 50.
    tE = 20
    mag = magnitude_model([15, u0, t0, tE], mjd)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.errorbar(mjd, mag, sigma)
    
    light_curve = SimulatedLightCurve(mjd=mjd, error=sigma, mag=15)
    light_curve.addMicrolensingEvent(t0=t0, u0=u0, tE=tE)
    light_curve.plot(ax)
    
    plt.show()    

def ln_likelihood(p, mag, mjd, sigma):
    """ We use chi-squared as our likelihood function because in *most cases*, 
        the magnitude errors are approximately Gaussian.
    """
    diff = (mag - magnitude_model(p, mjd)) / sigma
    return -0.5 * np.sum(diff*diff)

def test_ln_likelihood():
    mjd = np.linspace(0., 100., 50)
    sigma = np.random.uniform(0.05, 0.15, size=len(mjd))
    
    light_curve = SimulatedLightCurve(mjd=mjd, error=sigma, mag=15)
    light_curve.addMicrolensingEvent(t0=50., u0=0.3, tE=20)
    
    likelihood1 = ln_likelihood(p=[15, 0.3, 50., 20.], mag=light_curve.mag, mjd=mjd, sigma=sigma)
    likelihood2 = ln_likelihood(p=[15, 0.3, 1., 20.], mag=light_curve.mag, mjd=mjd, sigma=sigma)
    likelihood3 = ln_likelihood(p=[15, 0.3, 50., 50.], mag=light_curve.mag, mjd=mjd, sigma=sigma)
    
    print "Likelihood with model at correct peak: {}".format(likelihood1)
    print "Likelihood with model way off peak: {}".format(likelihood2)
    print "Likelihood with model at peak, wrong tE: {}".format(likelihood3)

def ln_p_u0(u0):
    """ Prior on impact parameter u0 """
    if u0 <= 0 or u0 > 1.34:
        return -np.inf
    else:
        return -np.log(1.34)

def ln_p_t0(t0, tmin, tmax):
    """ Prior on time of event t0 """
    if t0 < tmin or t0 > tmax:
        return -np.inf
    else:
        return -np.log(tmax-tmin)

def ln_p_tE(tE):
    """ Prior on duration of event tE """
    
    # TODO: For now this is uninformative, but I may want to penalize 
    #       shorter events?
    if tE < 1. or tE > 1000:
        return -np.inf
    else:
        #return -np.log(1000-1.)
        return np.log(1/2.17) + np.exp(-0.5*(np.log10(tE)-1.4)**2)

def ln_p_m0(m0, median_mag, mag_rms):
    """ Prior on reference magnitude m0. This will be a Gaussian around 
        the median magnitude of the light curve
    """
    return -0.5*np.log(2.*np.pi*mag_rms) - (m0 - median_mag)**2 / (2.*mag_rms**2)

def ln_prior(p, mag, mjd):
    m0, u0, t0, tE = p
    
    p_u0 = ln_p_u0(u0)
    p_t0 = ln_p_t0(t0, mjd.min(), mjd.max())
    p_tE = ln_p_tE(tE)
    p_m0 = ln_p_m0(m0, np.median(mag), np.std(mag))
    
    #logger.debug("Prior on u0: {}".format(p_u0))
    #logger.debug("Prior on t0: {}".format(p_t0))
    #logger.debug("Prior on tE: {}".format(p_tE))
    #logger.debug("Prior on m0: {}".format(p_m0))
    
    return ln_p_u0(u0) + ln_p_t0(t0, mjd.min(), mjd.max()) + ln_p_tE(tE) + ln_p_m0(m0, np.median(mag), np.std(mag))

def ln_posterior(p, mag, mjd, sigma):
    return ln_prior(p, mag, mjd) + ln_likelihood(p, mag, mjd, sigma)

def test_ln_prior():
    # First, test that using an out of bounds value breaks it
    mjd = np.linspace(0., 100., 50)
    sigma = np.random.uniform(0.05, 0.15, size=len(mjd))
    
    light_curve = SimulatedLightCurve(mjd=mjd, error=sigma, mag=15)
    light_curve.addMicrolensingEvent(t0=50., u0=0.3, tE=20)
    
    print "Good values:", ln_prior([15, 0.3, 50., 20.], light_curve.mag, light_curve.mjd)
    print "Test values:", ln_prior([15, 0.3, 50., 900.], light_curve.mag, light_curve.mjd)
    
    print "Bad m0:", ln_prior([25, 0.3, 50., 20.], light_curve.mag, light_curve.mjd)
    
    print "Bad u0:", ln_prior([15, -0.3, 50., 20.], light_curve.mag, light_curve.mjd)
    print "Bad u0:", ln_prior([15, 1.9, 50., 20.], light_curve.mag, light_curve.mjd)
    
    print "Bad t0:", ln_prior([15, 0.3, 500., 20.], light_curve.mag, light_curve.mjd)
    print "Bad t0:", ln_prior([15, 0.3, -50., 20.], light_curve.mag, light_curve.mjd)
    
    print "Bad tE:", ln_prior([15, 0.3, 50., 0.2], light_curve.mag, light_curve.mjd)
    print "Bad tE:", ln_prior([15, 0.3, 50., 2000.], light_curve.mag, light_curve.mjd)

def fit_model_to_light_curve(light_curve, nwalkers=200, nburn_in=100, nsamples=250):
    """ Fit a microlensing model to a given light curve using Emcee
        
    """
    
    #p0 = np.array([[np.random.normal(np.median(light_curve.mag), np.std(light_curve.mag)),
    #                np.random.uniform(0., 1.34),
    #                np.random.normal(light_curve.mjd[np.argmin(light_curve.mag)], 50),
    #                10**np.random.uniform(0., 3.)] for ii in range(nwalkers)])
    p0 = np.array([[np.random.normal(np.median(light_curve.mag), np.std(light_curve.mag)),
                    np.random.uniform(0., 1.34),
                    np.random.uniform(light_curve.mjd.min()-10., light_curve.mjd.min()+10.),
                    10**np.random.uniform(0.3, 3.)] for ii in range(nwalkers)])
    ndim = p0.shape[1]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, 
                                    args=[light_curve.mag, light_curve.mjd, light_curve.error], 
                                    threads=4)
    pos, prob, state = sampler.run_mcmc(p0, nburn_in)
    sampler.reset()
    
    sampler.run_mcmc(pos, nsamples)
    
    logger.info("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    
    return sampler

if __name__ == "__main__":
    
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("--test", action="store_true", dest="test", default=False,
                    help="Run tests, then exit.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    if args.test:
        test_magnitude_model()
        test_ln_likelihood()
        test_ln_prior()
        sys.exit(0)
    
    field = pdb.Field(100043, "R")
    for ccd in field.ccds.values():
        chip = ccd.read()
        
        sources = chip.sources.readWhere("(ngoodobs > 25) & (vonNeumannRatio < 1.0) & (vonNeumannRatio > 0)")
        logger.info("{} sources".format(len(sources)))
        
        for source in sources:
            light_curve = ccd.light_curve(source["matchedSourceID"], clean=True, barebones=True)
            
            if len(light_curve.mjd) > 100:
                logger.debug(source["matchedSourceID"])
                sampler = fit_model_to_light_curve(light_curve, nwalkers=500, nsamples=1000)
                
                plt.clf()
                for i,name in enumerate(["m0", "u0", "t0", "tE"]):
                    plt.subplot(2,2,i+1)
                    plt.hist(sampler.flatchain[:,i], 100, color="k", histtype="step")
                    plt.title("{}".format(name))
                    plt.ylabel("Posterior Probability")
                
                plt.savefig("plots/event_fitter/field{}_ccd{}_source{}_posterior.png".format(field.id, ccd.id, source["matchedSourceID"]))
                
                fig = plt.figure()
                ax = fig.add_subplot(111)
                mjd = np.arange(light_curve.mjd.min(), light_curve.mjd.max(), 0.2)
                for link in sampler.flatchain[-25:]:
                    m0, u0, t0, tE = link
                    s_light_curve = SimulatedLightCurve(mjd=mjd, error=np.zeros_like(mjd), mag=m0)
                    s_light_curve.addMicrolensingEvent(t0=t0, u0=u0, tE=tE)
                    ax.plot(s_light_curve.mjd, s_light_curve.mag, "k-", alpha=0.2)
                
                light_curve.plot(ax)
                ax.set_xlabel("MJD")
                ax.set_ylabel("R")
                fig.savefig("plots/event_fitter/field{}_ccd{}_source{}_fit.png".format(field.id, ccd.id, source["matchedSourceID"]))
        