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

def chi(p, mag, mjd, sigma):
    return (mag - magnitude_model(p, mjd)) / sigma

def ln_likelihood(p, mag, mjd, sigma):
    """ We use chi-squared as our likelihood function because in *most cases*, 
        the magnitude errors are approximately Gaussian.
    """
    diff = chi(p, mag, mjd, sigma)
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

def ln_p_t0(t0, mjd, mag, t0_sigma=25.):
    """ Prior on time of event t0 """
    return -0.5 * (np.log(2.*np.pi*t0_sigma) + (t0 - mjd[np.argmin(mag)])**2 / t0_sigma**2)

def ln_p_tE(tE, tE_sigma=1.3):
    """ Prior on duration of event tE """
    return -0.5 * (np.log(2.*np.pi*tE_sigma) + (np.log(tE) - np.log(30.))**2 / tE_sigma**2) # Log-normal centered at 30 days

def ln_p_m0(m0, median_mag, mag_rms):
    """ Prior on reference magnitude m0. This will be a Gaussian around 
        the median magnitude of the light curve
    """
    mag_rms *= 5
    return -0.5 * (np.log(2.*np.pi*mag_rms) + (m0 - median_mag)**2 / (2.*mag_rms)**2)

def ln_prior(p, mag, mjd):
    m0, u0, t0, tE = p    
    return ln_p_u0(u0) + ln_p_t0(t0, mjd, mag) + ln_p_tE(tE) + ln_p_m0(m0, np.median(mag), np.std(mag))

def plot_priors(light_curve, ln=False):
    mag = light_curve.mag
    mjd = light_curve.mjd
    
    plt.figure(figsize=(15,15))
    
    plt.subplot(221)
    x = np.linspace(np.median(mag) - 5., np.median(mag) + 5., 100)
    if ln: 
        plt.plot(x, ln_p_m0(x, np.median(mag), np.std(mag)), "k-")
        plt.xlabel(r"$\log m_0$")
    else:
        plt.plot(x, np.exp(ln_p_m0(x, np.median(mag), np.std(mag))), "k-")
        plt.xlabel(r"$m_0$")
    
    plt.subplot(222)
    x = np.linspace(0., 1.34, 100)
    if ln: 
        plt.plot(x, [ln_p_u0(xi) for xi in x], "k-")
        plt.xlabel(r"$\log u_0$")
    else:
        plt.plot(x, np.exp([ln_p_u0(xi) for xi in x]), "k-")
        plt.xlabel(r"$u_0$")
    
    plt.subplot(223)
    x = np.linspace(mjd.min(), mjd.max(), 100)
    if ln: 
        plt.plot(x, ln_p_t0(x, mjd, mag), "k-")
        plt.xlabel(r"$\log t_0$")
    else:
        plt.plot(x, np.exp(ln_p_t0(x, mjd, mag)), "k-")
        plt.xlabel(r"$t_0$")
    
    plt.subplot(224)
    x = 10**np.linspace(0, 3, 100)
    if ln: 
        plt.plot(x, ln_p_tE(x), "k-")
        plt.xlabel(r"$\log t_E$")
    else:
        plt.plot(x, np.exp(ln_p_tE(x)), "k-")
        plt.xlabel(r"$t_E$")
    
    plt.savefig("plots/test.png")

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
                    np.random.uniform(light_curve.mjd.min()-50., light_curve.mjd.min()+50.),
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
    import matplotlib.gridspec as gridspec
    
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("--test", action="store_true", dest="test", default=False,
                    help="Run tests, then exit.")
    
    parser.add_argument("-f", dest="field_id", default=None, required=True, type=int,
                    help="Field ID")
    parser.add_argument("-c", dest="ccd_id", default=None, required=True, type=int,
                    help="CCD ID")
    parser.add_argument("-s", dest="source_id", default=None, required=True, type=int,
                    help="Source ID")
    parser.add_argument("--clean", dest="clean", action="store_true", default=False,
                    help="Clean the light curve")
    
    parser.add_argument("--walkers", dest="walkers", default=200, type=int,
                    help="Number of walkers")
    parser.add_argument("--steps", dest="steps", default=1000, type=int,
                    help="Number of steps to take")
    parser.add_argument("--burn-in", dest="burn_in", default=100, type=int,
                    help="Number of steps to burn in")
    
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
    
    field = pdb.Field(args.field_id, "R")
    ccd = field.ccds[args.ccd_id]
    light_curve = ccd.light_curve(args.source_id, clean=args.clean, barebones=True)
    if args.source_id == 5466:
        idx = light_curve.mag > 18.25
        light_curve.mjd = light_curve.mjd[idx]
        light_curve.mag = light_curve.mag[idx]
        light_curve.error = light_curve.error[idx]
    sampler = fit_model_to_light_curve(light_curve, nwalkers=args.walkers, nsamples=args.steps, nburn_in=args.burn_in)
    end_chain = sampler.flatchain[-args.walkers*100:]
    
    plt.clf()
    plt.figure(figsize=(10,14))
    param_to_label = {"m0" : r"$M_0$", "u0" : r"$u_0$", "t0" : r"$t_0$", "tE" : r"$t_E$"}
    for i,name in enumerate(["m0", "u0", "t0", "tE"]):
        
        mn = np.mean(end_chain,axis=0)[i]
        std = np.std(end_chain,axis=0)[i]
        bins = np.linspace(mn - 5*std, mn + 5*std, 100)
            
        plt.subplot(2,2,i+1)
        plt.hist(sampler.flatchain[:,i], bins=bins, color="k", histtype="step")
        plt.title("{}".format(param_to_label[name]), fontsize=26)
        plt.gca().yaxis.set_ticks([])
    
    plt.suptitle("Posterior Probability Distributions\nField: {}, CCD: {}, Source ID: {}".format(field.id, ccd.id, args.source_id), fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("plots/event_fitter/field{}_ccd{}_source{}_posterior.pdf".format(field.id, ccd.id, args.source_id))
    
    # LIGHT CURVE FIGURE
    
    fig = plt.figure(figsize=(8,12))
    fig.suptitle("Field: {}, CCD: {}, Source ID: {}".format(field.id, ccd.id, args.source_id), fontsize=23)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
    ax = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    #ax = fig.add_subplot(211)
    #ax2 = fig.add_subplot(212)
    #ax_inset = fig.add_axes([0.65, 0.1, 0.25, 0.25])
    
    mjd = np.arange(light_curve.mjd.min()-1000., light_curve.mjd.max()+1000., 0.2)
    first_t0 = None
    for ii in np.random.randint(len(end_chain), size=100):
        link = end_chain[ii]
        m0, u0, t0, tE = link
        
        # More than 100% error
        if np.any(np.fabs(link - np.mean(end_chain,axis=0)) / np.mean(end_chain,axis=0) > 0.75) or tE < 8:
            continue
        
        if first_t0 == None:
            first_t0 = t0
        s_light_curve = SimulatedLightCurve(mjd=mjd, error=np.zeros_like(mjd), mag=m0)
        s_light_curve.addMicrolensingEvent(t0=t0, u0=u0, tE=tE)
        ax.plot(s_light_curve.mjd-first_t0, s_light_curve.mag, "r-", alpha=0.05)
        #ax_inset.plot(s_light_curve.mjd-first_t0, s_light_curve.mag, "r-", alpha=0.05)
    
    mean_m0, mean_u0, mean_t0, mean_tE = np.median(end_chain,axis=0)
    std_m0, std_u0, std_t0, std_tE = np.std(end_chain,axis=0)
    
    light_curve.mjd -= mean_t0
    zoomed_light_curve = light_curve.slice_mjd(-3.*mean_tE, 3.*mean_tE)
    zoomed_light_curve.plot(ax)
    light_curve.plot(ax2)
    ax.set_xlim(-3.*mean_tE, 3.*mean_tE)
    ax.text(-2.5*mean_tE, np.min(s_light_curve.mag), r"$u_0=${u0:.3f}$\pm${std:.3f}".format(u0=u0, std=std_u0) + "\n" + r"$t_E=${tE:.1f}$\pm${std:.1f} days".format(tE=mean_tE, std=std_tE), fontsize=19)
    #ax.set_xlabel(r"time-$t_0$ [days]", fontsize=24)
    ax.set_ylabel(r"$R$ (mag)", fontsize=24)
    
    ax2.set_xlabel(r"time-$t_0$ [days]", fontsize=24)
    ax2.set_ylabel(r"$R$ (mag)", fontsize=24)
    
    #light_curve.plot(ax_inset)
    #ax_inset.set_ylim(s_light_curve.mag.min()+abs(s_light_curve.mag.min()-m0)/5., s_light_curve.mag.min()-0.05)
    #ax_inset.set_xlim(-0.3*tE, 0.3*tE)
    #ax_inset.set_xticklabels([])
    #ax_inset.set_yticklabels([])
    
    fig.savefig("plots/event_fitter/field{}_ccd{}_source{}.pdf".format(field.id, ccd.id, args.source_id))
        