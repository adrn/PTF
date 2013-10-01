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
np.seterr(invalid="ignore")
import emcee
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project
from ptf.lightcurve import PTFLightCurve, SimulatedLightCurve
from ptf.util import index_to_label
import ptf.db.photometric_database as pdb
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
    light_curve.add_microlensing_event(t0=t0, u0=u0, tE=tE)
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
    light_curve.add_microlensing_event(t0=50., u0=0.3, tE=20)

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
    if tE < 0:
        return -np.inf
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
    #print p, ln_prior(p, mag, mjd), ln_likelihood(p, mag, mjd, sigma)
    return ln_prior(p, mag, mjd) + ln_likelihood(p, mag, mjd, sigma)

def test_ln_prior():
    # First, test that using an out of bounds value breaks it
    mjd = np.linspace(0., 100., 50)
    sigma = np.random.uniform(0.05, 0.15, size=len(mjd))

    light_curve = SimulatedLightCurve(mjd=mjd, error=sigma, mag=15)
    light_curve.add_microlensing_event(t0=50., u0=0.3, tE=20)

    print "Good values:", ln_prior([15, 0.3, 50., 20.], light_curve.mag, light_curve.mjd)
    print "Test values:", ln_prior([15, 0.3, 50., 900.], light_curve.mag, light_curve.mjd)

    print "Bad m0:", ln_prior([25, 0.3, 50., 20.], light_curve.mag, light_curve.mjd)

    print "Bad u0:", ln_prior([15, -0.3, 50., 20.], light_curve.mag, light_curve.mjd)
    print "Bad u0:", ln_prior([15, 1.9, 50., 20.], light_curve.mag, light_curve.mjd)

    print "Bad t0:", ln_prior([15, 0.3, 500., 20.], light_curve.mag, light_curve.mjd)
    print "Bad t0:", ln_prior([15, 0.3, -50., 20.], light_curve.mag, light_curve.mjd)

    print "Bad tE:", ln_prior([15, 0.3, 50., 0.2], light_curve.mag, light_curve.mjd)
    print "Bad tE:", ln_prior([15, 0.3, 50., 2000.], light_curve.mag, light_curve.mjd)

def fit_model_to_light_curve(light_curve, nwalkers=200, nburn_in=100, nsamples=250, seed=None):
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
    pos, prob, state = sampler.run_mcmc(p0, nburn_in, rstate0=seed)

    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, nsamples)

    logger.info("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))

    return sampler

def make_chain_distribution_figure(light_curve, sampler, title="", filename=None):
    chain = sampler.flatchain

    fig, axes = plt.subplots(2, 2, figsize=(10,12))
    param_to_label = {"m0" : r"$M_0$", "u0" : r"$u_0$", "t0" : r"$t_0$", "tE" : r"$t_E$"}
    params = ["m0", "u0", "t0", "tE"]

    #params = np.array([["m0", "u0"], ["t0", "tE"]])
    max_idx = np.ravel(sampler.lnprobability).argmax()

    # m0
    idx = 0
    mn = np.mean(chain,axis=0)[idx]
    std = np.std(chain,axis=0)[idx]
    bins = np.linspace(mn - 1.5*std, mn + 1.5*std, 100)

    axes[0,0].hist(chain[:,idx], bins=bins, color="k", histtype="step")
    axes[0,0].axvline(chain[:,idx][max_idx], color='r', linestyle='--')
    axes[0,0].set_title("{}".format(param_to_label[params[idx]]), fontsize=26)

    # u0
    idx = 1
    mn = np.mean(chain,axis=0)[idx]
    std = np.std(chain,axis=0)[idx]
    #bins = np.linspace(mn - 5*std, mn + 5*std, 100)
    bins = np.linspace(0., 1.5, 100)

    axes[0,1].hist(chain[:,idx], bins=bins, color="k", histtype="step")
    axes[0,1].axvline(chain[:,idx][max_idx], color='r', linestyle='--')
    axes[0,1].set_title("{}".format(param_to_label[params[idx]]), fontsize=26)

    # t0
    idx = 2
    mn = np.mean(chain,axis=0)[idx]
    std = np.std(chain,axis=0)[idx]
    bins = np.linspace(mn - 1*std, mn + 1*std, 100)

    axes[1,0].hist(chain[:,idx], bins=bins, color="k", histtype="step")
    axes[1,0].axvline(chain[:,idx][max_idx], color='r', linestyle='--')
    axes[1,0].set_title("{}".format(param_to_label[params[idx]]), fontsize=26)

    # tE
    idx = 3
    mn = np.mean(chain,axis=0)[idx]
    std = np.std(chain,axis=0)[idx]
    bins = np.logspace(0., 3., 100)

    axes[1,1].hist(chain[:,idx], bins=bins, color="k", histtype="step")
    axes[1,1].axvline(chain[:,idx][max_idx], color='r', linestyle='--')
    axes[1,1].set_title("{}".format(param_to_label[params[idx]]), fontsize=26)
    axes[1,1].set_xscale("log")

    for ax in np.ravel(axes):
        ax.yaxis.set_ticks([])
        #ax.set_yscale("log")

    fig.suptitle(title, fontsize=24)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)

    if filename == None:
        return fig, axes
    else:
        fig.savefig(os.path.join("plots/fit_events", filename))

def make_light_curve_figure(light_curve, sampler, filename=None, title="", x_axis_labels=True, y_axis_labels=True, chains=True):
    chain = sampler.flatchain

    fig = plt.figure(figsize=(10,12))
    fig.suptitle(title, fontsize=23)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1,2])
    ax = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[0])

    # "best" parameters
    max_idx = np.ravel(sampler.lnprobability).argmax()
    best_m0, best_u0, best_t0, best_tE = chain[max_idx]
    
    mjd = np.arange(light_curve.mjd.min()-1000., light_curve.mjd.max()+1000., 0.2)
    
    if chains:
        first_t0 = None
        for ii in range(150):
            walker_idx = np.random.randint(sampler.k)
            chain = sampler.chain[walker_idx][-100:]
            probs = sampler.lnprobability[walker_idx][-100:]
            link_idx = np.random.randint(len(chain))
    
            prob = probs[link_idx]
            link = chain[link_idx]
            m0, u0, t0, tE = link
    
            if prob/probs.max() > 2:
                continue
    
            # More than 100% error
            if np.any(np.fabs(link - np.mean(chain,axis=0)) / np.mean(chain,axis=0) > 0.25) or tE < 8 or tE > 250.:
                continue
    
            if np.fabs(t0 - best_t0) > 20.:
                continue
    
            if first_t0 == None:
                first_t0 = t0
            s_light_curve = SimulatedLightCurve(mjd=mjd, error=np.zeros_like(mjd), mag=m0)
            s_light_curve.add_microlensing_event(t0=t0, u0=u0, tE=tE)
            ax.plot(s_light_curve.mjd-first_t0, s_light_curve.mag, linestyle="-", color="#666666", alpha=0.1)
            #ax_inset.plot(s_light_curve.mjd-first_t0, s_light_curve.mag, "r-", alpha=0.05)

    mean_m0, mean_u0, mean_t0, mean_tE = np.median(chain,axis=0)
    std_m0, std_u0, std_t0, std_tE = np.std(chain,axis=0)
    
    print("m0 = ", mean_m0, std_m0)
    print("u0 = ", mean_u0, std_u0)
    print("tE = ", mean_tE, std_tE)
    print("t0 = ", mean_t0, std_t0)
    
    light_curve.mjd -= mean_t0
    zoomed_light_curve = light_curve.slice_mjd(-3.*mean_tE, 3.*mean_tE)
    if len(zoomed_light_curve) == 0:
        mean_t0 = light_curve.mjd.min()
        mean_tE = 10.
        
        light_curve.mjd -= mean_t0
        zoomed_light_curve = light_curve.slice_mjd(-3.*mean_tE, 3.*mean_tE)
    
    zoomed_light_curve.plot(ax)
    light_curve.plot(ax2)
    ax.set_xlim(-3.*mean_tE, 3.*mean_tE)
    #ax.text(-2.5*mean_tE, np.min(s_light_curve.mag), r"$u_0=${u0:.3f}$\pm${std:.3f}".format(u0=u0, std=std_u0) + "\n" + r"$t_E=${tE:.1f}$\pm${std:.1f} days".format(tE=mean_tE, std=std_tE), fontsize=19)
    
    if chains:
        fig.text(0.15, 0.48,
             r"$u_0=${u0:.3f}$\pm${std:.3f}".format(u0=mean_u0, std=std_u0) + 
             "\n" + r"$t_E=${tE:.1f}$\pm${std:.1f} days".format(tE=mean_tE, std=std_tE), fontsize=19)
    
    if y_axis_labels:
        ax.set_ylabel(r"$R$ (mag)", fontsize=24)
    if x_axis_labels:
        ax.set_xlabel(r"time-$t_0$ [days]", fontsize=24)

    # Now plot the "best fit" line in red
    if chains:
        s_light_curve = SimulatedLightCurve(mjd=mjd, error=np.zeros_like(mjd), mag=best_m0)
        s_light_curve.add_microlensing_event(t0=best_t0, u0=best_u0, tE=best_tE)
        ax.plot(s_light_curve.mjd-best_t0, s_light_curve.mag, "r-", alpha=0.6, linewidth=2)

    if y_axis_labels:
        ax2.set_ylabel(r"$R$ (mag)", fontsize=24)

    #light_curve.plot(ax_inset)
    #ax_inset.set_ylim(s_light_curve.mag.min()+abs(s_light_curve.mag.min()-m0)/5., s_light_curve.mag.min()-0.05)
    #ax_inset.set_xlim(-0.3*tE, 0.3*tE)
    #ax_inset.set_xticklabels([])
    #ax_inset.set_yticklabels([])
    
    fig.subplots_adjust(hspace=0.1, left=0.12)
    
    if filename == None:
        return fig, axes
    else:
        fig.savefig(os.path.join("plots/fit_events", filename))

if __name__ == "__main__":
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

    parser.add_argument("-p", dest="plot", action="store_true", default=False,
                    help="Plot or not")

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

    if args.plot:
        make_light_curve_figure(light_curve, sampler, filename="{0}_{1}_{2}_lc.png".format(light_curve.field_id, light_curve.ccd_id, light_curve.source_id))
        make_chain_distribution_figure(light_curve, sampler, filename="{0}_{1}_{2}_dist.png".format(light_curve.field_id, light_curve.ccd_id, light_curve.source_id))
