# coding: utf-8
from __future__ import division

""" This module contains routines for generating figures for the paper that don't really
    belong anywhere else
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import sys
import logging
import cPickle as pickle
import time

# Third-party
import matplotlib
#matplotlib.use("TkAgg")
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np

# Create logger for this module
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Project
from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
try:
    import ptf.photometricdatabase as pdb
    import ptf.analyze.analyze as pa
    import survey_coverage
except ImportError:
    logger.warning("photometric database modules failed to load! If this is on Navtara, you made a boo-boo.")

def make_survey_sampling_figure(N=10):
    """ Generate the survey sampling figure that shows how different the
        time sampling is for different light curves in PTF.
    """
    
    np.random.seed(42)
    
    """
    R_info = survey_coverage.SurveyInfo("R")
    fields = R_info.fields(100)
    for ii in range(N):
        idx = np.random.randint(len(fields))
        rand_fields.append(fields[idx])
    """
    field_ids = [100031, 2794, 4252, 100085, 110002, 3685]
    rand_fields = [pdb.Field(f, "R") for f in field_ids]
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    min_mjd = 55250
    max_mjd = min_mjd+365
    for ii,field in enumerate(rand_fields):
        ccd_idx = field.ccds.keys()[np.random.randint(len(field.ccds))]
        ccd = field.ccds[ccd_idx]
        chip = ccd.read()
        
        source_ids = chip.sources.readWhere("nobs > 25")["matchedSourceID"]
        np.random.shuffle(source_ids)
        source_id = source_ids[0]
        
        logger.info("Reading source {} on {}, {}".format(source_id, field, ccd))
        light_curve = ccd.light_curve(source_id, clean=False, barebones=True, where=["(mjd >= {})".format(min_mjd), "(mjd <= {})".format(max_mjd)])
    
        ax.plot(light_curve.mjd, [ii]*len(light_curve.mjd), color="black", alpha=0.3, marker="o", markersize=7, linestyle="none", markeredgecolor="none")
        ax.text(55300, ii+0.1, "{} observations".format(len(light_curve.mjd)), size=14)
        ccd.close()
    
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_yticks([])
    
    ax.set_xticks(np.linspace(min_mjd, max_mjd, 10, endpoint=True))
    ax.set_xticklabels(["{}".format(int(x)) for x in np.linspace(0, 365, 10, endpoint=True)])
    
    ax.set_xlim(min_mjd, max_mjd)
    ax.set_ylim(-0.5, ii+0.5)
    
    ax.set_xlabel("Time [day]", fontsize=20)
    fig.savefig("plots/sampling_figure.pdf", facecolor="white", bbox_inches="tight")

def microlensing_event_sim():
    """ Create the multi-panel figure with simulated microlensing events for a single
        'typical' PTF light curve.
    """
    with open("data/paper_figures/sample_lightcurve.pickle") as f:
        light_curve = pickle.load(f)
    
    num = 4
    fig, axes = plt.subplots(num,1, sharex=True, figsize=(11,15))
    
    sim_light_curve = SimulatedLightCurve(mjd=light_curve.mjd, mag=light_curve.mag, error=light_curve.error)
    
    t0 = sim_light_curve.mjd[int(len(sim_light_curve.mjd)/2)]
    
    kwarg_list = [None, {"u0" : 1.0, "t0" : t0, "tE" : 20},
                        {"u0" : 0.5, "t0" : t0, "tE" : 20}, 
                        {"u0" : 0.01, "t0" : t0, "tE" : 20}]
    
    args_list = [(17.3, "a)"), (17., "b)"), (16.6, "c)"), (13, "d)")]
    
    for ii in range(num):
        axes[ii].xaxis.set_visible(False)        
        # TODO: text letter
        
        if ii != 0:
            sim_light_curve.reset()
            sim_light_curve.addMicrolensingEvent(**kwarg_list[ii])
        
        sim_light_curve.plot(axes[ii], marker="o", ms=3, alpha=0.75)
        
        axes[ii].axhline(14.3, color='r', linestyle="--")
        
        if kwarg_list[ii] == None:
            u0_str = ""
        else:
            u0 = kwarg_list[ii]["u0"]
            u0_str = r"$u_0={:.2f}$".format(u0)
        axes[ii].set_ylabel(u0_str, rotation="horizontal")
        
        if ii % 2 != 0:
            axes[ii].yaxis.tick_right()
        else:    
            axes[ii].yaxis.set_label_position("right")
        
        axes[ii].text(55985, *args_list[ii], fontsize=18)
    
    fig.suptitle("PTF light curve with simulated microlensing events", fontsize=24)
    
    fig.subplots_adjust(hspace=0.0, left=0.1, right=0.9)
    fig.savefig("plots/simulated_events.pdf", bbox_inches="tight", facecolor="white")

def median_maximum_indices_plot(field_id, ccd_id=5):
    """ Given a field ID, produce a figure with 10 panels: left column are light curves
        with median values of the variability indices, right column are light curves with
        maximum outlier values of the indices.
    """
    
    pickle_filename = "data/paper_figures/var_indices_field{}_ccd{}.pickle".format(field_id, ccd_id)
    
    indices = ["eta", "delta_chi_squared", "sigma_mu", "j", "k", "con"]
    if not os.path.exists(pickle_filename):
        field = pdb.Field(field_id, "R")
        ccd = field.ccds[ccd_id]
        chip = ccd.read()
        
        source_ids = chip.sources.readWhere("ngoodobs > 25")["matchedSourceID"]
        
        var_indices = []
        
        count = 0
        for source_id in source_ids:
            light_curve = ccd.light_curve(source_id, clean=True, barebones=True)
            
            if len(light_curve.mjd) > 25:
                try:
                    lc_var_indices = analyze.compute_variability_indices(light_curve, indices, return_tuple=True)
                except:
                    logger.warning("Failed to compute variability indices for simulated light curve!")
                    continue
                
                var_indices.append(lc_var_indices)
                count += 1 
        
        var_indices = np.array(var_indices, dtype=zip(indices, [float]*len(indices)))
        
        f = open(pickle_filename, "w")
        pickle.dump(var_indices, f)
        f.close()
    
    f = open(pickle_filename, "r")
    var_indices = pickle.load(f)
    f.close()
        
    for index in indices:
        print index, np.median(var_indices[index]), var_indices[index].min(), var_indices[index].max()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite any existing / cached files")
                    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    #make_survey_sampling_figure(10)
    #microlensing_event_sim()
    median_maximum_indices_plot(100300)