# coding: utf-8
from __future__ import division

""" This module contains routines for generating figures for the paper that don't really
    belong anywhere else
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import cPickle as pickle
import time

# Third-party
import matplotlib
matplotlib.use("Agg")
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
from ptf.ptflightcurve import PTFLightCurve

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
    
    field = pdb.Field(field_id, "R")
    ccd = field.ccds[ccd_id]
    
    pickle_filename = "data/paper_figures/var_indices_field{}_ccd{}.pickle".format(field_id, ccd_id)
    
    indices = ["eta", "delta_chi_squared", "sigma_mu", "j", "k", "con"]
    if not os.path.exists(pickle_filename):
        chip = ccd.read()
        
        source_ids = chip.sources.readWhere("ngoodobs > 25")["matchedSourceID"]
        logger.info("{} source ids selected".format(len(source_ids)))
        
        var_indices = []
        for source_id in source_ids:
            light_curve = ccd.light_curve(source_id, clean=True, barebones=True)
            
            if len(light_curve.mjd) > 25:
                logger.debug("Source ID {} is good".format(source_id))
                try:
                    lc_var_indices = (source_id,) + pa.compute_variability_indices(light_curve, indices, return_tuple=True)
                except:
                    logger.warning("Failed to compute variability indices for simulated light curve!")
                    continue
                
                var_indices.append(lc_var_indices)
        
        var_indices = np.array(var_indices, dtype=zip(["source_id"] + indices, [int] + [float]*len(indices)))
        
        f = open(pickle_filename, "w")
        pickle.dump(var_indices, f)
        f.close()
    
    f = open(pickle_filename, "r")
    var_indices = pickle.load(f)
    f.close()
        
    print "{} light curves from this CCD are good".format(len(var_indices))    
    
    min_max = {"eta" : "min", "delta_chi_squared" : "max", "sigma_mu" : "max", "j" : "max", "k" : "min", "con" : "max"}
    
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(20,15))
    for ii,index in enumerate(indices):
        med = np.median(var_indices[index])
        w, = np.where((var_indices[index] > (med-0.01*med)) & (var_indices[index] > (med+0.01*med))) 
        med_var_indices = var_indices[w[np.random.randint(len(w))]]
        
        if min_max[index] == "min":
            w, = np.where(var_indices[index] == var_indices[index].min())
            this_var_indices = var_indices[w[np.random.randint(len(w))]]
        elif min_max[index] == "max":
            w, = np.where(var_indices[index] == var_indices[index].max())
            this_var_indices = var_indices[w[np.random.randint(len(w))]]
        
        med_light_curve = ccd.light_curve(med_var_indices["source_id"], clean=True, barebones=True)
        this_light_curve = ccd.light_curve(this_var_indices["source_id"], clean=True, barebones=True)
        
        med_light_curve, this_light_curve = intersect_light_curves(med_light_curve, this_light_curve)
        
        new_med_mag = med_light_curve.mag + (np.median(this_light_curve.mag) - np.median(med_light_curve.mag))
        
        print np.median(new_med_mag), np.median(this_light_curve.mag)
        
        axes[ii].errorbar(med_light_curve.mjd, this_light_curve.mag - new_med_mag, np.sqrt(this_light_curve.error**2 + med_light_curve.error**2), color="black", marker="o", linestyle="none", ecolor='0.6', capsize=0)
        axes[ii].set_xlim(55350, 55600)
        
        #med_light_curve.plot(axes[ii, 0], ms=4)
        #this_light_curve.plot(axes[ii, 1], ms=4)
    
    fig.savefig("plots/median_max_indices.pdf", bbox_inches="tight")
    
def median_maximum_indices_plot_pdb(field_id):
    """ Same as above function, but uses variability indices computed by the photometric database instead of my own """
    
    # TODO: "sigma_mu", 
    min_max = {"eta" : "min", "delta_chi_squared" : "max", "sigma_mu" : "max", "j" : "max", "k" : "min", "con" : "max"}
    indices = ["eta", "delta_chi_squared", "j", "k", "con"]
    pdb_indices = ["bestVonNeumannRatio", "bestChiSQ", "bestStetsonJ", "bestStetsonK", "bestCon"]
    field = pdb.Field(field_id, "R")
    
    fig, axes = plt.subplots(len(indices), 2, sharex=True, figsize=(20,15))
    
    for ii, (index, pdb_index) in enumerate(zip(indices, pdb_indices)):
        all_med_sources = []
        all_outlier_sources = []
        all_ccds = []
        for ccd in field.ccds.values():
            chip = ccd.read()
            
            sources = chip.sources.readWhere("(nbestobs > 100) & ({} != 0)".format(pdb_index))
            
            if len(sources) == 0: 
                logger.debug("Skipping CCD {}".format(ccd.id))
                continue
            
            med = np.median(sources[pdb_index])
            w, = np.where((sources[pdb_index] > (med-0.1*med)) & (sources[pdb_index] > (med+0.1*med))) 
            
            if len(sources[w]) == 0:
                logger.debug("Skipping CCD {} because no mean values for idx {}".format(ccd.id, index))
                continue
            
            best_med_source = sources[w][sources[w]["ngoodobs"].argmax()]
            
            if min_max[index] == "min":
                w, = np.where(sources[pdb_index] == sources[pdb_index].min())
                best_outlier_source = sources[w][sources[w]["ngoodobs"].argmax()]
            elif min_max[index] == "max":
                w, = np.where(sources[pdb_index] == sources[pdb_index].max())
                best_outlier_source = sources[w][sources[w]["ngoodobs"].argmax()]
            
            all_med_sources.append(best_med_source)
            all_outlier_sources.append(best_outlier_source)
            all_ccds.append(ccd)
        
        all_med_sources = np.array(all_med_sources, dtype=sources.dtype)
        all_outlier_sources = np.array(all_outlier_sources, dtype=sources.dtype)
        
        best_med_source = all_med_sources[all_med_sources["ngoodobs"].argmax()]
        best_med_lightcurve = all_ccds[all_med_sources["ngoodobs"].argmax()].light_curve(best_med_source["matchedSourceID"], clean=True, barebones=True)
        
        if min_max[index] == "min":
            best_outlier_source = all_outlier_sources[all_outlier_sources[pdb_index].argmin()]
            best_outlier_lightcurve = all_ccds[all_outlier_sources[pdb_index].argmin()].light_curve(best_outlier_source["matchedSourceID"], clean=True, barebones=True)
        elif min_max[index] == "max":
            best_outlier_source = all_outlier_sources[all_outlier_sources[pdb_index].argmax()]
            best_outlier_lightcurve = all_ccds[all_outlier_sources[pdb_index].argmax()].light_curve(best_outlier_source["matchedSourceID"], clean=True, barebones=True)
        
        best_med_lightcurve.plot(axes[ii, 0], ms=4)
        best_outlier_lightcurve.plot(axes[ii, 1], ms=4)
        
        axes[ii, 0].set_title(index)
        #axes[ii, 0].set_xlim(55350, 55600)
        #axes[ii, 1].set_xlim(55350, 55600)
    
    fig.savefig("plots/median_max_indices_field{}_pdb.pdf".format(field_id), bbox_inches="tight")

def intersect_light_curves(light_curve1, light_curve2):
    """ Returns two light curves that have the same time measurements """
    
    mjd_set = set(light_curve1.mjd)
    common_mjd = np.array(list(mjd_set.intersection(set(light_curve2.mjd))))
    
    light_curve1_idx = np.in1d(light_curve1.mjd, common_mjd)
    light_curve2_idx = np.in1d(light_curve2.mjd, common_mjd)
    
    new_light_curve1 = PTFLightCurve(mjd=common_mjd, mag=light_curve1.mag[light_curve1_idx], error=light_curve1.error[light_curve1_idx])
    new_light_curve2 = PTFLightCurve(mjd=common_mjd, mag=light_curve2.mag[light_curve2_idx], error=light_curve2.error[light_curve2_idx])
    
    return (new_light_curve1, new_light_curve2)

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
    
    np.random.seed(104)
    #make_survey_sampling_figure(10)
    #microlensing_event_sim()
    #median_maximum_indices_plot(100300)
    median_maximum_indices_plot_pdb(100043)