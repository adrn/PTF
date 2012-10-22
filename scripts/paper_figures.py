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
import ptf.globals as pg
import ptf.variability_indices as vi

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

'''
def maximum_outlier_indices_plot(field_id, ccd_id=5):
    """ Given a field ID, produce a figure with light curves for the 
        maximum outlier values of the indices.
    """
    
    field = pdb.Field(field_id, "R")
    ccd = field.ccds[ccd_id]
    
    pickle_filename = "data/paper_figures/var_indices_field{}_ccd{}.pickle".format(field_id, ccd_id)
    
    indices = ["eta", "delta_chi_squared", "sigma_mu", "j", "k"]
    if not os.path.exists(pickle_filename):
        chip = ccd.read()
        
        source_ids = chip.sources.readWhere("ngoodobs > 600")["matchedSourceID"]
        logger.info("{} source ids selected".format(len(source_ids)))
        
        var_indices = []
        for source_id in source_ids:
            light_curve = ccd.light_curve(source_id, clean=True, barebones=True)
            
            if len(light_curve.mjd) > 500:
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
    
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(20,15))
    
    lc = ccd.light_curve(var_indices[var_indices["eta"].argmin()]["source_id"], clean=True, barebones=True)
    lc.plot(axes[0])
    
    lc = ccd.light_curve(var_indices[var_indices["j"].argmax()]["source_id"], clean=True, barebones=True)
    lc.plot(axes[1])
    
    lc = ccd.light_curve(var_indices[var_indices["k"].argmin()]["source_id"], clean=True, barebones=True)
    lc.plot(axes[2])
    
    lc = ccd.light_curve(var_indices[var_indices["sigma_mu"].argmax()]["source_id"], clean=True, barebones=True)
    lc.plot(axes[3])
    
    lc = ccd.light_curve(var_indices[var_indices["delta_chi_squared"].argmax()]["source_id"], clean=True, barebones=True)
    lc.plot(axes[4])
    
    fig.savefig("plots/maximum_outlier_light_curves.pdf", bbox_inches="tight")
'''

def maximum_outlier_indices_plot(field_id):
    """ Same as above function, but uses variability indices computed by the photometric database instead of my own """
    
    indices = ["eta", "delta_chi_squared", "j", "k", "sigma_mu", "con"]
    min_max = {"eta" : "min", 
               "delta_chi_squared" : "max", 
               "j" : "max", 
               "k" : "min", 
               "sigma_mu": "max",
               "con" : "max"}
    pdb_indices = ["bestVonNeumannRatio", "bestChiSQ", "bestStetsonJ", "bestStetsonK", ["bestMagRMS", "bestMedianMag"], "bestCon"]
    field = pdb.Field(field_id, "R")
    
    fig, axes = plt.subplots(len(indices), 1, sharex=True, figsize=(18,24))
    
    for ii, (index, pdb_index) in enumerate(zip(indices, pdb_indices)):
        all_outlier_sources = []
        all_ccds = []
        for ccd in field.ccds.values():
            chip = ccd.read()
            
            if index == "sigma_mu":
                sources = chip.sources.readWhere("(nbestobs > 100) & ({} != 0)".format(pdb_index[0]))
            else:
                sources = chip.sources.readWhere("(nbestobs > 100) & ({} != 0)".format(pdb_index))
            
            if len(sources) == 0: 
                logger.debug("Skipping CCD {}".format(ccd.id))
                continue
            
            if index == "sigma_mu":
                sigma_mus = sources[pdb_index[0]] / sources[pdb_index[1]]
                if min_max[index] == "min":
                    w, = np.where(sigma_mus == sigma_mus.min())
                    best_outlier_source = sources[w][sources[w]["ngoodobs"].argmax()]
                elif min_max[index] == "max":
                    w, = np.where(sigma_mus == sigma_mus.max())
                    best_outlier_source = sources[w][sources[w]["ngoodobs"].argmax()]
            
            else:
                if min_max[index] == "min":
                    w, = np.where(sources[pdb_index] == sources[pdb_index].min())
                    best_outlier_source = sources[w][sources[w]["ngoodobs"].argmax()]
                elif min_max[index] == "max":
                    w, = np.where(sources[pdb_index] == sources[pdb_index].max())
                    best_outlier_source = sources[w][sources[w]["ngoodobs"].argmax()]
            
            all_outlier_sources.append(best_outlier_source)
            all_ccds.append(ccd)
    
        all_outlier_sources = np.array(all_outlier_sources, dtype=sources.dtype)
        
        if index == "sigma_mu":
            clean = False
            idx_vals = all_outlier_sources[pdb_index[0]] / all_outlier_sources[pdb_index[1]]
        else:
            clean = True
            idx_vals = all_outlier_sources[pdb_index]
        
        while True:
            if min_max[index] == "min":
                best_outlier_source = all_outlier_sources[idx_vals.argmin()]
                best_outlier_lightcurve = all_ccds[idx_vals.argmin()].light_curve(best_outlier_source["matchedSourceID"], clean=clean, barebones=True)
            elif min_max[index] == "max":
                best_outlier_source = all_outlier_sources[idx_vals.argmax()]
                best_outlier_lightcurve = all_ccds[idx_vals.argmax()].light_curve(best_outlier_source["matchedSourceID"], clean=clean, barebones=True)
            
            try:
                best_outlier_lightcurve.plot(axes[ii], ms=4)
                break
            except ValueError:
                if min_max[index] == "min":
                    idx_vals[idx_vals.argmin()] = 1E8
                elif min_max[index] == "max":
                    idx_vals[idx_vals.argmax()] = -1E8
        
        axes[ii].set_title(pg.index_to_label[index], fontsize=24)
        #axes[ii].set_xlim(best_outlier_lightcurve.mjd.min()-2, best_outlier_lightcurve.mjd.max()+2)
        #axes[ii, 1].set_xlim(55350, 55600)
    
    axes[-1].set_xlabel("MJD", fontsize=24)
    fig.subplots_adjust(hspace=0.2, top=0.95, bottom=0.08)
    fig.savefig("plots/max_outlier_light_curves.pdf".format(field_id))#, bbox_inches="tight")

def intersect_light_curves(light_curve1, light_curve2):
    """ Returns two light curves that have the same time measurements """
    
    mjd_set = set(light_curve1.mjd)
    common_mjd = np.array(list(mjd_set.intersection(set(light_curve2.mjd))))
    
    light_curve1_idx = np.in1d(light_curve1.mjd, common_mjd)
    light_curve2_idx = np.in1d(light_curve2.mjd, common_mjd)
    
    new_light_curve1 = PTFLightCurve(mjd=common_mjd, mag=light_curve1.mag[light_curve1_idx], error=light_curve1.error[light_curve1_idx])
    new_light_curve2 = PTFLightCurve(mjd=common_mjd, mag=light_curve2.mag[light_curve2_idx], error=light_curve2.error[light_curve2_idx])
    
    return (new_light_curve1, new_light_curve2)

def variability_indices_distributions(field_id=100018, overwrite=False):
    field = pdb.Field(field_id, "R")
    
    indices = ["eta", "j", "delta_chi_squared", "sigma_mu", "k"]
    number_of_microlensing_light_curves = 1000
    number_of_microlensing_simulations_per_light_curve = 100
    min_number_of_good_observations = 100
    
    # Convenience variables for filenames
    file_base = "field{:06d}_Nperccd{}_Nevents{}".format(field.id, number_of_microlensing_light_curves, number_of_microlensing_simulations_per_light_curve) + ".{ext}"
    pickle_filename = os.path.join("data", "var_indices", file_base.format(ext="pickle"))
    plot_filename = os.path.join("plots", "var_indices", file_base.format(ext="pdf"))
    
    if not os.path.exists(os.path.dirname(pickle_filename)):
        os.mkdir(os.path.dirname(pickle_filename))
    
    if not os.path.exists(os.path.dirname(plot_filename)):
        os.mkdir(os.path.dirname(plot_filename))
    
    if os.path.exists(pickle_filename) and overwrite:
        logger.debug("Data file exists, but you want to overwrite it!")
        os.remove(pickle_filename)
        logger.debug("Data file deleted...")
    
    # If the cache pickle file doesn't exist, generate the data
    if not os.path.exists(pickle_filename):
        logger.info("Data file {} not found. Generating data...".format(pickle_filename))
        
        # Initialize my PDB statistic dictionary
        # I use a dictionary here because after doing some sub-selection the index arrays may 
        #   have difference lengths.
        pdb_statistics = dict()
        for index in indices:
            pdb_statistics[index] = np.array([])
            
        for ccd in field.ccds.values():            
            print "Starting with CCD {}".format(ccd.id)
            chip = ccd.read()
            
            pdb_statistics_array = []
                        
            logger.info("Starting microlensing event simulations")
            # Keep track of how many light curves we've used, break after we reach the specified number
            light_curve_count = 0            
            for source in chip.sources.where("(ngoodobs > {})".format(min_number_of_good_observations)):
                source_id = source["matchedSourceID"]
                
                light_curve = ccd.light_curve(source_id, barebones=True, clean=True)
                if len(light_curve.mjd) < min_number_of_good_observations: 
                    continue
                
                # Add the pre-simulation statistics to an array
                lc_var_indices = pa.compute_variability_indices(light_curve, indices, return_tuple=True)
                pdb_statistics_array.append(lc_var_indices)
                
                one_light_curve_statistics = vi.simulate_events_compute_indices(light_curve, events_per_light_curve=number_of_microlensing_simulations_per_light_curve, indices=indices)
                try:
                    simulated_microlensing_statistics = np.hstack((simulated_microlensing_statistics, one_light_curve_statistics))
                except NameError:
                    simulated_microlensing_statistics = one_light_curve_statistics

                light_curve_count += 1                
                if light_curve_count >= number_of_microlensing_light_curves:
                    break
            
            pdb_statistics_array = np.array(pdb_statistics_array, dtype=[(index,float) for index in indices])
            
            try:
                all_pdb_statistics_array = np.hstack((all_pdb_statistics_array, pdb_statistics_array))
            except NameError:
                all_pdb_statistics_array = pdb_statistics_array
            
            ccd.close()
        
        f = open(pickle_filename, "w")
        pickle.dump((all_pdb_statistics_array, simulated_microlensing_statistics), f)
        f.close()       
        
    f = open(pickle_filename, "r")
    all_pdb_statistics_array, simulated_microlensing_statistics = pickle.load(f)
    f.close()
    
    index_pairs = [("eta", "delta_chi_squared"), ("eta", "j"), ("delta_chi_squared", "j")]
    
    nbins = 100
    for x_index, y_index in index_pairs:
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(20,10))

        # Variable data
        x = simulated_microlensing_statistics[x_index]
        y = simulated_microlensing_statistics[y_index]
        
        pos_x = x[(x > 0) & (y > 0)]
        pos_y = y[(x > 0) & (y > 0)]
        
        xbins_pos = np.logspace(np.log10(pos_x.min()), np.log10(pos_x.max()), nbins)
        ybins_pos = np.logspace(np.log10(pos_y.min()), np.log10(pos_y.max()), nbins)
        
        #print pos_x, pos_y, xbins_pos, ybins_pos
        H_pos, xedges_pos, yedges_pos = np.histogram2d(pos_x, pos_y, bins=[xbins_pos, ybins_pos])
        
        # Non-variable data
        x = all_pdb_statistics_array[x_index]
        y = all_pdb_statistics_array[y_index]
        
        pos_x = x[(x > 0) & (y > 0)]
        pos_y = y[(x > 0) & (y > 0)]
        
        H_pos_boring, xedges_pos, yedges_pos = np.histogram2d(pos_x, pos_y, bins=[xedges_pos, yedges_pos])
        
        ax1 = axes[0]
        #ax1.imshow(np.log10(H), interpolation="none", cmap=cm.gist_heat)
        ax1.pcolormesh(xedges_pos, yedges_pos, np.where(H_pos > 0, np.log10(H_pos), 0.).T, cmap=cm.Blues)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(xedges_pos[0], xedges_pos[-1])
        ax1.set_ylim(yedges_pos[0], yedges_pos[-1])
        
        ax1.set_xlabel(pg.index_to_label[x_index], fontsize=28)
        ax1.set_ylabel(pg.index_to_label[y_index], fontsize=28)
        
        ax2 = axes[1]
        ax2.pcolormesh(xedges_pos, yedges_pos, np.where(H_pos_boring > 0, np.log10(H_pos_boring), 0.).T, cmap=cm.Blues)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim(xedges_pos[0], xedges_pos[-1])
        ax2.set_ylim(yedges_pos[0], yedges_pos[-1])
        
        ax2.set_xlabel(pg.index_to_label[x_index], fontsize=28)
        
        fig.savefig("plots/var_indices/{}_vs_{}.pdf".format(x_index, y_index), bbox_inches="tight")
        
def variability_indices_distributions_easy(field_id=4588):
    indices = ["eta", "j", "delta_chi_squared", "sigma_mu", "k"]
    number_of_microlensing_light_curves = 1000
    number_of_microlensing_simulations_per_light_curve = 100
    min_number_of_good_observations = 100
    
    # This version just uses the PDB statistic values and the microlensing simulated values from the detection efficiency simulation
    field = pdb.Field(field_id, "R")
    
    for ccd in field.ccds.values():
        chip = ccd.read()
        sources = chip.sources.readWhere("ngoodobs > 100")
        ccd.close()
        
        idx = []
        for source in sources:
            idx.append((source["vonNeumannRatio"], source["stetsonJ"], source["chiSQ"], source["magRMS"]/source["referenceMag"], source["stetsonK"]))
        
        arr = np.array(idx, dtype=[(index,float) for index in indices])
        try:
            all_pdb_statistics_array = np.hstack((all_pdb_statistics_array, arr))
        except NameError:
            all_pdb_statistics_array = arr
    
    all_pdb_statistics_array = np.array(all_pdb_statistics_array, dtype=[(index,float) for index in indices])
    
    # Convenience variables for filenames
    file_base = "field{:06d}_Nperccd{}_Nevents{}".format(field.id, number_of_microlensing_light_curves, number_of_microlensing_simulations_per_light_curve) + ".{ext}"
    pickle_filename = os.path.join("data", "new_detection_efficiency", file_base.format(ext="pickle"))
        
    f = open(pickle_filename, "r")
    (simulated_microlensing_statistics, selection_criteria) = pickle.load(f)
    f.close()
    
    index_pairs = [("eta", "delta_chi_squared"), ("eta", "j"), ("delta_chi_squared", "j")]
    
    nbins = 100
    for x_index, y_index in index_pairs:
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(14,5))

        # Variable data
        x = simulated_microlensing_statistics[x_index]
        y = simulated_microlensing_statistics[y_index]
        
        pos_x = x[(x > 0) & (y > 0)]
        pos_y = y[(x > 0) & (y > 0)]
        
        xbins_pos = np.logspace(np.log10(pos_x.min()), np.log10(pos_x.max()), nbins)
        ybins_pos = np.logspace(np.log10(pos_y.min()), np.log10(pos_y.max()), nbins)
        
        #print pos_x, pos_y, xbins_pos, ybins_pos
        H_pos, xedges_pos, yedges_pos = np.histogram2d(pos_x, pos_y, bins=[xbins_pos, ybins_pos])
        
        # Non-variable data
        x = all_pdb_statistics_array[x_index]
        y = all_pdb_statistics_array[y_index]
        
        pos_x = x[(x > 0) & (y > 0)]
        pos_y = y[(x > 0) & (y > 0)]
        
        H_pos_boring, xedges_pos, yedges_pos = np.histogram2d(pos_x, pos_y, bins=[xedges_pos, yedges_pos])
        
        ax1 = axes[1]
        #ax1.imshow(np.log10(H), interpolation="none", cmap=cm.gist_heat)
        ax1.pcolormesh(xedges_pos, yedges_pos, np.where(H_pos > 0, np.log10(H_pos), 0.).T, cmap=cm.Blues)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(xedges_pos[0], xedges_pos[-1])
        ax1.set_ylim(yedges_pos[0], yedges_pos[-1])
        
        ax1.set_xlabel(pg.index_to_label[x_index], fontsize=28)
        ax1.set_ylabel(pg.index_to_label[y_index], fontsize=28)
        
        ax2 = axes[0]
        ax2.pcolormesh(xedges_pos, yedges_pos, np.where(H_pos_boring > 0, np.log10(H_pos_boring), 0.).T, cmap=cm.Blues)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim(xedges_pos[0], xedges_pos[-1])
        ax2.set_ylim(yedges_pos[0], yedges_pos[-1])
        
        ax2.set_xlabel(pg.index_to_label[x_index], fontsize=28)
        
        fig.savefig("plots/derp_{}_{}.png".format(x_index, y_index))

def num_observations_distribution():
    """ This figure is (top) just a histogram of all fields binned by the number of observations,
        and (bottom) binned by baseline.
    """
    datafile = "data/paper_figures/exposures_baselines.pickle"
    fig = plt.figure(figsize=(15,10))
    
    if not os.path.exists(datafile):
        R_info = survey_coverage.SurveyInfo("R")
        fields = R_info.fields(1)
        
        num_exp_baseline = []
        for field in fields:
            num_exp_baseline.append((field.number_of_exposures, field.baseline[field.baseline.keys()[0]]))
            field.close()
        
        num_exp_baseline = np.array(num_exp_baseline, dtype=[("num_exp", int), ("baseline", float)])
        f = open(datafile, "w")
        pickle.dump(num_exp_baseline, f)
        f.close()
    
    f = open(datafile, "r")
    num_exp_baseline = pickle.load(f)
    f.close()
    
    # Top panel: binned by number of observations
    ax_top = fig.add_subplot(211)
    bins = np.logspace(0, 3.5, 50)
    ax_top.hist(num_exp_baseline["num_exp"], bins=bins, color="k", histtype="step")
    ax_top.set_ylim((-2.,ax_top.get_ylim()[1]))
    ax_top.set_xscale("log")
    ax_top.set_xlabel(r"Number of $R$-band Exposures")
    
    ax_bottom = fig.add_subplot(212)
    ax_bottom.hist(num_exp_baseline["baseline"], bins=np.linspace(0, 1300, 30), color="k", histtype="step")
    ax_bottom.set_ylim((-4.,ax_bottom.get_ylim()[1]))
    ax_bottom.set_xlabel(r"Baseline ($t_{max} - t_{min}$)")
    fig.savefig("plots/derp.png")


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
    #maximum_outlier_indices_plot(100101)
    
    #variability_indices_distributions()
    #Don't use this! variability_indices_distributions_easy()
    num_observations_distribution()