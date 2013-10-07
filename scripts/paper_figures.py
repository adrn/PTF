# coding: utf-8
from __future__ import division

""" This module contains routines for generating figures for the paper that don't really
    belong anywhere else
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import glob
import logging
import cPickle as pickle
import time

# Third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

# Create logger for this module
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Project
from ptf.lightcurve import PTFLightCurve, SimulatedLightCurve
import ptf.globals as pg
import ptf.util as pu
import ptf.variability_indices as vi
import ptf.db.mongodb as mongo

import scripts.event_fitter as fit

try:
    import ptf.db.photometric_database as pdb
    import ptf.analyze as pa
    import survey_coverage
except ImportError:
    logger.warning("photometric database modules failed to load! If this is on Navtara, you made a boo-boo.")

tick_font_size = 14

ptfid_to_longname = {"PTFS1203am" : "PTF1J033545.80-041849.1",
                     "PTFS1206i" : "PTF1J061800.25+203142.5",
                     "PTFS1212ak" : "PTF1J120642.60-192016.4", 
                     "PTFS1213ap" : "PTF1J131500.09+715032.5",
                     "PTFS1215aj" : "PTF1J153202.91+674825.1",
                     "PTFS1216bt" : "PTF1J161502.39+540053.8",
                     "PTFS1217ce" : "PTF1J171606.82+474423.7",
                     "PTFS1217cv" : "PTF1J172826.08+692501.1",
                     "PTFS1217df" : "PTF1J173301.07+374311.9",
                     "PTFS1217dk" : "PTF1J174736.51+300506.9",
                     "PTFS1219ad" : "PTF1J193330.11+451501.1"}

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

    fig = plt.figure(figsize=(12,8))
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
        light_curve = ccd.light_curve(source_id, clean=False, barebones=True)#, where=["(mjd >= {})".format(min_mjd), "(mjd <= {})".format(max_mjd)])

        ax.plot(light_curve.mjd, [ii]*len(light_curve.mjd), color="black", alpha=0.3, marker="o", markersize=7, linestyle="none", markeredgecolor="none")
        ax.text(55300, ii+0.1, "{} observations".format(len(light_curve.mjd)), size=14)
        ccd.close()

    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_yticks([])

    ax.set_xticks(np.linspace(min_mjd, max_mjd, 10, endpoint=True))
    ax.set_xticklabels(["{}".format(int(x)) for x in np.linspace(0, 365, 10, endpoint=True)])

    ax.set_xlim(min_mjd, max_mjd)
    ax.set_ylim(-0.5, ii+0.5)

    ax.set_xlabel("Time [days]", fontsize=20)
    fig.savefig(os.path.join(pg.plots_path, "sampling_figure.pdf"), facecolor="white", bbox_inches="tight")

def microlensing_event_sim():
    """ Create the multi-panel figure with simulated microlensing events for a single
        'typical' PTF light curve.
    """

    #field = pdb.Field(100062, "R")
    #ccd = field.ccds[1]
    #chip = ccd.read()
    #sources = chip.sources.readWhere("(ngoodobs > 300) & (vonNeumannRatio > 1.235)")
    #light_curve = ccd.light_curve(sources["matchedSourceID"][np.random.randint(0, len(sources))], clean=True)
    #print sources["matchedSourceID"]
    light_curve = pdb.get_light_curve(100062, 1, 13268, clean=True)

    num = 4
    fig, axes = plt.subplots(num,1, sharex=True, figsize=(11,15))

    sim_light_curve = SimulatedLightCurve(mjd=light_curve.mjd, mag=light_curve.mag, error=light_curve.error)

    t0 = sim_light_curve.mjd[int(len(sim_light_curve.mjd)/2)]

    kwarg_list = [None, {"u0" : 1.0, "t0" : t0, "tE" : 20},
                        {"u0" : 0.5, "t0" : t0, "tE" : 20},
                        {"u0" : 0.01, "t0" : t0, "tE" : 20}]

    args_list = [(16.66, "a)"), (16.4, "b)"), (16.0, "c)"), (12, "d)")]
    args_list2 = [16.68, 16.5, 16.2, 13]

    for ii in range(num):
        axes[ii].xaxis.set_visible(False)

        if ii != 0:
            #sim_light_curve.reset()
            sim_light_curve = SimulatedLightCurve(mjd=light_curve.mjd, mag=light_curve.mag, error=light_curve.error)
            sim_light_curve.add_microlensing_event(**kwarg_list[ii])

        sim_light_curve.plot(axes[ii], marker="o", ms=3, alpha=0.75)

        axes[ii].axhline(14.3, color='r', linestyle="--")

        if kwarg_list[ii] == None:
            u0_str = ""
        else:
            u0 = kwarg_list[ii]["u0"]
            u0_str = r"$u_0={:.2f}$".format(u0)
        #axes[ii].set_ylabel(u0_str, rotation="horizontal")

        #for tick in axes[ii].yaxis.get_major_ticks():
        #    tick.label.set_fontsize(tick_font_size)

        if ii == 0:
            [tick.set_visible(False) for jj,tick in enumerate(axes[ii].get_yticklabels()) if jj % 2 != 0]

        if ii % 2 != 0:
            axes[ii].yaxis.tick_right()
        else:
            axes[ii].yaxis.set_label_position("right")

        if ii == 0:
            axes[ii].set_ylabel(r"$R$", rotation="horizontal", fontsize=26)
            axes[ii].yaxis.set_label_position("left")

        axes[ii].text(56100, *args_list[ii], fontsize=24)
        axes[ii].text(56100, args_list2[ii], u0_str, fontsize=24)

    #fig.suptitle("PTF light curve with simulated microlensing events", fontsize=24)

    for ax in fig.axes:
        for ticklabel in ax.get_yticklabels():
            ticklabel.set_fontsize(18)

    fig.subplots_adjust(hspace=0.0, left=0.1, right=0.9)
    fig.savefig(os.path.join(pg.plots_path, "paper_figures", "simulated_events.pdf"), bbox_inches="tight", facecolor="white")

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

    fig, axes = plt.subplots(len(indices), 1, sharex=True, figsize=(15,20))

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

        #axes[ii].set_xlim(best_outlier_lightcurve.mjd.min()-2, best_outlier_lightcurve.mjd.max()+2)
        #axes[ii, 1].set_xlim(55350, 55600)
        axes[ii].set_ylabel("$R$ [mag]", fontsize=26)
        yticks = np.linspace(min(axes[ii].get_yticks()), max(axes[ii].get_yticks()), 5)[1:-1]
        axes[ii].set_yticks([round(yt,2) for yt in yticks])
        for ticklabel in axes[ii].get_yticklabels():
            ticklabel.set_fontsize(22)
        
        axes[ii].text(55900, yticks[0], "{0}({1})".format(min_max[index], pu.index_to_label(index)), fontsize=30)

    for ax in fig.axes[:-1]:
        ax.xaxis.set_visible(False)
        #ax.yaxis.set_visible(False)

    #fig.axes[-1].yaxis.set_visible(False)
    for ticklabel in fig.axes[-1].get_xticklabels():
        ticklabel.set_fontsize(22)

    xticks = fig.axes[-1].get_xticks()
    axes[-1].set_xticklabels(["{0:d}".format(int(x-min(xticks))) for x in xticks])

    axes[-1].set_xlabel("time [days]", fontsize=26)
    fig.subplots_adjust(hspace=0.0, top=0.95, bottom=0.08)
    fig.savefig(os.path.join(pg.plots_path, "paper_figures", "max_outlier_light_curves.pdf".format(field_id)))#, bbox_inches="tight")

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

    selection_criteria = {
		"eta" : 0.16167735855516213,
		"delta_chi_squared" : 1.162994709319348,
		"j" : 1.601729135628142
	}

    index_pairs = [("eta", "delta_chi_squared"), ("eta", "j"), ("delta_chi_squared", "j")]

    nbins = 100
    for x_index, y_index in index_pairs:
        fig, axes = plt.subplots(1, 2, sharey=True, figsize=(15,7.5))

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

        ax1.set_xlabel(pu.index_to_label(x_index), fontsize=28)
        ax1.axhline(10.**selection_criteria[y_index], color='r', linestyle='--')
        ax1.axvline(10.**selection_criteria[x_index], color='r', linestyle='--')

        if x_index == "eta":
            ax1.fill_between([xedges_pos[0], 10.**selection_criteria[x_index]], 10.**selection_criteria[y_index], yedges_pos[-1], facecolor='red', alpha=0.1)
        elif x_index == "delta_chi_squared":
            ax1.fill_between([10.**selection_criteria[x_index], xedges_pos[-1]], 10.**selection_criteria[y_index], yedges_pos[-1], facecolor='red', alpha=0.1)

        ax2 = axes[0]
        ax2.pcolormesh(xedges_pos, yedges_pos, np.where(H_pos_boring > 0, np.log10(H_pos_boring), 0.).T, cmap=cm.Blues)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax2.set_xlim(xedges_pos[0], xedges_pos[-1])
        ax2.set_ylim(yedges_pos[0], yedges_pos[-1])

        ax2.set_xlabel(pu.index_to_label(x_index), fontsize=28)
        ax2.set_ylabel(pu.index_to_label(y_index), fontsize=28)
        ax2.axhline(10.**selection_criteria[y_index], color='r', linestyle='--')
        ax2.axvline(10.**selection_criteria[x_index], color='r', linestyle='--')

        if x_index == "eta":
            ax2.fill_between([xedges_pos[0], 10.**selection_criteria[x_index]], 10.**selection_criteria[y_index], yedges_pos[-1], facecolor='red', alpha=0.1)
        elif x_index == "delta_chi_squared":
            ax2.fill_between([10.**selection_criteria[x_index], xedges_pos[-1]], 10.**selection_criteria[y_index], yedges_pos[-1], facecolor='red', alpha=0.1)

        for ax in fig.axes:
            for ticklabel in ax.get_xticklabels()+ax.get_yticklabels():
                ticklabel.set_fontsize(18)

        fig.savefig(os.path.join(pg.plots_path, "paper_figures", "{}_vs_{}.pdf".format(x_index, y_index)), bbox_inches="tight")

def num_observations_distribution():
    """ This figure is (top) just a histogram of all fields binned by the number of observations,
        and (bottom) binned by baseline.
    """
    datafile = os.path.join(pg.data_path, "paper_figures/exposures_baselines.pickle")
    plotfile = os.path.join(pg.plots_path, "paper_figures/num_observations_baseline.pdf")

    if not os.path.exists(os.path.split(datafile)[0]):
        os.makedirs(os.path.split(datafile)[0])

    if not os.path.exists(os.path.split(plotfile)[0]):
        os.makedirs(os.path.split(plotfile)[0])

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

    fig = plt.figure(figsize=(11,11))
    # Top panel: binned by number of observations
    gs = gridspec.GridSpec(4,4)

    ax_top = fig.add_subplot(gs[0, 0:-1])
    bins = np.logspace(0, 3.5, 50)
    ax_top.hist(num_exp_baseline["num_exp"], bins=bins, color="k", histtype="step")
    ax_top.set_ylim((1.,ax_top.get_ylim()[1]))
    ax_top.set_xscale("log")
    [tick.set_visible(False) for tick in ax_top.get_xticklabels()]

    ax_side = fig.add_subplot(gs[1:, -1])
    ax_side.hist(num_exp_baseline["baseline"], bins=np.linspace(0, 1300, 30), color="k", histtype="step", orientation="horizontal")
    ax_side.set_ylim((-10,ax_side.get_ylim()[1]))
    [tick.set_visible(False) for tick in ax_side.get_yticklabels()]
    [tick.set_visible(False) for ii,tick in enumerate(ax_side.get_xticklabels()) if ii % 2 != 0 or ii == 0]

    ax_bottom = fig.add_subplot(gs[1:,:-1])
    ax_bottom.scatter(num_exp_baseline["num_exp"], num_exp_baseline["baseline"], color='k', marker=".", alpha=0.5)
    #ax_bottom.hexbin(np.log10(num_exp_baseline["num_exp"]), num_exp_baseline["baseline"])
    """
    xbins = np.logspace(np.log10(1), np.log10(ax_top.get_xlim()[1]), 100)
    ybins = np.linspace(1, ax_side.get_ylim()[1], 100)
    H, xedges, yedges = np.histogram2d(num_exp_baseline["num_exp"], num_exp_baseline["baseline"], bins=(xbins, ybins), normed=True)
    ax_bottom.imshow(np.log10(H.T), interpolation="none", origin="lower")
    """
    #ax_bottom.set_xlim(map(np.log10, ax_top.get_xlim()))
    ax_bottom.set_xscale("log")
    ax_bottom.set_xlim(ax_top.get_xlim())
    ax_bottom.set_ylim(ax_side.get_ylim())

    ax_bottom.set_xlabel(r"Number of $R$-band Exposures", fontsize=24)
    ax_bottom.set_ylabel(r"Baseline ($t_{max} - t_{min}$)", fontsize=24)

    for ax in fig.axes:
        for ticklabel in ax.get_yticklabels()+ax.get_xticklabels():
            ticklabel.set_fontsize(18)

    fig.subplots_adjust(hspace=0.02, wspace=0.02)
    fig.suptitle("Number of Exposures vs. Baseline for all PTF fields ($R$-band)", fontsize=22)
    fig.savefig(plotfile)

def num_observations_distribution_90deg():
    """ This figure is (bottom) just a histogram of all fields binned by the number of observations,
        and (top) binned by baseline.
    """
    datafile = os.path.join(pg.data_path, "paper_figures/exposures_baselines.pickle")
    plotfile = os.path.join(pg.plots_path, "paper_figures/num_observations_baseline_90deg.pdf")

    if not os.path.exists(os.path.split(datafile)[0]):
        os.makedirs(os.path.split(datafile)[0])

    if not os.path.exists(os.path.split(plotfile)[0]):
        os.makedirs(os.path.split(plotfile)[0])

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

    fig = plt.figure(figsize=(11,11))
    # Top panel: binned by number of observations
    gs = gridspec.GridSpec(4,4)

    ax_top = fig.add_subplot(gs[0, 0:-1])
    ax_side = fig.add_subplot(gs[1:, -1])
    bins = np.logspace(0, 3.5, 50)

    ax_side.hist(num_exp_baseline["num_exp"], bins=bins, color="k", histtype="step", orientation="horizontal")
    ax_side.set_ylim((1.,ax_side.get_ylim()[1]))
    ax_side.set_yscale("log")
    ax_side.xaxis.tick_top()
    [tick.set_visible(False) for tick in ax_side.get_yticklabels()]
    ax_side.get_xticklabels()[0].set_visible(False)

    ax_top.hist(num_exp_baseline["baseline"], bins=np.linspace(0, 1300, 30), color="k", histtype="step")
    ax_top.set_ylim((-10,ax_top.get_ylim()[1]))
    ax_top.yaxis.tick_right()
    [tick.set_visible(False) for tick in ax_top.get_xticklabels()]
    [tick.set_visible(False) for ii,tick in enumerate(ax_top.get_yticklabels()) if ii % 2 != 0 or ii == 0]

    ax_bottom = fig.add_subplot(gs[1:,:-1])
    ax_bottom.scatter(num_exp_baseline["baseline"], num_exp_baseline["num_exp"], color='k', marker=".", alpha=0.5)
    #ax_bottom.hexbin(np.log10(num_exp_baseline["num_exp"]), num_exp_baseline["baseline"])
    """
    xbins = np.logspace(np.log10(1), np.log10(ax_top.get_xlim()[1]), 100)
    ybins = np.linspace(1, ax_side.get_ylim()[1], 100)
    H, xedges, yedges = np.histogram2d(num_exp_baseline["num_exp"], num_exp_baseline["baseline"], bins=(xbins, ybins), normed=True)
    ax_bottom.imshow(np.log10(H.T), interpolation="none", origin="lower")
    """
    #ax_bottom.set_xlim(map(np.log10, ax_top.get_xlim()))
    ax_bottom.set_yscale("log")
    ax_bottom.set_xlim(ax_top.get_xlim())
    ax_bottom.set_ylim(ax_side.get_ylim())

    ax_bottom.set_xlabel(r"Baseline ($t_{max} - t_{min}$) [days]", fontsize=24)
    ax_bottom.set_ylabel(r"Number of $R$-band Exposures", fontsize=24)

    for ax in fig.axes:
        for ticklabel in ax.get_yticklabels()+ax.get_xticklabels():
            ticklabel.set_fontsize(18)

    fig.subplots_adjust(hspace=0.03, wspace=0.03)
    fig.suptitle("Number of Exposures vs. Baseline for all PTF fields ($R$-band)", fontsize=22)
    fig.savefig(plotfile)

def after_eta_cut_num_observations():
    ptf = mongo.PTFConnection()
    field_collection = ptf.fields

    field_ids = [3437, 1947, 100031]

    for field_id in field_ids:
        field = pdb.Field(field_id, "R")
        print(field)
        selection_criteria = field_collection.find_one({"_id" : field.id}, fields=["selection_criteria"])["selection_criteria"]["eta"]
        eta_cut = 10**selection_criteria

        ccd = field.ccds[0]
        chip = ccd.read()

        sources = chip.sources.readWhere("(ngoodobs > {}) & \
                                          (vonNeumannRatio > 0.0) & \
                                          (vonNeumannRatio < {})".format(10, eta_cut))

        apw_ngood = []
        for source in sources:
            light_curve = ccd.light_curve(source["matchedSourceID"], barebones=True, clean=True)
            apw_ngood.append(len(light_curve))

        apw_ngood = np.array(apw_ngood)

        plt.clf()
        plt.figure(figsize=(14,10))
        plt.subplot(121)
        plt.hist(sources["ngoodobs"], bins=25, color="#3182BD", linewidth=2., histtype="step", label=r"$N_{good,PDB}$")
        plt.hist(apw_ngood, bins=25, color="#CA0020", linewidth=2., histtype="step", label=r"$N_{good,APW}$")
        plt.xlabel(r"$N_{good}$")
        plt.xlim()

        plt.subplot(122)
        plt.hist(sources["ngoodobs"]/sources["nobs"], bins=25, color="#3182BD", linewidth=2., histtype="step", label=r"$N_{good,PDB}/N_{tot}$")
        plt.hist(apw_ngood/sources["nobs"], bins=25, color="#CA0020", linewidth=2., histtype="step", label=r"$N_{good,APW}/N_{tot}$")
        plt.xlabel(r"$N_{good}/N_{tot}$")
        plt.savefig(os.path.join(pg.plots_path, "paper_figures/after_eta_cut_{0}.pdf".format(field.id)))

    return
    plt.clf()
    num = 0
    for source in sources[sources["ngoodobs"] < 25]:
        light_curve = ccd.light_curve(source["matchedSourceID"], clean=True)

        if len(light_curve) < 10:
            continue

        num += 1
        plt.clf()
        light_curve.plot()
        plt.savefig("/home/aprice-whelan/projects/ptf/plots/test_lc_{0}_{1}.png".format(field.id, source["matchedSourceID"]))

        if num >= 10: break
    sys.exit(0)

def ml_parameter_distributions(overwrite=False):
    """ Compute distributions of microlensing event parameter fits by doing my MCMC
        fit to all candidate, qso, not interesting, bad data, transient, and supernova
        tagged light curves.

        There are about ~4500

        NEW: Hmm...maybe forget this, I keep breaking navtara
    """

    ptf = mongo.PTFConnection()
    light_curve_collection = ptf.light_curves

    Nwalkers = 100
    Nsamples = 1000
    Nburn = 100

    searched = []
    max_parameters = []
    for tag in ["candidate", "qso", "not interesting", "transient", "supernova", "bad data"]:
        for lc_document in list(light_curve_collection.find({"tags" : tag})):
            if str(lc_document["_id"]) in searched and not overwrite:
                continue

            light_curve = pdb.get_light_curve(lc_document["field_id"], lc_document["ccd_id"], lc_document["source_id"], clean=True)
            sampler = fit.fit_model_to_light_curve(light_curve, nwalkers=Nwalkers, nsamples=Nsamples, nburn_in=Nburn)
            max_idx = np.ravel(sampler.lnprobability).argmax()

            # Turn this on to dump plots for each light curve
            #fit.make_chain_distribution_figure(light_curve, sampler, filename="{0}_{1}_{2}_dists.png".format(lc_document["field_id"], lc_document["ccd_id"], lc_document["source_id"]))
            #fit.make_light_curve_figure(light_curve, sampler, filename="{0}_{1}_{2}_lc.png".format(lc_document["field_id"], lc_document["ccd_id"], lc_document["source_id"]))

            max_parameters.append(list(sampler.flatchain[max_idx]))
            searched.append(str(lc_document["_id"]))

            if len(searched) == 500:
                break

        if len(searched) == 500:
            break

    max_parameters = np.array(max_parameters)
    fig, axes = plt.subplots(2, 2, figsize=(14,14))

    for ii, ax in enumerate(np.ravel(axes)):
        if ii == 3:
            bins = np.logspace(min(max_parameters[:,ii]), max(max_parameters[:,ii]), 25)
            ax.hist(max_parameters[:,ii], bins=bins, color="k", histtype="step")
        else:
            ax.hist(max_parameters[:,ii], color="k", histtype="step")
        ax.set_yscale("log")

    fig.savefig("plots/fit_events/all_parameters.png")

def lightcurve_from_filename(filename):
    import astropy.io.ascii as ascii
    
    with open(filename) as f:
        data_start = 0
        for line in f:
            if line.startswith("#"):
                data_start += 1
    
    candidate = ascii.read(filename, header_start=data_start-1)#, data_start=data_start)
    c = candidate[(candidate['filterID'] == 2) & (candidate['mag'] > 0.)]
    flag_idx = c['relPhotFlags']==0
    
    light_curve = PTFLightCurve(mjd=c['epoch'], mag=c['mag'], error=c['magerr'])
    
    return light_curve

def fit_candidates(Nburn, Nsamples, Nwalkers, seed=None):
    final_candidates = ["PTFS1206i", "PTFS1216bt", "PTFS1217cv"]
    
    fig1 = plt.figure(figsize=(32,12))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1,2])
    
    u0s,t0s,m0s,tEs = [],[],[],[]
    du0s,dt0s,dm0s,dtEs = [],[],[],[]
    
    # Make 3-panel figure with candidates
    for ii,PTFID in enumerate(final_candidates):
        if ii == 2:
            Nsamples = 1000
            
        chains = True
        filename = "data/candidates/lc_{0}.dat".format(PTFID)
        figtext = ptfid_to_longname[PTFID] #candidate["long_ptf_name"]
        light_curve = lightcurve_from_filename(filename)
        
        if ii == 0:
            y_axis_labels = True
        else:
            y_axis_labels = False
        
        if ii == 1:
            x_axis_labels = True
        else:
            x_axis_labels = False
            
        logger.info("Fitting {0}".format(PTFID))
        sampler = fit.fit_model_to_light_curve(light_curve, nwalkers=Nwalkers, nsamples=Nsamples, nburn_in=Nburn, seed=seed)
        chain = sampler.flatchain
        print "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))

        ax = plt.subplot(gs[1,ii])
        ax2 = plt.subplot(gs[0,ii])
    
        # "best" parameters
        flatchain = np.vstack(sampler.chain)
        best_m0, best_u0, best_t0, best_tE = np.median(flatchain, axis=0)
        
        mjd = np.arange(light_curve.mjd.min()-1000., light_curve.mjd.max()+1000., 0.2)
        
        first_t0 = None
        for jj in range(100):
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
            ax.plot(s_light_curve.mjd-first_t0, s_light_curve.mag, linestyle="-", color="#ababab", alpha=0.05, marker=None)
            #ax_inset.plot(s_light_curve.mjd-first_t0, s_light_curve.mag, "r-", alpha=0.05)
    
        mean_m0, mean_u0, mean_t0, mean_tE = np.median(chain,axis=0)
        std_m0, std_u0, std_t0, std_tE = np.std(chain,axis=0)
        
        m0s.append(mean_m0)
        dm0s.append(std_m0)
        t0s.append(mean_t0+54832.) # HJD to MJD
        dt0s.append(std_t0)
        tEs.append(mean_tE)
        dtEs.append(std_tE)
        u0s.append(mean_u0)
        du0s.append(std_u0)
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
        
        ylim, xlim = ax.get_ylim(), ax.get_xlim()
        dx, dy = (xlim[1]-xlim[0]), (ylim[0]-ylim[1])
        ax.text(xlim[0]+dx/20., ylim[1]+dy/8., 
                 r"$u_0=${u0:.3f}$\pm${std:.3f}".format(u0=mean_u0, std=std_u0) + 
                 "\n" + r"$t_E=${tE:.1f}$\pm${std:.1f} days".format(tE=mean_tE, std=std_tE), fontsize=22)
        
        ylim, xlim = ax2.get_ylim(), ax2.get_xlim()
        dx, dy = (xlim[1]-xlim[0]), (ylim[0]-ylim[1])
        if ii == 2:
            ax2.text((xlim[0]+xlim[1])/2.,ylim[1]+dy/8.,figtext,fontsize=22)
        else:
            ax2.text(xlim[0]+dx/20., ylim[1]+dy/8., figtext,fontsize=22)
        
        if y_axis_labels:
            ax.set_ylabel(r"$R$ (mag)", fontsize=24)
        if x_axis_labels:
            ax.set_xlabel(r"time-$t_0$ [days]", fontsize=24)
    
        # Now plot the "best fit" line in red
        s_light_curve = SimulatedLightCurve(mjd=mjd, error=np.zeros_like(mjd), mag=best_m0)
        s_light_curve.add_microlensing_event(t0=best_t0, u0=best_u0, tE=best_tE)
        ax.plot(s_light_curve.mjd-best_t0, s_light_curve.mag, "k-", alpha=0.6, linewidth=3)
    
        if y_axis_labels:
            ax2.set_ylabel(r"$R$ (mag)", fontsize=24)
    
    import astropy.table as at
    from astropy.io import ascii
    tbl = at.Table([at.Column(data=u0s, name="u0"),
                    at.Column(data=du0s, name="e_u0"),
                    at.Column(data=tEs, name="tE"),
                    at.Column(data=dtEs, name="e_tE"),
                    at.Column(data=t0s, name="t0"),
                    at.Column(data=dt0s, name="e_t0"),
                    at.Column(data=m0s, name="m0"),
                    at.Column(data=dm0s, name="e_m0")])
    
    ascii.write(tbl, output=os.path.join("plots/fit_events/metadata.tex"), Writer=ascii.Latex)
    
    fig1.subplots_adjust(hspace=0.1, wspace=0.12, left=0.04, right=0.98, top=0.98, bottom=0.07)
    fig1.savefig(os.path.join("plots/fit_events", "fig10.eps"))     
    
    # ----------------------------------------------------------------

def fit_non_candidates(Nburn, Nsamples, Nwalkers):
    # Now, the other ones...
    
    
    non_candidates = [("PTFS1203am", "PTFS1212ak", "PTFS1213ap"), 
                      ("PTFS1215aj","PTFS1217ce", "PTFS1217df"), 
                      ("PTFS1217dk", "PTFS1219ad")]
    
    for kk,row in enumerate(non_candidates):
        fig1 = plt.figure(figsize=(32,12))
        gs = gridspec.GridSpec(2, 3, height_ratios=[1,2])
        
        # Make 6-panel figure with non-candidates
        for ii,PTFID in enumerate(row):
            if ii == 1 and kk == 2:
                Nsamples = 1000
                
            filename = "data/candidates/lc_{0}.dat".format(PTFID)
            figtext = ptfid_to_longname[PTFID] #candidate["long_ptf_name"]
            
            light_curve = lightcurve_from_filename(filename)
            
            if ii == 0:
                y_axis_labels = True
            else:
                y_axis_labels = False
            
            if kk == 2 or (kk == 1 and ii == 2):
                x_axis_labels = True
            else:
                x_axis_labels = False
                
            logger.info("Fitting {0}".format(PTFID))
            sampler = fit.fit_model_to_light_curve(light_curve, nwalkers=Nwalkers, nsamples=Nsamples, nburn_in=Nburn)
            chain = sampler.flatchain
            print "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
    
            ax = plt.subplot(gs[1,ii])
            ax2 = plt.subplot(gs[0,ii])
        
            # "best" parameters
            flatchain = np.vstack(sampler.chain)
            best_m0, best_u0, best_t0, best_tE = np.median(flatchain, axis=0)
            
            mjd = np.arange(light_curve.mjd.min()-1000., light_curve.mjd.max()+1000., 0.2)
        
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
            
            ylim, xlim = ax2.get_ylim(), ax2.get_xlim()
            dx, dy = (xlim[1]-xlim[0]), (ylim[0]-ylim[1])
            if (kk == 0 and ii == 1) or (kk == 1 and ii == 0) or (kk == 2 and ii == 0):
                ax2.text((xlim[0]+xlim[1])/2.,ylim[1]+dy/8.,figtext,fontsize=22)
            else:
                ax2.text(xlim[0]+dx/20., ylim[1]+dy/8., figtext,fontsize=22)
            
            if y_axis_labels:
                ax.set_ylabel(r"$R$ (mag)", fontsize=24)
            if x_axis_labels:
                ax.set_xlabel(r"time-$t_0$ [days]", fontsize=24)
        
            if y_axis_labels:
                ax2.set_ylabel(r"$R$ (mag)", fontsize=24)
            
        fig1.subplots_adjust(hspace=0.1, wspace=0.12, left=0.04, right=0.98, top=0.98, bottom=0.07)
        #fig1.savefig(os.path.join("plots/fit_events", "non_candidates_{0}.eps".format(kk)))
        fig1.savefig(os.path.join("plots/fit_events", "fig13_{0}.eps".format(kk)))
    
    
    
    """
    for ii,filename in enumerate(glob.glob("data/candidates/*.dat")):
        basename,ext = os.path.splitext(os.path.basename(filename))
        xx, PTFID = basename.split("_")
        
        if PTFID in final_candidates:
            chains = True
        else:
            chains = False
        
        with open(filename) as f:
            data_start = 0
            for line in f:
                if line.startswith("#"):
                    data_start += 1
        
        candidate = ascii.read(filename, header_start=data_start-1)#, data_start=data_start)
        c = candidate[(candidate['filterID'] == 2) & (candidate['mag'] > 0.)]
        flag_idx = c['relPhotFlags']==0
        
        light_curve = PTFLightCurve(mjd=c['epoch'], mag=c['mag'], error=c['magerr'])
        
        if PTFID in ["PTFS1203am", "PTFS1213ap", "PTFS1217ce", "PTFS1217dk"] \
            or PTFID in final_candidates:
            y_axis_labels = True
        else:
            y_axis_labels = False
        
        if PTFID in ["PTFS1215aj", "PTFS1213ap", "PTFS1219ad", "PTFS1217dk"] \
            or PTFID in final_candidates:
            x_axis_labels = True
        else:
            x_axis_labels = False
            
        logger.info("Fitting {0}".format(PTFID))
        sampler = fit.fit_model_to_light_curve(light_curve, nwalkers=Nwalkers, nsamples=Nsamples, nburn_in=Nburn)
        print "Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction))
        
        title = ptfid_to_longname[PTFID] #candidate["long_ptf_name"]

        fit.make_chain_distribution_figure(light_curve, sampler, 
            filename="{0}_dists.png".format(PTFID),
            title=title)
        fit.make_light_curve_figure(light_curve, sampler, 
            filename="{0}_lc.png".format(PTFID),
            title=title,
            x_axis_labels=x_axis_labels, y_axis_labels=y_axis_labels,
            chains=chains)
    """
    
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

    if not os.path.exists(os.path.join(pg.plots_path)):
        os.mkdir(pg.plots_path)
        
    # TODO: better interface here!!

    #make_survey_sampling_figure(10)
    #microlensing_event_sim()
    maximum_outlier_indices_plot(100101)
    #variability_indices_distributions()
    #num_observations_distribution()
    #num_observations_distribution_90deg()
    #after_eta_cut_num_observations()
    
    #np.random.seed(400)
    #fit_candidates(Nsamples=1000, Nburn=500, Nwalkers=100)
    
    #np.random.seed(104)
    #fit_non_candidates(Nsamples=500, Nburn=500, Nwalkers=100)
