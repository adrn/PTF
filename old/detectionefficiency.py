""" 
    Functions to help compute a detection efficiency
"""
from __future__ import division

# Standard library
import copy
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
from apwlib.globals import greenText
import numpy as np
from scipy.stats import scoreatpercentile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project
from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
import ptf.photometricdatabase as pdb
import ptf.analyze.analyze as analyze
from ptf.globals import index_to_label
import ptf.variability_indices as vi

#from var_indices_figure import VIFigure

def compute_false_positive_rate(var_indices, var_indices_simulated, indices, Nsigmas):
    """ """
    
    false_positive_rate = dict()
    for index_name in indices:
        Nsigma = Nsigmas[index_name]
        mu, sigma = np.mean(var_indices[index_name]), np.std(var_indices[index_name])
        idx = index_to_selection_indices(index_name, var_indices_simulated, mu, sigma, Nsigma)
        logger.debug("{} : sigma={}, Nsigma={}, {}".format(index_name, sigma, Nsigma, mu+Nsigma*sigma))
        
        false_positive_rate[index_name] = len(var_indices_simulated[index_name][idx]) / len(var_indices_simulated[index_name])
    
    return false_positive_rate

def compute_detection_efficiency(var_indices, var_indices_with_events, indices, Nsigmas):
    """ """
    
    # Will be a boolean array to select out only rows where event_added == True
    events_added_idx = var_indices_with_events["event_added"]
    
    timescale_bins = np.logspace(0, 3, 100) # from 1 day to 1000 days, 100 bins
    total_counts_per_bin, tE_bin_edges = np.histogram(var_indices_with_events[events_added_idx]["tE"], bins=timescale_bins)
    
    data = dict(total_counts_per_bin=total_counts_per_bin, \
                bin_edges=tE_bin_edges)
    
    for index_name in indices:
        Nsigma = Nsigmas[index_name]
        
        mu, sigma = np.mean(var_indices[index_name]), np.std(var_indices[index_name])       
        idx = index_to_selection_indices(index_name, var_indices_with_events, mu, sigma, Nsigma)
            
        detections_per_bin, tE_bin_edges = np.histogram(var_indices_with_events[idx]["tE"], bins=timescale_bins)
        
        total_efficiency = sum(var_indices_with_events[idx & events_added_idx]["event_added"]) / sum(events_added_idx)
        #num_false_positives = len(var_indices_with_events[idx & (events_added_idx == False)])
        
        data[index_name] = dict(total_efficiency=total_efficiency, \
                                num_false_positives=num_false_positives, \
                                detections_per_bin=detections_per_bin)
    
    return data

def determine_Nsigma(selection_indices, var_indices_simulated, indices):
    """ """

    Nsigmas_for_indices = {"eta" : 1.0, "delta_chi_squared" : 0.0, "j" : 0.0, "k" : 1.0, "con" : 0.0, "corr" : 0.0, "sigma_mu" : 0.0}
    
    while True:
        false_positive_rates = compute_false_positive_rate(selection_indices, var_indices_simulated, indices, Nsigmas=Nsigmas_for_indices)
        
        should_break = True
        for index in indices:
            if false_positive_rates[index] > 0.01:
                Nsigmas_for_indices[index] += 0.01
                should_break = False
        
        if should_break:
            break
    
    logger.debug("Final FPR: {}".format(false_positive_rates))
    return Nsigmas_for_indices

def compare_detection_efficiencies_on_field(field, light_curves_per_ccd, events_per_light_curve, overwrite=False, indices=["j","k","eta","sigma_mu","delta_chi_squared"], u0s=[], limiting_mags=[]):
    """ TODO: document """   
    
    if u0s == None or len(u0s) == 0:
        u0s = [None]
    
    if limiting_mags == None or len(limiting_mags) == 0:
        limiting_mags = [14.3, 21]
    
    file_base = "field{:06d}_Nperccd{}_Nevents{}_u0_{}_m{}".format(field.id, light_curves_per_ccd, events_per_light_curve, "-".join(map(str,u0s)), "-".join(map(str,limiting_mags))) + ".{ext}"
    pickle_filename = os.path.join("data", "detectionefficiency", file_base.format(ext="pickle"))
    plot_filename = os.path.join("plots", "detectionefficiency", file_base.format(ext="png"))
    fpr_plot_filename = os.path.join("plots", "detectionefficiency", "fpr_{}".format(file_base.format(ext="png")))
    
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
        
        # Conditions for reading from the 'sources' table
        #   - Only select sources with enough good observations (>25)
        #   - Omit sources with large amplitude variability so they don't mess with our simulation
        wheres = ["(ngoodobs > 25)", "(stetsonJ < 100)", "(vonNeumannRatio > 1.0)", "(stetsonJ > 0)"]
        
        # Keep track of calculated var indices for each CCD
        var_indices = dict()
        var_indices_with_events = dict()
        
        for ccd in field.ccds.values():
            logger.info(greenText("Starting with CCD {}".format(ccd.id)))
            
            # Get the chip object for this CCD
            chip = ccd.read()
            
            for ii, limiting_mag in enumerate(limiting_mags[:-1]):
                # Define bin edges for selection on reference magnitude
                limiting_mag1 = limiting_mag
                limiting_mag2 = limiting_mags[ii+1]
                mag_key = (limiting_mag1,limiting_mag2)
                logger.info("\tMagnitude range: {:.2f} - {:.2f}".format(limiting_mag1, limiting_mag2))
                
                if not var_indices_with_events.has_key(mag_key):
                    var_indices_with_events[mag_key] = dict()
                    
                read_wheres = wheres + ["(referenceMag >= {:.3f})".format(limiting_mag1)]
                read_wheres += ["(referenceMag < {:.3f})".format(limiting_mag2)]
                
                # Read information from the 'sources' table
                source_ids = chip.sources.readWhere(" & ".join(read_wheres))["matchedSourceID"]
                
                # Randomly shuffle the sources
                np.random.shuffle(source_ids)
                logger.info("\t\tSelected {} source ids".format(len(source_ids)))
                
                dtype = zip(indices, [float]*len(indices))
                count = 0
                good_source_ids = []
                for source_id in source_ids:
                    logger.debug("\t\t\tSource ID: {}".format(source_id))
                    light_curve = ccd.light_curve(source_id, clean=True, barebones=True)
                    
                    # After my quality cut, if light curve has less than 25 observations, skip it!
                    if len(light_curve.mjd) < 25:
                        continue
                    
                    # Run simulation to compute false positive variability indices
                    logger.debug("Starting false positive simulation")
                    these_indices = simulate_light_curves_compute_indices(light_curve, num_simulated=events_per_light_curve, indices=indices)
                    try:
                        var_indices_simulated = np.hstack((var_indices_simulated, these_indices))
                    except NameError:
                        var_indices_simulated = these_indices
                    
                    these_var_indices = np.array([analyze.compute_variability_indices(light_curve, indices, return_tuple=True)], dtype=dtype)
                    try:
                        var_indices[mag_key] = np.hstack((var_indices[mag_key], these_var_indices))
                    except KeyError:
                        var_indices[mag_key] = these_var_indices
                    
                    for u0 in u0s:
                        logger.debug("Starting detection efficiency computation for u0={}".format(u0))
                        these_indices = simulate_events_compute_indices(light_curve, events_per_light_curve=events_per_light_curve, indices=indices, u0=u0)
                        try:
                            var_indices_with_events[mag_key][u0] = np.hstack((var_indices_with_events[mag_key][u0], these_indices))
                        except KeyError:
                            var_indices_with_events[mag_key][u0] = these_indices
                    
                    good_source_ids.append(source_id)
                    count += 1
                    if count >= light_curves_per_ccd: break
                
                if len(good_source_ids) == 0:
                    logger.error("No good sources selected from this CCD for mag range {:.2f}-{:.2f}!".format(limiting_mag1, limiting_mag2))
                    continue
                
                logger.info("\t\t{} good light curves selected".format(count))
                
                # HACK: This is super hacky...
                # ----------------------------------------------
                while count < light_curves_per_ccd:
                    idx = np.random.randint(len(good_source_ids))
                    source_id = good_source_ids[idx]
                    
                    logger.debug("\t\t\tSource ID: {}".format(source_id))
                    light_curve = ccd.light_curve(source_id, clean=True, barebones=True)
                    #light_curve.shuffle()
                    
                    these_var_indices = np.array([analyze.compute_variability_indices(light_curve, indices, return_tuple=True)], dtype=dtype)
                    try:
                        var_indices[mag_key] = np.hstack((var_indices[mag_key], these_var_indices))
                    except KeyError:
                        var_indices[mag_key] = these_var_indices
                    
                    for u0 in u0s:
                        these_indices = simulate_events_compute_indices(light_curve, events_per_light_curve=events_per_light_curve, indices=indices, u0=u0)
                        try:
                            var_indices_with_events[mag_key][u0] = np.hstack((var_indices_with_events[mag_key][u0], these_indices))
                        except KeyError:
                            var_indices_with_events[mag_key][u0] = these_indices
                    
                    count += 1
                # ----------------------------------------------   
                    
            ccd.close()
        
        f = open(pickle_filename, "w")
        pickle.dump((var_indices, var_indices_with_events, var_indices_simulated), f)
        f.close()
    else:
        logger.info("Data file {} already exists".format(pickle_filename))
    
    logger.info(greenText("Starting plot routine!") + "\n  Data source: {}\n  Plotting and saving to: {}".format(pickle_filename, plot_filename))
    logger.debug("\t\tReading in data file...")
    f = open(pickle_filename, "r")
    var_indices, var_indices_with_events, var_indices_simulated = pickle.load(f)
    f.close()
    logger.debug("\t\tData loaded!")
    
    # Styling for lines: J, K, eta, sigma_mu, delta_chi_squared
    line_styles = { "j" : {"lw" : 1, "ls" : "-", "color" : "r", "alpha" : 0.5}, \
                   "k" : {"lw" : 1, "ls" : "-", "color" : "g", "alpha" : 0.5}, \
                   "eta" : {"lw" : 3, "ls" : "-", "color" : "k"}, \
                   "sigma_mu" : {"lw" : 1, "ls" : "-", "color" : "b", "alpha" : 0.5}, \
                   "delta_chi_squared" : {"lw" : 3, "ls" : "--", "color" : "k"}, 
                   "con" : {"lw" : 3, "ls" : ":", "color" : "k"},
                   "corr" : {"lw" : 3, "ls" : ":", "color" : "r"}}
    
    num_u0_bins = len(u0s)
    num_mag_bins = len(limiting_mags)-1
    
    # Detection efficiency figure, axes
    eff_fig, eff_axes = plt.subplots(num_u0_bins, num_mag_bins, sharex=True, sharey=True, figsize=(25,25))
    
    for ii, limiting_mag_pair in enumerate(sorted(var_indices.keys())):
        selection_indices = var_indices[limiting_mag_pair]
        for jj, u0 in enumerate(sorted(var_indices_with_events[limiting_mag_pair].keys())):

            Nsigmas = determine_Nsigma(selection_indices, var_indices_simulated, indices)

            # Log the values of N x sigma
            for index,Nsigma in Nsigmas.items():
                logger.info("{} = {}sigma".format(index, Nsigma))
                
            data = compute_detection_efficiency(selection_indices, var_indices_with_events[limiting_mag_pair][u0], indices, Nsigmas=Nsigmas)
            
            try:
                eff_ax = eff_axes[ii, jj]
            except:
                # If only one axis
                eff_ax = eff_axes

            for index_name in indices:
                eff_ax.semilogx((data["bin_edges"][1:]+data["bin_edges"][:-1])/2, \
                             data[index_name]["detections_per_bin"] / data["total_counts_per_bin"], \
                             label=r"{}".format(index_to_label[index_name]), \
                             **line_styles[index_name])
                #label=r"{}: $\varepsilon$={:.3f}, $F$={:.1f}%".format(index_to_label[index_name], data[index_name]["total_efficiency"], data[index_name]["num_false_positives"]/(11.*events_per_light_curve*light_curves_per_ccd)*100), \
            
            # Fix the y range and modify the tick lines
            eff_ax.set_ylim(0., 1.0)
            eff_ax.tick_params(which='major', length=10, width=2)
            eff_ax.tick_params(which='minor', length=5, width=1)
            
            if ii == 0:
                try: eff_ax.set_title("$u_0$={:.2f}".format(u0), size=36, y=1.1)
                except: pass
                
            if jj == (num_mag_bins-1):
                eff_ax.set_ylabel("{:.1f}<R<{:.1f}".format(*limiting_mag_pair), size=32, rotation="horizontal")
                eff_ax.yaxis.set_label_position("right")
            
            plt.setp(eff_ax.get_xticklabels(), visible=False)
            plt.setp(eff_ax.get_yticklabels(), visible=False)
            
            if jj == 0:
                plt.setp(eff_ax.get_yticklabels(), visible=True, size=24)
                eff_ax.set_ylabel(r"$\varepsilon$", size=38)
            if ii == (num_u0_bins-1):
                eff_ax.set_xlabel(r"$t_E$ [days]", size=34)
                plt.setp(eff_ax.get_xticklabels(), visible=True, size=24)
            
            if jj == (num_mag_bins-1) and ii == (num_u0_bins-1):
                legend = eff_ax.legend(loc="upper right")
                legend_text  = legend.get_texts()
                plt.setp(legend_text, fontsize=36)
    
    eff_fig.subplots_adjust(hspace=0.1, wspace=0.1, right=0.88)
    logger.debug("Saving figure and cleaning up!")
    eff_fig.savefig(plot_filename) 
    #fpr_fig.savefig(fpr_plot_filename)
    
    vifigure = VIFigure(indices=indices, figsize=(22,22))
    vifigure.scatter(var_indices_simulated, alpha=0.1)
    #vifigure.contour(var_indices, nbins=50)
    vifigure.beautify()
    plot_path = os.path.join("plots", "var_indices")
    vifigure.save(os.path.join(plot_path, "field{}_Nperccd{}_Nevents{}.png".format(field.id, light_curves_per_ccd, events_per_light_curve)))

def example_light_curves(field, u0s=[], limiting_mags=[]):
    """ Given a field, a list of microlensing event impact parameters,
        and a list of limiting magnitudes, generate a grid of sample light 
        curves with simulated microlensing events.
    """
    
    num_u0_bins = len(u0s)
    num_mag_bins = len(limiting_mags)-1
    fig, axes = plt.subplots(num_u0_bins, num_mag_bins, figsize=(30,30))
    
    file_base = "sample_light_curves_field{:06d}_u0_{}_m{}".format(field.id, "-".join(map(str,u0s)), "-".join(map(str,limiting_mags))) + ".{ext}"
    plot_filename = os.path.join("plots", "detectionefficiency", file_base.format(ext="png"))
    
    all_ylims = []
    for ii, limiting_mag1 in enumerate(limiting_mags[:-1]):
        limiting_mag2 = limiting_mags[ii+1]
        
        # Hack to shrink the range of magnitudes to select from
        diff = limiting_mag2 - limiting_mag1
        limiting_mag1 += diff/2.
        limiting_mag2 -= diff/4.
        
        ylims = []
        light_curve = None
        for jj, u0 in enumerate(u0s):            
            logger.debug("limiting_mags = {:.2f},{:.2f}, u0 = {:.2f}".format(limiting_mag1,limiting_mag2,u0))
            # Get the axis object for these indices in the grid
            ax = axes[ii, jj]
            
            if light_curve == None:
                # Get a random CCD
                ccd_key = field.ccds.keys()[np.random.randint(len(field.ccds))]
                ccd = field.ccds[ccd_key]
                
                # Get the chip object for this CCD
                chip = ccd.read()
                
                # Read information from the 'sources' table
                read_wheres = ["(ngoodobs > 100)", "(stetsonJ < 100)", "(stetsonJ > 0)", "(vonNeumannRatio > 1.0)", "(referenceMag < {:.3f})".format(limiting_mag2)]
                sources = chip.sources.readWhere(" & ".join(read_wheres))
                source_ids = sources["matchedSourceID"]
                
                counter = 0
                while True:
                    idx = np.random.randint(len(source_ids))
                    source_id = source_ids[idx]
                    light_curve = ccd.light_curve(source_id, clean=True, barebones=True)
                    counter += 1
                    
                    if len(light_curve.mjd) > 100:
                        break
                    elif counter >= len(source_ids):
                        logger.error("Light curve not found!")
                        sys.exit(0)           
            
            sim_light_curve = SimulatedLightCurve(light_curve.mjd, light_curve.mag, light_curve.error)
            
            #for tE in [1.0, 10., 100.]:
            for tE in [20.]:
                sim_light_curve.reset()
                sim_light_curve.addMicrolensingEvent(t0=np.mean(sim_light_curve.mjd), u0=u0, tE=tE)
                sim_light_curve.plot(ax=ax)
                
            ylims.append(ax.get_ylim())
            
            if ii == 0:
                ax.set_title("$u_0$={:.2f}".format(u0), size=36, y=1.1)
                
            if jj == (num_mag_bins-1):
                ax.set_ylabel("R={:.2f}".format(float(sources[idx]["referenceMag"])), size=36, rotation="horizontal")
                ax.yaxis.set_label_position("right")
            
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            
            if jj == 0:
                plt.setp(ax.get_yticklabels(), visible=True, size=24)
                ax.set_ylabel(r"$R$", size=38)
            if ii == (num_u0_bins-1):
                ax.set_xlabel(r"Time", size=34)
                #plt.setp(ax.get_xticklabels(), visible=True, size=24)
        
        all_ylims.append(ylims)
    
    for ii,row in enumerate(all_ylims):
        bottoms = [x[0] for x in row]
        tops = [y[1] for y in row]
        
        for jj in range(len(u0s)):
            axes[ii,jj].set_ylim(max(bottoms), min(tops))
    
    fig.subplots_adjust(hspace=0.1, wspace=0.1, right=0.88)
    fig.savefig(plot_filename)
    
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite any existing / cached files")
    parser.add_argument("-N", "--N", dest="N", default=100, type=int,
                    help="The number of microlensing events to be added to each light curve")
    parser.add_argument("-f", "--field-id", dest="field_id", default=None, type=int,
                    help="The PTF field ID to run on")
    parser.add_argument("--limit", dest="limit", default=None, type=int,
                    help="The number of light curves to select from each CCD in a field")
    parser.add_argument("--test", dest="test", action="store_true", default=False,
                    help="Run tests")
    parser.add_argument("--u0", dest="u0", nargs="+", type=float, default=None,
                    help="Only add microlensing events with the specified impact parameter.")
    parser.add_argument("--mag", dest="limiting_mag", nargs="+", type=float, default=None,
                    help="Specify the magnitude bin edges, e.g. 6 bin edges specifies 5 bins.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    if args.test:
        np.random.seed(42)
        
        print "\n\n\n"
        greenText("/// Tests Complete! ///")
        sys.exit(0)
    
    np.random.seed(42)
    field = pdb.Field(args.field_id, filter="R")
    compare_detection_efficiencies_on_field(field, indices=["eta","delta_chi_squared", "con", "j","k","sigma_mu"], \
                                            events_per_light_curve=args.N,
                                            light_curves_per_ccd=args.limit,
                                            overwrite=args.overwrite,
                                            u0s=args.u0,
                                            limiting_mags=args.limiting_mag)
    #example_light_curves(field, u0s=args.u0, limiting_mags=args.limiting_mag)
