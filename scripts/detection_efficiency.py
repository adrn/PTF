""" 
    Functions to help compute the detection efficiency -- new style! The procedure should
    look like this:
        1) Select field
        2) For each CCD:
            - Get variability index distributions from PDB
                -> Convert to log distributions:
                    + eta : only values > 0
                    + sigma_mu : all values
                    + j : only values > 0
                    + k : all values
                -> Store these as a dictionary keyed by the index name
            - Run false positive rate simulation to get the Nsigma values for FPR = 0.01
                -> For each index, do a random walk to get to the range  0.08 < FPR < 0.12
            - Once I have the Nsigma values, compute detection efficiency for each index
            - Then produce u0 and tE distributions from detected events
"""
from __future__ import division

# Standard library
import copy
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
from apwlib.globals import greenText
import numpy as np
from scipy.stats import scoreatpercentile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project
from ptf.lightcurve import SimulatedLightCurve
import ptf.db.photometric_database as pdb
import ptf.analyze.analyze as analyze
#from ptf.globals import index_to_label
import ptf.util as pu
import ptf.variability_indices as vi

index_to_label = pu.index_to_label

best = False

if best:
    pdb_index_name = dict(eta="vonNeumannRatio", j="stetsonJ", k="stetsonK", delta_chi_squared="chiSQ", sigma_mu=["magRMS","referenceMag"])
else:
    pdb_index_name = dict(eta="bestVonNeumannRatio", j="bestStetsonJ", k="bestStetsonK", delta_chi_squared="bestChiSQ", sigma_mu=["bestMagRMS","referenceMag"])

def source_index_name_to_pdb_index(source, index_name):
    """ Given a source (a row from chip.sources) and an index name (e.g. eta),
        return the value of the statistic. This is particularly needed for a
        computed index like sigma_mu.
    """
    if index_name == "sigma_mu":
        return source[pdb_index_name[index_name][0]] / source[pdb_index_name[index_name][1]]            
    else:
        return source[pdb_index_name[index_name]]

def prune_index_distribution(index, index_array):
    if index == "eta":
        return np.log10(index_array[index_array > 0])
    elif index == "sigma_mu":
        return np.log10(np.fabs(index_array))
    elif index == "j":
        return np.log10(index_array[index_array > 0])
    elif index == "k":
        return np.log10(index_array)
    elif index == "delta_chi_squared":
        return np.log10(index_array[index_array > 0])
    else:
        return

def detection_efficiency_for_field(field, ccds=range(12), config=dict(), overwrite=False, indices=["eta","sigma_mu","j","k", "delta_chi_squared"], plot=True):
    """ Run a detection efficiency simulation for a PTF field """   
    
    # Get configuration variables or defaults
    min_number_of_good_observations = config.get("min_number_of_good_observations", 100)
    number_of_fpr_light_curves = config.get("number_of_fpr_light_curves", 10)
    number_of_fpr_simulations_per_light_curve = config.get("number_of_fpr_simulations_per_light_curve", 10)
    number_of_microlensing_light_curves = config.get("number_of_microlensing_light_curves", 10)
    number_of_microlensing_simulations_per_light_curve = config.get("number_of_microlensing_simulations_per_light_curve", 10)
    
    # Convenience variables for filenames
    file_base = "field{:06d}_Nperccd{}_Nevents{}".format(field.id, number_of_microlensing_light_curves, number_of_microlensing_simulations_per_light_curve) + ".{ext}"
    pickle_filename = os.path.join("data", "new_detection_efficiency", file_base.format(ext="pickle"))
    plot_filename = os.path.join("plots", "new_detection_efficiency", file_base.format(ext="pdf"))
    
    if not os.path.exists(os.path.dirname(pickle_filename)):
        os.mkdir(os.path.dirname(pickle_filename))
    
    if not os.path.exists(os.path.dirname(plot_filename)):
        os.mkdir(os.path.dirname(plot_filename))
    
    if os.path.exists(pickle_filename) and overwrite:
        logger.debug("Data file exists, but you want to overwrite it!")
        os.remove(pickle_filename)
        logger.debug("Data file deleted...")
    
    #print(pickle_filename, os.path.exists(pickle_filename))

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
            if ccd.id not in ccds: continue
            
            logger.info(greenText("Starting with CCD {}".format(ccd.id)))
            chip = ccd.read()
            
            logger.info("Getting variability statistics from photometric database")
            source_ids = []
            pdb_statistics_array = []
            for source in chip.sources.where("(ngoodobs > {})".format(min_number_of_good_observations)):
                pdb_statistics_array.append(tuple([source_index_name_to_pdb_index(source,index) for index in indices]))
                source_ids.append(source["matchedSourceID"])
            pdb_statistics_array = np.array(pdb_statistics_array, dtype=[(index,float) for index in indices])
            
            logger.debug("Selected {} statistics".format(len(pdb_statistics_array)))
            
            # I use a dictionary here because after doing some sub-selection the index arrays may 
            #   have difference lengths.
            for index in indices:
                this_index_array = pdb_statistics_array[index]
                
                # This is where I need to define the selection distributions for each index.
                pdb_statistics[index] = np.append(pdb_statistics[index], prune_index_distribution(index, this_index_array))
            
            # Randomize the order of source_ids to prune through
            np.random.shuffle(source_ids)
            
            logger.info("Simulating light curves for false positive rate calculation")
            # Keep track of how many light curves we've used, break after we reach the specified number
            light_curve_count = 0
            for source_id in source_ids:
                light_curve = ccd.light_curve(source_id, barebones=True, clean=True)
                if len(light_curve.mjd) < min_number_of_good_observations: 
                    logger.debug("\tRejected source {}".format(source_id))
                    continue
                    
                logger.debug("\tSelected source {}".format(source_id))
                these_indices = vi.simulate_light_curves_compute_indices(light_curve, num_simulated=number_of_fpr_simulations_per_light_curve, indices=indices)
                try:
                    simulated_light_curve_statistics = np.hstack((simulated_light_curve_statistics, these_indices))
                except NameError:
                    simulated_light_curve_statistics = these_indices
                    
                light_curve_count += 1
                
                if light_curve_count >= number_of_fpr_light_curves:
                    break
                        
            logger.info("Starting microlensing event simulations")
            # Keep track of how many light curves we've used, break after we reach the specified number
            light_curve_count = 0            
            for source_id in source_ids:
                light_curve = ccd.light_curve(source_id, barebones=True, clean=True)
                if len(light_curve.mjd) < min_number_of_good_observations: 
                    logger.debug("\tRejected source {}".format(source_id))
                    continue
                
                logger.debug("\tSelected source {}".format(source_id))
                one_light_curve_statistics = vi.simulate_events_compute_indices(light_curve, events_per_light_curve=number_of_microlensing_simulations_per_light_curve, indices=indices)
                try:
                    simulated_microlensing_statistics = np.hstack((simulated_microlensing_statistics, one_light_curve_statistics))
                except NameError:
                    simulated_microlensing_statistics = one_light_curve_statistics

                light_curve_count += 1                
                if light_curve_count >= number_of_microlensing_light_curves:
                    break
            
            ccd.close()
        
        logger.info("Starting false positive rate calculation to get Nsigmas")
        # Now determine the N in N-sigma by computing the false positive rate and getting it to be ~0.01 (1%) for each index
        selection_criteria = {}
        for index in indices:
            logger.debug("\tIndex: {}".format(index))
            # Get the mean and standard deviation of the 'vanilla' distributions to select with
            mu,sigma = np.mean(pdb_statistics[index]), np.std(pdb_statistics[index])
            logger.debug("\t mu={}, sigma={}".format(mu, sigma))
            
            # Get the simulated statistics for this index
            these_statistics = np.log10(simulated_light_curve_statistics[index])
            
            # Start by selecting with Nsigma = 0
            Nsigma = 0.
            
            # Nsteps is the number of steps this routine has to take to converge -- just used for diagnostics
            Nsteps = 0
            while True:
                fpr = np.sum((these_statistics > (mu + Nsigma*sigma)) | (these_statistics < (mu - Nsigma*sigma))) / float(len(these_statistics))
                logger.debug("Step: {}, FPR: {}".format(Nsteps, fpr))
                
                # WARNING: If you don't use enough simulations, this may never converge!
                if fpr > 0.012: 
                    Nsigma += np.random.uniform(0., 0.05)
                elif fpr < 0.008:
                    Nsigma -= np.random.uniform(0., 0.05)
                else:
                    break
                
                Nsteps += 1
                
                if Nsteps > 1000:
                    logger.warn("{} didn't converge!".format(index))
                    break
                
            logger.info("{} -- Final Num. steps: {}, Final FPR: {}".format(index, Nsteps, fpr))
            logger.info("{} -- Final Nsigma={}, Nsigma*sigma={}".format(index, Nsigma, Nsigma*sigma))
            
            selection_criteria[index] = dict()
            selection_criteria[index]["upper"] = mu + Nsigma*sigma
            selection_criteria[index]["lower"] = mu - Nsigma*sigma
        
        f = open(pickle_filename, "w")
        pickle.dump((simulated_microlensing_statistics, selection_criteria), f)
        f.close()       
        
    f = open(pickle_filename, "r")
    (simulated_microlensing_statistics, selection_criteria) = pickle.load(f)
    f.close()
    
    # Now compute the detection efficiency of each index using the selection criteria from the false positive rate simulation
    selected_distributions = {}
    detection_efficiencies = {}
    for index in indices:
        #this_index_values = simulated_microlensing_statistics[index]
        this_index_values = np.log10(simulated_microlensing_statistics[index])
        
        """
        if index == "eta":
            selection = this_index_values > 0
            this_index_values = np.log10(this_index_values[selection])
        elif index == "sigma_mu":
            selection = np.ones_like(this_index_values).astype(bool)
            this_index_values = np.log10(np.fabs(this_index_values))
        elif index == "j":
            selection = this_index_values > 0
            this_index_values = np.log10(this_index_values[selection])
        elif index == "k":
            selection = np.ones_like(this_index_values).astype(bool)
            this_index_values = np.log10(this_index_values)
        elif index == "delta_chi_squared":
            selection = this_index_values > 0
            this_index_values = np.log10(this_index_values[selection])
        """
        
        selected_ml_statistics = simulated_microlensing_statistics[(this_index_values > selection_criteria[index]["upper"]) | (this_index_values < selection_criteria[index]["lower"])]
        selected_distributions[index] = selected_ml_statistics
        
        total_detection_efficiency = len(selected_ml_statistics) / float(len(simulated_microlensing_statistics[index]))
        print "{}, eff={}".format(index, total_detection_efficiency)
        detection_efficiencies[index] = total_detection_efficiency
    
    if plot:
        plot_distributions(selected_distributions, simulated_microlensing_statistics, detection_efficiencies, params=["tE", "u0", "m"], filename=plot_filename, indices=indices)
    
    return simulated_microlensing_statistics, selected_distributions

def plot_distributions(selected_distributions, simulated_microlensing_statistics, detection_efficiencies, params=["u0", "tE"], Nbins=50, filename=None, indices=None):
    """ Plot the distributions of u0 and tE for all selected events for the indices """
    
    if filename == None:
        #os.path.join(plot_path, )
        raise NotImplementedError()
    
    fig,axes = plt.subplots(len(params), len(selected_distributions.keys()), figsize=(18,12))
    param_bins = {"tE" : 10**np.linspace(0.3, 3., Nbins),
                  "u0" : np.linspace(0., 1.34, Nbins),
                  "m" : np.linspace(14.3, 21.5, Nbins)}
    
    if indices == None:
        indices = selected_distributions.keys()
    
    params_to_label = {"u0" : "$u_0$", "t0" : "$t_0$", "tE" : "$t_E$", "m" : "$m_0$"}
    for ii,param in enumerate(params):
        bins = None
        for jj,index in enumerate(indices):
            n, bins, patches = axes[jj,ii].hist(simulated_microlensing_statistics[param], bins=param_bins[param], histtype="step", color="k", linestyle="dashed")
            try:
                n, bins, patches = axes[jj,ii].hist(selected_distributions[index][param], bins=param_bins[param], histtype="step", color="r", alpha=0.7)
            except ValueError:
                logging.warn("Failed to create histogram for {}".format(index))
            
            plt.setp(axes[jj,ii].get_yticklabels(), visible=False)
            
            if ii == 0:
                axes[jj,ii].set_ylabel(index_to_label(index), fontsize=24, rotation=0)
            
            if ii == 0:
                ylim = axes[jj,ii].get_ylim()
                text_y = ylim[1] - (ylim[1]-ylim[0])/5.
                axes[jj,ii].text(3, text_y, r"Total $\varepsilon$={:.1f}%".format(detection_efficiencies[index]*100.), fontsize=18)
            
            if jj != (len(params)-1):
                plt.setp(axes[jj,ii].get_xticklabels(), visible=False)
            
            if jj == (len(params)-1):
                axes[jj,ii].set_xlabel(params_to_label[param], fontsize=24)
            
            if param == "tE":
                axes[jj,ii].set_xscale("log")
    
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    #fig.suptitle("Field: {}".format(field.id), fontsize=24)
    #fig.savefig(filename, bbox_inches="tight")
    fig.savefig(filename)

def get_best_light_curve(field, ccd_id):
    print(field.ccds)
    sys.exit(0)
    ccd = field.ccds[ccd_id]
    chip = ccd.read()
    sources = chip.sources.readWhere("ngoodobs > 100")
    obses = [s["ngoodobs"] for s in sources]
    idx = np.array(obses).argmax()
    source = sources[idx]
    light_curve = ccd.light_curve(source["matchedSourceID"], clean=True, barebones=True)
    ccd.close()    
    return light_curve

def plot_best_light_curve(field_id, ccd_id):
    field = pdb.Field(field_id, "R")
    lc = get_best_light_curve(field, ccd_id)
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    lc.plot(ax)
    #ax.yaxis.set_ticks([])
    from pylab import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter(False))
    ax.set_xlabel("MJD", fontsize=20)
    ax.set_ylabel("$R$ [mag]", fontsize=20)
    fig.savefig("plots/new_detection_efficiency/example_light_curve_f{}_ccd{}.pdf".format(field.id, ccd_id), bbox_inches="tight")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite any existing / cached files")
    parser.add_argument("-f", "--field-id", dest="field_id", default=None, type=int,
                    help="The PTF field ID to run on")
    parser.add_argument("--plot", dest="plot", action="store_true", default=False,
                    help="Make plots")
        
    parser.add_argument("--limit", dest="limit", default=100, type=int,
                    help="The number of light curves to select from each CCD in a field")
    parser.add_argument("-N", "--N", dest="N", default=100, type=int,
                    help="The number of microlensing events to be added to each light curve")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    #indices = ["eta","sigma_mu","j","k", "delta_chi_squared"]
    indices = ["eta","delta_chi_squared", "j"]
    
    config = dict()
    config["number_of_fpr_light_curves"] = args.limit
    config["number_of_fpr_simulations_per_light_curve"] = args.N
    config["number_of_microlensing_light_curves"] = args.limit*10
    config["number_of_microlensing_simulations_per_light_curve"] = args.N
    
    np.random.seed(42)
    if args.plot:
        plot_best_light_curve(args.field_id, 2)

    field = pdb.Field(int(args.field_id), filter="R")
    simulated_microlensing_statistics, selected_distributions = detection_efficiency_for_field(field, \
                                                                                               config=config, \
                                                                                               overwrite=args.overwrite, \
                                                                                               indices=indices,
                                                                                               plot=args.plot)
    
    sys.exit(0)
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(15,10))
    for ii,field_id in enumerate([4588, 100152, 3756]):
        field = pdb.Field(field_id, "R")
        lc = get_best_light_curve(field, 2)
        axes.plot(lc.mjd, [ii]*len(lc.mjd), "ko", ms=6, alpha=0.15)
        axes.text(55200, ii+0.1, str(field_id))
    axes.set_ylim(-0.2, 2.2)
    fig.savefig("plots/new_detection_efficiency/example_light_curves.png")


'''
#
# Some older stuff
#

# This function computes a binned detection efficiency by timescale
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
'''
