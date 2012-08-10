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

# Create logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def detection_efficiency(statistic, indices_with_events, lower_cut=None, upper_cut=None):
    """ Compute the detection efficiency of a specific variability statistic.
        
        This function computes the detection efficiency by accepting a column
        of variability statistic values, and upper and lower bounds, and computes
        the fraction of events detected. 
    """
    
    and_it = False
    if lower_cut != None:
        lower_idx = indices_with_events[statistic] < lower_cut
    else:
        lower_idx = np.array([True]*len(indices_with_events))
        and_it = True
    
    if upper_cut != None:
        upper_idx = indices_with_events[statistic] > upper_cut
    else:
        upper_idx = np.array([True]*len(indices_with_events))
        and_it = True
    
    if and_it:
        idx = lower_idx & upper_idx
    else:
        idx = lower_idx | upper_idx
        
    selected = indices_with_events[idx]
    
    efficiency = sum(selected["event_added"] == True) / sum(indices_with_events["event_added"])
    N_false_positives = sum(selected["event_added"] == False)
    
    return (efficiency, N_false_positives)

def simulate_events_compute_indices(light_curves, events_per_light_curve=100, indices=["j","k","eta","sigma_mu","delta_chi_squared"], u0=None):
    """ """
    var_indices_with_events = []
    
    # For each light curve, add [100] different microlensing events and recompute indices
    for light_curve in (SimulatedLightCurve(lc.mjd, mag=lc.mag, error=lc.error) for lc in light_curves):
        reference_mag = np.median(light_curve.mag)
        for event_id in range(events_per_light_curve):
            # Reset the simulated light curve back to the original data, e.g. erase any previously
            #   added microlensing events
            light_curve.reset()
            event_added = False
            
            # Only add events 50% of the time, to allow for false positives
            if np.random.uniform() > 0.5:
                tE = 10**np.random.uniform(0., 3.)
                light_curve.addMicrolensingEvent(tE=tE, u0=u0)
                event_added = True
            
            try:
                lc_var_indices = analyze.compute_variability_indices(light_curve, indices, return_tuple=True)
            except:
                break
            var_indices_with_events.append(lc_var_indices + (light_curve.tE, light_curve.u0, reference_mag, event_added))
    
    names = indices + ["tE", "u0", "m", "event_added"]
    dtypes = [float]*len(indices) + [float,float,float,bool]
    var_indices_with_events = np.array(var_indices_with_events, dtype=zip(names,dtypes))
    
    return var_indices_with_events
    
def compare_detection_efficiencies(light_curves, events_per_light_curve=100, indices=["j","k","eta","sigma_mu","delta_chi_squared"], u0=None):
    """ This figure should show the detection efficiency curve for all indices by
        injecting events into a random sampling of light_curves.
        
            1) Given a list of light curves, compute the variability indices for these "vanilla" light curves
            2) For each light curve, add 'events_per_light_curve' different events to each light curve, and 
                recompute the indices. Then use the selection criteria from the original light curves to define
                the detection efficiency and false positive rate for each index as a function of event timescale.
                
    """
    
    logger.debug(greenText("/// compare_detection_efficiencies ///"))
    logger.debug("Computing indices {} for {} light curves".format(",".join(indices), len(light_curves)))
    
    var_list = []
    for light_curve in light_curves:
        try:
            var_list.append(analyze.compute_variability_indices(light_curve, indices, return_tuple=True))
        except:
            pass
            
    selection_var_indices = np.array(var_list, dtype=zip(indices, [float]*len(indices)))
    simulated_var_indices = simulate_events_compute_indices(light_curves, events_per_light_curve, indices=indices, u0=u0)
    
    timescale_bins = np.logspace(0, 3, 100) # from 1 day to 1000 days
    total_counts_per_bin, tE_bin_edges = np.histogram(simulated_var_indices[simulated_var_indices["event_added"]]["tE"], bins=timescale_bins)
    
    data = dict(total_counts_per_bin=total_counts_per_bin, \
                bin_edges=tE_bin_edges)
    for index_name in indices:
        mu, sigma = np.mean(selection_var_indices[index_name]), np.std(selection_var_indices[index_name])
        idx = (simulated_var_indices[index_name] > (mu+2.*sigma)) | (simulated_var_indices[index_name] < (mu-2.*sigma))
        detections_per_bin, tE_bin_edges = np.histogram(simulated_var_indices[idx]["tE"], bins=timescale_bins)
        
        total_efficiency = sum(simulated_var_indices[idx & simulated_var_indices["event_added"]]["event_added"]) / sum(simulated_var_indices["event_added"])
        num_false_positives = len(simulated_var_indices[idx & (simulated_var_indices["event_added"] == False)])
        
        data[index_name] = dict(total_efficiency=total_efficiency, \
                                num_false_positives=num_false_positives, \
                                detections_per_bin=detections_per_bin)
    
    return data

def compare_detection_efficiencies_on_field(field, light_curves_per_ccd, events_per_light_curve, overwrite=False, indices=["j","k","eta","sigma_mu","delta_chi_squared"], u0s=[], limiting_mags=[]):
    """ Compare the detection efficiencies of various variability statistics using the
        light curves from an entire PTF field.
    """
    if not isinstance(field, pdb.Field):
        raise ValueError("field parameter must be a Field() object!")
    
    logger.debug(greenText("/// compare_detection_efficiencies_on_field ///"))
    logger.debug("Computing indices {} for {} light curves on each CCD of field {}".format(",".join(indices), light_curves_per_ccd, field))
    
    file_base = "field{field_id:06d}_Nperccd{lc_per_ccd}_Nevents{events_per_lc}"
    
    # If u0 is specified, only add microlensing events with this impact parameter, but vary other parameters
    if len(u0s) > 0:
        file_base += "_u0{u0:.2f}"
    
    # If limiting_mag is specified, only select light curves brighter than this limiting magnitude
    if len(limiting_mags) > 0:
        file_base += "_m{limiting_mag[0]:.1f}-{limiting_mag[1]:.1f}"
    
    for ii,limiting_mag in enumerate(limiting_mags[:-1]):
        limiting_mag_1 = limiting_mag
        limiting_mag_2 = limiting_mags[ii+1]
        
        logger.debug("Starting m={:.2f}-{:.2f}".format(limiting_mag1,limiting_mag2))
        # For a given limiting magnitude, we can use the same light curves we selected over and over, since that's a bottleneck
        light_curves = []
        for u0 in u0s:
            logger.debug("Starting u0={:.2f}".format(u0))
            formatted_file_base = file_base.format(field_id=field.id, lc_per_ccd=light_curves_per_ccd, events_per_lc=events_per_light_curve, u0=u0, limiting_mag=(limiting_mag1,limiting_mag2))
            
            if overwrite and os.path.exists("data/detectionefficiency/{}.pickle".format(formatted_file_base)):
                logger.debug("File {} already exists, and you want to overwrite it -- so I'm going to remove it, fine!".format("data/detectionefficiency/{}.pickle".format(formatted_file_base)))
                os.remove("data/detectionefficiency/{}.pickle".format(formatted_file_base))
            
            if not os.path.exists("data/detectionefficiency/{}.pickle".format(formatted_file_base)):
                logger.debug("File {} doesn't exist...generating data...".format("data/detectionefficiency/{}.pickle".format(formatted_file_base)))
                
                if len(light_curves) == 0:
                    for ccdid, ccd in field.ccds.items():
                        light_curve_batch = []
                        
                        chip = ccd.read()
                        # Read in all of the source IDs for this chip
                        if limiting_mag != None:
                            sources = chip.sources.readWhere("(referenceMag >= {:.3f}) & (referenceMag < {:.3f}) & (ngoodobs > 25)".format(limiting_mag1, limiting_mag2)) #["matchedSourceID"]
                        else:
                            sources = chip.sources[:] #.col("matchedSourceID")
                        
                        # Shuffle them about / randomize the order
                        np.random.shuffle(sources)
                        
                        for source in sources:
                            # Get a LightCurve object for this source id on this ccd
                            lc = field.ccds[0].light_curve(source["matchedSourceID"], clean=True)
                            
                            # If the light curve has more than 25 observations, include it
                            if len(lc.mjd) > 25: light_curve_batch.append(lc)
                            if len(light_curve_batch) >= light_curves_per_ccd: break
                        
                        logger.debug("Adding another batch of {} light curves...".format(len(light_curve_batch)))
                        light_curves += light_curve_batch
                
                    logger.debug("All {} light curves loaded, about to run simulation...".format(len(light_curves)))
                else:
                    logger.debug("Light already curves loaded, starting simulation...")
                
                data = compare_detection_efficiencies(light_curves, events_per_light_curve=events_per_light_curve, indices=indices, u0=u0)
                
                logger.debug("Done running detection efficiency simulation, writing to file...")
                f = open("data/detectionefficiency/{}.pickle".format(formatted_file_base), "w")
                pickle.dump(data, f)
                f.close()
    
    # do plotting
    
    # Styling for lines: J, K, eta, sigma_mu, delta_chi_squared
    line_styles = { "j" : {"lw" : 1, "ls" : "-", "color" : "r", "alpha" : 0.5}, \
                   "k" : {"lw" : 1, "ls" : "-", "color" : "g", "alpha" : 0.5}, \
                   "eta" : {"lw" : 3, "ls" : "-", "color" : "k"}, \
                   "sigma_mu" : {"lw" : 1, "ls" : "-", "color" : "b", "alpha" : 0.5}, \
                   "delta_chi_squared" : {"lw" : 3, "ls" : "--", "color" : "k"}}
    
    
    #fig = plt.figure(figsize=(30,30))
    fig, axes = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(30,30))
    
    for ii, limiting_mag in enumerate(limiting_mags):
        for jj, u0 in enumerate(u0s):
            formatted_file_base = file_base.format(field_id=field.id, lc_per_ccd=light_curves_per_ccd, events_per_lc=events_per_light_curve, u0=u0, limiting_mag=limiting_mag)
            
            logger.debug("Opening existing data file...")
            f = open("data/detectionefficiency/{}.pickle".format(formatted_file_base), "r")
            data = pickle.load(f)
            f.close()
            
            ax = axes[ii, jj]
            
            for index_name in ["eta", "delta_chi_squared", "j", "k", "sigma_mu"]:
                ax.semilogx((data["bin_edges"][1:]+data["bin_edges"][:-1])/2, \
                             data[index_name]["detections_per_bin"] / data["total_counts_per_bin"], \
                             label=r"{}".format(index_to_label[index_name]), \
                             **line_styles[index_name])
                #label=r"{}: $\varepsilon$={:.3f}, $F$={:.1f}%".format(index_to_label[index_name], data[index_name]["total_efficiency"], data[index_name]["num_false_positives"]/(11.*events_per_light_curve*light_curves_per_ccd)*100), \
            
            ax.set_ylim(0., 1.0)
            
            if ii == 0:
                ax.set_title("$u_0$={:.2f}".format(u0), size=36, y=1.1)
                
            if jj == 4:
                ax.set_ylabel("{:.1f}<M<{:.1f}".format(limiting_mag-1.5, limiting_mag), size=36, rotation="horizontal")
                ax.yaxis.set_label_position("right")
            
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            if jj == 0 and ii == 4:
                ax.set_xlabel(r"$t_E$", size=34)
                plt.setp(ax.get_xticklabels(), visible=True, size=24)
                plt.setp(ax.get_yticklabels(), visible=True, size=24)
            
            if jj == 4 and ii == 4:
                legend = ax.legend(loc="upper right")
                legend_text  = legend.get_texts()
                plt.setp(legend_text, fontsize=36)
    
    fig.subplots_adjust(hspace=0.04, wspace=0.04)
    logger.debug("Saving figure and cleaning up!")
    fig.savefig("data/detectionefficiency/field{field_id:06d}_Nperccd{lc_per_ccd}_Nevents{events_per_lc}.png".format(field_id=field.id, lc_per_ccd=light_curves_per_ccd, events_per_lc=events_per_light_curve))

def test_compare_detection_efficiencies():
    field = pdb.Field(110002, filter="R")
    chip = field.ccds[0].read()
    source_ids = chip.sources.col("matchedSourceID")
    np.random.shuffle(source_ids)
    
    light_curves = []
    for sid in source_ids:
        lc = field.ccds[0].light_curve(sid)
        
        if len(lc.mjd) > 25:
            light_curves.append(lc)
        
        if len(light_curves) == 100:
            break
    
    logger.debug("Read in {} light curves".format(len(light_curves)))
    field.ccds[0].close()
    
    data = compare_detection_efficiencies(light_curves)
    print data.keys()
       
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
        test_compare_detection_efficiencies()
        
        print "\n\n\n"
        greenText("/// Tests Complete! ///")
        sys.exit(0)
    
    field = pdb.Field(args.field_id, filter="R")
    compare_detection_efficiencies_on_field(field, events_per_light_curve=args.N, light_curves_per_ccd=args.limit, overwrite=args.overwrite, u0s=args.u0, limiting_mags=args.limiting_mag)