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
    
    # For each light curve, add 100 different microlensing events and recompute indices
    for light_curve in (SimulatedLightCurve(lc.mjd, mag=lc.mag, error=lc.error) for lc in light_curves):
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
            var_indices_with_events.append(lc_var_indices + (light_curve.tE, event_added))
    
    names = indices + ["tE", "event_added"]
    dtypes = [float]*len(indices) + [float,bool]
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

def compare_detection_efficiencies_on_field(field, light_curves_per_ccd, events_per_light_curve, overwrite=False, indices=["j","k","eta","sigma_mu","delta_chi_squared"], u0=None):
    """ Compare the detection efficiencies of various variability statistics using the
        light curves from an entire PTF field.
    """
    if not isinstance(field, pdb.Field):
        raise ValueError("field parameter must be a Field() object!")
    
    logger.debug(greenText("/// compare_detection_efficiencies_on_field ///"))
    logger.debug("Computing indices {} for {} light curves on each CCD of field {}".format(",".join(indices), light_curves_per_ccd, field))
    
    if u0 == None:
        file_base = "field{:06d}_Nperccd{}_Nevents{}".format(field.id, light_curves_per_ccd, events_per_light_curve)
    else:
        file_base = "field{:06d}_Nperccd{}_Nevents{}_u0{:.3f}".format(field.id, light_curves_per_ccd, events_per_light_curve, u0)
    
    if overwrite and os.path.exists("data/detectionefficiency/{}.pickle".format(file_base)):
        logger.debug("File {} already exists, and you want to overwrite it -- so I'm going to remove it, fine!".format("data/detectionefficiency/{}.pickle".format(file_base)))
        os.remove("data/detectionefficiency/{}.pickle".format(file_base))
    
    if not os.path.exists("data/detectionefficiency/{}.pickle".format(file_base)):
        logger.debug("File {} doesn't exist...generating data...".format("data/detectionefficiency/{}.pickle".format(file_base)))
        
        light_curves = []
        for ccdid, ccd in field.ccds.items():
            light_curve_batch = []
            
            chip = ccd.read()
            # Read in all of the source IDs for this chip
            source_ids = chip.sources.col("matchedSourceID")
            
            # Shuffle them about / randomize the order
            np.random.shuffle(source_ids)
            
            for sid in source_ids:
                # Get a LightCurve object for this source id on this ccd
                lc = field.ccds[0].light_curve(sid, clean=True)
                
                # If the light curve has more than 25 observations, include it
                # HACK
                if len(lc.mjd) > 25: light_curve_batch.append(lc)
                if len(light_curve_batch) >= light_curves_per_ccd: break
            
            logger.debug("Adding another batch of {} light curves...".format(len(light_curve_batch)))
            light_curves += light_curve_batch
            ccd.close()
        
        logger.debug("All {} light curves loaded, about to run simulation...".format(len(light_curves)))
        data = compare_detection_efficiencies(light_curves, events_per_light_curve=events_per_light_curve, indices=indices, u0=u0)
        
        logger.debug("Done running detection efficiency simulation, writing to file...")
        f = open("data/detectionefficiency/{}.pickle".format(file_base), "w")
        pickle.dump(data, f)
        f.close()
    
    logger.debug("Opening existing data file...")
    f = open("data/detectionefficiency/{}.pickle".format(file_base), "r")
    data = pickle.load(f)
    f.close()
    
    # do plotting
    fig1 = plt.figure(figsize=(20,15))
    ax1 = fig1.add_subplot(111)
    
    # Styling for lines: J, K, eta, sigma_mu, delta_chi_squared
    line_styles = [{"lw" : 2, "ls" : "-.", "color" : "c"}, \
                   {"lw" : 2, "ls" : ":", "color" : "m"}, \
                   {"lw" : 3, "ls" : "-", "color" : "k"}, \
                   {"lw" : 1.5, "ls" : "--", "color" : "y"}, \
                   {"lw" : 3, "ls" : "--", "color" : "k"}]
    
    for ii,index_name in enumerate(indices):
        ax1.semilogx((data["bin_edges"][1:]+data["bin_edges"][:-1])/2, \
                     data[index_name]["detections_per_bin"] / data["total_counts_per_bin"], \
                     label=r"{}: $\varepsilon$={:.3f}, $F$={:.1f}%".format(index_to_label[index_name], data[index_name]["total_efficiency"], data[index_name]["num_false_positives"]/(11.*events_per_light_curve*light_curves_per_ccd)*100), \
                     **line_styles[ii])
        
    ax1.legend(loc="upper left")
    ax1.set_ylim(0., 1.0)
    
    logger.debug("Saving figure and cleaning up!")
    fig1.savefig("data/detectionefficiency/{}.png".format(file_base))

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
    parser.add_argument("--u0", dest="u0", type=float, default=None,
                    help="Only add microlensing events with the specified impact parameter.")
    
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
    compare_detection_efficiencies_on_field(field, events_per_light_curve=args.N, light_curves_per_ccd=args.limit, overwrite=args.overwrite, u0=args.u0)