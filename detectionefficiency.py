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
#ch = logging.StreamHandler()
#formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
#ch.setFormatter(formatter)
#logger.addHandler(ch)

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

'''
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
'''

def simulate_events_worker(sim_light_curve, tE, u0, reference_mag, indices):
    """ Only to be used in the multiprocessing pool below! """
    
    light_curve = copy.copy(sim_light_curve)
    
    # Reset the simulated light curve back to the original data, e.g. erase any previously
    #   added microlensing events
    light_curve.reset()
    
    light_curve.addMicrolensingEvent(tE=tE, u0=u0)
    
    try:
        lc_var_indices = analyze.compute_variability_indices(light_curve, indices, return_tuple=True) + (light_curve.tE, light_curve.u0, reference_mag, True)
        return lc_var_indices
    except:
        logger.warning("Failed to compute variability indices for simulated light curve!")
    

def simulate_events_compute_indices(light_curve, events_per_light_curve=100, indices=["j","k","eta","sigma_mu","delta_chi_squared"], u0=None):
    """ """
    #logger.debug("\t\tsimulate_events_compute_indices")
    
    # Create a SimulatedLightCurve object for the light_curve. This object has the addMicrolensingEvent() method.
    sim_light_curve = SimulatedLightCurve(mjd=light_curve.mjd, mag=light_curve.mag, error=light_curve.error)
    sim_light_curve.reset()
    
    # Estimate the reference magnitude using the median magnitude
    reference_mag = np.median(sim_light_curve.mag)
    
    var_indices_with_events = []
    def callback(result):
        var_indices_with_events.append(result)
    
    # Pre-compute the variability indices to add when we *don't* add an event, to compute false positive rate
    vanilla_var_indices = analyze.compute_variability_indices(sim_light_curve, indices, return_tuple=True)
    
    pool = multiprocessing.Pool(processes=8)
    for event_id in range(events_per_light_curve):
        # Only add events 50% of the time, to allow for false positives
        if np.random.uniform() > 0.5:
            tE = 10**np.random.uniform(0., 3.)
            pool.apply_async(simulate_events_worker, args=(sim_light_curve, tE, u0, reference_mag, indices), callback=callback)
        else:
            var_indices_with_events.append(vanilla_var_indices + (sim_light_curve.tE, sim_light_curve.u0, reference_mag, False))
        
        """
        # Reset the simulated light curve back to the original data, e.g. erase any previously
        #   added microlensing events
        sim_light_curve.reset()
        event_added = False
        
        # Only add events 50% of the time, to allow for false positives
        if np.random.uniform() > 0.5:
            tE = 10**np.random.uniform(0., 3.)
            sim_light_curve.addMicrolensingEvent(tE=tE, u0=u0)
            event_added = True
            #logger.debug("\t\t {} -- event added tE={lc.tE:.2f}, t0={lc.t0:.2f}, u0={lc.u0:.2f}".format(event_id, lc=sim_light_curve))
        
        try:
            lc_var_indices = analyze.compute_variability_indices(sim_light_curve, indices, return_tuple=True)
        except:
            logger.warning("Failed to compute variability indices for simulated light curve!")
            continue
            
        var_indices_with_events.append(lc_var_indices + (sim_light_curve.tE, sim_light_curve.u0, reference_mag, event_added))
        """
    
    pool.close()
    pool.join()
    
    names = indices + ["tE", "u0", "m", "event_added"]
    dtypes = [float]*len(indices) + [float,float,float,bool]
    var_indices_with_events = np.array(var_indices_with_events, dtype=zip(names,dtypes))
    
    return var_indices_with_events
    
    # FAST WAY
    #return np.array(var_indices_with_events)

'''
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
'''

def compute_detection_efficiency(var_indices, var_indices_with_events, indices):
    """ """
    
    # Will be a boolean array to select out only rows where event_added == True
    events_added_idx = var_indices_with_events["event_added"]
    
    timescale_bins = np.logspace(0, 3, 100) # from 1 day to 1000 days, 100 bins
    total_counts_per_bin, tE_bin_edges = np.histogram(var_indices_with_events[events_added_idx], bins=timescale_bins)
    
    data = dict(total_counts_per_bin=total_counts_per_bin, \
                bin_edges=tE_bin_edges)
    for index_name in indices:
        mu, sigma = np.mean(var_indices[index_name]), np.std(var_indices[index_name])
        idx = (var_indices_with_events[index_name] > (mu+2.*sigma)) | (var_indices_with_events[index_name] < (mu-2.*sigma))
        detections_per_bin, tE_bin_edges = np.histogram(var_indices_with_events[idx]["tE"], bins=timescale_bins)
        
        total_efficiency = sum(var_indices_with_events[idx & events_added_idx]["event_added"]) / sum(events_added_idx)
        num_false_positives = len(var_indices_with_events[idx & (events_added_idx == False)])
        
        data[index_name] = dict(total_efficiency=total_efficiency, \
                                num_false_positives=num_false_positives, \
                                detections_per_bin=detections_per_bin)
    
    return data

'''
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
        limiting_mag1 = limiting_mag
        limiting_mag2 = limiting_mags[ii+1]
        
        logger.debug("Starting m={:.2f}-{:.2f}".format(limiting_mag1,limiting_mag2))
        
        for u0 in copy.copy(u0s):
            formatted_file_base = file_base.format(field_id=field.id, lc_per_ccd=light_curves_per_ccd, events_per_lc=events_per_light_curve, u0=u0, limiting_mag=(limiting_mag1,limiting_mag2))
            if overwrite and os.path.exists("data/detectionefficiency/{}.pickle".format(formatted_file_base)):
                logger.debug("File {} already exists, and you want to overwrite it -- so I'm going to remove it, fine!".format("data/detectionefficiency/{}.pickle".format(formatted_file_base)))
                os.remove("data/detectionefficiency/{}.pickle".format(formatted_file_base))
            
            if os.path.exists("data/detectionefficiency/{}.pickle".format(formatted_file_base)):
                logger.debug("File {} already exists...removing from queue".format("data/detectionefficiency/{}.pickle".format(formatted_file_base)))
                u0s.pop(u0s.index(u0))
        
        if len(u0s) == 0: continue
        
        light_curves = []
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
                # TODO: BIG BUG!!! ccdid wasn't there before!
                lc = field.ccds[ccdid].light_curve(source["matchedSourceID"], clean=True, barebones=True)
                
                # If the light curve has more than 25 observations, include it
                if len(lc.mjd) > 25: light_curve_batch.append(lc)
                if len(light_curve_batch) >= light_curves_per_ccd: break
            
            logger.debug("Adding another batch of {} light curves...".format(len(light_curve_batch)))
            light_curves += light_curve_batch
                
        logger.debug("All {} light curves loaded, about to run simulation...".format(len(light_curves)))
        
        # TODO: For testing memory usage
        sys.exit(0)
        
        # Multiprocessing stuff
        if len(u0s) > 8:
            pool_size = 8
        elif u0s == None:
            pool_size = 1
        else:
            pool_size = len(u0s)
            
        pool = multiprocessing.Pool(processes=pool_size)
        
        # The light curve objects later get turning into SimulatedLightCurve objects, so it's ok to
        #   pass the same list of light curves to all processes
        datas = pool.map(compare_detection_efficiencies_worker, [{"light_curves" : light_curves, "events_per_light_curve" : events_per_light_curve, "indices" : indices, "u0" : u0} for u0 in u0s])
        
        for u0,data in zip(u0s,datas):
            formatted_file_base = file_base.format(field_id=field.id, lc_per_ccd=light_curves_per_ccd, events_per_lc=events_per_light_curve, u0=u0, limiting_mag=(limiting_mag1,limiting_mag2))
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
'''

def compare_detection_efficiencies_on_field(field, light_curves_per_ccd, events_per_light_curve, overwrite=False, indices=["j","k","eta","sigma_mu","delta_chi_squared"], u0s=[], limiting_mags=[]):
    
    # TODO:
    #   - Write in file caching so I can save var_indices, var_indices_with_event to .npy files!! No more pickles!!
    #   - Figure out plotting stuff...
    #   - Document the new code!
    #   - Clean up / delete old stuff
    
    if u0s == None or len(u0s) == 0:
        u0s = [None]
    
    if limiting_mags == None or len(limiting_mags) == 0:
        limiting_mags = [14.3, 21]
    
    # Conditions for reading from the 'sources' table
    #   - Only select sources with enough good observations (>25)
    #   - Omit sources with large amplitude variability so they don't mess with our simulation
    wheres = ["(ngoodobs > 25)", "(stetsonJ < 100)", "(vonNeumannRatio > 1.0)"]
    
    # Keep track of calculated var indices for each CCD
    #all_var_indices = []
    all_var_indices_with_events = []
    var_indices_with_events = dict()
    
    for ccd in field.ccds.values():
        logger.info(greenText("Starting with CCD {}".format(ccd.id)))
        
        # Get the chip object for this CCD
        chip = ccd.read()
        
        for ii, limiting_mag in enumerate(limiting_mags[:-1]):
            # Define bin edges for selection on reference magnitude
            limiting_mag1 = limiting_mag
            limiting_mag2 = limiting_mags[ii+1]
            logger.info("\tMagnitude range: {:.2f} - {:.2f}".format(limiting_mag1, limiting_mag2))
            
            read_wheres = wheres + ["(referenceMag >= {:.3f})".format(limiting_mag1)]
            read_wheres += ["(referenceMag < {:.3f})".format(limiting_mag2)]
            
            # Read information from the 'sources' table
            source_ids = chip.sources.readWhere(" & ".join(read_wheres))["matchedSourceID"]
            
            # Randomly shuffle the sources
            np.random.shuffle(source_ids)
            logger.info("\t\tSelected {} source ids".format(len(source_ids)))
            
            dtype = zip(indices, [float]*len(indices))
            count = 0
            for source_id in source_ids:
                logger.debug("\t\t\tSource ID: {}".format(source_id))
                light_curve = ccd.light_curve(source_id, clean=True, barebones=True)
                
                # After quality cut, if light curve has less than 25 observations, skip it!
                if len(light_curve.mjd) < 25:
                    continue
                
                these_indices = np.array([analyze.compute_variability_indices(light_curve, indices, return_tuple=True)], dtype=dtype)
                try:
                    var_indices = np.hstack((var_indices, these_indices))
                except NameError:
                    var_indices = these_indices
                
                # FAST WAY
                """
                arr_idx1 = count*events_per_light_curve
                arr_idx2 = arr_idx1 + events_per_light_curve
                
                for u0 in u0s:
                    try:
                        var_indices_with_events[u0][arr_idx1:arr_idx2] = simulate_events_compute_indices(light_curve, events_per_light_curve=events_per_light_curve, indices=indices, u0=u0)
                    except KeyError:
                        var_indices_with_events[u0] = np.zeros((min(len(source_ids), light_curves_per_ccd)*events_per_light_curve,9))
                        var_indices_with_events[u0][arr_idx1:arr_idx2] = simulate_events_compute_indices(light_curve, events_per_light_curve=events_per_light_curve, indices=indices, u0=u0)
                """
                for u0 in u0s:
                    these_indices = simulate_events_compute_indices(light_curve, events_per_light_curve=events_per_light_curve, indices=indices, u0=u0)
                    try:
                        var_indices_with_events[u0] = np.hstack((var_indices_with_events[u0], these_indices))
                    except KeyError:
                        var_indices_with_events[u0] = these_indices

                count += 1
                if count >= light_curves_per_ccd: break
                if count == 10: sys.exit(0)
            
            # FAST WAY
            #if count != light_curves_per_ccd:
            #    for u0 in u0s:
            #        var_indices_with_events[u0] = var_indices_with_events[u0][:count*events_per_light_curve]
            
            logger.info("\t\t{} good light curves used".format(count))
            
            #all_var_indices_with_events.append(var_indices_with_events)
            
        ccd.close()
    
    for u0 in u0s:
        print compute_detection_efficiency(var_indices, var_indices_with_events[u0], indices=indices)["eta"]["total_efficiency"]
    
    print var_indices["j"], len(var_indices["j"])
    print var_indices_with_events[0.01]["j"], len(var_indices_with_events[0.01]["j"])
       
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
    compare_detection_efficiencies_on_field(field, events_per_light_curve=args.N, light_curves_per_ccd=args.limit, overwrite=args.overwrite, u0s=args.u0, limiting_mags=args.limiting_mag)