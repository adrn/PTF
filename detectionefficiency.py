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
import time

# Third-party
import apwlib.geometry as g
import apwlib.convert as c
from apwlib.globals import greenText
import numpy as np
from scipy.stats import scoreatpercentile

# Project
from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
import ptf.photometricdatabase as pdb
import ptf.analyze.analyze as analyze

# Create logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def variability_indices_detection_efficiency(light_curves, events_per_light_curve=100, indices=["j","k","eta","sigma_mu","delta_chi_squared"]):
    """ This figure should show the detection efficiency curve for all indices by
        injecting events into a random sampling of light_curves.
        
            1) Given a list of light curves, compute the variability indices for these "vanilla" light curves
            2) For each light curve, add 'events_per_light_curve' different events to each light curve, and 
                recompute the indices. Then use the selection criteria from the original light curves to define
                the detection efficiency and false positive rate for each index as a function of event timescale.
                
    """
    
    logger.debug(greenText("/// variability_indices_detection_efficiency ///"))
    logger.debug("Analyzing {} light curves".format(len(light_curves)))
    var_indices = np.array([analyze.compute_variability_indices(light_curve, indices, return_tuple=True) for light_curve in light_curves], dtype=float, names=indices)
    
    sim_light_curves = [SimulatedLightCurve(lc.mjd, mag=lc.mag, error=lc.error) for lc in light_curves]
    
    var_indices_with_events = []
    # For each light curve, add 100 different microlensing events and recompute indices
    for light_curve in sim_light_curves:
        for event_id in range(events_per_light_curve):
            light_curve.reset()
            event_added = False
            
            if np.random.uniform() > 0.5:
                light_curve.addMicrolensingEvent()
                event_added = True
            
            var_indices_with_events.append(analyze.compute_variability_indices(light_curve, indices, return_tuple=True) + (event_added,))
    
    names = indices + ["event_added"]
    dtypes = [float]*len(indices) + [bool]
    var_indices_with_events = np.array(var_indices_with_events, dtype=zip(names,dtypes))
    
    print len(var_indices_with_events["j"])
    
    return

field = pdb.Field(110002, filter="R")
chip = field.ccds[0].read()
source_ids = chip.sources.readWhere("matchedSourceID < 200")["matchedSourceID"]

light_curves = []
for sid in source_ids:
    lc = field.ccds[0].light_curve(sid)
    
    if len(lc.mjd) > 0:
        light_curves.append(lc)

field.ccds[0].close()

variability_indices_detection_efficiency(light_curves)

sys.exit(0)

if False:    
    # Help with plotting
    line_styles = [(3,"-."), (3,":"), (3,"--"), (1.5,"--"), (2,"-")]
    line_colors = ["c", "m", "g", "y", "k"]
    
    plt.figure(figsize=(15,15))
    for ii,idx in enumerate(indices):
        values = [getattr(x, idx) for x in var_indices]
        sigma = np.std(values)
        mu = np.mean(values)
        
        sim_results[np.isnan(sim_results["tE"])] = 0.
        
        detections = sim_results[(np.fabs(sim_results[idx]) > (mu + 2.*sigma)) | (np.fabs(sim_results[idx]) < (mu - 2.*sigma))]
        detections = detections[detections["event_added"] == True]
        
        tE_counts, tE_bin_edges = np.histogram(detections["tE"], bins=timescale_bins)
        total_counts, total_bin_edges = np.histogram(sim_results[sim_results["event_added"] == True]["tE"], bins=timescale_bins)
        
        lw,ls = styles[ii]
        plt.semilogx((total_bin_edges[1:]+total_bin_edges[:-1])/2, tE_counts / total_counts, c=colors[ii], lw=lw, label=r"{}".format(parameter_to_label[idx]), ls=ls)
        
    plt.xlabel(r"$t_E$ [days]", size=label_font_size)
    plt.ylabel(r"Detection Efficiency $\mathcal{E}(t_E)$", size=label_font_size)
    plt.ylim(0., 1.0)
    t = plt.title("PTF Detection Efficiency for Praesepe Light Curves", size=title_font_size)
    t.set_y(1.04)
    
    # Change tick label size
    ax = plt.gca()
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(tick_font_size)
    
    leg = plt.legend(shadow=True, fancybox=True)
    legendtext = leg.get_texts()
    plt.setp(legendtext, fontsize=label_font_size)
    plt.tight_layout()
    plt.savefig("plots/aas_var_indices_detection_efficiency.png")
    #plt.show()    

       
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite the file")
    parser.add_argument("-N", "--N", dest="N", default=100, type=int,
                    help="The number of microlensing events to be added to each light curve")
    parser.add_argument("--limit", dest="limit", default=0, type=int,
                    help="The number of light curves to select")
    parser.add_argument("-f", "--filename", dest="filename", required=True, type=str,
                    help="The filename to save the results to.")
    parser.add_argument("--random", dest="random", action="store_true", default=False,
                    help="Draw a random sample of light curves")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if args.limit > 0:
        limit = args.limit
    else:
        limit = None
    
    run_praesepe(args.N, args.filename, overwrite=args.overwrite, limit=limit, random=args.random)