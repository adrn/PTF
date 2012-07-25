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
import esutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import scoreatpercentile

# Project
import ptf.photometricdatabase as pdb

def variability_indices_detection_efficiency(light_curves, indices=["j","k","eta","sigma_mu","delta_chi_squared"]):
    """ This figure should show the detection efficiency curve for all indices by
        injecting events into a random sampling of light_curves.
    """
    
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

def test_variability_indices_detection_efficiency_eta():
    # Select some light curves
    pdb
    
    variability_indices_detection_efficiency(










#############################################
# Multiprocessing stuff
#

import ptf.ext.delta_chi_squared as dcs

def worker(lc):
    lc.reset() # lc = copy.copy(simulated_light_curve)
    
    if np.random.uniform() >= 0.5:
        # Only add the event 50% of the time
        tE = 10.**np.random.uniform(-2, 3)
        lc.addMicrolensingEvent(tE=tE)
        event_added = True
    else:
        event_added = False
        tE = 0.
    
    return simu.compute_variability_indices(lc, indices=["j", "k", "sigma_mu", "eta", "delta_chi_squared", "mu"]) + (tE, event_added)

def worker2(lc):
    lc.reset() # lc = copy.copy(simulated_light_curve)
    tE = 10.**np.random.uniform(-2, 3)
    lc.addMicrolensingEvent(tE=tE) # ???
    
    return (dcs.compute_delta_chi_squared(lc.mjd,lc.mag,lc.error), np.mean(lc.mag), lc.tE)

def process_light_curve_chunk(light_curves, N):
    chunk_data = []
    for light_curve in light_curves:
        simulated_light_curve = simu.SimulatedLightCurve.fromDBLightCurve(light_curve)
        
        # Multiprocessing way:
        #   EDIT: Actually ended up being slower...
        #pool = multiprocessing.Pool(4)
        #results = pool.map(worker, [simulated_light_curve]*N)
        #chunk_data += results
        
        # Cython way, also slower..
        #for ii in range(N):
        #    chunk_data += [worker2(simulated_light_curve)]
        
        # Pure Python way:
        for ii in range(N):
            chunk_data.append(worker(simulated_light_curve))
    
    return chunk_data
        
    """
    for ii in range(N):
        # Add a microlensing event to the light curve, recompute delta chi-squared
        lc.reset() # lc = copy.copy(simulated_light_curve)
        tE = 10.**np.random.uniform(-2, 3)
        lc.addMicrolensingEvent(tE=tE) # ???
        
        data.append((simu.compute_delta_chi_squared(lc), np.mean(lc.mag), lc.tE))
    """
            
def run_simulation(light_curve_generator, N):
    """ Given a generator to get chunks of light curves, perform a detection
        efficiency simulation with the light curves. This function will
        write the result out to 
    """
    # 'data' will get turned in to a numpy recarray with names=["delta_chi_squared", "mean_mag", "tE"]
    names=["j", "k", "sigma_mu", "eta", "delta_chi_squared", "mean_mag", "tE", "event_added"]
    dtypes = [float, float, float, float, float, float, float, bool]
    data = []
    
    # For each light curve, add N different microlensing events to the light
    #   curve, and re-measure the delta chi-squared
    a = time.time()
    while True:
        try:
            light_curves = light_curve_generator.next()
        except StopIteration:
            break
        
        chunk = process_light_curve_chunk(light_curves, N)
        data += chunk
        print len(chunk)
    
    logging.debug("Loop took {} seconds".format(time.time() - a))
    
    return np.array(data, dtype=zip(names, dtypes)).view(np.recarray)

def run_praesepe(N, filename="data/praesepe_detection_efficiency.npy", limit=None, overwrite=False, random=False):
    """ Run the detection efficiency simulation for the praesepe light curves """
    
    if os.path.exists(filename):
        if overwrite:
            os.remove(filename)
    
    if not os.path.exists(filename):
        # Select out just the Praesepe light curves (objid < 100000)
        #database_light_curves = session.query(LightCurve).filter(LightCurve.objid < 100000).order_by(LightCurve.objid).all()
        light_curve_generator = AllPraesepeLightCurves(limit=limit, random=random)

        data = run_simulation(light_curve_generator, N)
        
        np.save(filename, data)
    
    # Get the RMS scatter of delta chi-squared for the vanilla light curves
    dcs = [x[0] for x in session.query(VariabilityIndices.delta_chi_squared).join(LightCurve).filter(LightCurve.objid < 100000).all()]
    sigma = np.std(dcs)
    
    bins = np.logspace(np.log10(1), np.log10(1000), 100)
    compute_detection_efficiency(filename, sigma, bins)
        
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