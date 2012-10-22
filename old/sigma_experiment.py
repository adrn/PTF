""" 
    In this script I will try to justify my sigma-cut detection algorithm. The idea is to
    take a sample of light curves from a CCD and compute their variability statistic 
    distributions. I'll then add microlensing events with the **same** impact parameter
    to all light curves, and see where the distribution goes. 
"""

# Standard library
import sys, os
import cPickle as pickle
import logging

# Third-party
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import apwlib.geometry as g
from apwlib.globals import redText, greenText, yellowText

# PTF
from ptf.simulation.simulatedlightcurve import SimulatedLightCurve
import ptf.photometricdatabase as pdb
import ptf.analyze.analyze as analyze

indices = ["j", "k", "eta", "sigma_mu", "delta_chi_squared"]

def select_light_curves(field, N=1000):
    """ Fetch a sample of light curves from the photometric database from the
        specified Field.
        
        Parameters
        ----------
        field : Field
            Represents a PTF Field
        N : int
            The number of light curves to fetch from each CCD
    """
    
    if not isinstance(field, pdb.Field):
        raise ValueError("field parameter must be a Field object")
    
    light_curves = []
    for ccdid, ccd in field.ccds.items():
        logging.debug("CCD {}".format(ccdid))
        
        chip = ccd.read()
        # Read in all of the source IDs for this chip
        source_ids = chip.sources.col("matchedSourceID")
        
        # Shuffle them about / randomize the order
        np.random.shuffle(source_ids)
        
        count = 0
        for sid in source_ids:
            # Get a LightCurve object for this source id on this ccd
            lc = field.ccds[0].light_curve(sid, clean=True)
            
            # If the light curve has more than 25 observations, include it
            # HACK alert
            if len(lc.mjd) > 25:
                light_curve = SimulatedLightCurve(lc.mjd, mag=lc.mag, error=lc.error)
                light_curves.append(light_curve)
                count += 1
            
            if count >= N: break
        
        ccd.close()
    
    return light_curves
        
def experiment(light_curves, u0=1.0, events_per_light_curve=100):
    """ Fetch a sample of light curves from the photometric database and compare the vanilla
        variability statistic distributions to those with events added. Here the events **all
        have the same impact parameter**.
        
        Parameters
        ----------
        light_curves : list
            A list of light_curve objects
        u0 : float
            The impact parameter of events to add to the light curves
        events_per_light_curve : int
            Number of events to add to each light curve
    """
    
    var_indices = []
    var_indices_with_event = []
    
    for light_curve in light_curves:
        var_indices.append(analyze.compute_variability_indices(light_curve, indices=indices, return_tuple=True))
        for event_id in range(events_per_light_curve):
            # Reset the simulated light curve back to the original data, e.g. erase any previously
            #   added microlensing events
            light_curve.reset()
            tE = 10**np.random.uniform(0., 3.)
            light_curve.addMicrolensingEvent(tE=tE, u0=u0)
            
            try:
                lc_var_indices = analyze.compute_variability_indices(light_curve, indices, return_tuple=True)
            except:
                break
            var_indices_with_event.append(lc_var_indices)
        
    dtype = zip(indices, [float]*len(indices))
    var_indices = np.array(var_indices, dtype=dtype)
    var_indices_with_event = np.array(var_indices_with_event, dtype=dtype)
    
    return var_indices, var_indices_with_event
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        field_id = int(sys.argv[1])
    except:
        print redText("You must specify a field as a command line arg")
        raise
    
    field = pdb.Field(field_id, filter="R")
    u0s = [0.01, 0.1, 0.5, 1.0]#, 1.34]
    filename_base = "data/sigma_experiment_{:06d}_{:.3f}.pickle"
    
    generate_data = False
    for u0 in u0s:
        filename = filename_base.format(field.id, u0)
        print filename, os.path.exists(filename)
        if not os.path.exists(filename):
            generate_data = True
    
    if generate_data:
        light_curves = select_light_curves(field, N=1000)
        
        for u0 in u0s:
            filename = filename_base.format(field.id, u0)
            if not os.path.exists(filename):
                var_indices, var_indices_with_event = experiment(light_curves=light_curves, u0=u0, events_per_light_curve=100)

                f = open(filename, "w")
                pickle.dump((var_indices, var_indices_with_event), f)
                f.close()
                
    first = True
    bounds = dict()
    for ii,u0 in enumerate(u0s):  
        logging.debug("Reading in data file...")
        filename = filename_base.format(field.id, u0)
        f = open(filename, "r")
        (var_indices, var_indices_with_event) = pickle.load(f)
        f.close()
        
        # There's a problem with the saved data and for some reason only the first var_indices
        #   is good, so this is a hack to solve that 
        if first:
            good_var_indices = var_indices
        
        logging.info("Plotting...")
        for index_name in indices:
            fig = plt.figure(figsize=(15,10))
            ax = fig.add_subplot(111)
            
            mins = [good_var_indices[index_name].min(), var_indices_with_event[index_name].min()]
            maxs = [good_var_indices[index_name].max(), var_indices_with_event[index_name].max()]
            
            # Do this so I can have the same X axis on all plots
            if first: 
                bounds[index_name] = dict(x=(min(mins), max(maxs)))
            
            if index_name == "j" or index_name == "delta_chi_squared":
                bins = list(-np.logspace(-1, 6, 100))[::-1] + list(np.linspace(-0.1,0.1,10)) + list(np.logspace(-1, 6, 100))
                #bins = 100
                ax.set_xscale("symlog", linthreshx=0.1)
            else:
                bins = np.linspace(bounds[index_name]["x"][0], bounds[index_name]["x"][1], 100)
            
            ax.hist(good_var_indices[index_name], bins=bins, color="k", alpha=0.5, log=True)
            ax.hist(var_indices_with_event[index_name], bins=bins, color="r", alpha=0.5, log=True)
            
            if first:
                bounds[index_name]["y"] = ax.get_ylim()
            
            ax.set_ylim(bounds[index_name]["y"])
            
            mu = np.mean(good_var_indices[index_name])
            sigma = np.std(good_var_indices[index_name])
            ax.axvline(mu + sigma, color="b", ls="--", alpha=0.7)
            ax.axvline(mu - sigma, color="b", ls="--", alpha=0.7)
            ax.axvline(mu + 2.*sigma, color="g", ls="--", alpha=0.7)
            ax.axvline(mu - 2.*sigma, color="g", ls="--", alpha=0.7)
            
            ax.set_xlabel(index_name)
            ax.set_ylabel("Normalized number of light curves")
            ax.set_title(r"Field: {}, $u_0 = {}$".format(field.id, u0))
            
            fig.savefig("data/sigma_experiment_{:06d}_{}_{:.3f}.png".format(field.id, index_name, u0))
        
        first = False