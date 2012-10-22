""" 
    TODO:
"""

# Standard library
import sys, os
import multiprocessing

# Third-party
import numpy as np

# PTF
from ..lightcurve import SimulatedLightCurve
from ..analyze import compute_variability_indices
from ..util import get_logger
logger = get_logger(__name__)

__all__ = ["simulate_events_compute_indices"]

def _simulate_events_worker(sim_light_curve, tE, reference_mag, indices):
    """ The function that takes the input light curve and microlensing event parameters and
        simulates new light curves with injected microlensing events. This function is meant 
        to be the 'worker' function in the multiprocessing pool below.
    """
    
    light_curve = copy.copy(sim_light_curve)
    light_curve.add_microlensing_event(tE=tE, u0=u0)
    computed_var_indices = compute_variability_indices(light_curve, indices, return_tuple=True) + (light_curve.tE, light_curve.u0, reference_mag)
    
    return computed_var_indices

def simulate_events_compute_indices(light_curve, num, indices):
    """ Given a real light_curve from the photometric database, add simulated microlensing
        events to the light curve and compute the variability indices for each injected event.
        
        Parameters
        ----------
        light_curve : PTFLightCurve
            The input light curve object to grab the MJD and error information
        num : int
            The number of times to simulate events.
        indices : list
            Which indices to compute for these light curves.
    """
    # Create a SimulatedLightCurve object for the light_curve. This object has the addMicrolensingEvent() method.
    sim_light_curve = SimulatedLightCurve.from_ptflightcurve(light_curve)
    sim_light_curve.reset()
    
    # Estimate the reference magnitude using the median magnitude
    reference_mag = np.median(sim_light_curve.mag)
    
    var_indices_with_events = []
    def callback(result):
        var_indices_with_events.append(result)
    
    pool = multiprocessing.Pool(processes=8)
    for event_id in range(events_per_light_curve):
        # Draw tE from a uniform distribution from 1-1000 days
        tE = 10**np.random.uniform(0., 3.)
        pool.apply_async(simulate_events_worker, args=(sim_light_curve, tE, reference_mag, indices), callback=callback)
        #callback(simulate_events_worker(sim_light_curve, tE, reference_mag, indices))
    
    pool.close()
    pool.join()
    
    dtypes = [(index,float) for index in indices] + [("tE",float),("u0",float),("m",float)]
    var_indices_with_events = np.array(var_indices_with_events, dtype=dtypes)
    
    return var_indices_with_events
