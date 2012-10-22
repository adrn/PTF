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

def _simulate_light_curves_worker(sim_light_curve, indices):
    """ The function that takes the input light curve and simulates new light curves
        based on the MJD and error arrays. This function is meant to be the 'worker'
        function in a multiprocessing pool.
    """
    # Copy the input light curve object so we can overwrite the data
    new_light_curve = copy.copy(sim_light_curve)
    
    # Simulate a new array of magnitudes using the errors from the input light curve
    mag = np.random.normal(np.median(new_light_curve.mag), new_light_curve.error)
    
    new_light_curve.mag = mag
    
    # Assume that we only know our errors to 5% by adding some scatter to the error array
    new_light_curve.error += np.random.normal(0., sim_light_curve.error/20.)
    
    computed_var_indices = compute_variability_indices(new_light_curve, indices, return_tuple=True)
    return computed_var_indices

def simulate_light_curves_compute_indices(light_curve, num, indices):
    """ Given a real light_curve from the photometric database, simulate a bunch of 
        flat light curves (with some scatter) to use for estimating the false positive 
        rate for each index. 
        
        Parameters
        ----------
        light_curve : PTFLightCurve
            The input light curve object to grab the MJD and error information
        num : int
            The number of light curves to simulate based on this sampling and error.
        indices : list
            Which indices to compute for these light curves.
    """
    
    # Create a SimulatedLightCurve object for the light_curve
    sim_light_curve = SimulatedLightCurve.from_ptflightcurve(light_curve)
    
    var_indices_simulated = []
    def callback(result):
        var_indices_simulated.append(result)
      
    pool = multiprocessing.Pool(processes=8)
    for event_id in range(num_simulated):
        pool.apply_async(_simulate_light_curves_worker, args=(sim_light_curve, indices), callback=callback)
        #callback(simulate_light_curves_worker(sim_light_curve, indices))
    
    pool.close()
    pool.join()
    
    # Create a numpy structured array so I can access the arrays of index values using ["eta"] notation, for example.
    var_indices_simulated = np.array(var_indices_simulated, dtype=[(index, float) for index in indices])

    return var_indices_simulated