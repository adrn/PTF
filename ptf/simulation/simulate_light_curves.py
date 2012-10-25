""" 
    TODO:
"""

# Standard library
import sys, os
import copy
import multiprocessing

# Third-party
import numpy as np

# PTF
from ..lightcurve import SimulatedLightCurve
from ..analyze import compute_variability_indices
from ..util import get_logger
logger = get_logger(__name__)

__all__ = ["simulate_light_curves_compute_indices"]

def _simulate_light_curves_worker(args):
    """ The function that takes the input light curve and simulates new light curves
        based on the MJD and error arrays. This function is meant to be the 'worker'
        function in a multiprocessing pool.
    """
    #sim_light_curve, indices = args
    new_light_curve, indices, num = args
    
    for ii in range(num):
        # Simulate a new array of magnitudes using the errors from the input light curve
        mag = np.random.normal(np.median(new_light_curve.mag), new_light_curve.error)
        
        new_light_curve.mag = mag
        
        # Assume that we only know our errors to 5% by adding some scatter to the error array
        new_light_curve.error += np.random.normal(0., new_light_curve.error/20.)
        
        one_computed_var_indices = compute_variability_indices(new_light_curve, indices, return_tuple=True)
        try:
            computed_var_indices = np.hstack((computed_var_indices, one_computed_var_indices))
        except NameError:
            computed_var_indices = one_computed_var_indices
            
    return computed_var_indices

def simulate_light_curves_compute_indices(light_curves, num, indices):
    """ Given a real light_curve from the photometric database, simulate a bunch of 
        flat light curves (with some scatter) to use for estimating the false positive 
        rate for each index. 
        
        Parameters
        ----------
        light_curves : PTFLightCurve
            The input light curve objects to grab the MJD and error information.
        num : int
            The number of light curves to simulate based on this sampling and error.
        indices : list
            Which indices to compute for these light curves.
    """
    
    # Create a SimulatedLightCurve object for the light_curve
    #sim_light_curve = SimulatedLightCurve.from_ptflightcurve(light_curve)
    
    all_indices = []
    for args in [(SimulatedLightCurve.from_ptflightcurve(light_curve), indices, num) for light_curve in light_curves]:
        all_indices.append(_simulate_light_curves_worker(args))
    
    sys.exit(0)
    
    pool = multiprocessing.Pool(processes=8)
    result = pool.map_async(_simulate_light_curves_worker, [(SimulatedLightCurve.from_ptflightcurve(light_curve), indices, num) for light_curve in light_curves])
    
    pool.close()
    pool.join()

    #result = []
    #for ii in range(num):
    #    result.append(_simulate_light_curves_worker((sim_light_curve, indices)))
    
    # Create a numpy structured array so I can access the arrays of index values using ["eta"] notation, for example.
    var_indices_simulated = np.array(result.get(), dtype=[(index, float) for index in indices])
    #var_indices_simulated = np.array(result, dtype=[(index, float) for index in indices])

    return var_indices_simulated