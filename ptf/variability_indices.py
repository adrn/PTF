from __future__ import division

# Standard library
import copy
import os
import sys
import cPickle as pickle
import multiprocessing

# Third-party
import numpy as np

from .analyze import compute_variability_indices
from .lightcurve import SimulatedLightCurve
from .util import get_logger
logger = get_logger(__name__)

'''
class VariabilityIndex(object):
    
    def __init__(self, representation, label, outlier_selection="both"):
        """ Represents a variability index such as eta, sigma_mu, etc. 
            
            Parameters
            ----------
            representation : str
                The raw representation of the statistic, e.g. 'eta'
            label : str
                The LaTeX code for the actual name of this statistic, e.g. r'$\eta$'
            outlier_selection : str
                Can be 'above', 'below', or 'both'. Select outliers above, below, or both
                from the given mean and standard deviation.
        """
        
        self._repr = str(representation)
        self._label = label
        
        if outlier_selection == "above":
            self.select_outliers = self._select_outliers_above
        elif outlier_selection == "below":
            self.select_outliers = self._select_outliers_below
        else:
            self.select_outliers = self._select_outliers_both
            
    def __str__(self):
        return self._label
    
    def __repr__(self):
        return self._repr
    
    def _select_outliers_above(self, var_indices, mu, sigma, Nsigma):
        return var_indices[self.representation] > (mu+Nsigma*sigma)
    
    def _select_outliers_below(self, var_indices, mu, sigma, Nsigma):
        return var_indices[self.representation] < (mu-Nsigma*sigma)
    
    def _select_outliers_both(self, var_indices, mu, sigma, Nsigma):
        return (var_indices[self.representation] > (mu+Nsigma*sigma)) | (var_indices[self.representation] < (mu-Nsigma*sigma))

eta = VariabilityIndex("eta", r"$\eta$", outlier_selection="below")
sigma_mu = VariabilityIndex("sigma_mu", r"$\sigma/\mu$", outlier_selection="both")
j = VariabilityIndex("j", r"$J$", outlier_selection="above")
k = VariabilityIndex("k", r"$K$", outlier_selection="both")
delta_chi_squared = VariabilityIndex("delta_chi_squared", r"$\Delta\chi^2$", outlier_selection="above")
con = VariabilityIndex("con", r"$Con$", outlier_selection="both")
corr = VariabilityIndex("corr", r"$Corr$", outlier_selection="below")
_indices = [eta, sigma_mu, j, k, delta_chi_squared, con, corr]

variability_indices = dict([(repr(id), id) for id in _indices])
'''

def simulate_events_worker(sim_light_curve, tE, u0, reference_mag, indices):
    """ Only to be used in the multiprocessing pool below! """
    
    light_curve = copy.copy(sim_light_curve)
    
    # Reset the simulated light curve back to the original data, e.g. erase any previously
    #   added microlensing events
    light_curve.reset()
    
    # APW HACK -- add microlensing events only on a data point, ignores the *survey* detection efficiency
    #t0 = light_curve.mjd[np.random.randint(len(light_curve.mjd))]
    t0 = None
    light_curve.addMicrolensingEvent(tE=tE, u0=u0, t0=t0)
    
    lc_var_indices = compute_variability_indices(light_curve, indices, return_tuple=True) + (light_curve.tE, light_curve.u0, reference_mag)
    
    return lc_var_indices

def simulate_events_compute_indices(light_curve, events_per_light_curve, indices, u0=None):
    """ Given a light_curve, simulate a bunch of microlensing events and compute the given
        variability indices for each simulated light curve.
    """
    # Create a SimulatedLightCurve object for the light_curve. This object has the addMicrolensingEvent() method.
    sim_light_curve = SimulatedLightCurve.from_ptflightcurve(light_curve)
    sim_light_curve.reset()
    
    # Estimate the reference magnitude using the median magnitude
    reference_mag = np.median(sim_light_curve.mag)
    
    var_indices_with_events = []
    def callback(result):
        var_indices_with_events.append(result)
    
    # APW HACK
    pool = multiprocessing.Pool(processes=8)
    for event_id in range(events_per_light_curve):
        tE = 10**np.random.uniform(0.3, 3.)
        pool.apply_async(simulate_events_worker, args=(sim_light_curve, tE, u0, reference_mag, indices), callback=callback)
        #callback(simulate_events_worker(sim_light_curve, tE, u0, reference_mag, indices))
    
    pool.close()
    pool.join()
    
    dtypes = [(index,float) for index in indices] + [("tE",float),("u0",float),("m",float)]
    var_indices_with_events = np.array(var_indices_with_events, dtype=dtypes)
    
    return var_indices_with_events

def simulate_light_curves_worker(sim_light_curve, indices):
    """ Only to be used in the multiprocessing pool below! """
    
    light_curve = copy.copy(sim_light_curve)
    mag = np.random.normal(np.median(light_curve.mag), light_curve.error)
    
    light_curve.mag = mag
    light_curve.error += np.random.normal(0., sim_light_curve.error/10.)
    
    lc_var_indices = compute_variability_indices(light_curve, indices, return_tuple=True)
    return lc_var_indices

def simulate_light_curves_compute_indices(light_curve, num_simulated, indices):
    """ Given a light_curve, simulate a bunch of *flat* + scatter light curves to use for estimating
        the false positive rate for each index.
    """
    # Create a SimulatedLightCurve object for the light_curve
    sim_light_curve = SimulatedLightCurve.from_ptflightcurve(light_curve)
    
    var_indices_simulated = []
    def callback2(result):
        var_indices_simulated.append(result)
      
    pool = multiprocessing.Pool(processes=8)
    for event_id in range(num_simulated):
        pool.apply_async(simulate_light_curves_worker, args=(sim_light_curve, indices), callback=callback2)
        #callback2(simulate_light_curves_worker(sim_light_curve, indices))
    
    pool.close()
    pool.join()
    
    var_indices_simulated = np.array(var_indices_simulated, dtype=[(index, float) for index in indices])

    return var_indices_simulated

def select_outliers(arr, lower_cut=None, upper_cut=None):
    """ Return a boolean array where arr is between the specified cuts. """
    
    if lower_cut != None:
        lower_idx = arr < lower_cut
    else:
        lower_idx = np.array([False]*len(arr))
    
    if upper_cut != None:
        upper_idx = arr > upper_cut
    else:
        upper_idx = np.array([False]*len(arr))
    
    idx = lower_idx | upper_idx
    
    return idx

def boolean_selection_from_index_name(index_name, var_indices, mu, sigma, Nsigma):
    """ Return a boolean array with indices to select events from. Different indices
        will have different criteria. For example, for J we only want positive outliers.
    """
    if index_name in ["j", "delta_chi_squared", "sigma_mu"]:
        # Only select outliers above the mean
        idx = select_outliers(var_indices[index_name], upper_cut=(mu+Nsigma*sigma))
    elif index_name in ["eta"]:
        # Only select outliers below the mean
        idx = select_outliers(var_indices[index_name], lower_cut=(mu-Nsigma*sigma))
    else:
        # Select all outliers
        idx = select_outliers(var_indices[index_name], upper_cut=(mu+Nsigma*sigma), lower_cut=(mu-Nsigma*sigma))
    
    return idx