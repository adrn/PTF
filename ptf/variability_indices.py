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