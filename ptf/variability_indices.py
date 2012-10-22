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

def prune_index_distribution(index_array, index):
    """ Certain values of indices are an indication that the data are bad or that
        the calculation to compute the index was wrong. Obviously the "invalid domain"
        will depend on the index considered, so this function will know how to do it
        for all indices.
        
        Parameters
        ----------
        index_array : numpy.ndarray
            The input array of values for the given variability index.
        index : str
            The name of the variability index.
    """
        
    
    if index == "eta":
        # Only values >0 make sense
        return index_array[index_array > 0]
        
    elif index == "sigma_mu":
        # Sign doesn't matter, we're just interested in the scatter / mean magnitude
        return np.fabs(index_array)
    
    elif index == "j":
        # For a microlensing event, J will be very negative because it causes a decrease in the 
        #   magnitude!
        return np.fabs(index_array[index_array < 0])
        
    elif index == "k":
        # K should always be greater than 0
        return index_array[index_array > 0]
        
    elif index == "delta_chi_squared":
        # When delta_chi_squared < 0, that means the constant line is a much better
        #   fit than the microlensing model, which either means the fits failed or 
        #   it is definitely not a microlensing event
        return index_array[index_array > 0]
    else:
        return

def var_indices_for_simulated_light_curves(field, number_of_light_curves, number_of_simulations_per_light_curve, indices):
    """ Given a Field, simulate some number of flat + Gaussian noise light curves, compute their
        variability indices, and return an array of these values.
        
        Parameters
        ----------
        field : Field instance
            A ptf.db.photometric_database.Field object.
        number_of_light_curves : int
            The number of light curves to select from the photometric database
        number_of_simulations_per_light_curve : int
            The number of simulated light curves to generate given each db light curve.
        indices : list
            A list of variability indices to run this for.
            
    """
    
    indent_level = "\t"
    # Check that the Field has some CCD objects associated with it
    if len(field.ccds) == 0:
        logger.warn(indent_level+"Field has no associated CCDs! {}".format(field))
        return None
    
    # I have to keep track of these for all CCDs, since we do selection on a *field* by field basis.
    db_statistic_values = dict()
    for index in indices:
        db_statistic_values[index] = np.array([])
        
    for ccd in field.ccds.values():
        logger.info(indent_level+"Starting with CCD {}".format(ccd.id))
        indent_level += "\t"
        chip = ccd.read()
        
        logger.debug(indent_level + "Getting variability statistics from photometric database")
        source_ids = []
        db_statistics_array = []
        for source in chip.sources.where("(ngoodobs > {})".format(min_number_of_good_observations)):
            db_statistics_array.append(tuple([source_index_name_to_pdb_index(source,index) for index in indices]))
            source_ids.append(source["matchedSourceID"])
        db_statistics_array = np.array(db_statistics_array, dtype=[(index,float) for index in indices])
        
        logger.debug(indent_level + "Selected {} statistics".format(len(db_statistics_array)))
        
        # I use a dictionary here because after doing some sub-selection the index arrays may 
        #   have difference lengths.
        for index in indices:
            # This is where I need to define the selection distributions for each index.
            db_statistic_values[index] = np.append(db_statistic_values[index], 
                                                   vi.prune_index_distribution(db_statistics_array[index], index))
        
        # Up to here I have just selected the original variability statistic distributions from the
        #   photometric database and appended them to db_statistic_values. Next I will run simulations
        #   and determine where I can cut each distribution so that I get a given false positive rate.
        
        # Randomize the order of source_ids
        np.random.shuffle(source_ids)
        
        logger.debug(indent_level + "Simulating light curves for false positive rate calculation")
        
        # Keep count of how many light curves we've used, break after we reach the specified number
        light_curve_count = 0
        for source_id in source_ids:
            light_curve = ccd.light_curve(source_id, barebones=True, clean=True) # clean applies a quality cut to the data
            
            if light_curve == None or len(light_curve) < min_number_of_good_observations: 
                # If the light curve is not found, or has too few observations, skip this source_id
                continue
                
            simulated_indices_for_this_light_curve = sim.simulate_light_curves_compute_indices(light_curve, \
                                                                                num=number_of_simulations_per_light_curve, \
                                                                                indices=indices)
            try:
                simulated_statistics = np.hstack((simulated_light_curve_statistics, simulated_indices_for_this_light_curve))
            except NameError:
                simulated_statistics = simulated_indices_for_this_light_curve
                
            light_curve_count += 1
            
            if light_curve_count >= number_of_light_curves:
                break
        
        # If we get through all the sources and have gone through less than requested, warn the user
        if light_curve_count < number_of_light_curves:
            logging.warn(indent_level + "Not enough good light curves on this CCD! Field {}, CCD {}".format(field.id, ccd.id))
    
    return {"db" : db_statistic_values, "simulated" : simulated_statistics}

def compute_selection_criteria(var_indices, indices, fpr=0.01):
    """ Compute the selection criteria for each variability index on a given field by running simulations
        to find the selection boundaries that produce a false positive rate of 'fpr'.
        
        Parameters
        ----------
        var_indices : dict
            A dictionary with 2 keys,
                'db' : the variability index values pulled right from the database
                'simulated' : the variability index values computed from simulated flat light curves
                              Gaussian scatter.
        indices : list
            A list of variability indices to run this for.
        fpr : float
            The false positive rate to cut at.
        
    """
    
    try:
        if var_indices == None or len(var_indices["simulated"]) == 0:
            raise ValueError
    except NameError, ValueError:
        logger.info(indent_level + "No statistics computed for this field! Something went wrong...")
        return None
    
    indent_level = "\t"
    logger.debug(indent_level + "Starting false positive rate calculation to determine selection criteria")
    
    # Now determine the N in N-sigma by computing the false positive rate and getting it to be ~0.01 (1%) for each index
    selection_criteria = {}
    for index in indices:
        logger.debug(indent_level + "Index: {}".format(index))
        
        # Get the mean and standard deviation of the 'vanilla' distributions
        db_mean, db_sigma = np.mean(np.log10(db_statistic_values[index])), np.std(np.log10(db_statistic_values[index]))
        logger.debug(indent_level + "\t mu={}, sigma={}".format(db_mean, db_sigma))
        
        # Get the simulated statistics for this index 
        #TODO: Check the validity of using prune_ here -- might be some issue with J
        these_statistics = np.log10(vi.prune_index_distribution(simulated_statistics[index]))
        
        # Start by selecting with Nsigma = 0
        Nsigma = 0.
        
        # Nsteps is the number of steps this routine has to take to converge -- just used for diagnostics
        Nsteps = 0
        while True:
            computed_fpr = np.sum((these_statistics > (db_mean + Nsigma*db_sigma)) | (these_statistics < (db_mean - Nsigma*db_sigma))) / float(len(these_statistics))
            logger.debug("Step: {}, FPR: {}".format(Nsteps, fpr))
            
            # WARNING: If you don't use enough simulations, this may never converge!
            if computed_fpr > (fpr + 0.002): 
                Nsigma += np.random.uniform(0., 0.05)
            elif computed_fpr < (fpr - 0.002):
                Nsigma -= np.random.uniform(0., 0.05)
            else:
                break
            
            Nsteps += 1
            
            if Nsteps > 1000:
                logger.warn("{} didn't converge!".format(index))
                break
            
        logger.debug("{} -- Final Num. steps: {}, Final FPR: {}".format(index, Nsteps, computed_fpr))
        logger.debug("{} -- Final Nsigma={}, Nsigma*sigma={}".format(index, Nsigma, Nsigma*db_sigma))
        
        selection_criteria[index] = dict()
        selection_criteria[index]["upper"] = db_mean + Nsigma*db_sigma
        selection_criteria[index]["lower"] = db_mean - Nsigma*db_sigma
    
    return selection_criteria


































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