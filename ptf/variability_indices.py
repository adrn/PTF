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
from .simulation import simulate_light_curves_compute_indices
from .util import get_logger, source_index_name_to_pdb_index, richards_qso
from .globals import min_number_of_good_observations
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

    indent_level += "\t"
    for ccd in field.ccds.values():
        logger.info(indent_level+"Starting with CCD {}".format(ccd.id))
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
                                                   prune_index_distribution(db_statistics_array[index], index))

        # Up to here I have just selected the original variability statistic distributions from the
        #   photometric database and appended them to db_statistic_values. Next I will run simulations
        #   and determine where I can cut each distribution so that I get a given false positive rate.

        # Randomize the order of source_ids
        np.random.shuffle(source_ids)

        logger.debug(indent_level + "Reading in light curves")
        light_curves = []
        for source_id in source_ids:
            if len(light_curves) >= number_of_light_curves:
                break

            light_curve = ccd.light_curve(source_id, barebones=True, clean=True) # clean applies a quality cut to the data

            if light_curve == None or len(light_curve) < min_number_of_good_observations:
                # If the light curve is not found, or has too few observations, skip this source_id
                continue

            light_curves.append(light_curve)

        logger.debug(indent_level + "Simulating light curves for false positive rate calculation")

        # If we get through all the sources and have gone through less than requested, warn the user
        if len(light_curves) < number_of_light_curves:
            logger.warn(indent_level + "Not enough good light curves on this CCD! Field {}, CCD {}".format(field.id, ccd.id))
            continue

        simulated_statistics = simulate_light_curves_compute_indices(light_curves, \
                                                                     num=number_of_simulations_per_light_curve, \
                                                                     indices=indices)

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

    indent_level = "\t"
    try:
        if var_indices == None or len(var_indices["simulated"]) == 0:
            raise ValueError
    except NameError, ValueError:
        logger.info(indent_level + "No statistics computed for this field! Something went wrong...")
        return None

    logger.debug(indent_level + "Starting false positive rate calculation to determine selection criteria")

    # Now determine the N in N-sigma by computing the false positive rate and getting it to be ~0.01 (1%) for each index
    selection_criteria = {}
    for index in indices:
        logger.debug(indent_level + "Index: {}".format(index))

        # Get the simulated statistics for this index
        these_statistics = np.log10(prune_index_distribution(var_indices["simulated"][index],index))

        # Nsteps is the number of steps this routine has to take to converge -- just used for diagnostics
        Nsteps = 1000
        Nstep = 0
        if index == "eta":
            selection_cutoff = 0.
            while Nstep < Nsteps:
                computed_fpr = np.sum(these_statistics < selection_cutoff) / float(len(these_statistics))

                # WARNING: If you don't use enough simulations, this may never converge!
                if computed_fpr > (fpr + 0.002):
                    selection_cutoff -= np.random.uniform(0., 0.05)
                elif computed_fpr < (fpr - 0.002):
                    selection_cutoff += np.random.uniform(0., 0.05)
                else:
                    break

                Nstep += 1

        elif index == "delta_chi_squared" or index == "j":
            selection_cutoff = 0.
            while Nstep < Nsteps:
                computed_fpr = np.sum(these_statistics > selection_cutoff) / float(len(these_statistics))

                # WARNING: If you don't use enough simulations, this may never converge!
                if computed_fpr > (fpr + 0.002):
                    selection_cutoff += np.random.uniform(0., 0.05)
                elif computed_fpr < (fpr - 0.002):
                    selection_cutoff -= np.random.uniform(0., 0.05)
                else:
                    break

                Nstep += 1

        else:
            raise ValueError("Selection boundaries only available for eta, delta_chi_squared and j!")

        if Nstep == Nsteps:
            logger.warn("**Selection boundary didn't converge!**")

        logger.debug("{} -- Final Num. steps: {}, Final FPR: {}".format(index, Nsteps, computed_fpr))
        logger.debug("{} -- Final selection boundary: {}".format(index, selection_cutoff))

        selection_criteria[index] = dict()
        selection_criteria[index] = selection_cutoff

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