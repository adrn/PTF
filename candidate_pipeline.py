# coding: utf-8
from __future__ import division

""" This module contains a pipeline for selecting candidate microlensing events from
    David Levitan's photometric database. The new pipeline looks something like this:
        For each field, we run simulations to determine the selection criteria for "interesting"
        variable sources, store those values in mongodb, then we use these criteria to go back
        to the PDB and select out candidates. 
        
    NEW IDEA! Today is 9/16/12. It is now 4:58 PM
    - When doing delta chi-squared, subtract off the fit, recompute eta and see if it's still an outlier.
        if it is, throw it away!
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import math
import logging
import cPickle as pickle

# Third-party
import numpy as np
import scipy.optimize as so

try:
    from apwlib.globals import greenText, yellowText, redText
except ImportError:
    raise ImportError("apwlib not found! \nDo: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install' to install.")

# Project
import ptf.photometricdatabase as pdb
import ptf.analyze.analyze as pa
from ptf.globals import index_to_label, source_index_name_to_pdb_index
import ptf.variability_indices as vi
import ptf.db.mongodb as mongo

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

file_hdlr = logging.FileHandler('logs/candidate_pipeline.log', mode="a")
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
file_hdlr.setFormatter(formatter)
file_hdlr.setLevel(logging.DEBUG)
logger.addHandler(file_hdlr) 

def prune_index_distribution(index, index_array):
    if index == "eta":
        return np.log10(index_array[index_array > 0])
    elif index == "sigma_mu":
        return np.log10(np.fabs(index_array))
    elif index == "j":
        return np.log10(index_array[index_array > 0])
    elif index == "k":
        return np.log10(index_array)
    elif index == "delta_chi_squared":
        return np.log10(index_array[index_array > 0])
    else:
        return

def compute_selection_criteria_for_field(field, fpr=0.01, number_of_fpr_light_curves=100, number_of_fpr_simulations_per_light_curve=100, indices=["eta"]):
    """ Compute the selection criteria for a given field by running false positive rate simulations
        to get the upper and lower cuts.
    """
    
    # Initialize my PDB statistic dictionary
    # I use a dictionary here because after doing some sub-selection the index arrays may 
    #   have difference lengths.
    pdb_statistics = dict()
    for index in indices:
        pdb_statistics[index] = np.array([])
    
    if len(field.ccds) == 0:
        logger.info("FUNKY FIELD {}".format(field))
        return None
    
    for ccd in field.ccds.values():
        logger.info(greenText("Starting with CCD {}".format(ccd.id)))
        chip = ccd.read()
        
        logger.info("Getting variability statistics from photometric database")
        source_ids = []
        pdb_statistics_array = []
        for source in chip.sources.where("(ngoodobs > {})".format(min_number_of_good_observations)):
            pdb_statistics_array.append(tuple([source_index_name_to_pdb_index(source,index) for index in indices]))
            source_ids.append(source["matchedSourceID"])
        pdb_statistics_array = np.array(pdb_statistics_array, dtype=[(index,float) for index in indices])
        
        logger.debug("Selected {} statistics".format(len(pdb_statistics_array)))
        
        # I use a dictionary here because after doing some sub-selection the index arrays may 
        #   have difference lengths.
        for index in indices:
            this_index_array = pdb_statistics_array[index]
            
            # This is where I need to define the selection distributions for each index.
            pdb_statistics[index] = np.append(pdb_statistics[index], prune_index_distribution(index, this_index_array))
        
        # Randomize the order of source_ids to prune through
        np.random.shuffle(source_ids)
        
        logger.info("Simulating light curves for false positive rate calculation")
        # Keep track of how many light curves we've used, break after we reach the specified number
        light_curve_count = 0
        for source_id in source_ids:
            light_curve = ccd.light_curve(source_id, barebones=True, clean=True)
            if len(light_curve.mjd) < min_number_of_good_observations: 
                logger.debug("\tRejected source {}".format(source_id))
                continue
                
            logger.debug("\tSelected source {}".format(source_id))
            these_indices = vi.simulate_light_curves_compute_indices(light_curve, num_simulated=number_of_fpr_simulations_per_light_curve, indices=indices)
            try:
                simulated_light_curve_statistics = np.hstack((simulated_light_curve_statistics, these_indices))
            except NameError:
                simulated_light_curve_statistics = these_indices
                
            light_curve_count += 1
            
            if light_curve_count >= number_of_fpr_light_curves:
                break
        
        #ccd.close()
    
    try:
        a = simulated_light_curve_statistics
    except NameError:
        logger.info("FUNKY FIELD {}".format(field))
        return None
    
    logger.info("Starting false positive rate calculation to get Nsigmas")
    # Now determine the N in N-sigma by computing the false positive rate and getting it to be ~0.01 (1%) for each index
    selection_criteria = {}
    for index in indices:
        logger.debug("\tIndex: {}".format(index))
        # Get the mean and standard deviation of the 'vanilla' distributions to select with
        mu,sigma = np.mean(pdb_statistics[index]), np.std(pdb_statistics[index])
        logger.debug("\t mu={}, sigma={}".format(mu, sigma))
        
        # Get the simulated statistics for this index
        these_statistics = np.log10(simulated_light_curve_statistics[index])
        
        # Start by selecting with Nsigma = 0
        Nsigma = 0.
        
        # Nsteps is the number of steps this routine has to take to converge -- just used for diagnostics
        Nsteps = 0
        while True:
            fpr = np.sum((these_statistics > (mu + Nsigma*sigma)) | (these_statistics < (mu - Nsigma*sigma))) / float(len(these_statistics))
            logger.debug("Step: {}, FPR: {}".format(Nsteps, fpr))
            
            # WARNING: If you don't use enough simulations, this may never converge!
            if fpr > 0.012: 
                Nsigma += np.random.uniform(0., 0.05)
            elif fpr < 0.008:
                Nsigma -= np.random.uniform(0., 0.05)
            else:
                break
            
            Nsteps += 1
            
            if Nsteps > 1000:
                logger.warn("{} didn't converge!".format(index))
                break
            
        logger.info("{} -- Final Num. steps: {}, Final FPR: {}".format(index, Nsteps, fpr))
        logger.info("{} -- Final Nsigma={}, Nsigma*sigma={}".format(index, Nsigma, Nsigma*sigma))
        
        selection_criteria[index] = dict()
        selection_criteria[index]["upper"] = mu + Nsigma*sigma
        selection_criteria[index]["lower"] = mu - Nsigma*sigma
    
    return selection_criteria

def save_selection_criteria(field, selection_criteria, selection_criteria_collection, overwrite=False):
    """ Update the selection criteria database entry for this field """
    
    index = "eta"
    search_existing = selection_criteria_collection.find_one({"field_id" : field.id, "index" : index})
        
    if search_existing != None and not overwrite:
        logger.warning("Criteria already exists in database for field {}, index {}".format(field.id, index))
        return False
        
    if search_existing != None and overwrite:
        selection_criteria_collection.remove({"field_id" : field.id, "index" : index})
        
    document = {"field_id" : field.id, \
                "index" : index, \
                "upper" : selection_criteria[index]["upper"], \
                "lower" : selection_criteria[index]["lower"]}
    selection_criteria_collection.insert(document)
    
    return True

def select_candidates(field, selection_criteria):
    """ Select candidates from a field given the log10(selection criteria) from mongodb. 
        
        The current selection scheme is to first select on eta, then to sanity check with
        delta chi-squared by making sure it's positive and >10.
    
    """
    
    lower_cut = 10**selection_criteria["lower"]
    
    candidates = []
    for ccd in field.ccds.values():
        logger.info(greenText("Starting with CCD {}".format(ccd.id)))
        chip = ccd.read()
        
        sources = chip.sources.readWhere("(ngoodobs > {}) & \
                                          (vonNeumannRatio > 0.0) & \
                                          ((vonNeumannRatio > {}) | \
                                          (vonNeumannRatio < {}))".format(min_number_of_good_observations, 10**selection_criteria["upper"], 10**selection_criteria["lower"]))
                                          
        logger.debug("Selected {} candidates from PDB".format(len(sources)))
        
        for source in sources:
            light_curve = ccd.light_curve(source["matchedSourceID"], barebones=True, clean=True)
            
            #if source["matchedSourceID"] == 2126:
            #    srcd = chip.sourcedata.readWhere("matchedSourceID == 2126")
            #    print light_curve.mag
            #    sys.exit(0)
            
            # If light curve doesn't have enough clean observations, skip it
            if len(light_curve.mjd) < min_number_of_good_observations: continue
            
            # Re-compute eta now that we've (hopefull) cleaned out any bad data
            clean_eta = pa.eta(light_curve)
            light_curve.eta = clean_eta

            # Compute delta chi-squared
            delta_chi_squared = pa.gaussian_line_delta_chi_squared(light_curve)
            light_curve.delta_chi_squared = delta_chi_squared
            
            # Try to fit a microlensing model, then subtract it, then recompute eta and see
            #   if it is still an outlier
            new_light_curve = pa.fit_subtract_microlensing(light_curve)
            new_eta = pa.eta(new_light_curve)
            baseline = light_curve.mjd.max() - light_curve.mjd.min()
            
            if (clean_eta <= lower_cut) and (new_eta >= lower_cut) and \
               (round(light_curve.u0, 2) < 1.34) and (round(light_curve.tE) > 2.) and \
               (delta_chi_squared > 10.) and (light_curve.tE < 2.*baseline) and (delta_chi_squared > 250):
                candidates.append(light_curve)
        
        ccd.close()
    
    return candidates

def save_candidates_to_mongodb(candidates, collection):
    """ Save a list of light curves to the given mongodb collection """
    
    for candidate in candidates:
        try:
            ra, dec = candidate.ra, candidate.dec
        except AttributeError:
            logging.error("Candidate light curve has no 'ra' or 'dec' attributes!")
            continue
        
        light_curve_document = mongo.light_curve_to_document(candidate)
        mongo.save_light_curve_document_to_collection(light_curve_document, collection)
    
    return True

if __name__ == "__main__":
    import pymongo
    from ptf.globals import config
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite any existing / cached files")
    parser.add_argument("-a", "--all", dest="all", default=False, action="store_true",
                    help="Run on all fields.")
    parser.add_argument("-f", "--field-id", dest="field_id", default=[], nargs="+", type=int,
                    help="The PTF field IDs to run on")
    
    parser.add_argument("--num-light-curves", dest="num_light_curves", default=100, type=int,
                    help="Number of light curves to select from each CCD.")
    parser.add_argument("--num-simulations", dest="num_simulations", default=100, type=int,
                    help="Number of simulations per light curve.")
    parser.add_argument("--min-observations", dest="min_num_observations", default=100, type=int,
                    help="The minimum number of observations")
    
    args = parser.parse_args()
    
    if args.verbose:
        stream_handler.setLevel(logging.DEBUG)
    elif args.quiet:
        stream_handler.setLevel(logging.ERROR)
    else:
        stream_handler.setLevel(logging.INFO)
        
    #connection = pymongo.Connection(config["db_address"], config["db_port"])
    #ptf = connection.ptf # the database
    #ptf.authenticate(config["db_user"], config["db_password"])
    ptf = mongo.PTFConnection()
    selection_criteria_collection = ptf.selection_criteria
    light_curve_collection = ptf.light_curves
    already_searched = ptf.already_searched
    
    fields = args.field_id
    min_number_of_good_observations = args.min_num_observations
    
    if args.all:
        all_fields = np.load("data/survey_coverage/fields_observations_R.npy")
        fields = all_fields[all_fields["num_exposures"] > args.min_num_observations]["field"]
        logger.info("Chose to run on all fields with >{} observations -- {} fields.".format(args.min_num_observations, len(fields)))
    
    for field_id in fields:
        # Skip field 101001 because it breaks the pipeline for some reason!
        if field_id == 101001: continue
        
        field = pdb.Field(field_id, "R")
        logger.info(redText("Field: {}".format(field.id)))
        
        lctest = light_curve_collection.find_one({"field_id" : field.id})
        was_searched = already_searched.find_one({"field_id" : field.id})
        if lctest != None or was_searched != None: 
            logger.info("Already found in candidate database!")
            continue
            
        selection_criteria = selection_criteria_collection.find_one({"field_id" : field.id, "index" : "eta"})
        
        if args.overwrite or selection_criteria == None:
            logger.info("Selection criteria not available for field.")
            selection_criteria = compute_selection_criteria_for_field(field, \
                                                                      number_of_fpr_light_curves=args.num_light_curves, \
                                                                      number_of_fpr_simulations_per_light_curve=args.num_simulations)
            
            if selection_criteria == None:
                already_searched.insert({"field_id" : field.id})
                continue
            logger.debug("Done with selection criteria simulation...saving to database")
            save_selection_criteria(field, selection_criteria, selection_criteria_collection, overwrite=args.overwrite)
        
        logger.debug("Selection criteria loaded")
        selection_criteria = selection_criteria_collection.find_one({"field_id" : field.id, "index" : "eta"})   
        candidates = select_candidates(field, selection_criteria)
        logger.info("Selected {} candidates after checking light curve data".format(len(candidates)))
        
        save_candidates_to_mongodb(candidates, light_curve_collection)
        already_searched.insert({"field_id" : field.id})