# coding: utf-8
from __future__ import division

""" This module contains a pipeline for selecting candidate microlensing events from
    David Levitan's photometric database. The new pipeline looks something like this:
        For each field, we run simulations to determine the selection criteria for "interesting"
        variable sources, store those values in mongodb, then we use these criteria to go back
        to the PDB and select out candidates. 
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import math
import logging
import cPickle as pickle
import multiprocessing

# Third-party
import numpy as np
import scipy.optimize as so
try:
    from apwlib.globals import greenText, yellowText, redText
except ImportError:
    raise ImportError("apwlib not found! \nDo: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install' to install.")

# Project
import ptf.db.photometric_database as pdb
import ptf.db.mongodb as mongo
import ptf.analyze as pa
import ptf.variability_indices as vi
import ptf.simulation as sim
from ptf.globals import min_number_of_good_observations
from ptf.util import get_logger, source_index_name_to_pdb_index, richards_qso
logger = get_logger(__name__)

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
        
        source_ids = chip.sources.readWhere("(ngoodobs > {}) & \
                                          (vonNeumannRatio > 0.0) & \
                                          ((vonNeumannRatio > {}) | \
                                          (vonNeumannRatio < {}))".format(min_number_of_good_observations, 10**selection_criteria["upper"], 10**selection_criteria["lower"]), \
                                          field="matchedSourceID")
                                          
        logger.debug("Selected {} candidates from PDB".format(len(source_ids)))
        
        for source_id in source_ids:
            # APW: TODO -- this is still the biggest time hog!!! It turns out it's still faster than reading the whole thing into memory, though!
            light_curve = ccd.light_curve(source_id, barebones=True, clean=True)
            
            # If light curve doesn't have enough clean observations, skip it
            if light_curve != None and len(light_curve) < min_number_of_good_observations: continue
            
            # Re-compute eta now that we've (hopefully) cleaned out any bad data
            indices = pa.compute_variability_indices(light_curve, indices=["eta", "delta_chi_squared", "j", "k", "sigma_mu"])
            light_curve.indices = indices
            
            # TODO: do something with this num_attempts!
            num_attempts = 5
            ml_chisq = 1E6
            for ii in range(num_attempts):
                params = pa.fit_microlensing_event(light_curve)
                new_chisq = params["result"].chisqr
                
                if new_chisq < ml_chisq:
                    ml_chisq = new_chisq
                    break
            
            # Try to fit a microlensing model, then subtract it, then recompute eta and see
            #   if it is still an outlier
            new_light_curve = pa.fit_subtract_microlensing(light_curve, fit_data=params)
            new_eta = pa.eta(new_light_curve)
            
            light_curve.tags = []
            if (indices["eta"] <= lower_cut) and (new_eta >= lower_cut) and \
               (round(light_curve.u0, 2) < 1.34) and (round(light_curve.tE) > 2.) and \
               (light_curve.tE < 2.*light_curve.baseline) and indices["delta_chi_squared"] > 100:
                
                # Count number of data points between t0-3*tE and t0+3*tE, make sure we have at least a few above 2sigma
                sliced_lc = light_curve.slice_mjd(params["t0"].value-3.*params["tE"].value, params["t0"].value+3.*params["tE"].value)
                if sum(sliced_lc.mag > 2.*np.median(light_curve.mag)) < 3:
                    logger.debug("Light curve has fewer than 3 observations in a peak > 2sigma: {}".format((field.id, ccd.id, light_curve.source_id)))
                    continue
                    
                if light_curve not in candidates: candidates.append(light_curve)

                # Here I can try this: light_curve.sdss_colors
                #sdss_colors = light_curve.sdss_colors("psf")
                #if sdss_colors != None and richards_qso(sdss_colors):
                #    light_curve.tags.append("quasar candidate")
            
            if (indices["eta"] <= lower_cut):
                try:
                    fp = pa.findPeaks_aov(light_curve.mjd.copy(), light_curve.mag.copy(), light_curve.error.copy(), 3, 1., 2.*light_curve.baseline, 1., 0.1, 20)
                except ZeroDivisionError:
                    continue
                    
                if (fp["peak_period"][0] < 2.*light_curve.baseline):
                    if max(fp["peak_power"]) > 25.:
                        light_curve.tags.append("variable star")
                        if light_curve not in candidates: candidates.append(light_curve)
        
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
        
        microlensing_fit = {}
        microlensing_fit["tE"] = candidate.tE
        microlensing_fit["t0"] = candidate.t0
        microlensing_fit["u0"] = candidate.u0
        microlensing_fit["m0"] = candidate.m0
        microlensing_fit["chisqr"] = candidate.chisqr
        light_curve_document = mongo.light_curve_to_document(candidate, indices=candidate.indices, microlensing_fit=microlensing_fit)
        try:
            light_curve_document["tags"] += candidate.tags
        except AttributeError:
            pass
        mongo.save_light_curve_document_to_collection(light_curve_document, collection)
    
    return True

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite any existing / cached files")
    parser.add_argument("--overwrite-light-curves", action="store_true", dest="overwrite_lcs", default=False,
                    help="Keep the selection criteria, just redo the search.")
                    
    parser.add_argument("-a", "--all", dest="all", default=False, action="store_true",
                    help="Run on all fields.")
    parser.add_argument("-f", "--field-id", dest="field_id", default=[], nargs="+", type=int,
                    help="The PTF field IDs to run on")
    
    parser.add_argument("--num-light-curves", dest="num_light_curves", default=100, type=int,
                    help="Number of light curves to select from each CCD.")
    parser.add_argument("--num-simulations", dest="num_simulations", default=100, type=int,
                    help="Number of simulations per light curve.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
        
    ptf = mongo.PTFConnection()
    light_curve_collection = ptf.light_curves
    field_collection = ptf.fields
    
    field_ids = args.field_id
    
    if args.all:
        all_fields = np.load("data/survey_coverage/fields_observations_R.npy")
        field_ids = all_fields[all_fields["num_exposures"] > args.min_num_observations]["field"]
        logger.info("Chose to run on all fields with >{} observations = {} fields.".format(args.min_num_observations, len(field_ids)))
    
    for field_id in sorted(field_ids):
        # Skip field 101001 because it breaks the pipeline for some reason!
        if field_id == 101001: continue
        
        field = pdb.Field(field_id, "R")
        logger.info("Field: {}".format(field.id))
        
        # There is some strangeness to getting the coordinates for a Field -- this just makes it stupidproof
        try:
            if field.ra == None or field.dec == None:
                raise AttributeError()
        except AttributeError:
            logger.warn("Failed to get coordinates for this field!")
            continue
        
        # See if field is in database, remove it if we need to overwrite
        if args.overwrite:
            field_collection.remove({"_id" : field.id})
        
        if args.overwrite_lcs:
            field_collection.update({"_id" : field.id}, {"$set" : {"already_searched" : False}})
            light_curve_collection.remove({"field_id" : field.id})
        
        # Try to get field from database, if it doesn't exist, create it and insert
        field_document = field_collection.find_one({"_id" : field.id})
        if field_document == None:
            logger.debug("Field document not found -- creating and loading into mongodb!")
            field_document = mongo.field_to_document(field)
            field_collection.insert(field_document)
        
        # If the field has "already_searched" = True, skip it
        field_doc = field_collection.find_one({"_id" : field.id}, fields=["already_searched"])
        if field_doc["already_searched"]: 
            logger.info("Field already searched and candidates added to candidate database!")
            continue
        
        # Check to see if the selection criteria for this field is already loaded into the database
        selection_criteria = field_collection.find_one({"_id" : field.id}, fields=["selection_criteria"])["selection_criteria"]
        
        if args.overwrite or selection_criteria == None:
            logger.info("Selection criteria not available for field.")
            
            # TODO: Do I want to write var_indices to a file?
            var_indices = vi.var_indices_for_simulated_light_curves(field, number_of_light_curves=number_of_light_curves, number_of_simulations_per_light_curve=number_of_simulations_per_light_curve, indices=indices)
            selection_criteria = compute_selection_criteria_for_field(var_indices, indices=args.indices, fpr=0.01)
            
            if selection_criteria == None:
                # Something went wrong, or there is no good data for this field?
                field_collection.update({"_id" : field.id}, {"$set" : {"already_searched" : True}})
                continue
                
            logger.debug("Done with selection criteria simulation...saving to database")
            selection_criteria_document = {}
            selection_criteria_document["upper"] = selection_criteria["eta"]["upper"]
            selection_criteria_document["lower"] = selection_criteria["eta"]["lower"]
            field_collection.update({"_id" : field.id}, {"$set" : {"selection_criteria" : selection_criteria_document}})
        
        logger.debug("Selection criteria loaded")
        selection_criteria_document = field_collection.find_one({"_id" : field.id}, fields=["selection_criteria"])["selection_criteria"]

        candidates = select_candidates(field, selection_criteria_document)
        logger.info("Selected {} candidates after checking light curve data".format(len(candidates)))
        
        save_candidates_to_mongodb(candidates, light_curve_collection)
        field_collection.update({"_id" : field.id}, {"$set" : {"already_searched" : True}})