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
import os, sys
import math
import logging
import cPickle as pickle
import multiprocessing
import warnings

warnings.simplefilter("ignore")

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
from ptf.globals import min_number_of_good_observations
from ptf.util import get_logger, source_index_name_to_pdb_index, richards_qso
logger = get_logger(__name__)

def select_candidates(field, selection_criteria, num_fit_attempts=10):
    """ Select candidates from a field given the log10(selection criteria) from mongodb.

        The current selection scheme is to first select on eta, then to sanity check with
        delta chi-squared by making sure it's positive and >10.

    """

    eta_cut = 10**selection_criteria

    light_curves = []
    for ccd in field.ccds.values():
        logger.info(greenText("Starting with CCD {}".format(ccd.id)))
        chip = ccd.read()

        source_ids = chip.sources.readWhere("(ngoodobs > {}) & \
                                          (vonNeumannRatio > 0.0) & \
                                          (vonNeumannRatio < {}) & \
                                          ((ngoodobs/nobs) > 0.5)".format(min_number_of_good_observations, eta_cut), \
                                          field="matchedSourceID")

        logger.info("\tSelected {} pre-candidates from PDB".format(len(source_ids)))

        for source_id in source_ids:
            # APW: TODO -- this is still the biggest time hog!!! It turns out it's still faster than reading the whole thing into memory, though!
            light_curve = ccd.light_curve(source_id, barebones=True, clean=True)

            # If light curve doesn't have enough clean observations, skip it
            if light_curve != None and len(light_curve) < min_number_of_good_observations: continue

            # Compute the variability indices for the freshly cleaned light curve
            try:
                indices = pa.compute_variability_indices(light_curve, indices=["eta", "delta_chi_squared", "j", "k", "sigma_mu"])
            except ValueError:
                logger.warning("Failed to compute variability indices for light curve! {0}".format(light_curve))
                return False
            light_curve.indices = indices

            light_curve.tags = []
            light_curve.features = {}

            if light_curve.sdss_type() == "galaxy":
                light_curve.tags.append("galaxy")
                continue

            # If the object is not a Galaxy or has no SDSS data, try to get the SDSS colors
            #    to see if it passes the Richards et al. QSO color cut.
            sdss_colors = light_curve.sdss_colors("psf")
            qso_status = richards_qso(sdss_colors)
            if sdss_colors != None and qso_status:
                light_curve.tags.append("qso")

            candidate_status = pa.iscandidate(light_curve, lower_eta_cut=eta_cut)

            if candidate_status == "candidate" and "qso" not in light_curve.tags:
                light_curve.tags.append("candidate")
                light_curves.append(light_curve)
                continue

            if candidate_status == "subcandidate" and light_curve.indices["eta"] < eta_cut and not qso_status:
                # Try to do period analysis with AOV
                try:
                    peak_period = light_curve.features["aov_period"]
                    peak_power = light_curve.features["aov_power"]
                except KeyError:
                    try:
                        fp = pa.findPeaks_aov(light_curve.mjd.copy(), light_curve.mag.copy(), light_curve.error.copy(), 3, 1., 2.*light_curve.baseline, 1., 0.1, 20)
                    except ZeroDivisionError:
                        continue

                    light_curve.features["aov_period"] = peak_period = fp["peak_period"][0]
                    light_curve.features["aov_power"] = peak_power = max(fp["peak_period"])

                if (peak_period < 2.*light_curve.baseline):
                    if peak_power > 25.:
                        light_curve.tags.append("variable star")

                        if "subcandidate" in light_curve.tags:
                            light_curve.tags.pop(light_curve.tags.index("subcandidate"))

                        if light_curve not in light_curves:
                            light_curves.append(light_curve)

        ccd.close()

    return light_curves

def save_candidates_to_mongodb(candidates, collection, overwrite=False):
    """ Save a list of light curves to the given mongodb collection """

    for candidate in candidates:
        db_lightcurve = mongo.get_light_curve_from_collection(candidate.field_id, candidate.ccd_id, candidate.source_id, collection)
        if db_lightcurve != None and not overwrite:
            continue
        elif db_lightcurve != None and overwrite:
            collection.remove({"_id" : db_lightcurve["_id"]})

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
        light_curve_document = mongo.light_curve_to_document(candidate, indices=candidate.indices, microlensing_fit=microlensing_fit, features=candidate.features)
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
    parser.add_argument("-r", "--field-range", dest="field_range", default=[],
                    help="A range of PTF field IDs to run on")

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

    #indices = ["eta", "delta_chi_squared", "j", "k", "sigma_mu"]
    #indices = ["eta", "j", "delta_chi_squared"]
    indices = ["eta"]

    ptf = mongo.PTFConnection()
    light_curve_collection = ptf.light_curves
    field_collection = ptf.fields

    field_ids = args.field_id

    if args.all:
        all_fields = np.load("data/survey_coverage/fields_observations_R.npy")
        field_ids = all_fields[all_fields["num_exposures"] > min_number_of_good_observations]["field"]
        logger.info("Chose to run on all fields with >{} observations = {} fields.".format(min_number_of_good_observations, len(field_ids)))

    if args.field_range:
        min_field_id, max_field_id = map(int, args.field_range.split("-"))

        all_fields = np.load("data/survey_coverage/fields_observations_R.npy")
        field_ids = all_fields[all_fields["num_exposures"] > min_number_of_good_observations]["field"]
        field_ids = field_ids[(field_ids >= min_field_id) & (field_ids < max_field_id)]

    for field_id in sorted(field_ids):
        # Skip field 101001 because the data hasn't been reduced by the PTF pipeline?
        if field_id == 101001: continue

        field = pdb.Field(field_id, "R")
        logger.info("Field: {}".format(field.id))

        # There is some strangeness to getting the coordinates for a Field 
        # -- this just makes it stupidproof
        try:
            if field.ra == None or field.dec == None:
                raise AttributeError()
        except AttributeError:
            logger.warn("Failed to get coordinates for this field!")
            continue

        # See if field is in database, remove it if we need to overwrite
        if args.overwrite:
            field_collection.remove({"_id" : field.id})
            #light_curve_collection.remove({"field_id" : field.id})

            # Only remove light curves that haven't been looked at
            for light_curve_document in light_curve_collection.find({"field_id" : field.id}):
                if not light_curve_document.has_key("viewed") or not light_curve_document["viewed"]:
                    light_curve_collection.remove({"_id" : light_curve_document["_id"]})
                else:
                    continue

        if args.overwrite_lcs:
            field_collection.update({"_id" : field.id}, 
                                    {"$set" : {"already_searched" : False}})
            #light_curve_collection.remove({"field_id" : field.id})

            # Only remove light curves that haven't been looked at
            for light_curve_document in light_curve_collection.find({"field_id" : field.id}):
                if not light_curve_document.has_key("viewed") or not light_curve_document["viewed"]:
                    light_curve_collection.remove({"_id" : light_curve_document["_id"]})
                else:
                    continue

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
        selection_criteria = field_collection.find_one({"_id" : field.id}, 
                                                       fields=["selection_criteria"])
        selection_criteria = selection_criteria["selection_criteria"]

        if args.overwrite or selection_criteria == None:
            logger.info("Selection criteria not available for field.")

            try:
                var_indices = vi.var_indices_for_simulated_light_curves(field, number_of_light_curves=args.num_light_curves, number_of_simulations_per_light_curve=args.num_simulations, indices=indices)
            except UnboundLocalError:
                logger.warn("Field is wonky! Skipping...")
                field_collection.update({"_id" : field.id}, {"$set" : {"already_searched" : False}})
                field.close()
                continue

            selection_criteria = vi.compute_selection_criteria(var_indices, indices=indices, fpr=0.01)

            if selection_criteria == None:
                # Something went wrong, or there is no good data for this field?
                field_collection.update({"_id" : field.id}, {"$set" : {"already_searched" : True}})
                continue

            logger.debug("Done with selection criteria simulation...saving to database")
            selection_criteria_document = {}
            for index in indices:
                selection_criteria_document[index] = dict()
                selection_criteria_document[index] = selection_criteria[index]

            #selection_criteria_document["upper"] = selection_criteria["eta"]["upper"]
            #selection_criteria_document["lower"] = selection_criteria["eta"]["lower"]
            field_collection.update({"_id" : field.id}, {"$set" : {"selection_criteria" : selection_criteria_document}})

        logger.debug("Selection criteria loaded")
        selection_criteria_document = field_collection.find_one({"_id" : field.id}, 
                                                                fields=["selection_criteria"])
        selection_criteria_document = selection_criteria_document["selection_criteria"]["eta"]
        
        # APW: ok so what is select_candidates doing?
        selected_light_curves = select_candidates(field, selection_criteria_document)
        logger.info("Selected {} light curves.".format(len(selected_light_curves)))

        save_candidates_to_mongodb(selected_light_curves, light_curve_collection)
        field_collection.update({"_id" : field.id}, {"$set" : {"already_searched" : True}})
