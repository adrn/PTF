# coding: utf-8
from __future__ import division

""" 
    Interface for the PTF microlensing event candidate database 
    (using MongoDB)
    
    Server: deimos.astro.columbia.edu
    Database: ptf
    Collections:
        light_curves -- where the actual light curve data is stored
        candidate_status -- used to store the status of a candidate, e.g. "boring", "interesting", etc.
        table_state -- used to store the state of the table on my PTF candidate website
        selection_criteria -- we run simulations for each field to compute the selection boundary for eta
    
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np
import pymongo

# Project 
from ..ptflightcurve import PDBLightCurve
from ..globals import config, _base_path
from ..analyze import analyze as pa
from ..photometricdatabase import Field

# Create logger for this module
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Database connection
def PTFConnection():
    connection = pymongo.Connection(config["db_address"], config["db_port"])
    ptf = connection.ptf # the database
    ptf.authenticate(config["db_user"], config["db_password"])
    return ptf

def light_curve_to_document(light_curve, **kwargs):
    """ Converts a PTFLightCurve object into a dictionary to be
        loaded into MongoDB.
    """
    document = dict()
    
    if not kwargs.has_key("indices"):
        # Compute indices
        document["indices"] = pa.compute_variability_indices(light_curve, indices=["eta", "j", "k", "delta_chi_squared", "sigma_mu"])
    else:
        document["indices"] = kwargs["indices"]
    
    if not kwargs.has_key("microlensing_fit"):
        # Compute microlensing fit parameters
        document["microlensing_fit"] = pa.fit_microlensing_event(light_curve)
    else:
        document["microlensing_fit"] = kwargs["microlensing_fit"]
    
    document["mjd"] = list(light_curve.mjd)
    document["mag"] = list(light_curve.mag)
    document["error"] = list(light_curve.error)
    document["filter"] = "R"
    
    try:
        document["field_id"] = light_curve.field_id
        document["ccd_id"] = light_curve.ccd_id
        document["source_id"] = light_curve.source_id
        document["ra"] = light_curve.ra
        document["dec"] = light_curve.dec
    except AttributeError:
        print "You must pass a PTF PDBLightCurve object in, not a PTFLightCurve object"
        raise
    
    document["tags"] = []
    
    return document

def document_to_light_curve(document):
    """ Converts a MongoDB document dictionary back into a PTF Light Curve object """
            
    return PDBLightCurve(**document)    

def get_light_curve_from_collection(field, ccd, source_id, collection, filter="R"):
    """ Get a light curve from MongoDB from the specified field, ccd, and source_id """
    
    if isinstance(field, int):
        field_id = field
    else:
        field_id = int(field.id)
        filter = field.filter.name
    
    if isinstance(ccd, int):
        ccd_id = ccd
    else:
        ccd_id = int(ccd.id)
    
    document = collection.find_one({"field_id" : field_id,
                                    "ccd_id" : ccd_id,
                                    "source_id" : int(source_id),
                                    "filter" : filter})
    
    if document == None:
        return None
    else:
        return document_to_light_curve(document)

def save_light_curve_document_to_collection(light_curve_document, collection, overwrite=False):
    """ Saves the given light curve to the specified mongodb collection """
    
    search_existing = collection.find_one({"field_id" : light_curve_document["field_id"],
                                           "ccd_id" : light_curve_document["ccd_id"],
                                           "source_id" : light_curve_document["source_id"],
                                           "filter" : light_curve_document["filter"]})
    
    if search_existing != None:
        if overwrite:
            logger.debug("Overwriting existing light curve {} {} {}".format(light_curve_document["field_id"], light_curve_document["ccd_id"], light_curve_document["source_id"]))
            collection.remove({"field_id" : light_curve_document["field_id"],
                               "ccd_id" : light_curve_document["ccd_id"],
                               "source_id" : light_curve_document["source_id"]})
        else:
            logger.debug("Light curve {} {} {} already exists in database!".format(light_curve_document["field_id"], light_curve_document["ccd_id"], light_curve_document["source_id"]))
            return False
    else:
        collection.insert(light_curve_document)
    
    return True

def update_light_curve_document_tags(light_curve_document, tags, light_curve_collection):
    """ Add a tag to the light curve object """
    
    if not isinstance(tags, list): return False
    
    light_curve_collection.update({"_id" : light_curve_document["_id"]},
                                  {"$set": { "tags" : [tag.lower() for tag in tags] }}
                                 );

    return True

def update_light_curves(collection):
    """ Update the data in all the light curves in the specified collection """
    # TODO!

def field_to_document(field, **kwargs):
    """ Turn a PDB Field object into a field document for mongodb """
    
    selection_criteria = kwargs.get("selection_criteria", None)
    
    new_field = {}
    new_field["_id"] = field.id
    new_field["selection_criteria"] = selection_criteria
    new_field["already_searched"] = False
    new_field["filter"] = field.filter.name
    new_field["ra"] = field.ra.degrees
    new_field["dec"] = field.dec.degrees
    new_field["exposures"] = {}

    exposures_per_ccd = field.exposures
    for ccd_id in range(12):
        try:
            new_field["exposures"][str(ccd_id)] = {"mjd" : list(exposures_per_ccd[ccd_id]["obsMJD"].astype(float)), \
                                              "background" : list(exposures_per_ccd[ccd_id]["obsMJD"].astype(float)), \
                                              "airmass" : list(exposures_per_ccd[ccd_id]["airmass"].astype(float)), \
                                              "seeing" : list(exposures_per_ccd[ccd_id]["seeing"].astype(float)), \
                                              "moonIllumFrac" : list(exposures_per_ccd[ccd_id]["moonIllumFrac"].astype(float))
                                             }
        except:
            continue
    
    return new_field

def load_all_fields(field_collection):
    """ Load all of the field information into mongodb """
    
    R_fields = np.load("data/survey_coverage/fields_observations_R.npy")
    loaded_field_ids = [a["_id"] for a in field_collection.find(fields=["_id"])]
    
    field_ids_to_load = list(set(R_fields["field"]).difference(set(loaded_field_ids)))
    logger.debug("{} fields to load".format(len(field_ids_to_load)))
    
    for field_id in field_ids_to_load:
        logger.debug("Field {}".format(field_id))
        field = Field(field_id, "R")
        
        if field.ra == None: 
            print field.id
            continue
        
        # Double check that it's not already in there!
        if field_collection.find_one({"_id" : field.id}) != None:
            logger.info("Field is already loaded! What happened?! {}".format(field.id))
            continue
        
        field_doc = field_to_document(field)
        field_collection.insert(field_doc)
        logger.debug("Field loaded into mongodb {}".format(field.id))
    
    return True
    