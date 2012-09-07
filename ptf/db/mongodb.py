# coding: utf-8
from __future__ import division

""" 
    Interface for the PTF microlensing event candidate database 
    (using MongoDB)
    
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import pymongo
import numpy as np

# Project 
from ..ptflightcurve import PDBLightCurve

# Create logger for this module
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)
   
def light_curve_to_document(light_curve, **kwargs):
    """ Converts a PTFLightCurve object into a dictionary to be
        loaded into MongoDB.
    """
    document = dict()
    
    for key,value in kwargs.items():
        document[key] = value
    
    document["mjd"] = list(light_curve.mjd)
    document["mag"] = list(light_curve.mag)
    document["error"] = list(light_curve.error)
    
    try:
        document["field_id"] = light_curve.field_id
        document["ccd_id"] = light_curve.ccd_id
        document["source_id"] = light_curve.source_id
    except AttributeError:
        print "You must pass a PTF PDBLightCurve object in, not a PTFLightCurve object"
        raise
    
    return document

def document_to_light_curve(document):
    """ Converts a MongoDB document dictionary back into a PTF Light Curve object """
            
    return PDBLightCurve(mjd=document["mjd"], mag=document["mag"], error=document["error"],
                         field_id=document["field_id"], ccd_id=document["ccd_id"], source_id=document["source_id"])
    

def get_light_curve_from_collection(field, ccd, source_id, collection):
    """ Get a light curve from MongoDB from the specified field, ccd, and source_id """
    
    if isinstance(field, int):
        field_id = field
    else:
        field_id = int(field.id)
    
    if isinstance(ccd, int):
        ccd_id = field
    else:
        ccd_id = int(ccd.id)
        
    document = collection.find_one({"field_id" : field_id,
                                    "ccd_id" : ccd_id,
                                    "source_id" : int(source_id)})    
    if document == None:
        return None
    else:
        return document_to_light_curve(document)

def save_light_curve_to_collection(light_curve, collection, **kwargs):
    """ Saves the given light curve to the specified mongodb collection """
    
    search_existing = collection.find_one({"field_id" : light_curve.field_id,
                                           "ccd_id" : light_curve.ccd_id,
                                           "source_id" : light_curve.source_id})
    
    if search_existing != None:
        logger.warning("Light curve {} {} {} already exists in database!".format(light_curve.field_id, light_curve.ccd_id, light_curve.source_id))
    else:
        document = light_curve_to_document(light_curve, **kwargs)
        collection.insert(document)
    
    return True