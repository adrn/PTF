# coding: utf-8
from __future__ import division

""" This module contains utilities for computing PTF's survey coverage for JavaScript visualizations. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import json
import math

# Third-party
import numpy as np
import apwlib.geometry as g

# Project
from ..util import get_logger
logger = get_logger(__name__)

__all__ = ["field_to_feature", "field_list_to_json"]

def field_to_feature(field):
    """ Converts a PTF Field object into a 'feature' object to be stuffed into JSON for the 
        interactive survey coverage viewer.
        
        Parameters
        ----------
        field : ptf.photometricdatabase.Field
            Must be a PTF Field object. See above module for details
    """
    feature = dict(type="Feature", id=str(field.id))
    properties = dict(name=str(field.id))
    geometry = dict(type="Polygon")
    
    ra = field.ra.degrees-180.0
    dec = field.dec.degrees
    
    ra_offset = camera_size_degrees[0]/math.cos(math.radians(dec))/2.
    dec_offset = camera_size_degrees[1]/2.
    
    min_ra = ra - ra_offset
    max_ra = ra + ra_offset
    
    min_dec = dec - dec_offset
    max_dec = dec + dec_offset
    
    coordinates = [[ [min_ra, min_dec], [max_ra, min_dec], [max_ra, max_dec], [min_ra, max_dec], [min_ra, min_dec] ]]
    geometry["coordinates"] = coordinates
    feature["geometry"] = geometry
    feature["properties"] = properties

    return feature

def field_list_to_json(fields, filename=None):
    """ Given a list of fields, create a FeatureCollection JSON file to use with the 
        interactive survey coverage viewer.
        
        The final structure should look like this, where "name" is the field id,
        and coordinates are a list of the coordinates of the 4 corners of the field.
        
            {"type" : "FeatureCollection",
                "features": [
                    {"type" : "Feature",
                     "properties" : { "name" : "2471"},
                                      "geometry": { "type" : "Polygon",
                                                    "coordinates" : [[ [100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0] ]]
                                                  },
                     "id":"2471"}, etc.
                            ]
            }      
        
        Parameters
        ----------
        fields : list, iterable
            Must be a list of PTF Field objects. See: ptf.photometricdatabase.Field
        filename : str (optional)
            If filename is specified, will save the JSON to the file. Otherwise, return the JSON
    """
    
    final_dict = dict(type="FeatureCollection", features=[])
    
    # Minimum and maximum number of observations for all fields
    min_obs = 1
    
    # There is one crazy outlier: the Orion field 101001, so I remove that for calculating the max...
    num_exposures = np.array([field.number_of_exposures for field in fields if field.id != 101001])
    max_obs = num_exposures.max()
    
    # Create a matplotlib lognorm scaler between these values
    scaler = matplotlib.colors.LogNorm(vmin=min_obs, vmax=max_obs)
    
    for field in fields:
        if field.number_of_exposures < 1:
            logger.debug("Skipping field {}".format(field))
            continue
        
        try:
            field.ra
            field.dec
        except AttributeError:
            this_field = all_fields[all_fields["id"] == field.id]
            if len(this_field) != 1: 
                logger.warning("Field {} is weird".format(field))
                continue
                
            field.ra = g.RA.fromDegrees(this_field["ra"][0])
            field.dec = g.Dec.fromDegrees(this_field["dec"][0])
        
        if field.dec.degrees < -40:
            logger.warning("Field {} is weird, dec < -40".format(field))
            continue
        
        feature = field_to_feature(field)
        
        # Determine color of field
        #rgb = cm.autumn(scaler(field.number_of_exposures))
        rgb = cm.gist_heat(scaler(field.number_of_exposures))
        feature["properties"]["color"] = mc.rgb2hex(rgb)
        feature["properties"]["alpha"] = scaler(field.number_of_exposures)*0.75 + 0.05
        feature["properties"]["number_of_observations"] = str(field.number_of_exposures)
        feature["properties"]["ra"] = "{:.5f}".format(field.ra.degrees)
        feature["properties"]["dec"] = "{:.5f}".format(field.dec.degrees)
        
        final_dict["features"].append(feature)
    
    blob = json.dumps(final_dict)
    
    if filename != None:
        f = open(filename, "wb")
        f.write(blob)
        f.close()
        
        return
    else:
        return blob