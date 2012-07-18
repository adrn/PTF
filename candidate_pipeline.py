# coding: utf-8
from __future__ import division

""" This module contains a pipeline for selecting candidate microlensing events from
    David Levitan's photometric database.
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import math
import logging

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

# Create logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def remove_bad_data(data):
    """ Given a data structure with named columns, e.g. anything
        that can be called like this: data['blah'], make cuts on
        the photometric error flags.
    """
    
    logger.debug(greenText("/// remove_bad_data ///"))
    logger.debug("Initial number of light curves: {}".format(len(np.unique(data["matchedSourceID"]))))
    
    # First remove any NaN of inf values from the data
    data = data[np.isfinite(data["mag"])]
    data = data[np.isfinite(data["mjd"])]
    data = data[np.isfinite(data["magErr"])]
    
    # Then remove points with huge error bars or magnitude values outside
    #   the linear regime
    data = data[data["magErr"] < 0.5]
    data = data[data["mag"] > 13.5]
    data = data[data["mag"] < 22]
    
    # Finally, make cuts on sextractor flags to remove blended light curves
    data = data[(data["sextractorFlags"] < 8) == 0]
    data = data[(data["sextractorFlags"] & 1) == 0]
    #data = data[(data["relPhotFlags"] & 5949) == 0]
    
    logger.debug("Final number of light curves: {}".format(len(np.unique(data["matchedSourceID"]))))
    
    return data

def select_candidates_from_ccd(ccd_data):
    """ Given a pytables object from the photometric database for one field/ccd,
        select out candidate microlensing events.
        
        e.g. if the file is 
            dbFile = tables.openFile('matches/match_02_100083_10.pytable')
        you should pass this object:
            dbFile.root.filter02.field100083.chip10
    """
    raise NotImplementedError()
    
    candidate_sources = ccd_data.sources.readWhere("")
    idx = np.in1d(ccd_data.sourcedata.col("matchedSourceID"), candidate_sources["matchedSourceID"])
    return chip.sourcedata[idx]

def run_pipeline(fields):
    """ Given a list of PTF fields (PTFField objects), run the candidate selection
        pipeline on each field.
    """
    raise NotImplementedError()