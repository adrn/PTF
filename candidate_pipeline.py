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
from ptf.ptflightcurve import PTFLightCurve
import ptf.globals as pg

# Create logger
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# TODO: 
# - I need to separate my selection criteria from this code. Then I need to have a function that 
#   computes the detection efficiency for a given field.
# - Rather than writing out a bunch of files, this should save candidates to a database (sqlite file)

def select_candidates_from_ccd(ccd):
    """ Given a pytables object from the photometric database for one field/ccd,
        select out candidate microlensing events.
        
        e.g. if the file is 
            dbFile = tables.openFile('matches/match_02_100083_10.pytable')
        you should pass this object:
            dbFile.root.filter02.field100083.chip10
    """
    logger.debug(greenText("/// select_candidates_from_ccd ///"))
    
    chip = ccd.read()
    
    # Read in sources where eta > 0.
    good_eta_sources = chip.sources.readWhere("(ngoodobs > 25) & (vonNeumannRatio > 0.)")
    
    # Read in the full vonNeumannRatio column to figure out the distribution of values for this parameter
    all_eta = good_eta_sources["vonNeumannRatio"]
    
    # Use the values of eta from all light curves to defined selection criteria
    mean_eta = np.mean(all_eta)
    sig_eta = np.std(all_eta)
    eta_significance = 2.
    selected_sources = chip.sources.readWhere("(ngoodobs > 25) & (vonNeumannRatio < {})".format(mean_eta-eta_significance*sig_eta))
    
    exposures = chip.exposures[:]
    
    light_curves = []
    for row in selected_sources:
        this_sourcedata = pdb.quality_cut(chip.sourcedata, source_id=row["matchedSourceID"])
        
        if len(this_sourcedata) > 25:
            light_curve = PTFLightCurve(mjd=this_sourcedata["mjd"], mag=this_sourcedata["mag"], error=this_sourcedata["magErr"])
            var_indices = pa.compute_variability_indices(light_curve, indices=["eta"])
            
            # Check that after cleaning up bad data, the light curve still passes the cut
            if (var_indices["eta"] < (mean_eta-eta_significance*sig_eta)): 
                light_curve.metadata = np.array(row)
                light_curve.exposures = exposures
                light_curves.append(light_curve)
    
    ccd.close()
    
    # TODO: Left off looking at thus function -- seems to be selecting way too many candidates! Look in to this..
    
    return light_curves

def save_light_curves(light_curves, path="data/candidates/light_curves"):
    """ Takes a list of PTFLightCurve objects and saves them each to .npy files """
    
    for light_curve in light_curves:
        filename = "field{:06d}_ccd{:02d}_source{:06d}.pickle".format(int(light_curve.exposures["fieldID"][0]), int(light_curve.exposures["ccdID"][0]), int(light_curve.metadata["matchedSourceID"]))
        print filename
        f = open(os.path.join(path, filename), "w")
        pickle.dump(light_curve, f)
        f.close()

def run_pipeline():
    """ Given a list of PTF fields (PTFField objects), run the candidate selection
        pipeline on each field.
    """
    logger.debug(greenText("/// Running Pipeline! ///"))

    import time    
    # Select out all fields that have been observed many times
    # TODO: for now, I just know 110002 is a good one!
    #fields = [pdb.Field(110001, filter=R), pdb.Field(110002, filter=R), pdb.Field(110003, filter=R), pdb.Field(110004, filter=R)]
    fields = [pdb.Field(3756, filter="R"), pdb.Field(100043, filter="R"), pdb.Field(120001, filter="R")]
    for field in fields:
        for ccd in field.ccds.values():
            a = time.time()
            candidate_light_curves = select_candidates_from_ccd(ccd)
            logger.info("CCD {}, {} light curves, took {} seconds".format(ccd.id, len(candidate_light_curves), time.time() - a))
            save_light_curves(candidate_light_curves)

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    run_pipeline()