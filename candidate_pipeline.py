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

def quality_cut(sourcedata, ccd_edge_cutoff=25):
    """ This function accepts a Pytables table object (from the 'sourcedata' table)
        and returns only the rows that pass the given quality cuts.
        
        Parameters
        ----------
        sourcedata : table
            A pytables Table object -> 'chip.sourcedata'
        ccd_edge_cutoff : int
            Define the cutoff for sources near the edge of a CCD. The cut will remove
            all data points where the source is nearer than this limit to the edge.
    """
    logger.debug(greenText("/// remove_bad_data ///"))
    
    x_cut1, x_cut2 = ccd_edge_cutoff, pg.ccd_size[0] - ccd_edge_cutoff
    y_cut1, y_cut2 = ccd_edge_cutoff, pg.ccd_size[1] - ccd_edge_cutoff
    
    sourcedata = sourcedata.readWhere('(sextractorFlags < 8) & \
                                       (x_image > {}) & (x_image < {}) & \
                                       (y_image > {}) & (y_image < {}) & \
                                       (magErr < 0.3) & \
                                       (mag > 13.5) & (mag < 22)'.format(x_cut1, x_cut2, y_cut1, y_cut2))
    
    sourcedata = sourcedata[(sourcedata["sextractorFlags"] & 1) == 0]
    sourcedata = sourcedata[np.isfinite(sourcedata["mag"]) & \
                            np.isfinite(sourcedata["mjd"]) & \
                            np.isfinite(sourcedata["magErr"])]
                            
    return sourcedata

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
    
    all_J = chip.sources.col("stetsonJ")
    all_eta = chip.sources.col("vonNeumannRatio")
    
    J = all_J[all_J > 0.]
    med_J = np.median(J)
    sig_J = np.std(J)
    J_significance = 3.
    
    eta = all_eta[all_eta > 0.]
    med_eta = np.median(eta)
    sig_eta = np.std(eta)
    eta_significance = 2.
    
    exposures = chip.exposures[:]
    
    sourcedata = quality_cut(chip.sourcedata)
    
    light_curves = []
    for row in chip.sources[:]:
        if (row["stetsonJ"] > (med_J+J_significance*sig_J)) and \
           ((med_eta-eta_significance*sig_eta) < row["vonNeumannRatio"] < (med_eta+eta_significance*sig_eta)) and \
           (row["ngoodobs"] > 25):
            
            # TODO: Look for clumps brighter or fainter -- if clump has less than 3 points, throw it out?
            this_source = sourcedata[sourcedata["matchedSourceID"] == row["matchedSourceID"]]
            faint_idx, = np.where(this_source["mag"] > (row["referenceMag"] + 4*row["magRMS"]))
            bright_idx, = np.where(this_source["mag"] < (row["referenceMag"] - 4*row["magRMS"]))
            
            if len(faint_idx) < 3:
                this_source = this_source[this_source["mag"] < (row["referenceMag"] + 4*row["magRMS"])]
            
            if len(bright_idx) < 3:
                this_source = this_source[this_source["mag"] > (row["referenceMag"] - 4*row["magRMS"])]
            
            light_curve = PTFLightCurve(mjd=this_source["mjd"], mag=this_source["mag"], error=this_source["magErr"])
            
            if len(light_curve.mjd) > 25:
                # TODO: Light curve object should have more metadata
                var_indices = pa.compute_variability_indices(light_curve, indices=["j", "k", "eta", "delta_chi_squared", "simga_mu"])
                
                if (var_indices["j"] > (med_J+J_significance*sig_J)) and \
                   ((med_eta-eta_significance*sig_eta) < var_indices["eta"] < (med_eta+eta_significance*sig_eta)):
                   
                    light_curve.metadata = np.array(row)
                    light_curve.exposures = exposures
                    light_curves.append(light_curve)
    
    ccd.close()
    
    return light_curves

def save_light_curves(light_curves, path="data/candidates/light_curves"):
    """ Takes a list of PTFLightCurve objects and saves them each to .npy files """
    
    for light_curve in light_curves:
        filename = "field{:06d}_ccd{:02d}_source{:06d}.pickle".format(light_curve.field_id, light_curve.ccd_id, light_curve.source_id)
        f = open(os.path.join(path, filename), "w")
        pickle.dump(light_curve, f)
        f.close()

def run_pipeline():
    """ Given a list of PTF fields (PTFField objects), run the candidate selection
        pipeline on each field.
    """
    logger.debug(greenText("/// Running Pipeline! ///"))
    
    R = pdb.Filter("R")
    import time
    
    # Select out all fields that have been observed many times
    # TODO: for now, I just know 110002 is a good one!
    #fields = [pdb.Field(110001, filter=R), pdb.Field(110002, filter=R), pdb.Field(110003, filter=R), pdb.Field(110004, filter=R)]
    fields = [pdb.Field(110002, filter=R)]
    for field in fields:
        for ccd in field.ccds.values():
            a = time.time()
            candidate_light_curves = select_candidates_from_ccd(ccd)
            print len(candidate_light_curves), time.time() - a
            save_light_curves(candidate_light_curves)

if __name__ == "__main__":
    logger.setLevel(logging.INFO)
    run_pipeline()