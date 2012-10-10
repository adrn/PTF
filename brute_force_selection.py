# coding: utf-8
from __future__ import division

""" Run through every field, look for any light curve with ngoodobs > 100,
    use Lev-Mar to fit a Gaussian (analytic derivatives) and a straight line,
    any large outliers get flagged. 
    
    In detail:
        - Choose a Field
        - Choose a CCD
        - Select all light curves with > 100 observations
        - Compute delta chi squared for all light curves that pass cut
        
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import time
import multiprocessing

# Third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from apwlib.globals import greenText

# Create logger for this module
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Project
from ptf.globals import config
from ptf.db.mongodb import *
try:
    import ptf.photometricdatabase as pdb
    import ptf.analyze.analyze as pa
except ImportError:
    logger.warning("photometric database modules failed to load! If this is on Navtara, you made a boo-boo.")
    
def process_light_curve(light_curve):
    """ Process an individual light curve from the multiprocessing pool """
    delta_chi_squared = pa.gaussian_line_delta_chi_squared(light_curve)
    light_curve.delta_chi_squared = delta_chi_squared
    
    return (light_curve, delta_chi_squared)

def find_candidates_on_field(field, min_num_obs=100, Nsigma=3.):
    """ Given a PTF Field object, find microlensing events """
    
    candidates = []
    for ccd in field.ccds.values():
        chip = ccd.read()
        
        source_ids = chip.sources.readWhere("(ngoodobs > {}) & \
                                             (referenceMag > 14.3) & \
                                             (referenceMag < 21)".format(min_num_obs), field="matchedSourceID")
                                             
        logger.info("{} sources with more than {} observations".format(len(source_ids), min_num_obs))
        
        delta_chi_squareds = []
        light_curves = []
        for light_curve in ccd.light_curves(source_ids):
            #logger.debug("Source: {}".format(light_curve.source_id))
            
            if len(light_curve.mjd) < 50: continue
            
            light_curves.append(light_curve)
        
        ccd_candidates = []
        delta_chi_squareds = []
        def callback(tup):
            candidate, delta_chi_squared = tup
            ccd_candidates.append(candidate)
            delta_chi_squareds.append(delta_chi_squared)
        
        pool = multiprocessing.Pool(processes=8)
        for lc in light_curves:
            pool.apply_async(process_light_curve, args=(lc,), callback=callback)
        
        pool.close()
        pool.join()
        
        print len(delta_chi_squareds)
        
        #delta_chi_squareds = np.array(delta_chi_squareds, dtype=[("source_id", int), ("dcs", float)])
        delta_chi_squareds = np.array(delta_chi_squareds)
        
        w, = np.where(delta_chi_squareds >= 0.)
        selected_dcs = delta_chi_squareds[w]
        selected_lcs = [ccd_candidates[ii] for ii in w]
        
        log_dcs = np.log10(selected_dcs)
        mu, sigma = np.mean(log_dcs), np.std(log_dcs)
        
        w, = np.where(log_dcs > (mu+Nsigma*sigma))
        logger.info("\t {} candidates selected".format(len(w)))
        candidates += [selected_lcs[ii] for ii in w]
        
        #return candidates
        # HACK
    
    field.close()
    return candidates
    
def save_candidates_to_mongodb(candidates, collection):
    """ Save a list of light curves to the given mongodb collection """
    
    for candidate in candidates:
        try:
            ra, dec = candidate.ra, candidate.dec
        except AttributeError:
            logging.error("Candidate light curve has no 'ra' or 'dec' attributes!")
            continue
            
        save_light_curve_to_collection(candidate, collection=collection, delta_chi_squared=candidate.delta_chi_squared, ra=ra, dec=dec)
    
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
    parser.add_argument("-a", "--all", dest="all", default=False, action="store_true",
                    help="Run on all fields.")
    parser.add_argument("-f", "--field-id", dest="field_id", default=[], nargs="+", type=int,
                    help="The PTF field IDs to run on")
    parser.add_argument("-N", "--Nsigma", dest="Nsigma", default=5, type=float,
                    help="")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
        
    connection = pymongo.Connection(config["db_address"], config["db_port"])
    ptf = connection.ptf # the database
    ptf.authenticate(config["db_user"], config["db_password"])
    light_curve_collection = ptf.light_curves  
    
    if args.all:
        all_fields = np.load("data/all_fields.npy")

    else:
        for field_id in args.field_id:
            logger.info(greenText("Starting with field {}".format(field_id)))
            field = pdb.Field(field_id, "R")
            candidates = find_candidates_on_field(field, Nsigma=5.)
            save_candidates_to_mongodb(candidates, light_curve_collection)
            del candidates