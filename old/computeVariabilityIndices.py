"""
    For each light curve in the local PTF database on Deimos, compute
    the variability indices
"""

# Standard library
import os, sys
import argparse
import logging

# Third party
import numpy as np

# Project
import ptf.simulation.util as simu
from ptf.db.DatabaseConnection import *

__author__ = "adrn@astro.columbia.edu"    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                        help="Be chatty!")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                        help="Be quiet!")
    # Integer example
    parser.add_argument("-o", "--num-trials", type=int, dest="num_trials", default=1024,
        				help="The number of trials (default = False)")
    # String example
    parser.add_argument("-k", "--krazy", type=str, dest="test", default="no",
				help="")
				
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.WARN)
    else:
        logging.basicConfig(level=logging.INFO)
    
    query = session.query(LightCurve).filter(LightCurve.variability_indices_pk == None).limit(1000)
    while True:
        session.begin()
        light_curves = query.all()
        if len(light_curves) == 0: break
        
        goodLightCurves = []
        for lc in light_curves:
            if len(lc.mjd) > 10:
                goodLightCurves.append(lc)
            else:
                session.delete(lc)
        
        logging.debug("Next 1000 light curves selected")
        for lc in goodLightCurves:
            lightCurve = simu.PTFLightCurve(lc.Rmjd, lc.Rmag, lc.Rerror)
            
            var_indices = simu.computeVariabilityIndices(lightCurve)
            variabilityIndices = VariabilityIndices()
            variabilityIndices.sigma_mu = var_indices["sigma_mu"]
            variabilityIndices.con = var_indices["con"]
            variabilityIndices.eta = var_indices["eta"]
            variabilityIndices.j = var_indices["J"]
            variabilityIndices.k = var_indices["K"]
            variabilityIndices.light_curve = lc
            session.add(variabilityIndices)
        
        session.commit()