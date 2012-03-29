"""
    Load light curves from the Large Survey Database 
    and save it into a pickle.
    
    ** This script should be run on deimos **

"""

# Standard library
import os, sys
from argparse import ArgumentParser
import logging
import pyfits as pf
import numpy as np

from ptf.db.DatabaseConnection import *
from ptf.db.NumpyAdaptors import *

def loadLightCurves(filename):
    """ """
    
    logging.debug("Opening {}...".format(filename))
    f = pf.open(filename)
    resultsArray = f[1].data.view(np.recarray)
    f.close()
    logging.debug("File loaded!")
    
    session.begin()
    logging.debug("Starting database load...")
    for objid in np.unique(resultsArray.objid):
        lightCurveData = resultsArray[resultsArray.objid == objid]
        if len(lightCurveData.mag) < 10: continue
        
        lightCurve = LightCurve()
        lightCurve.objid = np.array([objid]).astype(np.uint64)[0]
        lightCurve.mag = lightCurveData.mag
        lightCurve.mag_error = lightCurveData.mag_err
        lightCurve.mjd = lightCurveData.mjd
        lightCurve.sys_error = lightCurveData.sys_err
        lightCurve.ra = lightCurveData.ra[0]
        lightCurve.dec = lightCurveData.dec[0]
        lightCurve.flags = lightCurveData["flags"].astype(np.uint16)
        lightCurve.imaflags = lightCurveData.imaflags_iso.astype(np.uint16)
        lightCurve.filter_id = lightCurveData.filter_id.astype(np.uint8)
        session.add(lightCurve)
        
        if len(session.new) == 1000:
            session.commit()
            logging.debug("1000 light curves committed!")
            session.begin()
    
    logging.info("All light curves committed!")
    session.commit()

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    parser.add_argument("-f", "--filename", type=str, dest="filename", 
                    help="Data file containing unique field ids for dense fields")
    
    args = parser.parse_args()
    if args.verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
    
    loadLightCurves(args.filename)
    
else:
    raise ImportError("loadLightCurves.py should be run as a standalone script, not imported!")
