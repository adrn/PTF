# -*- coding: utf-8 -*-

""" Brief: This script searches within a radius of a given RA and Dec
        for any candidate microlensing events in PTF.
"""

# System libraries
import sys
import os
import cPickle as pickle
import logging
import argparse

import apwlib.geometry as g

# External packages
try:
    import lsd
    import lsd.bounds as lb
    db = lsd.DB("/scr4/bsesar")
    
except ImportError:
    raise ImportError("LSD package not found! Did you mean to run this on navtara?")
    
import numpy as np

def saveFieldLightCurves(fieldid, fieldDict):
    thisField = fieldDict[fieldid]
        
    radius = 0.75 #degrees
    bounds_t  = lb.intervalset((40000, 60000))
    for ccdid in range(0, 12):
        filename = "ccd_pickles/{0}_{1}.pickle".format(fieldid, ccdid)
        if os.path.exists(filename):
            logger.warn("{0} already exists! skipping...".format(filename))
            continue
        
        thisCCD = thisField[thisField.ccdid == ccdid]
        if len(thisCCD) == 0:
            logging.debug("CCD {0} has no data!".format(ccdid))
            continue
        ra = np.mean(thisCCD.ra)
        dec = np.mean(thisCCD.dec)
        
        logger.debug("CCD position: {0}, {1}".format(ra, dec))
        bounds_xy = lb.beam(ra, dec, radius)
        results = db.query("ptf_det.ra as ra, ptf_det.dec as dec, mjd, mag_abs/1000. as mag, magerr_abs/1000. as magErr, \
                apbsrms as sys_err, fid, obj_id, ptf_field, ccdid, flags, imaflags_iso \
                FROM ptf_exp, ptf_det, ptf_obj \
                WHERE ((ccdid == {0}) & (ptf_field == {1}))".format(ccdid, fieldid))\
            .fetch(bounds=[(bounds_xy, bounds_t)])
        
        resultsArray = np.array(results, dtype=[('ra', np.float64), ('dec', np.float64), ('mjd', np.float64), ('mag', np.float64), ('mag_err', np.float64), \
            ('sys_err', np.float32), ('filter_id', np.uint8), ('obj_id', np.uint64), ('field_id', np.uint32), ('ccd_id', np.uint8), ('flags', np.uint16), ('imaflags_iso', np.uint16)])
        resultsArray = resultsArray.view(np.recarray)
        numObservations = len(np.unique(resultsArray.mjd))
        logger.info("CCD {0} had {1} detected sources and {2} observations".format(ccdid, len(results)/numObservations, numObservations))
        
        if numObservations < 25:
            logger.info("This field only has {0} observations! Exiting this field...".format(numObservations))
            return False
            
        f = open(filename, "w")
        pickle.dump(resultsArray, f)
        f.close()
        
    return True
    
def getDenseFields():
    bounds_t  = lb.intervalset((40000, 60000))
    
    # Globulars:
    data = np.genfromtxt("globularClusters.txt", delimiter=",", dtype=str)
    globularData = np.array([(g.RA.fromHours(row[1]).degrees, g.Dec.fromDegrees(row[2]).degrees, float(row[3]), float(row[4])) for row in data], dtype=[("ra",float), ("dec",float), ("l",float), ("b",float)]).view(np.recarray)
    
    radius = 3.0
    allBounds = []
    for ra,dec in zip(globularData.ra, globularData.dec):
        allBounds.append((lb.beam(ra, dec, radius), bounds_t))
        
    results = db.query("ptf_field FROM ptf_exp").fetch(bounds=allBounds)
    
    fieldids = np.unique([r[0] for r in results])
    
    f = open("fieldDict.pickle")
    global fieldDict
    fieldDict = pickle.load(f)
    f.close()
    
    for fieldid in fieldids:
        logger.debug("Field {0}".format(fieldid))
        success = saveFieldLightCurves(fieldid, fieldDict)
        if success: break
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o","--clobber", help="Delete all generated files before re-running", \
            action='store_true', dest='clobber')
    parser.add_argument("-v","--verbose", help="Be chatty", \
            action='store_true', dest='verbose')
    parser.add_argument("-q","--quiet", help="Be quiet, dammit!", \
            action='store_true', dest='quiet')
    
    global args
    args = parser.parse_args()
    
    global logger
    logger = logging.getLogger("SaveFieldData")
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    
    if args.verbose: 
        #logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
        logger.debug("Verbose mode it is!")
    elif args.quiet: logger.setLevel(logging.ERROR)
    else: logger.setLevel(level=logging.INFO)
    
    getDenseFields()
