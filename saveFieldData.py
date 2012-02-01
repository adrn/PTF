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
        
    radius = 0.7 #degrees
    bounds_t  = lb.intervalset((40000, 60000))
    for ccdid in range(0, 12):
        ra = np.mean(thisField[thisField.ccdid == ccdid].ra)
        dec = np.mean(thisField[thisField.ccdid == ccdid].dec)
        
        bounds_xy = lb.beam(ra, dec, radius)
        results = db.query("ptf_det.ra as ra, ptf_det.dec as dec, mjd, mag_abs/1000. as mag, magerr_abs/1000. as magErr, \
                apbsrms as sys_err, fid, obj_id \
                FROM ptf_exp, ptf_det, ptf_obj \
                WHERE ((flags & 1) == 0) & ((imaflags_iso & 4053) == 0) & (flags < 8) & (apbsrms > 0) & (ccdid == {0}) & (ptf_field == {1})".format(thisCCD, fieldid))\
            .fetch(bounds=[(bounds_xy, bounds_t)])
        
        print len(results)
        sys.exit(0)

        """
        logging.info("CCD {0} had {1} detected sources".format(thisCCD, len(results)))
        resultsArray = np.array(results, dtype=[('ra', float), ('dec', float), ('mjd', float), ('mag', float), ('mag_err', float), \
            ('sys_err', np.float32), ('filterid', np.uint8), ('obj_id', np.uint64)])
        resultsArray = resultsArray.view(np.recarray)
        
        f = open("{0}_{1}.pickle".format(id, thisCCD), "w")
        pickle.dump(resultsArray, f)
        f.close()
        """
    
def getDenseFields():
    
    # Have to open list of fields, for now just do:
    fieldids = [3419]
    
    f = open("../microlensing/fieldDict.pickle")
    global fieldDict
    fieldDict = pickle.load(f)
    f.close()
    
    for fieldid in fieldids:
        logging.debug("Field {0}".format(fieldid))
        saveFieldLightCurves(fieldid, fieldDict)


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
    
    if args.verbose: logging.basicConfig(level=logging.DEBUG)
    elif args.quiet: logging.basicConfig(level=logging.ERROR)
    else: logging.basicConfig(level=logging.INFO)
    
    getDenseFields()
