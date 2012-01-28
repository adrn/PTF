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

def fieldToRADec(thisField):
    """ Takes a PTF Field ID and returns the RA/Dec of
        the field center.
    """
    
    ra = np.mean([thisField[thisField.ccdid == 8].ra[0], thisField[thisField.ccdid == 9].ra[0]])
    dec = np.mean([thisField[thisField.ccdid == 0].dec[0], thisField[thisField.ccdid == 6].dec[0]])
    logging.debug("Field center RA/Dec: {0}, {1}".format(ra, dec))
    
    return (ra, dec)
    
def saveLightCurves(field, id):
    """ Given a PTF Field ID, select all light curves from
        the field.
    """
    ra, dec = fieldToRADec(field)
    bounds_xy = lb.rectangle(ra-3.5/2, dec-2.31/2, ra+3.5/2, dec+2.31/2) # (ra,dec)_bottomleft, (ra,dec)_topright
    bounds_t  = lb.intervalset((40000, 60000))
    
    for thisCCD in range(1, 13):
        if os.path.exists("{0}_{1}.pickle".format(id, thisCCD)):
            logging.info("{0}_{1}.pickle exists! Skipping...".format(id, thisCCD))
            continue
    
        results = db.query("ptf_det.ra as ra, ptf_det.dec as dec, mjd, mag_abs/1000. as mag, magerr_abs/1000. as magErr, \
            apbsrms as sys_err, fid, obj_id \
            FROM ptf_exp, ptf_det, ptf_obj \
            WHERE ((flags & 1) == 0) & ((imaflags_iso & 4053) == 0) & (flags < 8) & (apbsrms > 0) & (ccdid == {0})".format(thisCCD))\
        .fetch(bounds=[(bounds_xy, bounds_t)])
        
        logging.info("CCD {0} had {1} detected sources".format(thisCCD, len(results)))
        resultsArray = np.array(results, dtype=[('ra', '<f8'), ('dec', '<f8'), ('mjd', '<f8'), ('mag', '<f8'), ('mag_err', '<f8'), \
            ('sys_err', '<f8'), ('filterid', int), ('obj_id', '<u8')])
        resultsArray = resultsArray.view(np.recarray)
        
        f = open("{0}_{1}.pickle".format(id, thisCCD), "w")
        pickle.dump(resultsArray, f)
        f.close()
    
def main(min_number_observations=150):
    f = open("fieldDict.pickle")
    fieldDict = pickle.load(f)
    f.close()
    
    for fieldID in fieldDict.keys():
        thisField = fieldDict[fieldID]
        logging.debug("Field {0}, {1} observations".format(fieldID, len(thisField[thisField.ccdid == 1])))
        
        if len(thisField[thisField.ccdid == 1]) > min_number_observations:
            saveLightCurves(thisField, fieldID)

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
    
    main()