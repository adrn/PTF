# -*- coding: utf-8 -*-

""" Brief: This script searches within a radius of a given RA and Dec
        for any candidate microlensing events in PTF.
"""

# System libraries
import sys
import os
import cPickle as pickle

# External packages
try:
    import lsd
    import lsd.bounds as lb
except ImportError:
    raise ImportError("LSD package not found! Did you mean to run this on navtara?")
    
import numpy as np

def fieldToRADec(thisField):
    """ Takes a PTF Field ID and returns the RA/Dec of
        the field center.
    """
    
    bottomLeftRA = thisField[thisField.ccdid == 9].ra[0]
    bottomRightRA, bottomRightDec = thisField[thisField.ccdid == 10].ra[0], thisField[thisField.ccdid == 10].dec[0]
    topRightDec = thisField[thisField.ccdid == 4].dec[0]
    
    ra = np.mean([bottomLeftRA, bottomRightRA])
    dec = np.mean([bottomRightDec, topRightDec])
    
    return (ra, dec)
    
def getLightCurves(field):
    """ Given a PTF Field ID, select all light curves from
        the field.
    """
    
    ra, dec = fieldToRADec(field)
    bounds_xy = lb.rectangle(ra-3.5/2, dec-2.31/2, ra+3.5/2, dec+2.31/2)                 # (ra,dec)_bottomleft, (ra,dec)_topright
    bounds_t  = lb.intervalset((40000, 60000))
    
    def row_counter_kernel(qresult):
        for rows in qresult:
                yield len(rows)
    
    for thisCCD in range(1, 13):
        query = db.query("ptf_det.ra as ra, ptf_det.dec as dec, mjd, mag_abs/1000. as mag, magerr_abs/1000. as magErr, \
                apbsrms as sys_err, appar02 as colortrm, fid, flags, imaflags_iso, obj_id, ptf_field, ccdid \
                FROM ptf_exp, ptf_det, ptf_obj \
                WHERE ((flags & 1) == 0) & ((imaflags_iso & 4053) == 0) & (flags < 8) & (apbsrms > 0) & (ccdid = {0})".format(thisCCD))
        
        total = 0
        for subtotal in query.execute([row_counter_kernel]):
            total += subtotal
        
        print "The total number of rows on ccd {0} is {1}".format(thisCCD, total)
        
    
    """
    results = db.query("ptf_det.ra as ra, ptf_det.dec as dec, mjd, mag_abs/1000. as mag, magerr_abs/1000. as magErr, \
            apbsrms as sys_err, appar02 as colortrm, fid, flags, imaflags_iso, obj_id, ptf_field, ccdid \
            FROM ptf_exp, ptf_det, ptf_obj \
            WHERE ((flags & 1) == 0) & ((imaflags_iso & 4053) == 0) & (flags < 8) & (apbsrms > 0) & (ccdid = {0})".format(thisCCD))\
        .fetch(bounds=[(bounds_xy, bounds_t)])
    """
    
    return results
    
def main(min_number_observations=150):
    f = open("fieldDict.pickle")
    fieldDict = pickle.load(f)
    f.close()
    
    for fieldID in fieldDict.keys():
        thisField = fieldDict[fieldID]
        if len(thisField[thisField.ccdid == 1]) > min_number_observations:
            getLightCurves(field)
            break

if __name__ == "__main__":
    pass