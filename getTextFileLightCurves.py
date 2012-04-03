"""
    Given a text file with ra,dec, get any light curves that match the positions
"""

# Standard library
import os, sys
from argparse import ArgumentParser
import logging

# Third-party
import sqlalchemy
import numpy as np
import pyfits as pf
import apwlib.geometry as g

import lsd
import lsd.bounds as lb
db = lsd.DB("/scr/bsesar/projects/DB")

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
    
    txtFile = np.genfromtxt(args.filename, usecols=[0,1], dtype=[('ra','|S20'), ('dec','|S20')]).view(np.recarray)
    
    bounds = []
    bounds_t  = lb.intervalset((40000, 60000)) # Cover the whole survey
    for line in txtFile:
        ra = g.RA(line.ra)
        dec = g.Dec(line.dec)
        
        bounds_xy = lb.beam(ra, dec, 4./3600.)
        bounds.append((bounds_xy, bounds_t))
    
    query = db.query("mjd, ptf_obj.ra, ptf_obj.dec, obj_id, mag_abs/1000. as mag, magerr_abs/1000. as magErr, apbsrms as sys_err, fid, flags, imaflags_iso \
                        FROM ptf_det, ptf_obj, ptf_exp")
    
    resultsArray = np.array(query.fetch(bounds=[(bounds_xy, bounds_t)]), dtype=[('mjd', np.float64), ('ra', np.float64), ('dec', np.float64), ('obj_id', np.int64), ('mag', np.float64), ('mag_err', np.float64), \
                        ('sys_err', np.float32), ('filter_id', np.uint8),  ('flags', np.uint16), ('imaflags_iso', np.uint16)])
    resultsArray = resultsArray.view(np.recarray)
    
    hdu = pf.BinTableHDU(resultsArray)
    hdu.writeto("tpagb_20120403.fits")