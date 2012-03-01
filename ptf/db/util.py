# -*- coding: utf-8 -*-

""" 
    This module provides some helper functions for downloading and copying light curve data
    from the Large Survey Database on navtara to the ptf_microlensing database on deimos.
"""

# System libraries
import sys, os, glob
import cPickle as pickle
import logging
import urllib, urllib2, base64
import gzip
import socket

# Third party libraries
import numpy as np
import pyfits as pf

# External packages
if socket.gethostname() == "kepler" or socket.gethostname() == "navtara":
    import lsd
    import lsd.bounds as lb
    db = lsd.DB("/scr/bsesar/projects/DB")

try:
    import apwlib.geometry as g
    import apwlib.geometry as c
except ImportError:
    logging.warn("apwlib not found! Some functionality may not work correctly.\nDo: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install' to install.")

try:
    import sqlalchemy
    from sqlalchemy import func
except ImportError:
    logging.warn("sqlalchemy not found! Postgres database functions won't work.")

# ==================================================================================================

def getLightCurvesRadial(ra, dec, radius):
    """ Selects light curves from the Large Survey Database (LSD) on kepler
        given an ra and dec in degrees, and a radius in degrees. The constraints
        in the query are taken from the LSD wiki:
            http://www.oir.caltech.edu/twiki_ptf/bin/viewauth/Main/LSDNavtara
        except here I allow for saturated magnitude measurements in case a 
        microlensing event causes a star to saturate temporarily.
    """    
    bounds_t  = lb.intervalset((40000, 60000)) # Cover the whole survey
    bounds_xy = lb.beam(ra, dec, radius)
    
    query = db.query("mjd, ptf_obj.ra, ptf_obj.dec, obj_id, mag_abs/1000. as mag, magerr_abs/1000. as magErr, apbsrms as sys_err, fid, flags, imaflags_iso \
                        FROM ptf_det, ptf_obj, ptf_exp\
                        WHERE ((flags & 1) == 0) & ((imaflags_iso & 3797) == 0) & (flags < 8) & (apbsrms > 0) & (fid == 2)")
    
    if radius > 0.5:
        raise ValueError("Radius is too large to do a straight query! Consider using 'getLightCurvesRadialBig' instead")
    else:
        resultsArray = np.array(results, dtype=[('mjd', np.float64), ('ra', np.float64), ('dec', np.float64), ('obj_id', np.uint64), ('mag', np.float64), ('mag_err', np.float64), \
                        ('sys_err', np.float32), ('filter_id', np.uint8),  ('flags', np.uint16), ('imaflags_iso', np.uint16)])
        resultsArray = resultsArray.view(np.recarray)
    
    logging.debug("Number of unique objid's: {0}".format(len(np.unique(resultsArray.obj_id))))
    
    return resultsArray
    
def saveLightCurvesRadial(ra, dec, filename="", radius=10, overwrite=False, skip=False):
    """ Given one set of coordinates, download all
        light curves around x arcminutes from the cluster center

        Parameters
        -----------
        filename : string
            The name of the output file
        ra : float, apwlib.geometry.RA
            A Right Ascension in decimal degrees
        dec : float, apwlib.geometry.Dec
            A Declination in decimal degrees
        radius : float, optional
            Radius in arcminutes to search for light curves around the given ra, dec
        overwrite : bool, optional
            If a pickle exists, do you want to overwrite it?
    """
    ra = c.parseDegrees(ra)
    dec = c.parseDegrees(dec)
    radiusDegrees = radius / 60.0
    
    logging.debug("{0},{1} with radius={2} deg".format(ra, dec, radiusDegrees))
    
    if filename.strip() == "":
        outputFilename = os.path.join("data", "lightcurves", "{0}_{1}.fits".format(ra.degrees, dec.degrees))
    else:
        outputFilename = filename
    logging.debug("Output file: {0}".format(outputFilename))
    
    if os.path.exists(outputFilename) and not overwrite:
        raise IOError("{0} already exists!".format(outputFilename))
    elif os.path.exists(outputFilename) and overwrite:
        logging.debug("You've chosen to overwrite the file!")
        os.remove(outputFilename)
        logging.debug("File deleted.")
    
    lightCurves = dbu.getLightCurvesRadial(ra, dec, radiusDegrees)
    
    hdu = pf.BinTableHDU(lightCurves)
    hdu.writeto(outputFilename)
    
    return

def getLightCurvesRadialBig(ra, dec, radius):
    """ Selects light curves from the Large Survey Database (LSD) on kepler
        given an ra and dec in degrees, and a radius in degrees. The constraints
        in the query are taken from the LSD wiki:
            http://www.oir.caltech.edu/twiki_ptf/bin/viewauth/Main/LSDNavtara
        except here I allow for saturated magnitude measurements in case a 
        microlensing event causes a star to saturate temporarily.
    """    
    bounds_t  = lb.intervalset((40000, 60000)) # Cover the whole survey
    bounds_xy = lb.beam(ra, dec, radius)
    
    query = db.query("mjd, ptf_obj.ra, ptf_obj.dec, obj_id, mag_abs/1000. as mag, magerr_abs/1000. as magErr, apbsrms as sys_err, fid, flags, imaflags_iso \
                        FROM ptf_det, ptf_obj, ptf_exp\
                        WHERE ((flags & 1) == 0) & ((imaflags_iso & 3797) == 0) & (flags < 8) & (apbsrms > 0) & (fid == 2)")
    
    logging.warn("Remember: the 'objid's saved in a FITS file are stored as 64-bit integers, but they are meant to be used as unsigned 64-bit integers. Make sure to convert properly!")
    for block in query.iterate(bounds=[(bounds_xy, bounds_t)], return_blocks=True):
        # *** objid should really be np.uint64, but FITS doesn't handle uint64 so remember to convert it! ***
        resultsArray = np.array(block, dtype=[('mjd', np.float64), ('ra', np.float64), ('dec', np.float64), ('objid', np.int64), ('mag', np.float64), ('mag_err', np.float64), \
                        ('sys_err', np.float32), ('filter_id', int),  ('flags', int), ('imaflags_iso', int)])
        resultsArray = resultsArray.view(np.recarray)
        logging.debug("Number of unique objid's: {0}".format(len(np.unique(resultsArray.objid))))
        yield resultsArray