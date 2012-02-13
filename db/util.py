# -*- coding: utf-8 -*-

""" 
    This module provides some helper functions for downloading and copying light curve data
    from the Large Survey Database on navtara to the ptf_microlensing database on deimos.
"""

# System libraries
import sys, os, glob
import cPickle as pickle
from cStringIO import StringIO
import argparse
import logging
import urllib, urllib2, base64
import gzip

# Third party libraries
import numpy as np
import pyfits as pf

# External packages
try:
    import lsd
    import lsd.bounds as lb
    db = lsd.DB("/scr/bsesar/projects/DB")
    
except ImportError:
    logging.warn("LSD package not found! Did you mean to run this on kepler?")

try:
    import apwlib.geometry as g
except ImportError:
    logging.warn("apwlib not found! Some functionality may not work correctly.\nDo: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install' to install.")

try:
    import sqlalchemy
    from sqlalchemy import func
except ImportError:
    logging.warn("sqlalchemy not found! Postgres database functions won't work.")
    
#from DatabaseConnection import *
#from NumpyAdaptors import *

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

def getLogger(verbosity, name="db.util"):
    """ Helper function for creating a Python logger """
    
    logger = logging.getLogger(name)
    logger.propagate = False
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    if verbosity == None:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(verbosity)
    
    return logger

def writeDenseCoordinatesFile(filename="data/denseCoordinates.pickle", overwrite=False, logger=None):
    """ Reads in data/globularClusters.txt and writes a pickle containing an array of ra,dec's for
        the given clusters, bulge, and M31    
    """
    if logger == None: logger = logging
    
    # Create empty lists to store all of the RA and Dec values
    allRAs = []
    allDecs = []
    
    # Check whether the file exists, and act accordingly
    if os.path.exists(filename) and not overwrite:
        return True
    elif os.path.exists(filename) and overwrite:
        if not os.path.splitext(filename)[1] == ".pickle":
            raise IOError("You can only overwrite .pickle files!")
        os.remove(filename)
    
    # Read in globular data from text file
    globularData = np.genfromtxt("data/globularClusters.txt", delimiter=",", usecols=[1,2], dtype=[("ra", "|S20"),("dec", "|S20")]).view(np.recarray)
    logger.debug("Globular data loaded...")
    
    arcmins = 10.
    for raStr,decStr in zip(globularData.ra, globularData.dec):
        # Use apwlib's RA/Dec parsers to convert string RA/Decs to decimal degrees
        ra = g.RA.fromHours(raStr).degrees
        dec = g.Dec.fromDegrees(decStr).degrees
        
        # Set bounds to check if any light curves exist within 10 arcminutes of the given positions
        bounds_t  = lb.intervalset((40000, 60000)) # Cover the whole survey
        bounds_xy = lb.beam(ra, dec, arcmins/60.)
        
        logger.debug("Checking whether {0},{1} is in survey footprint...".format(ra,dec))
        results = db.query("obj_id FROM ptf_obj").fetch(bounds=[(bounds_xy, bounds_t)])
        
        if len(results) > 1:
            logger.debug("Found {0} objects within {1} arcminutes of {2},{3}".format(len(results), arcmins, ra, dec))
            allRAs.append(ra)
            allDecs.append(dec)
        else:
            logger.debug("No objects found within {1} arcminutes of {2},{3}".format(len(results), arcmins, ra, dec))
        
    # M31
    ra = g.RA.fromHours("00 42 44.3").degrees
    dec = g.Dec.fromDegrees("+41 16 09").degrees
    allRAs.append(ra)
    allDecs.append(dec)
    
    # Bulge
    ra = g.RA.fromHours("17:45:40.04").degrees
    dec = g.Dec.fromDegrees("-29:00:28.1").degrees
    allRAs.append(ra)
    allDecs.append(dec)
    
    denseCoordinates = []
    for ra,dec in zip(allRAs, allDecs):
        denseCoordinates.append((ra,dec))
    
    f = open(filename, "w")
    pickle.dump(np.array(denseCoordinates, dtype=[("ra",float),("dec",float)]).view(np.recarray), f)
    f.close()
    
    return True

def matchRADecToImages(ra, dec, size=None, logger=None):
    """ This function is a wrapper around the IPAC PTF image server. This function 
        accepts an RA and Dec in degrees, and optionally a size, and returns a list 
        of images that overlap the given coordinates.
    """
    if logger == None: logger = logging
        
    if size == None: url = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS={0},{1}".format(ra,dec)
    else: url = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS={0},{1}&SIZE={2}".format(ra,dec,size)
    logger.debug("Image Search URL: {0}".format(url))
    
    request = urllib2.Request(url)
    base64string = base64.encodestring('%s:%s' % ("PTF", "palomar")).replace('\n', '')
    request.add_header("Authorization", "Basic %s" % base64string)
    file = StringIO(urllib2.urlopen(request).read())
    filenames = np.genfromtxt(file, skiprows=4, usecols=[20], dtype=str)
    logger.debug("Image downloaded.")
    logger.debug("{0} images in list".format(len(filenames)))
    
    return sorted(list(filenames))

def getAllImages(imageList, prefix, logger=None):
    """ Takes a list of PTF IPAC image basenames and downloads and saves them to the
        prefix directory.
    """
    if logger == None: logger = logging
    
    fieldList = dict()
    
    # Get any existing FITS files in the directory
    existingImages = glob.glob(os.path.join(prefix, "*.fits"))
    
    for image in existingImages:
        imageBase = os.path.splitext(os.path.basename(image))[0]
        ccd = imageBase.split("_")[-1]
        field = imageBase.split("_")[-2]
        
        try:
            fieldList[field].append(ccd)
        except KeyError:
            fieldList[field] = []
            fieldList[field].append(ccd)
    
    for image in imageList:
        imageBase = os.path.splitext(os.path.basename(image))[0]
        logging.debug("Image base: {0}".format(imageBase))
        
        if "scie" not in imageBase: continue

        # Extract ccd and fieldid from image filename
        ccd = imageBase.split("_")[-1]
        field = imageBase.split("_")[-2]
        
        logging.debug("Field: {0}, CCD: {1}".format(field,ccd))
        
        try:
            thisCCDList = fieldList[field]
            if ccd in thisCCDList: continue
        except KeyError:
            fieldList[field] = []
        
        file = os.path.join(prefix, os.path.basename(image))
        if os.path.exists(file):
            logging.info("File {0} already exists".format(file))
            continue
        
        imageURL = "http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/" + image
        
        request = urllib2.Request(imageURL)
        base64string = base64.encodestring('%s:%s' % ("PTF", "palomar")).replace('\n', '')
        request.add_header("Authorization", "Basic %s" % base64string)
        
        logging.debug("Image Full URL: {0}".format(request.get_full_url()))
        
        try:
            f = StringIO(urllib2.urlopen(request).read())
        except urllib2.HTTPError:
            continue
            
        try:
            gz = gzip.GzipFile(fileobj=f, mode="rb")
            gz.seek(0)
            
            fitsFile = StringIO(gz.read())
        except IOError:
            fitsFile = f
        
        fitsFile.seek(0)
        
        hdulist = pf.open(fitsFile, mode="readonly")
        hdulist.writeto(file)
        fieldList[field].append(ccd)
    
def getFITSCutout(ra, dec, size=0.5, save=False, logger=None):
    """ This function is a wrapper around the IPAC PTF image server cutout feature. 
        Given an RA, Dec, and size in degrees, download a FITS cutout of the given
        coordinates +/- the size.
    """
    if logger == None: logger = logging
    
    images = matchRADecToImages(ra, dec, logger=logger, verbosity=verbosity)
    imageURL = "http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/" + images[-1]
    
    urlParams = {'center': '{0},{1}deg'.format(ra, dec),\
                 'size': '{0}deg'.format(size)}
    
    request = urllib2.Request(imageURL +"?{}".format(urllib.urlencode(urlParams)))
    base64string = base64.encodestring('%s:%s' % ("PTF", "palomar")).replace('\n', '')
    request.add_header("Authorization", "Basic %s" % base64string)
    
    logger.debug("Image Full URL: {0}".format(request.get_full_url()))
    
    f = StringIO(urllib2.urlopen(request).read())
    gz = gzip.GzipFile(fileobj=f, mode="rb")
    gz.seek(0)
    
    fitsFile = StringIO(gz.read())
    fitsFile.seek(0)
    
    hdulist = pf.open(fitsFile, mode="readonly")
    logger.debug("Image loaded. Size: {0} x {1}".format(*hdulist[0].data.shape))
    
    if save:
        ra = g.RA.fromDegrees(ra)
        dec = g.Dec.fromDegrees(dec)
        filename = os.path.join("images", "{0}_{1}_{2}x{3}deg.fits".format(ra.string(sep="-"), dec.string(sep="-"), size, size))
        logger.debug("Writing file to: {0}".format(filename))
        hdulist.writeto(filename)
        return True
    else:
        return hdulist

def getLightCurvesRadial(ra, dec, radius, logger=None):
    """ Selects light curves from the Large Survey Database (LSD) on kepler
        given an ra and dec in degrees, and a radius in degrees. The constraints
        in the query are taken from the LSD wiki:
            http://www.oir.caltech.edu/twiki_ptf/bin/viewauth/Main/LSDNavtara
        except here I allow for saturated magnitude measurements in case a 
        microlensing event causes a star to saturate temporarily.
    """
    if logger == None: logger = logging
    
    bounds_t  = lb.intervalset((40000, 60000)) # Cover the whole survey
    bounds_xy = lb.beam(ra, dec, radius)
    
    results = db.query("mjd, ptf_obj.ra, ptf_obj.dec, obj_id, mag_abs/1000. as mag, magerr_abs/1000. as magErr, apbsrms as sys_err, fid, flags, imaflags_iso \
                        FROM ptf_det, ptf_obj, ptf_exp\
                        WHERE ((flags & 1) == 0) & ((imaflags_iso & 3797) == 0) & (flags < 8) & (apbsrms > 0) & (fid == 2)").fetch(bounds=[(bounds_xy, bounds_t)])
    
    resultsArray = np.array(results, dtype=[('mjd', np.float64), ('ra', np.float64), ('dec', np.float64), ('obj_id', np.uint64), ('mag', np.float64), ('mag_err', np.float64), \
                        ('sys_err', np.float32), ('filter_id', np.uint8),  ('flags', np.uint16), ('imaflags_iso', np.uint16)])
    resultsArray = resultsArray.view(np.recarray)
    
    logger.debug("Number of unique objid's: {0}".format(len(np.unique(resultsArray.obj_id))))
    
    return resultsArray
    
def loadLightCurves(filename, logger=None):
    """ """
    
    if logger == None: logger = logging
    
    logger.debug("Opening {}...".format(filename))
    f = open(filename)
    results = pickle.load(f)
    f.close()
    logger.debug("File loaded!")
    
    resultsArray = np.array(results, dtype=[('ra', np.float64), ('dec', np.float64), ('mjd', np.float64), ('mag', np.float64), ('mag_err', np.float64), \
        ('sys_err', np.float32), ('filter_id', np.uint8), ('obj_id', np.uint64), ('field_id', np.uint32), ('ccd_id', np.uint8), ('flags', np.uint16), ('imaflags_iso', np.uint16)])
    
    resultsArray = resultsArray.view(np.recarray)
    logger.debug("Data converted to recarray")
    
    if len(np.unique(resultsArray.field_id)) > 1 or len(np.unique(resultsArray.ccd_id)) > 1: 
        raise ValueError("More than one field or ccd id for this pickle!")
    
    fieldid = resultsArray.field_id[0]
    ccdid = resultsArray.ccd_id[0]
    
    exposures = session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).all()
    existingObjids = session.query(LightCurve.objid).join(CCDExposureToLightcurve, CCDExposure).filter(CCDExposure.field_id == fieldid).\
                                                      filter(CCDExposure.ccd_id == ccdid).distinct().all()
    existingObjids = np.unique([x[0] for x in existingObjids])
    logger.debug("Existing objids: {0}".format(len(existingObjids)))
    notLoadedObjids = np.array(list(set(resultsArray.obj_id).symmetric_difference(set(existingObjids))))
    
    session.begin()
    logger.debug("Starting database load...")
    for objid in notLoadedObjids:
        lightCurveData = resultsArray[resultsArray.obj_id == objid]
        if len(lightCurveData) < 25: continue
        lightCurve = LightCurve()
        lightCurve.objid = objid
        lightCurve.mag = lightCurveData.mag
        lightCurve.mag_error = lightCurveData.mag_err
        lightCurve.mjd = lightCurveData.mjd
        lightCurve.sys_error = lightCurveData.sys_err
        lightCurve.ra = lightCurveData.ra
        lightCurve.dec = lightCurveData.dec
        lightCurve.flags = lightCurveData["flags"]
        lightCurve.imaflags = lightCurveData.imaflags_iso
        lightCurve.ccdExposures = exposures
        
        if len(session.new) == 1000:
            session.commit()
            logger.debug("1000 light curves committed!")
            session.begin()
        
    session.commit()
