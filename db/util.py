# -*- coding: utf-8 -*-

""" 
    This module provides some helper functions for downloading and copying light curve data
    from the Large Survey Database on navtara to the ptf_microlensing database on deimos.
"""

# System libraries
import sys
import os
import cPickle as pickle
import argparse
import logging

# Third party libraries
import numpy as np
import sqlalchemy
from sqlalchemy import func

# External packages
try:
    import lsd
    import lsd.bounds as lb
    db = lsd.DB("/scr4/bsesar")
    
except ImportError:
    logging.warn("LSD package not found! Did you mean to run this on navtara?")

try:
    import apwlib.geometry as g
except ImportError:
    raise ImportError("apwlib not found! Do: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install'")

from DatabaseConnection import *
from NumpyAdaptors import *

# ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-

def saveExposureData(filename="data/exposureData.pickle", overwrite=False, logger=None, verbosity=None):
    """ Queries LSD and saves the information to be loaded into
        the ccd_exposure table of ptf_microlensing.
    """
    
    if logger == None:
        logger = logging.getLogger("SaveExposureData")
        logger.propagate = False
        ch = logging.StreamHandler()
        logger.addHandler(ch)
        if verbosity == None:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(verbosity)
    
    if filename.strip == "":
        raise ValueError("filename parameter must not be empty!")
    elif os.path.exists(filename):
        if overwrite: os.remove(filename)
        else: raise FileError("File {0} already exists! If you want to overwrite it, set 'overwrite=True'".format(filename))
    
    db = lsd.DB("/scr4/bsesar")
    results = db.query("mjd, exp_id, ptf_field, ccdid, fid, ra, dec, l, b FROM ptf_exp").fetch()
    exposureData = [tuple(row) for row in results]
    logger.debug("saveExposureData: {0} rows returned from ptf_exp".format(len(exposureData)))
    
    exposureDataArray = np.array(exposureData, dtype=[("mjd", np.float64),\
                                                      ("exp_id", int), \
                                                      ("field_id", int), \
                                                      ("ccd_id", int), \
                                                      ("filter_id", int), \
                                                      ("ra", np.float64), \
                                                      ("dec", np.float64), \
                                                      ("l", np.float64), \
                                                      ("b", np.float64)]).view(np.recarray)
    
    logger.debug("saveExposureData: writing file {0}".format(filename))
    f = open(filename, "w")
    pickle.dump(exposureDataArray, f)
    f.close()
    logger.debug("saveExposureData: done!")
    
    return True

def loadExposureData(filename="data/exposureData.pickle", logger=None, verbosity=None):
    """ Loads a dump of the ptf_exp table into the ptf_microlensing databse """
    
    if logger == None:
        logger = logging.getLogger("LoadExposureData")
        logger.propagate = False
        ch = logging.StreamHandler()
        logger.addHandler(ch)
        if verbosity == None:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(verbosity)
    
    logger.debug("Loading file {0}...".format(filename))
    f = open(filename)
    exposureData = pickle.load(f)
    f.close()
    logger.info("Pickle loaded")
    
    fieldids = np.unique(exposureData.field_id)
    print fieldids
    sys.exit(0)
    
    for fieldid in fieldids:
        logger.debug("Processing field {0}".format(fieldid))
        
        # Start a sqlalchemy database session to allow for modifying and adding data to the DB
        Session.begin()
        
        # Select only the data for the current field
        thisData = exposureData[exposureData.field_id == fieldid]
        
        # Try to get the existing database information for the field, but if it doesn't
        #   exist this will create one and add it to the current session
        try:
            field = Session.query(Field).filter(Field.id == fieldid).one()
        except sqlalchemy.orm.exc.NoResultFound:
            logger.debug("Field not found in database")
            field = Field()
            field.id = fieldid
            session.add(field)
            session.flush()
        
        # See if the exposure data is already loaded into the database, or if only some of it is loaded
        for ccdid in range(12):
            ccdData = thisData[thisData.ccd_id == ccdid]
            numberOfExposuresInPickle = len(ccdData)
            numberOfExposuresInDatabase = Session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).count()
            
            # If there are more exposures in the file than in the database, this means there are new exposure
            #   in the file to be loaded into the database
            if numberOfExposuresInPickle > numberOfExposuresInDatabase:
                for exposure in ccdData:
                    if Session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).\
                                                  filter(CCDExposure.ccd_id == ccdid).\
                                                  filter(CCDExposure.exp_id == exposure["exp_id"]).count() == 0:
                        ccdExposure = CCDExposure()
                        ccdExposure.field_id = fieldid
                        ccdExposure.exp_id = exposure["exp_id"]
                        ccdExposure.mjd = exposure["mjd"]
                        ccdExposure.ccd_id = exposure["ccd_id"]
                        ccdExposure.filter_id = exposure["filter_id"]
                        ccdExposure.ra = exposure["ra"]
                        ccdExposure.dec = exposure["dec"]
                        ccdExposure.l = exposure["l"]
                        ccdExposure.b = exposure["b"]
                        session.add(ccdExposure)
                    else:
                        pass
                session.flush()
            
            # If the number of exposures are equal, the database is up to date
            elif numberOfExposuresInPickle == numberOfExposuresInDatabase:
                logger.debug("Field {0}, CCD {1}: All exposures loaded!".format(fieldid, ccdid))
                continue
            
            # If the number of exposures in the file is less than the number in the database, something went wrong!
            elif numberOfExposuresInPickle < numberOfExposuresInDatabase:
                logger.warning("Field {0}, CCD {1}: There are more entries in the database than in the load file!".format(fieldid, ccdid))
        
        Session.commit()
        logger.debug("Done with field {0}".format(fieldid))
            
    return True

def loadLightCurvesFromField(fieldid, minimumNumberOfExposures=25, ccdids=range(0,12), ogger=None, verbosity=None):
    """ Experimental function to directly load the ptf_microlensing database
        from navtara via a remote psql connection
    """
    
    if logger == None:
        logger = logging.getLogger("loadLightCurvesFromField")
        logger.propagate = False
        ch = logging.StreamHandler()
        logger.addHandler(ch)
        if verbosity == None:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(verbosity)
    
    logger.debug("Field: {0}".format(fieldid))
    
    radius = 0.75 # degrees, roughly one CCD
    bounds_t  = lb.intervalset((40000, 60000)) # Cover the whole survey
    
    for ccdid in ccdids:
        numberOfExposures = session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).count()
        if numberOfExposures < minimumNumberOfExposures:
            logger.info("This field {0} only has {1} observations! Exiting this field...".format(fieldid, numberOfExposures))
            return False
            
        logger.info("Field {0} has {1} observations...".format(fieldid, numberOfExposures))
        
        ra = session.query(func.avg(CCDExposure.ra)).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).all()[0][0]
        dec = session.query(func.avg(CCDExposure.dec)).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).all()[0][0]
        
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
        logger.info("CCD {0} had {1} detected sources and {2} observations".format(ccdid, len(results)/numberOfExposures, numberOfExposures))
            
        for objid in np.unique(resultsArray.obj_id):
            lightCurveData = resultsArray[resultsArray.obj_id == objid]
            lightCurve = LightCurve()
            lightCurve.objid = objid
            lightCurve.mag = lightCurveData.mag
            lightCurve.mag_error = lightCurveData.mag_err
            lightCurve.mjd = lightCurveData.mjd
            lightCurve.sys_error = lightCurveData.sys_err
            lightCurve.ra = lightCurveData.ra
            lightCurve.dec = lightCurveData.dec
            lightCurve.ccdExposures = session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).all()
        
    return True

def transferExposureData(overwrite=False, logger=None, verbosity=None):
    """ Queries LSD and then remotely loads the information into
        the ccd_exposure table of ptf_microlensing.
    """
    
    if logger == None:
        logger = logging.getLogger("SaveExposureData")
        logger.propagate = False
        ch = logging.StreamHandler()
        logger.addHandler(ch)
        if verbosity == None:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(verbosity)
        
    try:
        f = open("data/exposureData.pickle")
        exposureData = pickle.load(f) 
        f.close()
    except IOError:
        db = lsd.DB("/scr4/bsesar")
        results = db.query("mjd, exp_id, ptf_field, ccdid, fid, ra, dec, l, b FROM ptf_exp").fetch()
        exposureData = [tuple(exposure) for exposure in results]
        logger.debug("{0} exposures returned from ptf_exp".format(len(exposureData)))

        f = open("data/exposureData.pickle", "w")
        pickle.dump(exposureData, f)
        f.close()
    print exposureData[0]

    exposureData = np.array(exposureData, dtype=[("mjd", np.float64),\
                                                          ("exp_id", "|S20"), \
                                                          ("field_id", np.uint8), \
                                                          ("ccd_id", np.uint8), \
                                                          ("filter_id", np.uint8), \
                                                          ("ra", np.float64), \
                                                          ("dec", np.float64), \
                                                          ("l", np.float64), \
                                                          ("b", np.float64)]).view(np.recarray)
    
    fieldids = np.unique(exposureData.field_id)
    
    for fieldid in fieldids:
        logger.debug("Processing field {0}".format(fieldid))
        
        # Start a sqlalchemy database session to allow for modifying and adding data to the DB
        Session.begin()
        
        # Select only the data for the current field
        thisData = exposureData[exposureData.field_id == fieldid]
        
        # Try to get the existing database information for the field, but if it doesn't
        #   exist this will create one and add it to the current session
        try:
            field = Session.query(Field).filter(Field.id == fieldid).one()
        except sqlalchemy.orm.exc.NoResultFound:
            logger.debug("Field not found in database")
            field = Field()
            field.id = fieldid
            session.add(field)
        
        # See if the exposure data is already loaded into the database, or if only some of it is loaded
        for ccdid in range(12):
            ccdData = thisData[thisData.ccd_id == ccdid]
            numberOfExposuresInPickle = len(ccdData)
            numberOfExposuresInDatabase = Session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).count()
            
            # If there are more exposures in the file than in the database, this means there are new exposure
            #   in the file to be loaded into the database
            if numberOfExposuresInPickle > numberOfExposuresInDatabase:
                for exposure in ccdData:
                    if Session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).\
                                                  filter(CCDExposure.ccd_id == ccdid).\
                                                  filter(CCDExposure.exp_id == exposure["exp_id"]).count() == 0:
                        ccdExposure = CCDExposure()
                        ccdExposure.field_id = fieldid
                        ccdExposure.exp_id = exposure["exp_id"]
                        ccdExposure.mjd = exposure["mjd"]
                        ccdExposure.ccd_id = exposure["ccd_id"]
                        ccdExposure.filter_id = exposure["filter_id"]
                        ccdExposure.ra = exposure["ra"]
                        ccdExposure.dec = exposure["dec"]
                        ccdExposure.l = exposure["l"]
                        ccdExposure.b = exposure["b"]
                        session.add(ccdExposure)
                    else:
                        pass
            
            # If the number of exposures are equal, the database is up to date
            elif numberOfExposuresInPickle == numberOfExposuresInDatabase:
                logger.debug("Field {0}, CCD {1}: All exposures loaded!".format(fieldid, ccdid))
                continue
            
            # If the number of exposures in the file is less than the number in the database, something went wrong!
            elif numberOfExposuresInPickle < numberOfExposuresInDatabase:
                logger.warning("Field {0}, CCD {1}: There are more entries in the database than in the load file!".format(fieldid, ccdid))
        
            Session.flush()
        logger.debug("Done with field {0}".format(fieldid))
            
    return True
