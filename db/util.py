# -*- coding: utf-8 -*-

""" 
    This module provides some helper functions for downloading and copying light curve data
    from the Large Survey Database on navtara to the ptf_microlensing database on deimos.
"""

# System libraries
import sys
import os
import cPickle as pickle
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
                                                      ("exp_id", np.uint64), \
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
                logger.debug("Field {0}, CCD {1}: Exposures not loaded".format(fieldid, ccdid))
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
                session.commit()
                session.begin()
            
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

def saveLightCurvesFromField(fieldid, minimumNumberOfExposures=25, ccdids=range(0,12), logger=None, verbosity=None):
    """ Given a ptf field id, save all light curves if it has enough exposures """
    
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
        filename = "data/{0}_{1}.pickle".format(fieldid, ccdid)
        if os.path.exists(filename):
            logger.warn("{0} already exists!".format(filename))
            continue
        
        numberOfExposures = session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).count()
        if numberOfExposures < minimumNumberOfExposures:
            logger.info("This ccd {0} only has {1} observations! Exiting this field...".format(ccdid, numberOfExposures))
            continue
            
        logger.info("Field {0}:{1} has {2} observations...".format(fieldid, ccdid, numberOfExposures))
        
        ra = session.query(func.avg(CCDExposure.ra)).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).all()[0][0]
        dec = session.query(func.avg(CCDExposure.dec)).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == ccdid).all()[0][0]
        
        logger.debug("CCD position: {0}, {1}".format(ra, dec))
        bounds_xy = lb.beam(ra, dec, radius)
        results = db.query("ptf_det.ra as ra, ptf_det.dec as dec, mjd, mag_abs/1000. as mag, magerr_abs/1000. as magErr, \
                apbsrms as sys_err, fid, obj_id, ptf_field, ccdid, flags, imaflags_iso \
                FROM ptf_exp, ptf_det, ptf_obj \
                WHERE ((ccdid == {0}) & (ptf_field == {1}) & (flags & 1) == 0) & ((imaflags_iso & 3797) == 0) & (flags < 8) & (apbsrms > 0))".format(ccdid, fieldid))\
            .fetch(bounds=[(bounds_xy, bounds_t)])

        #resultsArray = np.array(results, dtype=[('ra', np.float64), ('dec', np.float64), ('mjd', np.float64), ('mag', np.float64), ('mag_err', np.float64), \
        #    ('sys_err', np.float32), ('filter_id', np.uint8), ('obj_id', np.uint64), ('field_id', np.uint32), ('ccd_id', np.uint8), ('flags', np.uint16), ('imaflags_iso', np.uint16)])
        #
        #resultsArray = resultsArray.view(np.recarray)
        #logger.info("CCD {0} had {1} detected sources and {2} observations".format(ccdid, len(results)/numberOfExposures, numberOfExposures))
        logger.info("Saving file...{0}".format(filename))
        
        f = open(filename, "w")
        #pickle.dump(resultsArray, f)
        pickle.dump(results, f)
        f.close()
        
        logger.info("Done with field {0} and ccd {1}!".format(fieldid, ccdid))
        
    return True

def saveWellSampledDenseFields(filename="data/denseFields.pickle", minimumNumberOfExposures=25, logger=None, verbosity=None, fieldid=None):
    """ """
    if logger == None:
        logger = logging.getLogger("saveWellSampledDenseFields")
        logger.propagate = False
        ch = logging.StreamHandler()
        logger.addHandler(ch)
        if verbosity == None:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(verbosity)
    
    
    if field == None:
        fieldids = []
    else:
        fieldids = list(field)
    
    if not os.path.exists(filename) and len(fieldids) == 0:
        # Globulars
        globularData = np.genfromtxt("data/globularClusters.txt", delimiter=",", usecols=[1,2], dtype=[("ra", "|S20"),("dec", "|S20")]).view(np.recarray)
        logger.debug("Globular data loaded...")
        
        for raStr,decStr in zip(globularData.ra, globularData.dec):
            ra = g.RA.fromHours(raStr).degrees
            dec = g.Dec.fromDegrees(decStr).degrees
            newFields = list(np.unique([x[0] for x in session.query(Field.id).join(CCDExposure).filter(func.q3c_radial_query(CCDExposure.ra, CCDExposure.dec, ra, dec, 20./60)).all()])) # 20 arcminutes
            fieldids += newFields
            if len(newFields) == 0:
                logger.debug("No fields for RA/Dec: {},{}".format(ra,dec))
            else:
                logger.debug("Fields: {0}, RA/Dec: {1},{2}".format(",".join([str(x) for x in newFields]), ra,dec))
            
        # M31
        ra = g.RA.fromHours("00 42 44.3").degrees
        dec = g.Dec.fromDegrees("+41 16 09").degrees
        fieldids += [x[0] for x in session.query(Field.id).join(CCDExposure).filter(func.q3c_radial_query(CCDExposure.ra, CCDExposure.dec, ra, dec, 5.)).all()] # ~5 degrees
        
        # Bulge
        fieldids += [x[0] for x in session.query(Field.id).join(CCDExposure).filter(func.q3c_radial_query(CCDExposure.l, CCDExposure.b, 0., 0., 15.)).all()] # ~15 degrees of galactic center
        
        f = open(filename, "w")
        pickle.dump(np.unique(fieldids), f)
        f.close()
    
    f = open(filename)
    fieldids = pickle.load(f)
    f.close()
    
    if os.path.exists("data/doneFields.pickle"):
        f = open("data/doneFields.pickle")
        doneFields = pickle.load(f)
        f.close()
    else:
        doneFields = []
    
    for fieldid in fieldids:
        if fieldid in doneFields: 
            logger.info("Field {0} has already been processed".format(fieldid))
            continue
        
        if session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).filter(CCDExposure.ccd_id == 0).count() > minimumNumberOfExposures:
            success = saveLightCurvesFromField(fieldid, logger=logger, verbosity=verbosity)
            if success:
                doneFields.append(fieldid)
                f = open("data/doneFields.pickle", "w")
                pickle.dump(doneFields, f)
                f.close()

def loadLightCurves(filename, logger=None, verbosity=None):
    """ """
    
    if logger == None:
        logger = logging.getLogger("saveWellSampledDenseFields")
        logger.propagate = False
        ch = logging.StreamHandler()
        logger.addHandler(ch)
        if verbosity == None:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(verbosity)
    
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
        print objid
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