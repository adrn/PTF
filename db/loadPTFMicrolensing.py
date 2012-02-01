import os, sys
import cPickle as pickle
import logging

import sqlalchemy
import numpy as np
from DatabaseConnection import *
from NumpyAdaptors import *

def loadFieldExposureData(filename):
    """ Given the filename to a .pickle file containing
        data from the PTF LSD database, load it into the 
        local ptf_microlensing Database
    """
    
    if not os.path.exists(filename) or len(filename.strip()) == 0:
        if session.query(Field).count() > 0 and session.query(CCDExposure).count() > 0: return True
        raise FileError("Error: You must specify the path to the data (as a pickle).\n\t e.g. -f /path/to/exposureData.pickle")
    
    if args.overwrite: session.query(Field).delete()
    
    logging.debug("Loading pickle...")
    f = open(filename)
    exposureData = pickle.load(f)
    f.close()
    logging.info("Pickle loaded")
    
    Session.begin()
    
    for fieldid in np.unique(exposureData.fieldid):
        logging.debug("Processing field {0}".format(fieldid))
        
        thisField = exposureData[exposureData.fieldid == fieldid]
        
        try:
            field = Session.query(Field).filter(Field.id == fieldid).one()
        except sqlalchemy.orm.exc.NoResultFound:
            logging.debug("Field not found in database. Loading new data...")
            field = Field()
            field.id = fieldid
            session.add(field)
            session.flush()
        
        numExposures = Session.query(CCDExposure).filter(CCDExposure.field_id == fieldid).count()
        if numExposures == len(thisField):
            logging.debug("CCDExposure data for field {0} found in database! Skipping...".format(fieldid))
            continue
        elif numExposures == 0:
            logging.debug("CCDExposure data not found in database. Loading new data...")
            
            for row in thisField:
                exposure = CCDExposure()
                exposure.field = field
                exposure.mjd = row["mjd"]
                exposure.ccd_id = row["ccdid"]
                exposure.filter_id = row["filterid"]
                exposure.ra = row["ra"]
                exposure.dec = row["dec"]
                exposure.l = row["l"]
                exposure.b = row["b"]
                Session.add(exposure)
            
            Session.commit()
            Session.begin()
            
        else:
            logging.error("ERROR: CCDExposure data lenght does not match file!")
    
    session.close()
        
    logging.debug("Committed Exposure and Field data!")
            
    return True

def loadLightCurves(filename):
    """ Given a filename that points to a pickle with PTF data, 
        load the Detection and Object tables with PTF light curves
    """
    
    if not os.path.exists(filename) or len(filename.strip()) == 0:
        raise FileError("Error: You must specify the path to the data (as a pickle).\n\t e.g. -f /path/to/FIELDID_CCDID.pickle")
    
    # no overwrite currently
    
    logging.debug("Loading pickle...")
    f = open(filename)
    lightCurveData = pickle.load(f)
    f.close()
    logging.info("Pickle loaded")
    
    
    
    session.add(lightCurve)
    
    # Get the CCD Exposures for this Light Curve
    ccdExposures = session.query(CCDExposure).join(Field).filter(Field.id == fieldid).filter(CCDExposure.ccd_id == ccdid).all()
    lightCurve.ccdExposures = ccdExposures
    session.flush()    
    
    

if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite all data (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_argument("-f", "--exposure-file", dest="ccdExposureFile", default="",
                    help="The path to the .pickle file that contains the CCDExposure information")
    
    global args
    args = parser.parse_args()
    if args.verbose: logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else: logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    loadFieldExposureData(args.ccdExposureFile)
    #loadLightCurves