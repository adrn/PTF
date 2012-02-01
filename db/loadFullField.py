""" 
    Contains a function that accepts a filename representing a chunk of 
    PTF data and loads it into the light_curve table in the 
    ptf_microlensing database.
    
    Take PTF light curve data from ./lightcurves/*.pickle and load it
    into a postgres database.
    
    ** SQL TO CREATE TABLE IS AT BOTTOM **
    
    Note: the 'candidate' field has various meanings:
        0 -> Light curve has not been analyzed
        1 -> Light curve has been looked at
        2 -> Light curve is a candidate
    
"""
import glob, os, sys
import cPickle as pickle
import logging
from argparse import ArgumentParser

import numpy as np
import sqlalchemy

from DatabaseConnection import *
from NumpyAdaptors import *

def loadData(filename):
    """ Given the filename of a pickle containing PTF
        light curve data, load it into the light_curve
        table of ptf_microlensing
    """
    if args.overwrite:
        

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite all data in the database (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_argument("-f", "--file", dest="file", default="",
                    help="The path to the .pickle file that contains the PTF data")
    
    global args
    args = parser.parse_args()
    if args.verbose: logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else: logging.basicConfig(level=logging.INFO, format='%(message)s')

if args.overwrite:
    session.query(LightCurve).delete()

# Find all light curve pickles
files = glob.glob("fullfields/*.pickle")

session.begin()

for file in files:
    if file == "fullfields/fieldDict.pickle": continue
    
    fn, ext = os.path.basename(file).split(".")
    field, ccdid = map(int, fn.split("_"))
    
    logging.info("Opening file: {0}".format(file))
    f = open(file)
    data = pickle.load(f)
    f.close()
    logging.info("File opened!")
    
    objids = np.unique(data.obj_id)
    logging.info("{0} unique objid's".format(len(objids)))
    
    numLCs = session.query(LightCurve).filter(LightCurve.fieldid == field).filter(LightCurve.ccdid == ccdid).count()
    
    if numLCs == len(objids): 
        logging.info("Field is already loaded!")
        continue
    
    for objid in objids:
        try:
            lc = session.query(LightCurve).filter(LightCurve.objid == objid).one()
            logging.debug("Light curve found in database. Skipping add...")
            continue
        except sqlalchemy.orm.exc.NoResultFound:
            logging.debug("Light curve NOT found in database...")    
        
        thisData = data[data.obj_id == objid]
        
        lightCurve = LightCurve()
        lightCurve.fieldid = field
        lightCurve.objid = objid
        lightCurve.ccdid = ccdid
        
        lightCurve.mjd = list(thisData.mjd)
        lightCurve.mag = list(thisData.mag)
        lightCurve.mag_error = list(thisData.mag_err)
        lightCurve.sys_error = list(thisData.sys_err)
        lightCurve.filterid = list(thisData.filterid)
        
        lightCurve.ra = list(thisData.ra)
        lightCurve.dec = list(thisData.dec)
        
        session.add(lightCurve)
    
        if len(session.new) >= 1000:
            session.commit()
            logging.info("Committed 1000 light curves!")
            session.begin()

session.commit()
engine.dispose()

""" SQL TO CREATE TABLE:

CREATE TABLE light_curve
(
  mjd double precision[] NOT NULL,
  mag double precision[] NOT NULL,
  mag_error double precision[] NOT NULL,
  sys_error double precision[] NOT NULL,
  ra double precision[] NOT NULL,
  "dec" double precision[] NOT NULL,
  fieldid integer NOT NULL,
  filterid smallint[] NOT NULL,
  objid numeric NOT NULL,
  ccdid smallint NOT NULL,
  candidate smallint;
  CONSTRAINT ptf_light_curve_pk PRIMARY KEY (objid)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE light_curve OWNER TO adrian;

CREATE INDEX ptf_light_curve_objid_idx
  ON light_curve
  USING btree
  (objid);

CREATE INDEX light_curve_fieldid_idx
  ON light_curve
  USING btree
  (fieldid);
  
CREATE INDEX light_curve_candidate_idx
  ON light_curve
  USING btree
  (candidate);

"""
