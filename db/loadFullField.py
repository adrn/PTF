""" 
    Take PTF light curve data from ./lightcurves/*.pickle and load it
    into a postgres database.
    
    ** SQL TO CREATE TABLE IS AT BOTTOM **
    
"""
import glob, os
import cPickle as pickle
import logging
from argparse import ArgumentParser

import numpy as np
import sqlalchemy
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.schema import Table

from DatabaseConnection import *
from NumpyAdaptors import *
session = Session

parser = ArgumentParser(description="Populate the BOSS part of the spectradb database")
parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                help="Overwrite all data in the database (default = False)")
parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                help="Be chatty (default = False)")

args = parser.parse_args()
if args.verbose: logging.basicConfig(level=logging.DEBUG, format='%(message)s')
else: logging.basicConfig(level=logging.INFO, format='%(message)s')

if args.overwrite:
    session.query(LightCurve).delete()

# Find all light curve pickles
files = glob.glob("fullfields/*.pickle")

session.begin()

for file in files:
    fn, ext = os.path.basename(file).split(".")
    field, ccdid = map(int, fn.split("_"))
    
    logging.info("Opening file: {0}".format(file))
    f = open(file)
    data = pickle.load(f)
    f.close()
    logging.info("File opened!")
    
    objids = np.unique(data.obj_id)
    print objids, data.obj_id
    print data.ra, data.dec
    sys.exit(0)
    logging.info("{0} unique objid's".format(len(objids)))
    
    for objid in objids:
        try:
            lc = session.query(LightCurve).filter(LightCurve.objid == objid).one()
            logging.debug("Light curve found in database. Skipping add...")
            continue
        except sqlalchemy.orm.exc.NoResultFound:
            logging.debug("Light curve NOT found in database...")    
        
        thisData = data[data.obj_id == objid]
        
        lightCurve = LightCurve()
        lightCurve.field = field
        lightCurve.objid = objid
        lightCurve.ccdid = ccdid
        
        lightCurve.mjd = list(data.mjd)
        lightCurve.mag = list(data.mag)
        lightCurve.mag_error = list(data.mag_err)
        lightCurve.sys_error = list(data.sys_err)
        lightCurve.filterid = list(data.filterid)
        
        lightCurve.ra = list(data.ra)
        lightCurve.dec = list(data.dec)
        
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

"""
