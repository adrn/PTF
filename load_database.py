""" 
    Take PTF light curve data from ./lightcurves/*.pickle and load it
    into a postgres database.
"""
import glob, os
import cPickle as pickle
import logging
from optparse import OptionParser

import sqlalchemy
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.schema import Table

# Postgresql Database Connection
database_connection_string = 'postgresql://%s:%s@%s:%s/%s' \
	% ('adrian','','localhost','5432','ptf_microlensing')

engine = create_engine(database_connection_string, echo=False)
metadata = MetaData()
metadata.bind = engine
Base = declarative_base(bind=engine)
Session = scoped_session(sessionmaker(bind=engine, autocommit=True, autoflush=False))
session = Session()

# Model Class for light curves
class LightCurve(Base):
	__tablename__ = 'light_curve'
	__table_args__ = {'autoload' : True}
	
	def __repr__(self):
		return self.__class__.__name__

parser = OptionParser(description="Populate the BOSS part of the spectradb database")
parser.add_option("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                help="Overwrite all data in the database (default = False)")
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                help="Be chatty (default = False)")

(options, args) = parser.parse_args()
if options.verbose: logging.basicConfig(level=logging.DEBUG, format='%(message)s')
else: logging.basicConfig(level=logging.INFO, format='%(message)s')

if options.overwrite:
    session.query(LightCurve).delete()

# Find all light curve pickles
files = glob.glob("lightcurves/*.pickle")

session.begin()

for file in files:
    fn, ext = os.path.basename(file).split(".")
    field, id = map(int, fn.split("_"))
    
    f = open(file)
    data = pickle.load(f)
    f.close()
    
    try:
        lc = session.query(LightCurve).filter(LightCurve.objid == id).one()
        logging.debug("Light curve found in database. Skipping add...")
        continue
    except sqlalchemy.orm.exc.NoResultFound:
        logging.debug("Light curve NOT found in database...")    
    
    lightCurve = LightCurve()
    lightCurve.field = field
    lightCurve.objid = id
    lightCurve.mjd = list(data.mjd)
    lightCurve.mag = list(data.mag)
    lightCurve.mag_error = list(data.mag_err)
    
    try:
        lightCurve.ra = data.ra
        lightCurve.dec = data.dec
    except AttributeError:
        pass
        
    session.add(lightCurve)
    
    if len(session.new) >= 1000:
        session.commit()
        logging.info("Committed 1000 light curves!")
        session.begin()

session.commit()
session.rollback()
engine.dispose()
