import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, BigInteger, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, deferred, relationship
from sqlalchemy.schema import Column
from sqlalchemy.types import Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql.expression import func

import numpy as np

__all__ = ["session", "Session", "Base", "engine", "LightCurve", "CCDExposure", "Field"]

class Singleton(type):
	def __init__(cls, name, bases, dict):
		super(Singleton, cls).__init__(name, bases, dict)
		cls.instance = None
	
	def __call__(cls, *args, **kw):
		if cls.instance is None:
			cls.instance = super(Singleton, cls).__call__(*args, **kw)
	
		return cls.instance
	
class DatabaseConnection(object):
	__metaclass__ = Singleton

	def __init__(self, database_connection_string=None):
		self.database_connection_string = database_connection_string
		self.engine = create_engine(self.database_connection_string, echo=False) # change 'echo' to print each SQL query (for debugging/optimizing/the curious)
		self.metadata = MetaData()
		self.metadata.bind = self.engine
		self.Base = declarative_base(bind=self.engine)
		self.Session = scoped_session(sessionmaker(bind=self.engine, autocommit=True, autoflush=False))

db_config = {
	'user'     : 'adrian',
	'password' : 'lateralus0',
	'database' : 'ptf_microlensing_test',
	'host'     : 'deimos.astro.columbia.edu',
	'port'     : 5432
}

database_connection_string = 'postgresql://%s:%s@%s:%s/%s' % (db_config["user"], db_config["password"], db_config["host"], db_config["port"], db_config["database"])

db = DatabaseConnection(database_connection_string=database_connection_string)
engine = db.engine
metadata = db.metadata
Session = db.Session
session = Session
Base = db.Base

# Model Class for true PTF light curves
class CCDExposure(Base):
    __tablename__ = 'ccd_exposure'
    __table_args__ = {'autoload' : True}

class Field(Base):
    __tablename__ = 'field'
    __table_args__ = {'autoload' : True}
    
    @property
    def numberOfExposures(self):
        session = Session.object_session(self)
        
        numExps = []
        for ccdid in range(12):
            numExps.append(session.query(Exposure).join(Field).filter(Exposure.ccdid == ccdid).filter(Field.id == self.id).count())
        
        return max(numExps)

class LightCurve(Base):
    __tablename__ = 'light_curve'
    __table_args__ = {'autoload' : True}
    
    #mjd = deferred(Column(ARRAY(Float)))
    #mag = deferred(Column(ARRAY(Float)))
    #mag_error = deferred(Column(ARRAY(Float)))
    
    def __repr__(self):
        return "<{0} -- objid: {1}>".format(self.__class__.__name__, self.objid)
    
    @property
    def amjd(self):
        return np.array(self.mjd)
    
    @property
    def amag(self):
        return np.array(self.mag)
    
    @property
    def amag_error(self):
        return np.array(self.mag_error)

class CCDExposureToLightcurve(Base):
    __tablename__ = 'ccd_exposure_to_light_curve'
    __table_args__ = {'autoload' : True}


CCDExposure.field = relationship(Field, backref="ccd_exposures")
CCDExposure.lightCurves = relationship(LightCurve,
                    secondary=CCDExposureToLightcurve.__table__,
                    backref="ccdExposures")