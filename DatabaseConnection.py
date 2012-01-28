import sqlalchemy
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, deferred
from sqlalchemy.schema import Column
from sqlalchemy.types import Float
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql.expression import func

import numpy as np

__all__ = ["Session", "Base", "engine", "LightCurve"]

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
	'database' : 'ptf_microlensing',
	'host'     : 'localhost',
	'port'     : 5432
}

database_connection_string = 'postgresql://%s:%s@%s:%s/%s' % (db_config["user"], db_config["password"], db_config["host"], db_config["port"], db_config["database"])

db = DatabaseConnection(database_connection_string=database_connection_string)
engine = db.engine
metadata = db.metadata
Session = db.Session
Base = db.Base

# Model Class for light curves
class LightCurve(Base):
    __tablename__ = 'light_curve'
    __table_args__ = {'autoload' : True}
    
    mjd = deferred(Column(ARRAY(Float)))
    mag = deferred(Column(ARRAY(Float)))
    mag_error = deferred(Column(ARRAY(Float)))
    
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
