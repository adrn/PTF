import warnings
warnings.filterwarnings(action="ignore", message="Skipped unsupported reflection of expression-based index q3c_ccd_exposure[_a-z]+")

import sqlalchemy
from sqlalchemy import create_engine, MetaData, Column, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, deferred, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql.expression import func

import numpy as np

__all__ = ["session", "Session", "Base", "engine", "LightCurve", "VariabilityIndices"]

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
	'database' : 'ptf',
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

class VariabilityIndices(Base):
    __tablename__ = 'variability_indices'
    __table_args__ = {'autoload' : True}
    
    @property
    def all_dict(self):
        return {"sigma_mu" : self.sigma_mu,\
                "con" : self.con,\
                "eta" : self.eta,\
                "J" : self.j,\
                "K" : self.k}
    
    @property
    def all_tuple(self):
        return (self.sigma_mu, self.con, self.eta, self.j, self.k)

class LightCurve(Base):
    __tablename__ = 'light_curve'
    __table_args__ = {'autoload' : True}
    
    mjd = deferred(Column(ARRAY(Float)))
    mag = deferred(Column(ARRAY(Float)))
    mag_error = deferred(Column(ARRAY(Float)))
    sys_error = deferred(Column(ARRAY(Float)))
    
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
    
    @property
    def asys_error(self):
        return np.array(self.sys_error)
    
    @property
    def error(self):
        return np.sqrt(self.amag_error**2 + self.asys_error**2)
    
    @property
    def goodMJD(self):
        return self.amjd[(self.error < 0.1) & (np.array(self.filter_id) == 2)]
    
    @property
    def goodMag(self):
        return self.amag[(self.error < 0.1) & (np.array(self.filter_id) == 2)]
    
    @property
    def goodError(self):
        return self.error[(self.error < 0.1) & (np.array(self.filter_id) == 2)]
    
    @property
    def aflags(self):
        return np.array(self.flags, dtype=int)
    
    @property
    def aimaflags_iso(self):
        return np.array(self.imaflags_iso, dtype=int)
    
    @property
    def afilter_id(self):
        return np.array(self.filter_id, dtype=int)
    
    @property
    def Rmjd(self):
        return self.amjd[self.afilter_id == 2]
    
    @property
    def gmjd(self):
        return self.amjd[self.afilter_id == 1]
    
    @property
    def Rmag(self):
        return self.amag[self.afilter_id == 2]
    
    @property
    def gmag(self):
        return self.amag[self.afilter_id == 1]
        
    @property
    def Rerror(self):
        return self.error[self.afilter_id == 2]
    
    @property
    def gerror(self):
        return self.error[self.afilter_id == 1]
    
    def plot(self, ax=None):
        ax_specified = ax == None
        if ax_specified:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
        # This should return a figure with 1 subplot if only R band, 2 subplots if R and g
        #ax.errorbar(self.mjd, self.mag, self.error, ls="none", marker=".", color="r")
        ax.errorbar(self.Rmjd, self.Rmag, self.Rerror, ls="none", marker="o", c='k', ecolor='0.7', capsize=0)
        ax.set_xlabel("MJD")
        ax.set_ylabel(r"$M_R$")
        ax.set_ylim(ax.get_ylim()[::-1])
        #ax.errorbar(self.gmjd, self.gmag, self.gerror, ls="none", marker="o", color="g", ms=5, ecolor='0.3')
        
        if ax_specified:
            plt.show()
        else:
            return ax
        
VariabilityIndices.light_curve = relationship(LightCurve, backref="variability_indices", uselist=False)

"""
CCDExposure.field = relationship(Field, backref="ccd_exposures")
CCDExposure.lightCurves = relationship(LightCurve,
                    secondary=CCDExposureToLightcurve.__table__,
                    backref="ccdExposures")
"""