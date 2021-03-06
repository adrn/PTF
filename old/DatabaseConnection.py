import warnings
warnings.filterwarnings(action="ignore", message="Skipped unsupported reflection of expression-based index q3c_ccd_exposure[_a-z]+")
import urllib

import sqlalchemy
from sqlalchemy import create_engine, MetaData, Column, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session, deferred, relationship
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql.expression import func

import numpy as np

import ptf.simulation.util as simu

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

user = "adrian"
password = 'cc.ppAPW1218+-*'
database_name = "ptf"
host = "deimos.astro.columbia.edu"
port = 5432

#database_connection_string = 'postgresql://%s:%s@%s:%s/%s' % (db_config["user"], db_config["password"], db_config["host"], db_config["port"], db_config["database"])
database_connection_string = "postgresql://{user}:{password}@{host}:{port}/{db}".format(user=user, password=urllib.quote(password), host=host, port=port, db=database_name)

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
    
    # TODO: This may not work...
    @property
    def mjd(self):
        return np.array(self.mjd)
    
    @property
    def mag(self):
        return np.array(self.mag)
    
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
    def afield(self):
        return np.array(self.field)
    
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
    
    def plot(self, ax=None, error_cut=None):
        ax_not_specified = ax == None
        if ax_not_specified:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
        
        if error_cut != None:
            idx = self.Rerror < error_cut
            mjd = self.Rmjd[idx]
            mag = self.Rmag[idx]
            error = self.Rerror[idx]
        else:
            mjd = self.Rmjd
            mag = self.Rmag
            error = self.Rerror
        
        if len(mjd) == 0: return
        
        # This should return a figure with 1 subplot if only R band, 2 subplots if R and g
        #ax.errorbar(self.mjd, self.mag, self.error, ls="none", marker=".", color="r")
        ax.errorbar(mjd, mag, error, ls="none", marker="o", c='k', ecolor='0.7', capsize=0)
        ax.set_xlabel("MJD")
        ax.set_ylabel(r"$M_R$")
        ax.set_ylim(ax.get_ylim()[::-1])
        #ax.errorbar(self.gmjd, self.gmag, self.gerror, ls="none", marker="o", color="g", ms=5, ecolor='0.3')
        
        if ax_not_specified:
            plt.show()
        else:
            return ax
            
    def indices(self, indices, recompute=False):
        """ Return a tuple with the values of the variability indices specified. 
        
            If recompute, it will recompute the values on the fly.
        """
        if recompute:
            return simu.compute_variability_indices(self, indices)
        else:
            vals = []
            for idx in indices:
                vals.append(getattr(self.variability_indices, idx))
        
        return tuple(vals)
        
VariabilityIndices.light_curve = relationship(LightCurve, backref="variability_indices", uselist=False)

"""
CCDExposure.field = relationship(Field, backref="ccd_exposures")
CCDExposure.lightCurves = relationship(LightCurve,
                    secondary=CCDExposureToLightcurve.__table__,
                    backref="ccdExposures")
"""