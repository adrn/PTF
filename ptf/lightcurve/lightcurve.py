""" Contains the PTFLightCurve class """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard Library
import os
import copy

# Third-party dependencies
import numpy as np
import matplotlib.pyplot as plt

from ..analyze import microlensing_model
from ..util import get_logger
logger = get_logger(__name__)

__all__ = ["PTFLightCurve", "PDBLightCurve", "SimulatedLightCurve"]

class PTFLightCurve(object):
    """ Represents a PTF Light Curve """
    
    def __init__(self, mjd, mag, error, metadata=None, exposures=None, **kwargs):
        """ Create a PTFLightCurve by passing equal-length arrays of mjd, magnitude, and
            magnitude errors. This object also accepts an optional metadata parameter, 
            which is any numpy recarray that contains extra information about the light
            curve data.
        """

        idx = np.argsort(mjd)
        self.amjd = self.mjd = np.array(mjd)[idx].astype(np.float64)
        self.amag = self.mag = np.array(mag)[idx].astype(np.float64)
        self.error = np.array(error)[idx].astype(np.float64)
        
        for key,val in kwargs.items():
            setattr(self, key, val)
        
        if metadata != None:
            self.metadata = np.array(metadata)
        else:
            self.metadata = None
    
        if exposures != None:
            self.exposures = np.array(exposures)
        else:
            self.exposures = None
    
    def shuffle(self):
        """ Randomly move around the magnitude values in the light curve to shuffle up
            the data points, but keep the mjd values fixed
        """
        idx = np.arange(len(self.mjd), dtype=int)
        
        np.random.shuffle(idx)
        self.mag = self.mag[idx]
        
        np.random.shuffle(idx)
        self.error = self.error[idx]
    
    def plot(self, ax=None, **kwargs):
        """ Either plots the light curve and show()'s it to the display, or plots it on 
            the given matplotlib Axes object
        """
        
        if not kwargs.has_key("ls") and not kwargs.has_key("linestyle"):
            kwargs["ls"] = 'none'
        
        if not kwargs.has_key("marker"):
            kwargs["marker"] = 'o'
        
        if not kwargs.has_key("c") and not kwargs.has_key("color"):
            kwargs["c"] = 'k'
        
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.errorbar(self.mjd, self.mag, self.error, ecolor='0.7', capsize=0, **kwargs)
            ax.set_xlim(min(self.mjd), max(self.mjd))
            ax.set_ylim(ax.get_ylim()[::-1])
            plt.show()
            return
        
        # APW Hack
        idx = np.where(self.error < 0.5)
        ax.errorbar(self.mjd[idx], self.mag[idx], self.error[idx], ecolor='0.7', capsize=0, **kwargs)
        ax.set_ylim(ax.get_ylim()[::-1])
        return ax
    
    def save(self, filename, overwrite=False):
        """ Save the light curve to a file """

        if os.path.exists(filename) and not overwrite:
            raise IOError("File {} already exists! You must specify overwrite=True or be more careful!".format(filename))
        elif os.path.exists(filename) and overwrite:
            os.remove(filename)
        
        # TODO: This is broken!
        xx, ext = os.path.splitext(filename)
        
        if ext == ".txt":
            np.savetxt(filename, np.transpose((self.mjd, self.mag, self.error)), fmt="%.5f\t%.5f\t%.5f")
        elif ext == ".pickle":
            import cPickle as pickle
            
            f = open(filename, "w")
            pickle.dump(self, f)
            f.close()
        elif ext == ".npy":
            arr = np.transpose((self.mjd, self.mag, self.error))
            arr = np.array(arr, dtype=[("mjd", float), ("mag", float), ("error", float)])
            np.save(filename, arr)
        else:
            raise ValueError("I don't know how to handle {} files!".format(ext))

    def slice_mjd(self, min=None, max=None):
        """ Select out a part of the light curve between two MJD values -- INCLUSIVE """
        idx = (self.mjd >= min) & (self.mjd <= max)
        
        return PTFLightCurve(mjd=self.mjd[idx], mag=self.mag[idx], error=self.error[idx])
    
    def __len__(self):
        return len(self.mjd)
    
    def sdss_colors(self, mag_type="psf"):
        """ Returns a dictionary with SDSS colors for this object. mag_type can be 'psf' or 'mod' """
        import galacticutils
        try:
            sdssData = galacticutils.querySDSSCatalog(self.ra, self.dec)
        except RuntimeError:
            return None
        
        if sdssData == None:
            return None
        
        mag_type = mag_type.lower()
        mags = {}
        
        for color in "ugriz":
            mags[color] = sdssData[color + mag_type.capitalize()][0]
            mags[color+"_err"] = sdssData[color + mag_type.capitalize()][1]
            
        return mags
    
    @property
    def baseline(self):
        return self.mjd.max()-self.mjd.min()
        
class PDBLightCurve(PTFLightCurve):
    """ Subclass of PTFLightCurve that requires a field_id, ccd_id, and source_id """
    
    def __init__(self, mjd, mag, error, field_id, ccd_id, source_id, **kwargs):
        self.field_id = int(field_id)
        self.ccd_id = int(ccd_id)
        self.source_id = int(source_id)
        
        super(PDBLightCurve, self).__init__(mjd, mag, error, **kwargs)

class SimulatedLightCurve(PTFLightCurve):
    
    @staticmethod
    def from_ptflightcurve(ptflightcurve):
        return SimulatedLightCurve(mjd=ptflightcurve.mjd, mag=ptflightcurve.mag, error=ptflightcurve.error)
    
    def __init__(self, mjd, mag=None, error=None, outliers=False):
        """ Creates a simulated PTF light curve
        
            Parameters
            ----------
            mjd : numpy.array
                An array of mjd values. If none, creates one internally.
            error : numpy.array
                An array of error values (sigmas). If none, creates one internally.
            mag : numpy.array
                An optional array of magnitude values
            outliers : bool, optional
                This controls whether to sample from an outlier distribution
                when creating magnitude values for the light curve
            
        """
        
        self.amjd = self.mjd = np.array(mjd)
        
        if error != None:
            self.error = np.array(error)
            if len(self.error) == 1:
                self.error = np.zeros_like(self.mjd) + error
            elif len(self.error) != len(self.mjd):
                raise ValueError("Error array should have same shape as mjd")
        else:
            self.error = np.zeros_like(self.mjd)
        
        if isinstance(mag, np.ndarray) or isinstance(mag, list):
            self.amag = self.mag = np.array(mag)
        elif isinstance(mag, int) or isinstance(mag, float):
            self.amag = self.mag = np.zeros(len(mjd)) + mag
            self._addNoise()
        else:
            if outliers:
                # Add ~1% outliers
                outlier_points = (np.random.uniform(0.0, 1.0, size=len(self.mjd)) < 0.01).astype(float) * np.random.normal(0.0, 2.0, size=len(self.mjd))
            else:
                outlier_points = np.zeros(len(self.mjd))
            
            self.F0 = np.random.uniform(0.1, 1.) / 100.
            self.mag = np.ones(len(self.mjd), dtype=float)*FluxToRMag(self.F0) + outlier_points
    
            self._addNoise()
        
        self._original_mag = copy.copy(self.mag)
    
    def reset(self):
        self.amag = self.mag = copy.copy(self._original_mag)
        self.u0 = None
        self.t0 = None
        self.tE = None
    
    def addMicrolensingEvent(self, u0=None, t0=None, tE=None, **kwargs):
        """ Adds a simulated microlensing event to the light curve
            
            u0 : float, optional
                The impact parameter for the microlensing event. If not specified,
                the value will be drawn from the measured u0 distribution 
            t0 : float, optional
                The peak time of the event (shouldn't really be specified)
                This is just drawn from a uniform distribution between mjd_min
                and mjd_max
            tE : float, optional
                The length of the microlensing event. If not specified,
                the value will be drawn from the measured tE distribution       
        """
        
        # If u0 is not specified, draw from u0 distribution
        #   - see for example Popowski & Alcock 
        if u0 == None: self.u0 = np.random.uniform(0., 1.34)
        else: self.u0 = float(u0)
        
        # If t0 is not specified, draw from uniform distribution between days
        if t0 == None: self.t0 = np.random.uniform(min(self.mjd), max(self.mjd))
        else: self.t0 = float(t0)
        
        # If tE is not specified, draw from tE distribution
        #   I use an estimate of Wood's "observed" distribution for now:
        #   http://onlinelibrary.wiley.com/store/10.1111/j.1365-2966.2005.09357.x/asset/j.1365-2966.2005.09357.x.pdf?v=1&t=h1whtf1h&s=7b4d93a69aa684387a49ece5fc33c32fa5037052
        if tE == None: self.tE = 10**np.random.normal(1.3, 0.5)
        else: self.tE = float(tE)
        
        params = {"t0" : self.t0, "tE" : self.tE, "u0" : self.u0, "m0" : 1.}
        self.amag = self.mag = self.mag+microlensing_model(params, self.mjd)
    
    def _addNoise(self):
        """ Add scatter to the light curve """
        if not np.all(self.error == 0.):
            self.mag += np.random.normal(0.0, self.error)
