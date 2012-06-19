import numpy as np
from ..ptflightcurve import PTFLightCurve

class SimulatedLightCurve(PTFLightCurve):
    
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
        else:
            # TODO: Implement this
            raise NotImplementedError("TODO!")
        
        if mag != None:
            self.amag = self.mag = np.array(mag)
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
    
    def addMicrolensingEvent(self, u0=None, t0=None, tE=None):
        """ Adds a simulated microlensing event to the light curve
            
            u0 : float, optional
                The impact parameter for the microlensing event. If not specified,
                the value will be drawn from the measured u0 distribution 
                [TODO: REFERENCE]
            t0 : float, optional
                The peak time of the event (shouldn't really be specified)
                This is just drawn from a uniform distribution between mjd_min
                and mjd_max
            tE : float, optional
                The length of the microlensing event. If not specified,
                the value will be drawn from the measured tE distribution 
                [TODO: REFERENCE]        
        """
        
        # If u0 is not specified, draw from u0 distribution
        #   - see for example Popowski & Alcock 
        #   - u0 maximum defined by detection limit of survey, but in our
        #       case assum the amplifcation should be >1.4. Using eq. 1 from
        #       Popowski & Alcock, this corresponds to a maximum u0 of ~0.9261
        if u0 == None: self.u0 = np.random.uniform()*0.9261
        else: self.u0 = float(u0)
        
        # If t0 is not specified, draw from uniform distribution between days
        if t0 == None: self.t0 = np.random.uniform(min(self.mjd), max(self.mjd))
        else: self.t0 = float(t0)
        
        if (self.t0 > max(self.mjd)) or (self.t0 < min(self.mjd)):
            logging.warn("t0 is outside of the mjd range for this light curve!")
        
        # If tE is not specified, draw from tE distribution
        #   I use an estimate of Wood's "observed" distribution for now:
        #   http://onlinelibrary.wiley.com/store/10.1111/j.1365-2966.2005.09357.x/asset/j.1365-2966.2005.09357.x.pdf?v=1&t=h1whtf1h&s=7b4d93a69aa684387a49ece5fc33c32fa5037052
        if tE == None: self.tE = 10**np.random.normal(1.3, 0.5)
        else: self.tE = float(tE)
        
        flux = fluxModel(self.mjd, u0=self.u0, t0=self.t0, tE=self.tE, F0=1.)#self.F0)
        self.mag = FluxToRMag(flux*RMagToFlux(self.mag))
        
        self.amag = self.mag
    
    def _addNoise(self):
        """ Add scatter to the light curve """
        self.mag += np.random.normal(0.0, self.error)
