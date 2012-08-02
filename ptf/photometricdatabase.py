""" 
    Various classes and functions for interfacing with the PTF photometric
    database (David Levitan's db), as described here: 
        http://www.astro.caltech.edu/~dlevitan/ptf/photomdb.html
"""

# Standard library
import sys, os
import logging

# Third-party
import numpy as np
import apwlib.geometry as g
from apwlib.globals import redText, greenText, yellowText
import pytest

try:
    import tables
except ImportError:
    logging.warn("PyTables not found! Some functionality won't work.\nTry running with: /scr4/dlevitan/sw/epd-7.1-1-rh5-x86_64/bin/python instead.")

# PTF
import globals
from ptflightcurve import PTFLightCurve
import analyze.analyze as analyze

#all_fields = np.load(os.path.join(ptf_params.config["PROJECTPATH"], "data", "all_fields.npy"))
all_fields = np.load(os.path.join("data", "all_fields.npy"))
match_path = "/scr4/dlevitan/matches"
pytable_base_string = os.path.join(match_path, "match_{filter.id:02d}_{field.id:06d}_{ccd.id:02d}.pytable")
filter_map = {"R" : 2, "g" : 1}
inv_filter_map = {2 : "R", 1 : "g"}

# ==================================================================================================
#   Classes
#

class Filter(object):
    
    def __init__(self, filter_id):
        """ Create a new filter object given a string or number id """
        
        if isinstance(filter_id, str):
            assert filter_id in filter_map.keys(), "Filter ID not valid: {}".format(filter_id)
            self.id = filter_map[filter_id]
            self.name = filter_id
        elif isinstance(filter_id, int):
            assert filter_id in inv_filter_map.keys(), "Filter ID not valid: {}".format(filter_id)
            self.id = filter_id
            self.name = inv_filter_map[filter_id]
        else:
            raise ValueError(redText("filter_id must be a number (e.g. 1, 2) or a string (e.g. 'g', 'R')"))
        
    def __repr__(self):
        return "<Filter: id={}, name={}>".format(self.id, self.name)

class Field(object):
    """ Represents PTF Field """
    
    def __repr__(self):
        return "<Field: id={}, filter={}>".format(self.id, self.filter.id)
    
    def __init__(self, field_id, filter):
        """ Create a field object given a PTF Field ID
            
            Parameters
            ----------
            field_id : int
                The PTF Field ID for a field.
            filter : Filter
                A PTF Filter object (e.g. R = 2, g = 1)
            
        """
        
        # Validate Field ID
        try:
            self.id = int(field_id)
        except ValueError:
            print redText("field_id must be an integer or parseable string, e.g. 110002 or '110002'")
            raise
        
        # Validate filter_id
        if isinstance(filter, Filter):
            self.filter = filter
        else:
            raise ValueError(redText("filter") + " parameter must be Filter object")
        
        self.ccds = dict()
        
        # Create new CCD objects for this field
        for ccd_id in range(12):
            try:
                self.ccds[ccd_id] = CCD(ccd_id, field=self, filter=self.filter)
            except ValueError:
                logging.debug("CCD {} not found for Field {}.".format(ccd_id, self))
        
        if len(self.ccds) == 0:
            logging.debug("No CCD data found for: {}".format(self))
    
    @property
    def number_of_exposures(self):
        """ To get the number of observations, this must be run on navtara so the
            script has access to the PTF Photometric Database. 
        """
        observations_per_ccd = dict()
        for ccdid,ccd in self.ccds.items():
            chip = ccd.read()
            observations_per_ccd[ccdid] = chip.exposures.nrows
        
        return observations_per_ccd
    
    @property
    def baseline(self):
        """ To get the baseline, this must be run on navtara so the
            script has access to the PTF Photometric Database. 
        """
        baseline_per_ccd = dict()
        for ccdid,ccd in self.ccds.items():
            chip = ccd.read()
            mjds = np.sort(chip.exposures.col("obsMJD"))
            baseline_per_ccd[ccdid] = mjds[-1]-mjds[0]
        
        return baseline_per_ccd
    
    def close(self):
        for ccd in self.ccds.values():
            ccd.close()
        
class CCD(object):
    _chip = None
    
    def __repr__(self):
        return "<CCD: id={}, field={}>".format(self.id, self.field)
    
    def __init__(self, ccd_id, field, filter):
        # Validate CCD ID
        try:
            self.id = int(ccd_id)
        except ValueError:
            print redText("ccd_id must be an integer or parseable string, e.g. 1 or '03'")
            raise
            
        if isinstance(field, Field):
            self.field = field
        else:
            raise ValueError("field parameter must be a Field object.")
        
        if isinstance(filter, Filter):
            self.filter = filter
        else:
            raise ValueError("filter parameter must be a Filter object.")
        
        self.filename = pytable_base_string.format(filter=self.filter, field=self.field, ccd=self)
        
        if not os.path.exists(self.filename):
            raise ValueError("CCD data file does not exist!")
    
    def read(self):
        self._file = tables.openFile(self.filename)
        
        if self._chip == None:
            self._chip = getattr(getattr(getattr(self._file.root, "filter{:02d}".format(self.filter.id)), "field{:06d}".format(self.field.id)), "chip{:02d}".format(self.id))
        
        return self._chip
    
    def close(self):
        self._file.close()
    
    def light_curve(self, source_id, mag_type="relative", clean=True):
        """ Get a light curve for a given source ID from this chip """
        # TODO: should this be here?

        chip = self.read()
        sourcedata = chip.sourcedata.readWhere('matchedSourceID == {}'.format(source_id))
        mjd = sourcedata["mjd"]
        
        if mag_type == 'relative':
            mag = sourcedata["mag"]
            mag_err = sourcedata["magErr"]
        elif mag_type == 'absolute':
            mag = sourcedata["mag_auto"]/1000.0 + sourcedata["absphotzp"]
            mag_err = sourcedata["magerr_auto"]/10000.0
        
        return PTFLightCurve(mjd=mjd, mag=mag, error=mag_err, metadata=sourcedata)
        

# ==================================================================================================
#   Convenience functions
#
def convert_all_fields_txt(txt_filename="ptf_allfields.txt", npy_filename="all_fields.npy"):
    """ Convert the text file containing all PTF Field ID's and positons to a 
        numpy .npy save file.
        
        It is much faster to load the .npy file than to generate an array on 
        the fly from the text file. The text file can be found here:
            http://astro.caltech.edu/~eran/PTF_AllFields_ID.txt
    """
    all_fields = np.genfromtxt(txt_filename, \
                           skiprows=4, \
                           usecols=[0,1,2,5,6,7], \
                           dtype=[("id", int), ("ra", float), ("dec", float), ("gal_lon", float), ("gal_lat", float), ("Eb_v", float)]).\
                           view(np.recarray)

    np.save(npy_filename, all_fields)
    
    return True    

def select_most_observed_fields(minimum_num_observations=25, limit=0):
    """ Return a list of PTFField objects for fields with more than 
        minimum_num_observations observations.
        
        There are 16598 fields in total.
    """
    
    filter_R = Filter("R")
    field = None
    fields = []
    for row in all_fields:
        if field is not None: field.close()
        field = Field(row["id"], filter=filter_R)

        if sum(np.array(field.number_of_exposures.values()) > minimum_num_observations) > 0:
            fields.append(field)
        
        if limit > 0:
            if len(fields) >= limit: break
    
    return fields

'''
def get_raw_light_curve(source_id, filter_id=2, field_id=None, ccd_id=None):
    """ Given a source_id, return a PTFLightCurve object for the 
        raw light curve. 
        
        If you don't specify field_id or ccd_id, it could take some
        time because it doesn't know where to look! It has to scan
        through every field table.
    """
    
    if field_id == None or ccd_id == None:
        # Do long search
        raise NotImplementedError()

    field = PTFField(field_id, filter_id=filter_id)
    ccd = field.ccds[ccd_id]
    chip = ccd.read()
    lc_data = chip.sourcedata.readWhere("matchedSourceID == {}".format(source_id))
    ccd.close()
    
    return PTFLightCurve(mjd=lc_data["mjd"], mag=lc_data["mag"], error=lc_data["magErr"])
'''

# ==================================================================================================
#   Test functions
#
def test_ptffield():
    # Test PTFField
    
    field_110001 = PTFField(110001)
    ccd0 = field_110001.ccds[0]
    print "CCDS: {}".format(",".join([str(x) for x in field_110001.ccds.keys()]))
    chip = ccd0.read()
    print "Number of observations on CCD 0:", chip.sources.col("nobs")
    print "Number of GOOD observations on CCD 0:", chip.sources.col("ngoodobs")
    ccd0.close()
    
    print "Number of observations per ccd:"
    print field_110001.number_of_exposures
    print "Baseline per ccd:"
    print field_110001.baseline

def test_select_most_observed_fields():
    for field in select_most_observed_fields(300):
        print field.id

def time_compute_var_indices():
    # Time computing variability indices for all light curves on a chip
    
    field = PTFField(110002)
    
    # Try to get light curve generator for a CCD
    import time
    a = time.time()
    N = 0
    for light_curve in field.ccds[0].light_curves("(ngoodobs > 15) & (matchedSourceID < 1000)"):
        indices = analyze.compute_variability_indices(light_curve, indices=["j","k","eta","sigma_mu","delta_chi_squared"])
        N += 1
    
    print "Took {} seconds to compute for {} light curves".format(time.time() - a, N)
    
def test_field():
    filter_R = Filter(filter_id="R")
    filter_g = Filter(filter_id="g")
    
    # Test all the various ways to initialize the object
    field = Field(field_id=110002, filter=filter_R)
    field = Field(field_id="110002", filter=filter_g)
    
    # Test number of exposures and baseline
    field = Field(field_id=110002, filter=filter_R)
    print field.number_of_exposures
    print field.baseline

def test_filter():
    filter = Filter(filter_id=2)
    filter = Filter(filter_id=1)
    
    filter = Filter(filter_id="R")
    filter = Filter(filter_id="g")
    
if __name__ == "__main__":
    #test_ptffield()
    #time_compute_var_indices()
    #select_most_observed_fields()
    
    logging.basicConfig(level=logging.INFO)
    #test_filter()
    #test_field()
    test_select_most_observed_fields()

"""
sources:
  "astrometricRMS": Float64Col(shape=(), dflt=0.0, pos=0),
  "bestAstrometricRMS": Float64Col(shape=(), dflt=0.0, pos=1),
  "bestChiSQ": Float32Col(shape=(), dflt=0.0, pos=2),
  "bestCon": Float32Col(shape=(), dflt=0.0, pos=3),
  "bestLinearTrend": Float32Col(shape=(), dflt=0.0, pos=4),
  "bestMagRMS": Float32Col(shape=(), dflt=0.0, pos=5),
  "bestMaxMag": Float32Col(shape=(), dflt=0.0, pos=6),
  "bestMaxSlope": Float32Col(shape=(), dflt=0.0, pos=7),
  "bestMeanMag": Float32Col(shape=(), dflt=0.0, pos=8),
  "bestMedianAbsDev": Float32Col(shape=(), dflt=0.0, pos=9),
  "bestMedianMag": Float32Col(shape=(), dflt=0.0, pos=10),
  "bestMinMag": Float32Col(shape=(), dflt=0.0, pos=11),
  "bestNAboveMeanByStd": UInt16Col(shape=(3,), dflt=0, pos=12),
  "bestNBelowMeanByStd": UInt16Col(shape=(3,), dflt=0, pos=13),
  "bestNMedianBufferRange": UInt16Col(shape=(), dflt=0, pos=14),
  "bestNPairPosSlope": UInt16Col(shape=(), dflt=0, pos=15),
  "bestPercentiles": Float32Col(shape=(12,), dflt=0.0, pos=16),
  "bestPeriodSearch": Float32Col(shape=(2,), dflt=0.0, pos=17),
  "bestSkewness": Float32Col(shape=(), dflt=0.0, pos=18),
  "bestSmallKurtosis": Float32Col(shape=(), dflt=0.0, pos=19),
  "bestStetsonJ": Float32Col(shape=(), dflt=0.0, pos=20),
  "bestStetsonK": Float32Col(shape=(), dflt=0.0, pos=21),
  "bestVonNeumannRatio": Float32Col(shape=(), dflt=0.0, pos=22),
  "bestWeightedMagRMS": Float32Col(shape=(), dflt=0.0, pos=23),
  "bestWeightedMeanMag": Float32Col(shape=(), dflt=0.0, pos=24),
  "chiSQ": Float32Col(shape=(), dflt=0.0, pos=25),
  "con": Float32Col(shape=(), dflt=0.0, pos=26),
  "dec": Float64Col(shape=(), dflt=0.0, pos=27),
  "linearTrend": Float32Col(shape=(), dflt=0.0, pos=28),
  "magRMS": Float32Col(shape=(), dflt=0.0, pos=29),
  "matchedSourceID": Int32Col(shape=(), dflt=0, pos=30),
  "maxMag": Float32Col(shape=(), dflt=0.0, pos=31),
  "maxSlope": Float32Col(shape=(), dflt=0.0, pos=32),
  "meanMag": Float32Col(shape=(), dflt=0.0, pos=33),
  "medianAbsDev": Float32Col(shape=(), dflt=0.0, pos=34),
  "medianMag": Float32Col(shape=(), dflt=0.0, pos=35),
  "minMag": Float32Col(shape=(), dflt=0.0, pos=36),
  "nAboveMeanByStd": UInt16Col(shape=(3,), dflt=0, pos=37),
  "nBelowMeanByStd": UInt16Col(shape=(3,), dflt=0, pos=38),
  "nMedianBufferRange": UInt16Col(shape=(), dflt=0, pos=39),
  "nPairPosSlope": UInt16Col(shape=(), dflt=0, pos=40),
  "nbestobs": UInt16Col(shape=(), dflt=0, pos=41),
  "ngoodobs": UInt16Col(shape=(), dflt=0, pos=42),
  "nobs": UInt16Col(shape=(), dflt=0, pos=43),
  "percentiles": Float32Col(shape=(12,), dflt=0.0, pos=44),
  "periodSearch": Float32Col(shape=(2,), dflt=0.0, pos=45),
  "ra": Float64Col(shape=(), dflt=0.0, pos=46),
  "referenceMag": Float32Col(shape=(), dflt=0.0, pos=47),
  "referenceMagErr": Float32Col(shape=(), dflt=0.0, pos=48),
  "skewness": Float32Col(shape=(), dflt=0.0, pos=49),
  "smallKurtosis": Float32Col(shape=(), dflt=0.0, pos=50),
  "stetsonJ": Float32Col(shape=(), dflt=0.0, pos=51),
  "stetsonK": Float32Col(shape=(), dflt=0.0, pos=52),
  "uncalibMeanMag": Float32Col(shape=(), dflt=0.0, pos=53),
  "vonNeumannRatio": Float32Col(shape=(), dflt=0.0, pos=54),
  "weightedMagRMS": Float32Col(shape=(), dflt=0.0, pos=55),
  "weightedMeanMag": Float32Col(shape=(), dflt=0.0, pos=56),
  "x": Float64Col(shape=(), dflt=0.0, pos=57),
  "y": Float64Col(shape=(), dflt=0.0, pos=58),
  "z": Float64Col(shape=(), dflt=0.0, pos=59)}

sourcedata:
  "a_world": Float32Col(shape=(), dflt=0.0, pos=0),
  "absphotzp": Float32Col(shape=(), dflt=0.0, pos=1),
  "alpha_j2000": Float64Col(shape=(), dflt=0.0, pos=2),
  "b_world": Float32Col(shape=(), dflt=0.0, pos=3),
  "background": Float32Col(shape=(), dflt=0.0, pos=4),
  "class_star": Float32Col(shape=(), dflt=0.0, pos=5),
  "delta_j2000": Float64Col(shape=(), dflt=0.0, pos=6),
  "errx2_image": Float32Col(shape=(), dflt=0.0, pos=7),
  "errxy_image": Float32Col(shape=(), dflt=0.0, pos=8),
  "erry2_image": Float32Col(shape=(), dflt=0.0, pos=9),
  "flux_auto": Float32Col(shape=(), dflt=0.0, pos=10),
  "flux_radius_1": Float32Col(shape=(), dflt=0.0, pos=11),
  "flux_radius_2": Float32Col(shape=(), dflt=0.0, pos=12),
  "flux_radius_3": Float32Col(shape=(), dflt=0.0, pos=13),
  "flux_radius_4": Float32Col(shape=(), dflt=0.0, pos=14),
  "flux_radius_5": Float32Col(shape=(), dflt=0.0, pos=15),
  "fluxerr_auto": Float32Col(shape=(), dflt=0.0, pos=16),
  "fwhm_image": Float32Col(shape=(), dflt=0.0, pos=17),
  "ipacFlags": Int16Col(shape=(), dflt=0, pos=18),
  "isoarea_world": Float32Col(shape=(), dflt=0.0, pos=19),
  "kron_radius": Float32Col(shape=(), dflt=0.0, pos=20),
  "mag": Float32Col(shape=(), dflt=0.0, pos=21),
  "magErr": Float32Col(shape=(), dflt=0.0, pos=22),
  "mag_aper_1": Int16Col(shape=(), dflt=0, pos=23),
  "mag_aper_2": Int16Col(shape=(), dflt=0, pos=24),
  "mag_aper_3": Int16Col(shape=(), dflt=0, pos=25),
  "mag_aper_4": Int16Col(shape=(), dflt=0, pos=26),
  "mag_aper_5": Int16Col(shape=(), dflt=0, pos=27),
  "mag_auto": Int16Col(shape=(), dflt=0, pos=28),
  "mag_iso": Int16Col(shape=(), dflt=0, pos=29),
  "mag_petro": Int16Col(shape=(), dflt=0, pos=30),
  "magerr_aper_1": Int16Col(shape=(), dflt=0, pos=31),
  "magerr_aper_2": Int16Col(shape=(), dflt=0, pos=32),
  "magerr_aper_3": Int16Col(shape=(), dflt=0, pos=33),
  "magerr_aper_4": Int16Col(shape=(), dflt=0, pos=34),
  "magerr_aper_5": Int16Col(shape=(), dflt=0, pos=35),
  "magerr_auto": Int16Col(shape=(), dflt=0, pos=36),
  "magerr_iso": Int16Col(shape=(), dflt=0, pos=37),
  "magerr_petro": Int16Col(shape=(), dflt=0, pos=38),
  "matchedSourceID": Int32Col(shape=(), dflt=0, pos=39),
  "mjd": Float64Col(shape=(), dflt=0.0, pos=40),
  "mu_max": Float32Col(shape=(), dflt=0.0, pos=41),
  "mu_threshold": Float32Col(shape=(), dflt=0.0, pos=42),
  "petro_radius": Float32Col(shape=(), dflt=0.0, pos=43),
  "pid": Int32Col(shape=(), dflt=0, pos=44),
  "relPhotFlags": Int16Col(shape=(), dflt=0, pos=45),
  "sextractorFlags": Int16Col(shape=(), dflt=0, pos=46),
  "sid": Int64Col(shape=(), dflt=0, pos=47),
  "theta_j2000": Float32Col(shape=(), dflt=0.0, pos=48),
  "threshold": Float32Col(shape=(), dflt=0.0, pos=49),
  "x": Float64Col(shape=(), dflt=0.0, pos=50),
  "x2_image": Float32Col(shape=(), dflt=0.0, pos=51),
  "x_image": Float32Col(shape=(), dflt=0.0, pos=52),
  "xpeak_image": Float32Col(shape=(), dflt=0.0, pos=53),
  "y": Float64Col(shape=(), dflt=0.0, pos=54),
  "y2_image": Float32Col(shape=(), dflt=0.0, pos=55),
  "y_image": Float32Col(shape=(), dflt=0.0, pos=56),
  "ypeak_image": Float32Col(shape=(), dflt=0.0, pos=57),
  "z": Float64Col(shape=(), dflt=0.0, pos=58)}
  
exposures:
  "absPhotZP": Float32Col(shape=(), dflt=0.0, pos=0),
  "absPhotZPRMS": Float32Col(shape=(), dflt=0.0, pos=1),
  "airmass": Float32Col(shape=(), dflt=0.0, pos=2),
  "background": Float32Col(shape=(), dflt=0.0, pos=3),
  "ccdID": UInt8Col(shape=(), dflt=0, pos=4),
  "decRMS": Float32Col(shape=(), dflt=0.0, pos=5),
  "expTime": Float32Col(shape=(), dflt=0.0, pos=6),
  "expid": Int32Col(shape=(), dflt=0, pos=7),
  "fieldID": UInt32Col(shape=(), dflt=0, pos=8),
  "filename": StringCol(itemsize=1024, shape=(), dflt='', pos=9),
  "filterID": UInt16Col(shape=(), dflt=0, pos=10),
  "infobits": Int32Col(shape=(), dflt=0, pos=11),
  "limitMag": Float32Col(shape=(), dflt=0.0, pos=12),
  "moonIllumFrac": Float32Col(shape=(), dflt=0.0, pos=13),
  "nSources": Int32Col(shape=(), dflt=0, pos=14),
  "nStars": Int32Col(shape=(), dflt=0, pos=15),
  "nTransients": Int32Col(shape=(), dflt=0, pos=16),
  "nightID": Int32Col(shape=(), dflt=0, pos=17),
  "obsDate": StringCol(itemsize=255, shape=(), dflt='', pos=18),
  "obsHJD": Float64Col(shape=(), dflt=0.0, pos=19),
  "obsMJD": Float64Col(shape=(), dflt=0.0, pos=20),
  "pid": Int32Col(shape=(), dflt=0, pos=21),
  "raRMS": Float32Col(shape=(), dflt=0.0, pos=22),
  "relPhotSatMag": Float32Col(shape=(), dflt=0.0, pos=23),
  "relPhotSysErr": Float32Col(shape=(), dflt=0.0, pos=24),
  "relPhotZP": Float32Col(shape=(), dflt=0.0, pos=25),
  "relPhotZPErr": Float32Col(shape=(), dflt=0.0, pos=26),
  "seeing": Float32Col(shape=(), dflt=0.0, pos=27)
"""