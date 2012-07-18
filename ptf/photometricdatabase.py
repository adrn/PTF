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

try:
    import tables
except ImportError:
    logging.warn("PyTables not found! Some functionality won't work.\nTry running with: /scr4/dlevitan/sw/epd-7.1-1-rh5-x86_64/bin/python instead.")

# PTF
import parameters as ptf_params
from ptflightcurve import PTFLightCurve

#all_fields = np.load(os.path.join(ptf_params.config["PROJECTPATH"], "data", "all_fields.npy"))
all_fields = np.load(os.path.join("data", "all_fields.npy"))
match_path = "/scr4/dlevitan/matches"
pytable_base_string = os.path.join(match_path, "match_{:02d}_{:06d}_{:02d}.pytable") #filterid, fieldid, ccdid
filter_map = {"R" : 2, "g" : 1}

# ==================================================================================================
#   Classes
#
class Field(object):
    """ Represents a generic telescope field of view """
    
    def fromHDU(self, hdu):
        """ Uses a FITS HDU with a WCS header to create a field object """
        raise NotImplementedError()
    
    def __init__(self, ra, dec, dimensions, rotation_angle):
        """ Create a field object given a center position and the
            size of the FOV.
            
            TODO: Should this accept WCS information?
            
            Parameters
            ----------
            ra : any apwlib.geometry.RA parsable
                The Right Ascension at field center. If you don't pass in a apwlib.geometry.RA
                object, the units are assumed to be hours.
            dec : any apwlib.geometry.Dec parsable
                The Declination at field center. If you don't pass in a apwlib.geometry.Dec
                object, the units are assumed to be degrees.
            dimensions : tuple
                The size of the field.
            rotation_angle : any apwlib.geometry.Angle parsable
                The rotation angle of the field with respect to North. If you don't pass in a 
                apwlib.geometry.Angle object, the units are assumed to be degrees.
        """
        self.ra = g.RA(ra)
        self.dec = g.Dec(dec)
        
        # TODO: check to make sure this is a tuple of numbers
        self.dimensions = dimensions
        
        if isinstance(rotation_angle, g.Angle):
            self.rotation_angle = rotation_angle
        else:
            self.rotation_angle = g.Angle.fromDegrees(rotation_angle)
        

class PTFField(Field):
    """ Represents a PTF field, e.g. 12 CCDs """
    
    def __init__(self, id, name=None, filter_id=2):
        self.id = int(id)
        
        if name == None:
            self.name = str(name)
        else:
            self.name = str(self.id)
        
        self.filter_id = filter_id
        
        # Look up field information from all_fields
        field = all_fields[all_fields["id"] == self.id]
        ra = g.RA.fromDegrees(field["ra"][0])
        dec = g.Dec.fromDegrees(field["dec"][0])
        
        super(PTFField, self).__init__(ra, dec, ptf_params.camera_size_degrees, 0.0)
        
        # Figure out which CCDs have data
        ccds = dict()
        for ccdid in range(12):
            try:
                ccds[ccdid] = PTFCCD(id=ccdid, field_id=self.id, filter_id=self.filter_id)
            except ValueError:
                logging.debug("Field {}, CCD {} not found".format(self.id, ccdid))
                continue
                
        self.ccds = ccds
    
    def __repr__(self):
        return "<PTFField ID: {}>".format(self.id)
        
    @property
    def number_of_exposures(self):
        """ To get the number of observations, this must be run on navtara so the
            script has access to the PTF Photometric Database. 
        """
        observations_per_ccd = dict()
        for ccdid,ccd in self.ccds.items():
            chip = ccd.read()
            observations_per_ccd[ccdid] = chip.exposures.nrows
            ccd.close()
        
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
            ccd.close()
        
        return baseline_per_ccd
        
class PTFCCD(object):
    
    def __init__(self, id, field_id, filter_id):
        self.id = id
        self.field_id = field_id
        self.filter_id = filter_id
        
        self.filename = pytable_base_string.format(self.filter_id, self.field_id, self.id)
        
        if not os.path.exists(self.filename):
            raise ValueError("CCD data file does not exist!")
    
    def read(self):
        self._file = tables.openFile(self.filename)
        return getattr(getattr(getattr(self._file.root, "filter{:02d}".format(self.filter_id)), "field{:06d}".format(self.field_id)), "chip{:02d}".format(self.id))
    
    def close(self):
        self._file.close()

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
    
    fields = []
    for row in all_fields:
        field = PTFField(row["id"])
        
        for ccd_num in field.number_of_exposures.values():
            if ccd_num > minimum_num_observations:
                fields.append(field)
                break
        
        if limit > 0:
            if len(fields) >= limit: break
    
    return fields

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
    
if __name__ == "__main__":
    test = False
    
    if test:
        test_ptffield()
    
    select_most_observed_fields()