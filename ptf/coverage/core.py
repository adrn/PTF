# coding: utf-8
from __future__ import division

""" 
    This module contains utilities for computing PTF's survey coverage, both in time and position. 
"""

# Standard library
import sys, os
import datetime
import time

# Third-party
import numpy as np

# PTF
from ..globals import ccd_size, all_fields, _base_path
from ..lightcurve import PTFLightCurve, PDBLightCurve
from ..analyze import compute_variability_indices
from ..util import get_logger
logger = get_logger(__name__)

__all__ = ["SurveyInfo"]

def get_fields_exposures(filter, filename=None, overwrite=False):
    """ Given a filter, go to the PTF photometric database and get information about all PTF
        fields for that filter, and the number of good exposures per field.
        
        Parameters
        ----------
        filter : ptf.photometricdatabase.Filter
            Must be a Filter object (see the above module)
        filename : str (optional)
            The filename to store this data to.
        overwrite : bool (optional)
            Overwrite 'filename' if it already exists.
        
    """
    
    #if not isinstance(filter, pdb.Filter):
    #    raise ValueError("Filter must be a valid Filter() object!")
    
    if filename == None:
        filename = os.path.join(_base_path, "data", "survey_coverage", "fields_observations_{}.npy".format(str(filter)))
    
    if os.path.exists(filename) and overwrite:
        logger.debug("Data file already exists, but you want to overwrite it")
        os.remove(filename)
        logger.debug("File {} deleted".format(filename))
    elif os.path.exists(filename) and not overwrite:
        logger.info("Data file already exists: {}".format(filename))
    
    if not os.path.exists(filename):
        logger.info("Data file doesn't exist -- it could take some time to create it!")
        
        fields = []
        exposures = []
        
        pattr = re.compile(".*match_(\d+)_(\d+)_(\d+)")
        for match_filename in glob.glob("/scr4/dlevitan/matches/match_{:02d}_*.pytable".format(filter.id)):
            logger.debug("Reading file: {}".format(match_filename))
            
            filter_id, field_id, ccd_id = map(int, pattr.search(match_filename).groups())
            
            if field_id in fields:
                continue
    
            try:
                file = tables.openFile(match_filename)
                chip = getattr(getattr(getattr(file.root, "filter{:02d}".format(filter_id)), "field{:06d}".format(field_id)), "chip{:02d}".format(ccd_id))
            except:
                continue
            
            fields.append(field_id)
            exposures.append(len(chip.exposures))
            
            file.close()
        
        fields_exposures = np.array(zip(fields, exposures), dtype=[("field", int), ("num_exposures", int)])
        logger.debug("Saving file {}".format(filename))
        np.save(filename, fields_exposures)
    
    fields_exposures = np.load(filename)
    logger.debug("Data file loaded!")
    
    return fields_exposures

class SurveyInfo(object):
    """ Some example use cases for this object:
        
        TODO: Fill this in! ???
    
    """
    def __init__(self, filter, overwrite=False):
        """ Convenience class for getting information about the PTF survey
            
            Parameters
            ----------
            filter : int, str, ptf.photometricdatabase.Filter
                Any parseable, unambiguous filter representation.
            overwrite : bool (optional)
                Overwrite any cache files if they already exist
        """
        
        from ..db.photometric_database import Filter
        if not isinstance(filter, Filter):
            self.filter = Filter(filter)
        
        cache_filename = os.path.join(os.path.split(_base_path)[0], "data", "survey_coverage", "fields_observations_{}.npy".format(str(self.filter)))
        self._fields_exposures = get_fields_exposures(self.filter, filename=cache_filename, overwrite=overwrite)
        self.timestamp = datetime.datetime.strptime(time.ctime(os.path.getmtime(cache_filename)), "%a %b %d %H:%M:%S %Y")
        
    def fields(self, min_num_observations):
        """ Return a list of fields with more than the above number of observations """
        
        from ..db.photometric_database import Field
        rows = self._fields_exposures[self._fields_exposures["num_exposures"] >= min_num_observations]
        fields = [Field(row["field"], self.filter, number_of_exposures=row["num_exposures"]) for row in rows]
        return [f for f in fields if f.ra != None]
        
