# coding: utf-8
from __future__ import division

""" This module contains utilities for computing PTF's survey coverage, both in time and position. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import datetime
import re
import glob
import cPickle as pickle
import logging
import time, datetime

# Third-party
import numpy as np
import tables

# Project
# TODO: add some kind of catch here -- if this import fails, it's probably because I'm trying to
#           run on the wrong machine...
import ptf.photometricdatabase as pdb

# Create logger for this module
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

class SurveyInfo(object):
    
    def __init__(self, filter, overwrite=False):
        """ Convenience class for getting information about the PTF survey
            
            Parameters
            ----------
            filter : int, str, ptf.photometricdatabase.Filter
                Any parseable, unambiguous filter representation.
            overwrite : bool (optional)
                Overwrite any cache files if they already exist
        """
        
        if not isinstance(filter, pdb.Filter):
            self.filter = pdb.Filter(filter)
        
        cache_filename = os.path.join("data", "survey_coverage", "fields_observations_{}.npy".format(str(self.filter)))
        self._fields_exposures = get_fields_exposures(self.filter, filename=cache_filename, overwrite=overwrite)
        self.timestamp = datetime.datetime.strptime(time.ctime(os.path.getmtime(cache_filename)), "%a %b %d %H:%M:%S %Y")
        
    def fields(self, min_num_observations):
        """ Return a list of fields with more than the above number of observations """
        
        field_ids = self._fields_exposures[self._fields_exposures["num_exposures"] > min_num_observations]["field"]
        return [pdb.Field(x, self.filter) for x in field_ids]

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
    
    if not isinstance(filter, pdb.Filter):
        raise ValueError("Filter must be a valid Filter() object!")
    
    if filename == None:
        filename = os.path.join("data", "survey_coverage", "fields_observations_{}.npy".format(str(filter)))
    
    if os.path.exists(filename) and overwrite:
        logger.debug("Data file already exists, but you want to overwrite it")
        os.remove(filename)
        logger.debug("File {} deleted".format(filename))
    elif os.path.exists(filename) and not overwrite:
        logger.info("Data file already exists.")
    
    if not os.path.exists(filename):
        logger.info("Data file doesn't exist -- it could take some time to create it!")
        
        fields = []
        exposures = []
        
        pattr = re.compile(".*match_(\d+)_(\d+)_(\d+)")
        for match_filename in glob.glob("/scr4/dlevitan/matches/match_{:02d}_*".format(filter.id)):
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
    
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite any existing / cached files")
                    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    R_info = SurveyInfo(filter="R", overwrite=args.overwrite)
    g_info = SurveyInfo(filter="g", overwrite=args.overwrite)
    
    
    """
    save_fields_exposures(filename=args.filename, overwrite=args.overwrite)
    
    f = open("data/field_exposures.pickle")
    data = pickle.load(f)
    f.close()
    
    print "Date: {}".format(data["date_stamp"])
    print "------------------------------------------"
    g_25 = data["g"][data["g"]["num_exposures"] > 25]
    g_100 = data["g"][data["g"]["num_exposures"] > 100]
    print "Number of g-band fields >25 observations:\t{}, {:.2f} sq. deg.".format(len(g_25), len(g_25)*7.26)
    print "Number of g-band fields >100 observations:\t{}, {:.2f} sq. deg.".format(len(g_100), len(g_100)*7.26)
    
    print "------------------------------------------"
    R_25 = data["R"][data["R"]["num_exposures"] > 25]
    R_100 = data["R"][data["R"]["num_exposures"] > 100]
    print "Number of R-band fields >25 observations:\t{}, {:.2f} sq. deg.".format(len(R_25), len(R_25)*7.26)
    print "Number of R-band fields >100 observations:\t{}, {:.2f} sq. deg.".format(len(R_100), len(R_100)*7.26)
    print 
    print
    
    num_lcs = 0
    for field_id in R_25["field"]:
        field = pdb.Field(field_id, "R")
        ccd = field.ccds.values()[0]
        chip = ccd.read()
        num_lcs += len(chip.sources.readWhere("ngoodobs > 25"))*11
        ccd.close()
    
    print "Total number of light curves with >25 good observations in R-band >25 observation fields is: {}".format(num_lcs)
    """