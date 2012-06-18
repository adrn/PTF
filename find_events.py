# -*- coding: utf-8 -*-

""" Search David Levitan's database for microlensing events. 
    
    This script gets a list of exposures from the LSD, and selects all PTF fields
    with more than 100 observations. Then, for each field selected, we select out
    all sources with η < 2σ and saves this into a large ?? file for further analysis.

"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard Library
import os, sys
from optparse import OptionParser
import logging

# Third-party dependencies
import numpy as np
try:
    import tables
except ImportError:
    raise ImportError("You have to run this with David Levitan's installation of Python here:\n\n \t/scr4/dlevitan/sw/epd-7.1-1-rh5-x86_64/bin/\n")

# PTF dependences

db_path = "/scr4/dlevitan/matches"

def compute_eta_2sigma(chip):
    """ Given a pytables object from DL's photometric database for one field/ccd,
        compute the 2σ levels for the von Neumann ratio to select out candidate
        microlensing events.
        
        e.g. if the file is 
            dbFile = tables.openFile('matches/match_02_100083_10.pytable')
        you should pass this object:
            dbFile.root.filter02.field100083.chip10
    """
    etas = chip.sources.col("vonNeumannRatio")
    mu = np.mean(etas[etas > 0.0])
    sigma = np.std(etas[etas > 0.0])
    
    return (mu - 2.*sigma, mu + 2.*sigma)

def select_candidates_from_chip(chip):
    """ Given a pytables object from DL's photometric database for one field/ccd,
        select out candidate microlensing events
        
        e.g. if the file is 
            dbFile = tables.openFile('matches/match_02_100083_10.pytable')
        you should pass this object:
            dbFile.root.filter02.field100083.chip10
    """
    sigma_cut = compute_eta_2sigma(chip)
    candidate_sources = chip.sources.readWhere("(vonNeumannRatio < {}) & (vonNeumannRatio > 0)".format(sigma_cut[0]))
    idx = np.in1d(chip.sourcedata.col("matchedSourceID"), candidate_sources["matchedSourceID"])    
    return chip.sourcedata[idx]

def save_candidate_light_curves_from_field(fieldid):
    """ Given a field id, return photometric information for candidate
        microlensing events selected on their von Neumann ratios.
    """
    
    field_candidates = None 
    for ccdid in range(12):
        file = "match_02_{:06d}_{:02d}.pytable".format(fieldid, ccdid)
        ccd_filename = os.path.join(db_path, file)
        
        if not os.path.exists(ccd_filename):
            logging.debug("Field: {}, CCD: {} table not found".format(fieldid, ccdid))
            continue
        
        dbFile = tables.openFile(ccd_filename)
        chip = getattr(getattr(dbFile.root.filter02, "field{:06d}".format(fieldid)), "chip{:02d}".format(ccdid))
        candidates = select_candidates_from_chip(chip)
        
        if field_candidates == None:
            field_candidates = candidates
        else:
            field_candidates = np.hstack(field_candidates, candidates)
    
    return field_candidates
        

if __name__ == "__main__":    
    parser = OptionParser(description="")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_option("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    
    (options, args) = parser.parse_args()
    if options.verbose: logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    elif options.quiet: logging.basicConfig(level=logging.ERROR, format='%(message)s')
    else: logging.basicConfig(level=logging.INFO, format='%(message)s')