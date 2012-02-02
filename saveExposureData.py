"""
    Extract the entire ptf_exp table from the PTF 
    Large Survey Database and save it into a pickle.
    
    ** This script should be run on navtara **

"""

# Standard library
import os, sys
from argparse import ArgumentParser
import logging

from db.util import saveExposureData

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite the file if it exists (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    
    
    args = parser.parse_args()
    if args.verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
    saveExposureData(overwrite=args.overwrite, verbosity=verbosity)
    
else:
    raise ImportError("saveExposureData.py should be run as a standalone script, not imported!")