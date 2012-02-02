"""
    Load the ptf_exp table dump from the PTF 
    Large Survey Database into the ptf database
    
    ** This script should be run on navtara **

"""

# Standard library
import os, sys
from argparse import ArgumentParser
import logging

from db.util import loadExposureData

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
    loadExposureData(overwrite=args.overwrite, verbosity=verbosity)
    
else:
    raise ImportError("loadExposureData.py should be run as a standalone script, not imported!")
