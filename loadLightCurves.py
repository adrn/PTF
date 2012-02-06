"""
    Load light curves from the Large Survey Database 
    and save it into a pickle.
    
    ** This script should be run on deimos **

"""

# Standard library
import os, sys
from argparse import ArgumentParser
import logging

from db.util import loadLightCurves

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    parser.add_argument("-f", "--filename", type=str, dest="filename", 
                    help="Data file containing unique field ids for dense fields")
    
    args = parser.parse_args()
    if args.verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
        
    loadLightCurves(args.filename, verbosity=verbosity)
    
else:
    raise ImportError("loadLightCurves.py should be run as a standalone script, not imported!")
