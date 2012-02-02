"""
    Extract the entire ptf_exp table from the PTF 
    Large Survey Database and save it into a pickle.
    
    ** This script should be run on navtara **

"""

# Standard library
import os, sys
from argparse import ArgumentParser
import logging

from db.util import saveWellSampledDenseFields

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("-m", "--minimum-exposures", type=int, dest="minimum", default=25,
                    help="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    
    args = parser.parse_args()
    if args.verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
        
    saveWellSampledDenseFields(args.minimum, verbosity=verbosity)
    
else:
    raise ImportError("saveDenseFieldLightCurves.py should be run as a standalone script, not imported!")
