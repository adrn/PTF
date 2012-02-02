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
    parser.add_argument("-f", "--file", dest="file", default="exposureTable.pickle",
                    help="The path to the output pickle file")
    
    args = parser.parse_args()
    
    logger = logging.getLogger("SaveFieldData")
    ch = logging.StreamHandler()
    logger.addHandler(ch)
    
    saveExposureData(filename=args.file, overwrite=args.overwrite)
    
else:
    raise ImportError("getExposureData.py should be run as a standalone script, not imported!")
