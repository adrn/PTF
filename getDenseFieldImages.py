"""
    Use the coordinates in denseCoordinates.pickle to download images
    from the IPAC interface so I can measure sizes of the clusters 
    for selecting light curves.

"""

# Standard library
import os, sys
from argparse import ArgumentParser
import logging
import cPickle as pickle

import db.util as dbu

def main():
    """ """
    try:
        f = open("data/denseCoordinates.pickle")
        denseCoordinates = pickle.load(f)
        f.close()
    except IOError:
        raise IOError("data/denseCoordinates.pickle doesn't exist!\n Did you 'git pull' from navtara?")
    
    

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite all image files (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    
    args = parser.parse_args()
    if args.verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
    loadExposureData(verbosity=verbosity)
    
else:
    raise ImportError("loadExposureData.py should be run as a standalone script, not imported!")
