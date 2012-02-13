"""
    DESCRIPTION!

"""

# Standard library
import os, sys
from argparse import ArgumentParser
import logging
import cPickle as pickle

# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
import apwlib.geometry as g
import aplpy

# Project-specific
import db.util as dbu

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite all image files (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    parser.add_argument("-r", "--ra", dest="dec", type=str, help="")
    parser.add_argument("-d", "--dec", dest="dec", type=str, help="")
    parser.add_argument("-s", "--size", dest="dec", type=float, help="")
    
    args = parser.parse_args()
    if args.verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
    
    ra = g.RA.fromDegrees(args.ra)
    dec = g.Dec.fromDegrees(args.dec)
    size = float(args.size)
    
    dbu.getFITSCutout(ra.degrees, dec.degrees, size=size, save=False)
