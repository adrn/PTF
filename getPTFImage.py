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
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
                    
    parser.add_argument("-r", "--ra", dest="ra", type=str, help="")
    parser.add_argument("-R", "--ra-units", dest="raUnits", type=str, default="degrees", \
                    help="")
    
    parser.add_argument("-d", "--dec", dest="dec", type=str, help="")
    parser.add_argument("-D", "--dec-units", dest="decUnits", type=str, default="degrees",\
                    help="")
    
    parser.add_argument("-s", "--size", dest="size", type=float, help="")
    parser.add_argument("-o", "--output-file", dest="output", type=str,
                    help="")
    
    args = parser.parse_args()

    if args.verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
    
    ra = g.RA(args.ra, units=args.raUnits)
    dec = g.Dec(args.dec, units=args.decUnits)
    size = float(args.size)
    
    hdulist = dbu.getFITSCutout(ra.degrees, dec.degrees, size=size, save=False)
    hdulist.writeto(args.output)
