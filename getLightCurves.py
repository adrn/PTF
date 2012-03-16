"""
    This is a standalone script that accepts an ra, dec, and radius and downloads all 
    light curves within the radius of the given coordinates. Optionally, you can specify
    a filename to dump to and a minimum number of observations to save 
        (see 'python getLightCurves.py --help')
"""

# Standard library
import os, sys, glob
from argparse import ArgumentParser
import logging
import cPickle as pickle

# Third-party
import sqlalchemy
import numpy as np
import apwlib.geometry as g

# Project
import ptf.db.util as dbu

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    parser.add_argument("-f", "--filename", type=str, dest="filename", 
                    help="Data file containing unique field ids for dense fields")
    parser.add_argument("-R", "--ra", type=str, dest="ra", 
                    help="The right ascension at the center of your field (HOURS)")
    parser.add_argument("-D", "--dec", type=str, dest="dec", 
                    help="The declination at the center of your field (DEGREES)")
    parser.add_argument("-r", "--radius", type=float, dest="radius", 
                    help="The radius of the area you want to select (DEGREES)")
    
    args = parser.parse_args()
    if args.verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
    
    ra = g.RA(args.ra)
    dec = g.Dec(args.dec)
    
    name = "{0}_{1}_{2}".format(ra.degrees, dec.degrees, args.radius)
    
    if args.radius > 0.25:
        outputFilename = os.path.join("data", "lightcurves", "{0}_{1}.fits")
        lightCurveGen = dbu.getLightCurvesRadialBig(ra.degrees, dec.degrees, args.radius)
    
        ii = 1
        while True:
            try:
                lightCurves = lightCurveGen.next()
                logging.debug("Output file: {0}".format(outputFilename.format(name, ii)))
                
                # Write to a FITS file instead!
                hdu = pf.BinTableHDU(lightCurves)
                hdu.writeto(outputFilename.format(name, ii))
            
                ii += 1
            except StopIteration:
                break
    else:
        outputFilename = os.path.join("data", "lightcurves", "{0}.fits")
        lightCurveArray = dbu.getLightCurvesRadial(ra.degrees, dec.degrees, args.radius)
        
        # Write to a FITS file instead!
        hdu = pf.BinTableHDU(lightCurves)
        hdu.writeto(outputFilename.format(name))
    
    print "Now you can do:\n\t python loadLightCurves.py -f {0}\n to load the data on Deimos.".format(args.filename)
else:
    raise ImportError("getLightCurves.py should be run as a standalone script, not imported!")
