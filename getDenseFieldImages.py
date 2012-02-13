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

# Third-party packages
import matplotlib.pyplot as plt
import numpy as np
import apwlib.geometry as g
import aplpy

# Project-specific
import db.util as dbu

def singleField(verbosity):
    """ """
    try:
        f = open("data/denseCoordinates.pickle")
        denseCoordinates = pickle.load(f)
        f.close()
    except IOError:
        raise IOError("data/denseCoordinates.pickle doesn't exist!\n Did you 'git pull' from navtara?")
    
    logger = dbu.getLogger(None, verbosity, name="getDenseFieldImages")
    size = 0.5
    
    for ra,dec in denseCoordinates:
        fig = plt.figure()
        try:
            hdulist = dbu.getFITSCutout(ra, dec, size=size, logger=logger, verbosity=verbosity, save=False) # 30 arcminute images
        except:
            print "FAIL\nFAIL\nFAIL\nFAIL\nFAIL\n"
            try:
                hdulist = dbu.getFITSCutout(np.floor(ra), np.floor(dec), size=size, logger=logger, verbosity=verbosity, save=False) # 30 arcminute images
            except:
                print "ANOTHER FAIL\nFAIL\nFAIL\nFAIL\nFAIL\n"
                continue

        fitsFigure = aplpy.FITSFigure(hdulist[0], figure=fig)
        fitsFigure.show_grayscale()
        ax = fig.get_axes()[0]
        pixRA, pixDec = fitsFigure.world2pixel(ra, dec)
        ax.plot([pixRA], [pixDec], "r+", ms=10)
        
        ra = g.RA.fromDegrees(ra)
        dec = g.Dec.fromDegrees(dec)
        filename = os.path.join("images", "{0}_{1}_{2}x{3}deg.png".format(ra.string(sep="-"), dec.string(sep="-"), size, size))
        plt.savefig(filename)
        del fig

def allFields(verbosity):
    """ """
    try:
        f = open("data/denseCoordinates.pickle")
        denseCoordinates = pickle.load(f)
        f.close()
    except IOError:
        raise IOError("data/denseCoordinates.pickle doesn't exist!\n Did you 'git pull' from navtara?")
    
    logger = dbu.getLogger(None, verbosity, name="getDenseFieldImages")
    size = 0.5
    
    for ra,dec in denseCoordinates:
        ra = g.RA.fromDegrees(ra)
        dec = g.Dec.fromDegrees(dec)
        prefix = "images/{0}_{1}".format(ra.string(sep="-"), dec.string(sep="-"))
        
        if not os.path.exists(prefix):
            os.mkdir(prefix)
        imageList = dbu.matchRADecToImages(ra.degrees, dec.degrees, size=size, logger=logger)
        
        dbu.getAllImages(imageList, prefix=prefix, logger=logger)

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite all image files (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    parser.add_argument("-a", "--all-fields", action="store_true", dest="all", default=False,
                    help="Get all fields around specified RA/Dec")
    
    args = parser.parse_args()
    if args.verbose:
        verbosity = logging.DEBUG
    else:
        verbosity = logging.INFO
    
    if args.all:
        allFields(verbosity)
    else:
        singleField(verbosity)
