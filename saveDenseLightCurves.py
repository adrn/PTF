"""
    Use the coordinates in data/globularClusters.txt to select light 
    curves from the LSD and save them to pickle files.
"""

# Standard library
import os, sys, glob
from argparse import ArgumentParser
import logging
import cPickle as pickle

# Third-party
import numpy as np
import apwlib.geometry as g
import apwlib.convert as c

# Project
import db.util as dbu

def getOneCluster(name, ra, dec, radius=10, overwrite=False, skip=False):
    """ Given one set of globular cluster coordinates, download all
        light curves around 12 arcminutes from the cluster center

        Parameters
        -----------
        name : string
            The name of the globular cluster
        ra : float, apwlib.geometry.RA
            A Right Ascension in decimal degrees
        dec : float, apwlib.geometry.Dec
            A Declination in decimal degrees
        radius : float, optional
            Radius in arcminutes to search for light curves around the given ra, dec
        overwrite : bool, optional
            If a pickle exists, do you want to overwrite it?
    """
    ra = c.parseDegrees(ra)
    dec = c.parseDegrees(dec)
    radiusDegrees = radius / 60.0
    
    logging.debug("{0},{1} with radius={2} deg".format(ra, dec, radiusDegrees))
    
    outputFilename = os.path.join("data", "lightcurves", "{0}.pickle".format(name))
    logging.debug("Output file: {0}".format(outputFilename))
    
    elif os.path.exists(outputFilename) and not overwrite:
        if skip: return
        
        logging.info("{0} already exists!".format(outputFilename))
        toOverwrite = raw_input("{0} already exists! Overwrite? y/[n]:".format(outputFilename))
        if toOverwrite.lower() == "y":
            logging.debug("You've chosen to overwrite the file!")
            os.remove(outputFilename)
            logging.debug("File deleted.")
        else:
            return
    elif os.path.exists(outputFilename) and overwrite:
        logging.debug("You've chosen to overwrite the file!")
        os.remove(outputFilename)
        logging.debug("File deleted.")
    
    lightCurves = dbu.getLightCurvesRadial(ra, dec, radiusDegrees)
    
    f = open(outputFilename, "w")
    pickle.dump(lightCurves, f)
    f.close()
    
    return

def loadLightCurves():
    pickles = glob.glob("data/lightcurves/*.pickle")
    
    for file in pickles:
        try:
            dbu.loadLightCurves(file)
        except sqlalchemy.exc.IntegrityError:
            logging.info("File already loaded!")
            

def saveLightCurves():
    globularData = np.genfromtxt("data/globularClusters.txt", delimiter=",", usecols=[0,1,2], dtype=[("name", "|S20"), ("ra", "|S20"),("dec", "|S20")]).view(np.recarray)
    
    if not os.path.exists("data/lightcurves"):
        os.mkdir("data/lightcurves")
    
    arcmins = 10.
    for row in globularData:
        name,raStr,decStr = row
        logging.debug("RA string: {0}, Dec string: {1}".format(raStr, decStr))
        
        # Use apwlib's RA/Dec parsers to convert string RA/Decs to decimal degrees
        ra = g.RA.fromHours(raStr)
        dec = g.Dec.fromDegrees(decStr)
        
        getOneCluster(name, ra, dec, radius=arcmins, skip=True)
    
    # M31
    ra = g.RA.fromHours("00 42 44.3")
    dec = g.Dec.fromDegrees("+41 16 09")
    getOneCluster("M31", ra, dec, radius=2.2*60, skip=True)
    
    # Bulge
    ra = g.RA.fromHours("17:45:40.04")
    dec = g.Dec.fromDegrees("-29:00:28.1")
    getOneCluster("M31", ra, dec, radius=10.*60, skip=True)
    

if __name__ == "__main__":
    lsdLogger = logging.getLogger("lsd.pool2")
    lsdLogger.setLevel(logging.ERROR)
    
    lsdLogger = logging.getLogger("lsd.table")
    lsdLogger.setLevel(logging.ERROR)
    
    parser = ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite all image files (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty!")
    parser.add_argument("-l", "--load", action="store_true", dest="load", default=False,
                    help="Instead of saving light curves, load them into the database.")
    
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if args.load:
        loadLightCurves()
    else:
        saveLightCurves()