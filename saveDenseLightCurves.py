"""
    Use the coordinates in data/globularClusters.txt to select light 
    curves from the LSD and save them to pickle files.
"""

# Standard library
import os, sys, glob
from argparse import ArgumentParser
import logging
import cPickle as pickle
import socket

# Third-party
import sqlalchemy
import numpy as np
import pyfits as pf
import apwlib.geometry as g
import apwlib.convert as c

# Project
import ptf.db.util as dbu

if socket.gethostname() != "kepler" and socket.gethostname() != "navtara":
    try:
        from ptf.db.DatabaseConnection import *
        from ptf.db.NumpyAdaptors import *
    except:
        logging.warn("Connection to deimos could not be established. Postgres database features won't work.")


def loadLightCurves(filename):
    """ """
    
    logging.debug("Opening {}...".format(filename))
    f = pf.open(filename)
    resultsArray = f[1].data.view(np.recarray)
    f.close()
    logging.debug("File loaded!")
    
    session.begin()
    logging.debug("Starting database load...")
    for objid in np.unique(resultsArray.objid):
        lightCurveData = resultsArray[resultsArray.objid == objid]
        if len(lightCurveData.mag) < 10: continue
        
        lightCurve = LightCurve()
        lightCurve.objid = np.array([objid]).astype(np.uint64)[0]
        lightCurve.mag = lightCurveData.mag
        lightCurve.mag_error = lightCurveData.mag_err
        lightCurve.mjd = lightCurveData.mjd
        lightCurve.sys_error = lightCurveData.sys_err
        lightCurve.ra = lightCurveData.ra[0]
        lightCurve.dec = lightCurveData.dec[0]
        lightCurve.flags = lightCurveData["flags"].astype(np.uint16)
        lightCurve.imaflags = lightCurveData.imaflags_iso.astype(np.uint16)
        lightCurve.filter_id = lightCurveData.filter_id.astype(np.uint8)
        session.add(lightCurve)
        
        if len(session.new) == 1000:
            session.commit()
            logging.debug("1000 light curves committed!")
            session.begin()
    
    logging.info("All light curves committed!")
    session.commit()

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
    
    if os.path.exists(outputFilename) and not overwrite:
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

def getBigCluster(name, ra, dec, radius=1.0):
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
            Radius in DEGREES to search for light curves around the given ra, dec
    """
    ra = c.parseDegrees(ra)
    dec = c.parseDegrees(dec)
    radiusDegrees = radius
    
    logging.debug("{0},{1} with radius={2} deg".format(ra, dec, radiusDegrees))
    
    outputFilename = os.path.join("data", "lightcurves", "{0}_{1}.fits")
    lightCurveGen = dbu.getLightCurvesRadialBig(ra, dec, radiusDegrees)
    
    ii = 1
    while True:
        try:
            lightCurves = lightCurveGen.next()
            logging.debug("Output file: {0}".format(outputFilename.format(name, ii)))
            
            #f = open(outputFilename.format(name, ii), "w")
            #pickle.dump(lightCurves, f)
            #f.close()
            
            # Write to a FITS file instead!
            hdu = pf.BinTableHDU(lightCurves)
            hdu.writeto(outputFilename.format(name, ii))
        
            ii += 1
        except StopIteration:
            return
    
    return

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
    getBigCluster("M31", ra, dec, radius=2.2)
    
    # Bulge
    ra = g.RA.fromHours("17:45:40.04")
    dec = g.Dec.fromDegrees("-29:00:28.1")
    getBigCluster("Bulge", ra, dec, radius=10.)

def loadM31LightCurves():
    try:
        f = open("data/lightcurves/alreadyLoadedFiles.pickle")
        alreadyLoadedFiles = pickle.load(f)
        f.close()
    except IOError:
        alreadyLoadedFiles = []
        
    m31Files = glob.glob("data/lightcurves/M31*.fits")
    
    for file in m31Files:
        if file in alreadyLoadedFiles: 
            logging.debug("File {0} already loaded!".format(file))
            continue
        
        logging.debug("loading file {0}".format(file))
        try:
            loadLightCurves(file)
            f = open("data/lightcurves/alreadyLoadedFiles.pickle", "w")
            alreadyLoadedFiles.append(file)
            pickle.dump(alreadyLoadedFiles, f)
            f.close()
        except sqlalchemy.exc.IntegrityError:
            logging.warn("Duplicate objid found!")
            session.close()

def loadBulgeLightCurves():
    try:
        f = open("data/lightcurves/alreadyLoadedFiles.pickle")
        alreadyLoadedFiles = pickle.load(f)
        f.close()
    except IOError:
        alreadyLoadedFiles = []
        
    bulgeFiles = glob.glob("data/lightcurves/Bulge*.fits")
    
    for file in bulgeFiles:
        logging.debug("loading file {0}".format(file))
        try:
            loadLightCurves(file)
            f = open("data/lightcurves/alreadyLoadedFiles.pickle", "w")
            alreadyLoadedFiles.append(file)
            pickle.dump(alreadyLoadedFiles, f)
            f.close()
        except sqlalchemy.exc.IntegrityError:
            logging.warn("Duplicate objid found!")
            session.close()


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
        loadM31LightCurves()
        loadBulgeLightCurves()
    else:
        saveLightCurves()