# -*- coding: utf-8 -*-

""" This module provides some generic utility functions for Adrian's PTF project(s) """

# System libraries
import sys, os, glob
import urllib, urllib2, base64
import cStringIO as StringIO
import logging
import gzip

# Third party libraries
import numpy as np
import pyfits as pf

try:
    import apwlib.geometry as g
except ImportError:
    logging.warn("apwlib not found! Some functionality may not work correctly.\nDo: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install' to install.")

__author__ = "adrn"

# ==================================================================================================

def matchRADecToImages(ra, dec, size=None):
    """ This function is a wrapper around the IPAC PTF image server. This function 
        accepts an RA and Dec in degrees, and optionally a size, and returns a list 
        of images that overlap the given coordinates.
    """
        
    if size == None: url = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS={0},{1}".format(ra,dec)
    else: url = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS={0},{1}&SIZE={2}".format(ra,dec,size)
    logging.debug("Image Search URL: {0}".format(url))
    
    request = urllib2.Request(url)
    base64string = base64.encodestring('%s:%s' % ("PTF", "palomar")).replace('\n', '')
    request.add_header("Authorization", "Basic %s" % base64string)
    file = StringIO(urllib2.urlopen(request).read())
    filenames = np.genfromtxt(file, skiprows=4, usecols=[20], dtype=str)
    logging.debug("Image downloaded.")
    logging.debug("{0} images in list".format(len(filenames)))
    
    return sorted(list(filenames))

def getAllImages(imageList, prefix):
    """ Takes a list of PTF IPAC image basenames and downloads and saves them to the
        prefix directory.
    """    
    fieldList = dict()
    
    # Get any existing FITS files in the directory
    existingImages = glob.glob(os.path.join(prefix, "*.fits"))
    
    for image in existingImages:
        imageBase = os.path.splitext(os.path.basename(image))[0]
        ccd = imageBase.split("_")[-1]
        field = imageBase.split("_")[-2]
        
        try:
            fieldList[field].append(ccd)
        except KeyError:
            fieldList[field] = []
            fieldList[field].append(ccd)
    
    for image in imageList:
        imageBase = os.path.splitext(os.path.basename(image))[0]
        logging.debug("Image base: {0}".format(imageBase))
        
        if "scie" not in imageBase: continue

        # Extract ccd and fieldid from image filename
        ccd = imageBase.split("_")[-1]
        field = imageBase.split("_")[-2]
        
        logging.debug("Field: {0}, CCD: {1}".format(field,ccd))
        
        try:
            thisCCDList = fieldList[field]
            if ccd in thisCCDList: continue
        except KeyError:
            fieldList[field] = []
        
        file = os.path.join(prefix, os.path.basename(image))
        if os.path.exists(file):
            logging.info("File {0} already exists".format(file))
            continue
        
        imageURL = "http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/" + image
        
        request = urllib2.Request(imageURL)
        base64string = base64.encodestring('%s:%s' % ("PTF", "palomar")).replace('\n', '')
        request.add_header("Authorization", "Basic %s" % base64string)
        
        logging.debug("Image Full URL: {0}".format(request.get_full_url()))
        
        try:
            f = StringIO(urllib2.urlopen(request).read())
        except urllib2.HTTPError:
            continue
            
        try:
            gz = gzip.GzipFile(fileobj=f, mode="rb")
            gz.seek(0)
            
            fitsFile = StringIO(gz.read())
        except IOError:
            fitsFile = f
        
        fitsFile.seek(0)
        
        hdulist = pf.open(fitsFile, mode="readonly")
        hdulist.writeto(file)
        fieldList[field].append(ccd)
    
def getFITSCutout(ra, dec, size=0.5, save=False):
    """ This function is a wrapper around the IPAC PTF image server cutout feature. 
        Given an RA, Dec, and size in degrees, download a FITS cutout of the given
        coordinates +/- the size.
    """
    
    images = matchRADecToImages(ra, dec)

    for image in images:
        if "scie" in image: 
            break
    imageURL = "http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/" + image
    
    urlParams = {'center': '{0},{1}deg'.format(ra, dec),\
                 'size': '{0}deg'.format(size)}
    
    request = urllib2.Request(imageURL +"?{}".format(urllib.urlencode(urlParams)))
    base64string = base64.encodestring('%s:%s' % ("PTF", "palomar")).replace('\n', '')
    request.add_header("Authorization", "Basic %s" % base64string)
    
    logging.debug("Image Full URL: {0}".format(request.get_full_url()))
    
    f = StringIO(urllib2.urlopen(request).read())
    gz = gzip.GzipFile(fileobj=f, mode="rb")
    gz.seek(0)
    
    fitsFile = StringIO(gz.read())
    fitsFile.seek(0)
    
    hdulist = pf.open(fitsFile, mode="readonly")
    logging.debug("Image loaded. Size: {0} x {1}".format(*hdulist[0].data.shape))
    
    if save:
        ra = g.RA.fromDegrees(ra)
        dec = g.Dec.fromDegrees(dec)
        filename = os.path.join("images", "{0}_{1}_{2}x{3}deg.fits".format(ra.string(sep="-"), dec.string(sep="-"), size, size))
        logging.debug("Writing file to: {0}".format(filename))
        hdulist.writeto(filename)
        return True
    else:
        return hdulist

def plotLightCurves():
    """ """
    pass
    