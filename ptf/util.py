# -*- coding: utf-8 -*-

""" This module provides some generic utility functions for Adrian's PTF project(s) """

# System libraries
import sys, os, glob
import urllib, urllib2, base64
import cStringIO as StringIO
import logging
import gzip
import datetime

# Third party libraries
import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt

try:
    import apwlib.geometry as g
    import apwlib.astrodatetime as adatetime
except ImportError:
    logging.warn("apwlib not found! Some functionality may not work correctly.\nDo: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install' to install.")

__author__ = "adrn"

# ==================================================================================================

class PTFImageQuery:
    
    def __init__(self):
        columns = ["obsdate", "crval1", "crval2", "filter", "ccdid", "ptffield", \
                   "seeing", "airmass", "pfilename", "afilename1", "obsmjd"]
                       
        self.search_clauses = []
        self.where_clauses = []
        self.base_url = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process"
        self.base_url += "?columns=" + ",".join(columns)
    
    def on_date(self, date):
        """ Add a date constraint to the query
            
            date : str, tuple, datetime.datetime
                If str, must be a date like YEAR-MONTH-DAY. If tuple, must be (year, month, day).
        """
        
        fmt_string = "where=obsdate+LIKE+'%25{}-{:02}-{:02}%25'"
        fmt_items = []
        
        if isinstance(date, str):
            fmt_items = date.split("-")
        elif isinstance(date, datetime.datetime):
            fmt_items.append(date.year)
            fmt_items.append(date.month)
            fmt_items.append(date.day)
            
            if not (date.hour == 0 and date.minute == 0 and date.second == 0):
                fmt_items.append(date.hour)
                fmt_items.append(date.minute)
                fmt_items.append(date.second)
                fmt_string = "obsdate%20LIKE%20%27{}-{:02}-{:02}%20{:02}:{:02}:{:02}%25%27"
                
        elif isinstance(date, tuple):
            fmt_items = list(date)
        else:
            raise ValueError("Invalid date type")
        
        self.where_clauses.append(fmt_string.format(*fmt_items))
    
    def on_mjd(self, mjd):
        """ Add a date constraint to the query with an MJD """
        
        if isinstance(mjd, int):
            self.where_clauses.append("obsmjd%20between%20{}%20and%20{}".format(mjd, mjd+1))
        elif isinstance(mjd, float):
            self.where_clauses.append("obsmjd%20between%20{}%20and%20{}".format(mjd-0.05, mjd+0.05))
        else:
            raise ValueError("Invalid MJD")
    
    def at_position(self, ra, dec):
        """ Add a position constraint to the query
            
            ra : anything that apwlib can parse (str, float, Angle, RA)
            dec : anything that apwlib can parse (str, float, Angle, Dec)
        """
        ra = g.RA(ra)
        dec = g.Dec(dec)
        
        self.search_clauses.append("POS={0.degrees},{1.degrees}".format(ra, dec))
    
    def size(self, size):
        """ Add a size constraint to the query 
            
            size : float, int
                A search size in degrees
        """
        self.search_clauses.append("SIZE={}".format(size))
    
    def field(self, fieldid):
        """ Add a field constraint to the query 
            
            field : int, list
                A PTF Field ID
        """
        if not isinstance(fieldid, list):
            fieldid = [fieldid]
        
        fieldid = [str(x) for x in fieldid]
        
        self.where_clauses.append("ptffield%20IN%20({})".format(",".join(fieldid)))
    
    def filter(self, filter):
        """ Constrain on R or g filter """
        self.where_clauses.append("filter=%27{}%27".format(filter))
    
    def ccds(self, ccds):
        """ Add a CCD constraint to the query 
            
            ccds : list
                A lsit of CCDs
        """
        if not isinstance(ccds, list):
            ccds = [ccds]
        
        ccds = [str(x) for x in ccds]
        
        self.where_clauses.append("ccdid%20IN%20({})".format(",".join(ccds)))
    
    @property
    def where(self):
        return "where=" + "%20and%20".join(self.where_clauses)
    
    @property
    def url(self):
        return self.base_url + "&".join(self.search_clauses) + "&{}".format(self.where)
    
    def __repr__(self):
        return "<PTFImageQuery:\t{}".format("\n\t\t".join(self.search_clauses + self.where_clauses)) + ">"
    
    def __str__(self):
        return self.url

class PTFImageList:
    
    @classmethod
    def fromQueryReturnFile(cls, filename):
        recarray_columns = ["date", "time", "ra", "dec", "filter", "ccdid", "fieldid", \
                            "seeing", "airmass", "data_filename", "mask_filename", "mjd"]
        recarray_dtypes = ["|S10", "|S12", float, float, "|S1", int, int, \
                           float, float, "|S120", "|S120", float]
        dtype = zip(recarray_columns, recarray_dtypes)
        
        table = np.genfromtxt(filename, skiprows=4, dtype=dtype).view(np.recarray)
        
        return cls(table)
    
    @classmethod
    def fromQueryReturn(cls, text_blob):
        recarray_columns = ["date", "time", "ra", "dec", "filter", "ccdid", "fieldid", \
                            "seeing", "airmass", "data_filename", "mask_filename", "mjd"]
        recarray_dtypes = ["|S10", "|S12", float, float, "|S1", int, int, \
                           float, float, "|S120", "|S120", float]
        dtype = zip(recarray_columns, recarray_dtypes)
        
        file = StringIO.StringIO(text_blob)
        table = np.genfromtxt(file, skiprows=4, dtype=dtype).view(np.recarray)
        
        return cls(table)
        
    @classmethod
    def fromImageQuery(cls, ptf_image_query):
        """ Given a PTFImageQuery object, run the query and get the image list back """
        
        recarray_columns = ["date", "time", "ra", "dec", "filter", "ccdid", "fieldid", \
                            "seeing", "airmass", "data_filename", "mask_filename", "mjd"]
        recarray_dtypes = ["|S10", "|S12", float, float, "|S1", int, int, \
                           float, float, "|S120", "|S120", float]
        dtype = zip(recarray_columns, recarray_dtypes)
        
        request = urllib2.Request(ptf_image_query.url)
        base64string = base64.encodestring('%s:%s' % ("PTF", "palomar")).replace('\n', '')
        request.add_header("Authorization", "Basic %s" % base64string)
        file = StringIO.StringIO(urllib2.urlopen(request).read())
        
        #self.table = np.genfromtxt(file, skiprows=4, usecols=range(len(recarray_columns)+3)[3:], dtype=dtype).view(np.recarray)
        table = np.genfromtxt(file, skiprows=4, dtype=dtype).view(np.recarray)
        
        return cls(table)
        
    def __init__(self, recarray):
        """ Given a numpy recarray, create an image list """
        self.table = recarray
    
    @property
    def scie(self):
        return self.table.data_filename
    
    @property
    def mask(self):
        return self.table.mask_filename
    
    def best_seeing_images(self):
        tb = self.table[np.logical_not(np.isnan(self.table.seeing))]
        return tb[tb.seeing == np.min(tb.seeing)]
        
#class PTFCCDImage:    
#    def __init__(self, ccdid, fieldid, filter

#class PTFFieldImage:
#    pass

def matchRADecToImages(ra, dec, size=None):
    """ This function is a wrapper around the IPAC PTF image server. This function 
        accepts an RA and Dec in degrees, and optionally a size, and returns a list 
        of images that overlap the given coordinates.
    """
    
    ra = g.RA(ra).degrees
    dec = g.Dec(dec).degrees
    
    if size == None: url = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS={0},{1}".format(ra,dec)
    else: url = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS={0},{1}&SIZE={2}".format(ra,dec,size)
    logging.debug("Image Search URL: {0}".format(url))
    
    request = urllib2.Request(url)
    base64string = base64.encodestring('%s:%s' % ("PTF", "palomar")).replace('\n', '')
    request.add_header("Authorization", "Basic %s" % base64string)
    file = StringIO.StringIO(urllib2.urlopen(request).read())
    filenames = np.genfromtxt(file, skiprows=4, usecols=[20,21], dtype=[("data","|S110"),("mask","|S110")])
    logging.debug("Image downloaded.")
    logging.debug("{0} images in list".format(len(filenames)))
    
    return list(filenames.data)

def getAllImages(imageList, prefix):
    """ Takes a list of PTF IPAC image basenames and downloads and saves them to the
        prefix directory.
    """    
    scieFieldList = dict()
    maskFieldList = dict()
    
    # Get any existing FITS files in the directory
    existingImages = glob.glob(os.path.join(prefix, "*.fits"))
    
    for image in existingImages:
        imageBase = os.path.splitext(os.path.basename(image))[0]
        ccd = imageBase.split("_")[-1]
        field = imageBase.split("_")[-2]
        
        if "scie" in image:
            try:
                scieFieldList[field].append(ccd)
            except KeyError:
                scieFieldList[field] = []
                scieFieldList[field].append(ccd)
        elif "mask" in image:
            try:
                maskFieldList[field].append(ccd)
            except KeyError:
                maskFieldList[field] = []
                maskFieldList[field].append(ccd)
    
    for image in imageList:
        imageBase = os.path.splitext(os.path.basename(image))[0]
        logging.debug("Image base: {0}".format(imageBase))

        # Extract ccd and fieldid from image filename
        ccd = imageBase.split("_")[-1]
        field = imageBase.split("_")[-2]
        
        logging.debug("Field: {0}, CCD: {1}".format(field,ccd))
        
        if "scie" in imageBase:
            try:
                thisCCDList = scieFieldList[field]
                if ccd in thisCCDList: continue
            except KeyError:
                scieFieldList[field] = []
        elif "mask" in imageBase:
            try:
                thisCCDList = maskFieldList[field]
                if ccd in thisCCDList: continue
            except KeyError:
                maskFieldList[field] = []
        
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
            f = StringIO.StringIO(urllib2.urlopen(request).read())
        except urllib2.HTTPError:
            continue
            
        try:
            gz = gzip.GzipFile(fileobj=f, mode="rb")
            gz.seek(0)
            
            fitsFile = StringIO.StringIO(gz.read())
        except IOError:
            fitsFile = f
        
        fitsFile.seek(0)
        
        hdulist = pf.open(fitsFile, mode="readonly")
        hdulist.writeto(file)
        
        if "scie" in imageBase:
            scieFieldList[field].append(ccd)
        elif "mask" in imageBase:
            maskFieldList[field].append(ccd)
    
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
    print request.get_full_url()
    
    f = StringIO.StringIO(urllib2.urlopen(request).read())
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