# coding: utf-8

""" This module contains classes and functions used to access PTF imaging data. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import abc
import base64
import os
import pytest
import re
import urllib, urllib2
import cStringIO as StringIO

# Third-party
import pyfits as pf
try:
    from apwlib.globals import greenText, yellowText, redText
    import apwlib.geometry as g
except ImportError:
    raise ImportError("apwlib not found! \nDo: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install' to install.")

# These URLs might change in the future, but they provide HTTP-level access to
#   the PTF imaging data API at IPAC.
IPAC_DATA_URL = "http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process"
IPAC_SEARCH_URL = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process"
try:
    IPAC_USER = os.environ["IPAC_USER"]
    IPAC_PASSWORD = os.environ["IPAC_PASSWORD"]
except KeyError:
    raise KeyError("Environment has no IPAC_USER or IPAC_PASSWORD. You must set these environment variables before running this script!")
SEARCH_COLUMNS = ['expid', 'obsdate', 'crval1', 'crval2', 'filter', 'ccdid', 'ptffield', 'seeing', 'airmass', 'moonillf', 'moonesb', 'photcalflag', 'infobits', 'nid', 'fieldid', 'ptfpid', 'pfilename', 'afilename1', 'obsmjd', 'ptfprpi', 'filtersl', 'moonra', 'moondec', 'moonphas', 'moonalt', 'ra1', 'dec1', 'ra2', 'dec2', 'ra3', 'dec3', 'ra4', 'dec4', 'gain', 'readnoi', 'darkcur']

def retrieve_ipac_file(url):
    """ Given a full URL, open an HTTP request to the url and return a StringIO file object
        with the contents of the response.
        
        Parameters
        ----------
        url : str
            A full IPAC url, e.g. http://kanaloa.ipac...etc.../process?query=blah

    """
    
    request = urllib2.Request(url)
    
    # Encode the username and password to send for authorization
    base64string = base64.encodestring('%s:%s' % (IPAC_USER, IPAC_PASSWORD)).replace('\n', '')
    request.add_header("Authorization", "Basic %s" % base64string)
    
    # Retrieve the response
    try:
        response = urllib2.urlopen(request)
    except urllib2.HTTPError, e:
        print "HTTPError: Authorization failed or request invalid.\n\t->HTTP Response returned error code {}".format(e.code)
        raise
    except urllib2.URLError, e:
        print "URLError: {}".format(e.reason)
        raise
    
    file = StringIO.StringIO(response.read())
    return file

def parse_ipac_table(table_file):
    """ Given a text file output from an IPAC query, parse it and return a list
        of metadata objects (currently dictionaries)
    """
    file_lines = table_file.readlines()
    if len(file_lines) < 5:
        raise ValueError("No images found!")
    
    columns = file_lines[0].replace("|","   ").split()
    
    # Each row in the table starting at index 4 is metadata for a new image / observation
    metadatas = []
    for image_data in file_lines[4:]:
        line = image_data.replace("|","   ").split()
        obsdate_idx = columns.index("obsdate")
        tmp = line[obsdate_idx] + " " + line[obsdate_idx+1]
        del line[obsdate_idx+1]
        line[obsdate_idx] = tmp
        metadatas.append(dict(zip(columns, line)))
    
    return metadatas

def download_images_from_metadatas(metadatas):
    """ Given a list of metadata instances, download the 'pfilename' files
        and return PTFImage objects for each file
    """
    ptf_images = []
    for metadata in metadatas:
        url = os.path.join(IPAC_DATA_URL, metadata["pfilename"])
        ptf_images.append(PTFImage(retrieve_ipac_file(url), metadata=metadata))
        print greenText("Image {} downloaded.".format(os.path.basename(metadata["pfilename"])))
    
    return ptf_images

class PTFImage(object):
    
    def __init__(self, fits_file, metadata=None):
        self.fits = pf.open(fits_file)
        self.metadata = metadata
    
    def save(self, path, filename=None, overwrite=False):
        """ Given a path, save the PTFImage as a FITS file in that path """
        
        if filename is None and self.metadata is None:
            raise ValueError("If the image has no 'metadata', you must specify a filename")
        elif filename is not None:
            pass
        elif filename is None and self.metadata is not None:
            filename = os.path.basename(self.metadata["pfilename"])
        
        full_image_path = os.path.join(path, filename)
        
        if overwrite and os.path.exists(full_image_path):
            os.remove(full_image_path)
            
        self.fits.writeto(full_image_path)
    
    def show(self):
        import aplpy
        ff = aplpy.FITSFigure(self.fits)
        ff.show_grayscale()
        return ff

def ptf_images_from_fieldid(fieldid, ccds=[], filter="R", epoch=None, number=None):
    """ Creates PTF FITS Images given a PTF Field ID and optionally CCD IDs.
        
        Parameters
        ----------
        fieldid : int
            A PTF Field ID
        ccds : (optional) list
            A list of CCD IDs
        filter : (optional) str
            Select only observations of this filter.
        epoch : (optional) float
            The MJD of the observation of the image. If not specified, the
            image with the best seeing is returned.
        number : (optional) int
            Constrain the number of images to return.
    """
    
    # Construct search URL with parameters
    where_str = "?where=filter IN ('{}')".format(filter)
    
    where_str += " AND ptffield IN ({})".format(fieldid)
    
    if len(ccds) > 0:
        where_str += " AND ccdid IN ({})".format(",".join([str(x) for x in ccds]))
    
    if epoch is not None:
        where_str += " AND obsmjd IN ({})".format(epoch)
    
    table_file = retrieve_ipac_file(IPAC_SEARCH_URL + urllib.quote(where_str) + "&columns={}".format(",".join(SEARCH_COLUMNS)))
    metadatas = parse_ipac_table(table_file)
    if number is not None and len(metadatas) > number:
        metadatas = metadatas[:number]
    ptf_images = download_images_from_metadatas(metadatas)
    
    return ptf_images

def ptf_images_from_name(name, size, intersect="overlaps", filter="R", epoch=None, number=None):
    """ Creates PTF FITS Images given a name of an astronomical object that can be
        resolved by SIMBAD.
        
        Parameters
        ----------
        name : str
            The name of an astronomical object.
        size : apwlib.geometry.Angle, tuple
            An angular extent on the sky, or a tuple of 2 angular extents
            representing a size in RA and a size in Dec.
        intersect : str
            See IPAC image query documentation:
                http://kanaloa.ipac.caltech.edu/ibe/queries.html
        filter : (optional) str
            Select only observations of this filter.
        epoch : (optional) float
            The MJD of the observation of the image. If not specified, the
            image with the best seeing is returned.
        number : (optional) int
            Constrain the number of images to return.
    """
    
    url = "http://simbad.u-strasbg.fr/simbad/sim-id?Ident={}&NbIdent=1&Radius=2&Radius.unit=arcmin&submit=submit+id&output.format=ASCII_TAB".format(urllib.quote(name))
    request = urllib2.Request(url)
    response = urllib2.urlopen(request)
    
    pattr = re.compile("Coordinates\(ICRS\,ep\=J2000\,eq\=2000\)\: ([0-9\s\+\-\.]+)\(")
    try:
        radec_str = pattr.search(response.read()).groups()[0]
    except AttributeError:
        raise ValueError(redText("Unable to resolve name '{}'".format(name)))
    
    ra,dec = radec_str.strip().split("  ")
    ra = g.RA(ra)
    dec = g.Dec(dec)
    
    return ptf_images_from_position(ra, dec, size=size, intersect=intersect, filter=filter, epoch=epoch, number=number)

def ptf_images_from_position(ra, dec, size, intersect="covers", filter="R", epoch=None, number=None):
    """ Creates PTF FITS Images given an equatorial position (RA/Dec) 
        and a size.
        
        Parameters
        ----------
        ra : apwlib.geometry.RA or any type parsable by apwlib
            A right ascension.
        dec : apwlib.geometry.Dec or any type parsable by apwlib
            A declination.
        size : apwlib.geometry.Angle, tuple
            An angular extent on the sky, or a tuple of 2 angular extents
            representing a size in RA and a size in Dec.
        intersect : str
            See IPAC image query documentation:
                http://kanaloa.ipac.caltech.edu/ibe/queries.html
        filter : (optional) str
            Select only observations of this filter.
        epoch : (optional) float
            The MJD of the observation of the image. If not specified, the
            image with the best seeing is returned.
        number : (optional) int
            Constrain the number of images to return.
    """
        
    ra = g.RA(ra)
    dec = g.Dec(dec)
    
    if isinstance(size, tuple) and isinstance(size[0], g.Angle)  and isinstance(size[1], g.Angle):
        size_str = "SIZE={w.degrees},{h.degrees}".format(w=size[0], h=size[1])
    elif isinstance(size, g.Angle):
        size_str = "SIZE={0.degrees}".format(size)
    else:
        raise TypeError("'size' must be a tuple of apwlib.geometry.Angle objects, or a single Angle object.") 
    
    # Construct search URL with parameters
    pos_str = "POS={ra.degrees},{dec.degrees}".format(ra=ra, dec=dec)
    
    if number == 1:
        intersect_str = "INTERSECT=CENTER&mcen"
    else:
        intersect_str = "INTERSECT={}".format(intersect.upper())
    
    search_url_append = "?{}&{}&{}&where=filter IN ('{}')".format(pos_str, size_str, intersect_str, filter)
    
    if epoch is not None:
        search_url_append += " AND obsmjd IN ({})".format(epoch)
    
    table_file = retrieve_ipac_file(IPAC_SEARCH_URL + urllib.quote(search_url_append) + "&columns={}".format(",".join(SEARCH_COLUMNS)))
    
    metadatas = parse_ipac_table(table_file)
    
    num = 0
    ptf_images = []
    for metadata in metadatas:
        cutout_url = os.path.join(IPAC_DATA_URL, metadata["pfilename"])
        cutout_url_query = "?center={ra.degrees},{dec.degrees}&{size}&gzip=false".format(ra=ra, dec=dec, size=size_str.lower())
        
        try:
            ptf_images.append(PTFImage(retrieve_ipac_file(cutout_url + cutout_url_query), metadata=metadata))
            num += 1
            print greenText("Image {} downloaded.".format(os.path.basename(ptf_images[-1].metadata["pfilename"])))
        except urllib2.HTTPError:    
            print yellowText("Image failed to download:\n\t{}".format(cutout_url + cutout_url_query))
            
        if number is not None and num >= number:
            break
    
    return ptf_images

# ===============================================
# ==================== Tests ====================
# ===============================================

def test_retrieve_ipac_file():
    # An example url
    url = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS=215.1351,31.1241"
    table_file = retrieve_ipac_file(url + "&columns={}".format(",".join(SEARCH_COLUMNS)))
    
    url = "http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process/proc/2010/02/15/f2/c0/p13/v1/PTF_201002153960_i_p_scie_t093015_u011664659_f02_p110001_c00.fits"
    image_file = retrieve_ipac_file(url)

def test_ptfimage():
    # Get an image:
    image = ptf_images_from_fieldid(110001, ccds=[0], epoch=55242.39601)[0]
    
    # Test save()
    image.save("/tmp/", overwrite=True)
    image.save("/tmp/", "ptf_test_image.fits", overwrite=True)
    
    assert os.path.exists("/tmp/{}".format(os.path.basename(image.metadata["pfilename"])))
    assert os.path.exists("/tmp/ptf_test_image.fits")

def test_ptf_images_from_fieldid():
    fieldid = 110001
    ptf_images_from_fieldid(fieldid, epoch=55242.39601)

def test_ptf_images_from_name():
    names = ["m42"]
    size = g.Angle.fromDegrees(0.05)
    
    for name in names:
        # Test a small patch, contained on one CCD
        ptf_images_from_name(name=name, size=size)
    
def test_ptf_images_from_position():
    ra = g.RA("14:21:17.1341")
    dec = g.Dec("+15:16:51.6246")
    x_size = g.Angle.fromDegrees(0.05)
    y_size = g.Angle.fromDegrees(1.0)
    
    # Test a small patch, contained on one CCD
    ptf_images_from_position(ra=ra, dec=dec, size=x_size)
    
    # Test a large patch, image is over multiple CCDs and size is a tuple
    ptf_images_from_position(ra=ra, dec=dec, size=(x_size,y_size), intersect="overlaps")
    
    with pytest.raises(ValueError):
        ptf_images_from_position(ra=ra, dec=dec, size=x_size, filter="g")