# coding: utf-8

""" This module contains classes and functions used to access PTF imaging data. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import abc
import base64
import os
import pytest
import urllib, urllib2
import cStringIO as StringIO

# Third-party
try:
    from apwlib.globals import greenText, yellowText, redText
    import apwlib.geometry as g
except ImportError:
    raise ImportError("apwlib not found! \nDo: 'git clone git@github.com:adrn/apwlib.git' and run 'python setup.py install' to install.")

# Project


# These URLs might change in the future, but they provide HTTP-level access to
#   the PTF imaging data API at IPAC.
IPAC_DATA_URL = "http://kanaloa.ipac.caltech.edu/ibe/data/ptf/dev/process"
IPAC_SEARCH_URL = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process"
try:
    IPAC_USER = os.environ["IPAC_USER"]
    IPAC_PASSWORD = os.environ["IPAC_PASSWORD"]
except KeyError:
    raise KeyError("Environment has no IPAC_USER or IPAC_PASSWORD. You must set these environment variables before running this script!")

def send_ipac_search_request(url):
    """ Given a full URL, open an HTTP request to the url and expect a table to be returned 
        
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

def test_send_ipac_search_request():
    # An example url
    url = "http://kanaloa.ipac.caltech.edu/ibe/search/ptf/dev/process?POS=215.1351,31.1241"
    table_file = send_ipac_search_request(url)

class PTFImage(object):
    
    def __init__(self, fits_file, mask_file=None):
        pass
    
    def from_metadata(metadata):
        """ Initialize a PTFImage object from image metadata """
        fits_filename = metadata["pfilename"]
        mask_filename = metadata["afilename1"]
        
        # Retrieve FITS files
        print IPAC_DATA_URL, fits_filename
        
        img = PTFImage(fits_file, mask_file)
        img.metadata = metadata
        return img

def test_ptfimage():
    # Test that we can't create an instance of the abstract class PTFImage
    with pytest.raises(TypeError):
        image = PTFImage()

class PTFFITSImage(PTFImage):
    
    def __init__(self):
        raise NotImplementedError()

class TestPTFFITSImage:
    ra = g.RA("14:21:17.1341")
    dec = g.Dec("+15:16:51.6246")
    x_size = g.Angle.fromDegrees(0.2)
    y_size = g.Angle.fromDegrees(0.1)
    
    def test_failures(self):
        with pytest.raises(TypeError):
            # Test failure when passing in float 'size'
            ptf_image = PTFFITSImage.from_position(ra=self.ra, dec=self.dec, size=0.1512)
            
    def test_api(self):
        # Test extracting square image
        ptf_image = PTFFITSImage.from_position(ra=self.ra, dec=self.dec, size=self.x_size)
        
        # Test creating rectangular image
        ptf_image = PTFFITSImage.from_position(ra=self.ra, dec=self.dec, size=(self.x_size,self.y_size))
        
        
""" def from_position(ra, dec, size, filter="R", epoch=None):
        return
    
    def from_name(name, size, filter="R", epoch=None):
        return
    
    def from_fieldid(fieldid, ccds, filter="R", epoch=None):
        return
"""
def ptf_images_from_position(ra, dec, size, filter="R", epoch=None):
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
        filter : (optional) str
            Select only observations of this filter.
        epoch : (optional) float
            The MJD of the observation of the image. If not specified, the
            image with the best seeing is returned.
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
    search_url_append =  "?{}&{}&where=filter IN ('{}')".format(pos_str, size_str, filter)
    
    if epoch is not None:
        search_url_append += " AND obsmjd IN ({})".format(epoch)
    
    table_file = send_ipac_search_request(IPAC_SEARCH_URL + urllib.quote(search_url_append))
    
    file_lines = table_file.readlines()
    if len(file_lines) < 5:
        raise ValueError("No images found!")
    
    columns = file_lines[0].replace("|","   ").split()
    
    # Each row in the table starting at index 4 is metadata for a new image / observation
    metadatas = []
    for image_data in file_lines[4:]:
        line = image_data.replace("|","   ").split()
        tmp = line[4] + " " + line[5]
        del line[5]
        line[4] = tmp
        metadatas.append(dict(zip(columns, line)))
    
    return [PTFFITSImage.from_metadata(metadata) for metadata in metadatas]
    
def test_ptf_images_from_position():
    ra = g.RA("14:21:17.1341")
    dec = g.Dec("+15:16:51.6246")
    x_size = g.Angle.fromDegrees(0.2)
    y_size = g.Angle.fromDegrees(0.1)
    
    ptf_images_from_position(ra=ra, dec=dec, size=x_size)
    
    with pytest.raises(ValueError):
        ptf_images_from_position(ra=ra, dec=dec, size=x_size, filter="g")
    
    return PTFFITSImage()