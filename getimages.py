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

# Project-specific
import ptf.util as pu

def download_one_field(path, field, ccd=None, filter=None, mask=False, mjd=None, best=False):
    """ Given a PTF Field ID, download the image for that field.
        
        Parameters
        ----------
        path : str
            Path to save the downloaded images to
        field : int
            The PTF Field ID
        ccd : (optional) int, list
            Only download one or a list of CCD's
        filter : (optional) str
            Specify only observations of a given filter
        mask : (optional) bool
            Also download the mask files.
        best : (optional) bool
            If True, only select the best images (lowest seeing)
    
    """
    image_query = pu.PTFImageQuery()
    
    if not field == int(field):
        raise ValueError("Invalid field id!")
    
    logging.debug("Field: {}".format(field))
    image_query.field(field)
    
    if ccd:
        if isinstance(ccd, int):
            ccd = [ccd]
        elif isinstance(ccd, list):
            ccd = map(int, ccd)
        else:
            raise ValueError("Invalid CCD ID! Must be a number (e.g. 4) or a list (e.g. [0, 3, 11])")
        
        ccds = list(set(ccd).intersection(list(range(12))))
        logging.debug("CCDs: {}".format(",".join(map(str,ccds))))
        image_query.ccds(ccds)
    
    if filter: 
        image_query.filter(filter.upper())
    
    if mjd and best:
        raise ValueError("You can't specify an MJD and best=True")
    
    if mjd:
        image_query.on_mjd(mjd)
    
    print image_query.url
    image_list = pu.PTFImageList.fromImageQuery(image_query)
    
    if best:
        best_mjd = image_list.best_seeing_images().mjd[0]
        image_query.on_mjd(best_mjd)
        image_list = pu.PTFImageList.fromImageQuery(image_query)
    
    pu.getAllImages(image_list, prefix=path)
    
    return 0
        

##########################
# DON'T use the code below
##########################
def singleField(verbosity):
    """ ??????? """
    try:
        f = open("data/denseCoordinates.pickle")
        denseCoordinates = pickle.load(f)
        f.close()
    except IOError:
        raise IOError("data/denseCoordinates.pickle doesn't exist!\n Did you 'git pull' from navtara?")
    
    logger = dbu.getLogger(verbosity, name="getDenseFieldImages")
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
    
    logger = dbu.getLogger(verbosity, name="getDenseFieldImages")
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
    parser.add_argument("-f", "--field", dest="field", type=int, default=None, required=True,
                    help="Get a single image of the specified field")
    parser.add_argument("-m", "--mjd", dest="mjd", default=None, type=float,
                    help="Specify the epoch of the field image to download.")
    parser.add_argument("--ccd", nargs="*", dest="ccd", default=list(range(12)),
                    help="Specify a ccd or list of ccds to download from this field")
    parser.add_argument("--filter", dest="filter", default="R",
                    help="Specify a filter (R or g)")
    parser.add_argument("--mask", dest="mask", action="store_true", default=False,
                    help="Also download the mask file for that image")
    parser.add_argument("--best", dest="best", action="store_true", default=False,
                    help="Only download the best observation")
    parser.add_argument("--path", dest="path", required=True,
                    help="Path to save the images")

    
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if args.field:
        download_one_field(args.path, args.field, ccd=args.ccd, filter=args.filter, mask=args.mask, mjd=args.mjd, best=args.best)
