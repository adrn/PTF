"""
    Extract the entire ptf_exp table from the PTF 
    Large Survey Database and save it into a pickle.
    
    ** This script should be run on navtara **

"""

# Standard library
import os, sys
import cPickle as pickle
from argparse import ArgumentParser

# Third-party dependencies 
import numpy as np
import lsd

if __name__ == "__main__":    
    parser = ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite the file if it exists (default = False)")
    parser.add_argument("-f", "--file", dest="file", default="exposureTable.pickle",
                    help="The path to the output pickle file")
    
    args = parser.parse_args()
                    
    db = lsd.DB("/scr4/bsesar")
    results = db.query("mjd, ptf_field, ccdid, fid, ra, dec, l, b, medfwhm, limitmag, mumax_med, mumax_rms FROM ptf_exp").fetch()
    exposureData = [tuple(row) for row in results]
    
    exposureDataArray = np.array(exposureData, dtype=[("mjd", float),\ 
                                                      ("fieldid", int), \
                                                      ("ccdid", int), \
                                                      ("filterid", int), \
                                                      ("ra", float), \
                                                      ("dec", float), \
                                                      ("l", float), \
                                                      ("b", float), \
                                                      ("medfwhm", float), \
                                                      ("mumax_med", float), \
                                                      ("mumax_rms", float)]).view(np.recarray)
    
    if os.path.exists(args.file) and not args.overwrite:
        raise IOError("File `{0}` already exists! Did you want to overwrite it? (use -o)".format(args.file))
    
    f = open(args.file, "w")
    pickle.dump(exposureDataArray, f)
    f.close()
    print "Finished! File located at: {0}".format(args.file)
    
else:
    raise ImportError("getExposureData.py should be run as a standalone script, not imported!")