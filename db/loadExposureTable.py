""" 
    Take PTF Exposure information from a pickle and load it
    into the `ptf_microlensing` database.
    
    ** SQL TO CREATE TABLE IS AT BOTTOM **
    
"""

import os, sys
import cPickle as pickle
import logging

import numpy as np
from DatabaseConnection import *
from NumpyAdaptors import *

def loadExposureData(filename):
    """ Given the filename to a .pickle file containing
        exposure table data from the PTF LSD database, 
        load it into the local ptf_microlensing Database
    """
    if not os.path.exists(filename) or len(filename.strip()) == 0:
        raise ValueError("Error: You must specify the path to the Exposure data (as a pickle).\n\t e.g. -f /path/to/exposure.pickle")
    
    f = open(filename)
    exposureData = pickle.load(f)
    f.close()
    
    


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite all data in this table (default = False)")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_argument("-f", "--file", dest="file", default="",
                    help="The path to the .pickle file that contains the Exposure information")
                    
    args = parser.parse_args()
    if args.verbose: logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    else: logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    if args.overwrite:
        session.query(Exposure).delete()
    
    loadExposureData(args.file)
    
    

""" SQL TO CREATE TABLE:

CREATE TABLE exposure
(
  pk serial NOT NULL,
  mjd double precision NOT NULL,
  field integer NOT NULL,
  ccdid smallint NOT NULL,
  filterid smallint NOT NULL,
  ra double precision NOT NULL,
  dec double precision NOT NULL,
  l double precision NOT NULL,
  b double precision NOT NULL,
  medfwhm double precision NOT NULL,
  limitmag double precision NOT NULL,
  mumax_med double precision NOT NULL,
  mumax_rms double precision NOT NULL,
  CONSTRAINT exposure_pk PRIMARY KEY (pk)
)
WITH (
  OIDS=FALSE
);
ALTER TABLE exposure OWNER TO adrian;

CREATE INDEX exposure_field_idx
  ON exposure
  USING btree
  (field);
  
CREATE INDEX exposure_ccdid_idx
  ON exposure
  USING btree
  (ccdid);
  
"""