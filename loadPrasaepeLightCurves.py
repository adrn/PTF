"""
    Load Marcel's Prasaepe light curves into my local Postgres database
    ** Table schema at bottom! **
"""

# Standard library
import os, sys, glob
from argparse import ArgumentParser
import logging

# Third-party
import sqlalchemy
import numpy as np
import apwlib.geometry as g
import apwlib.convert as c
import matplotlib.pyplot as plt

# Project
#from ptf.db.DatabaseConnection import *

def txtFileToRecarray(filename):
    """ All columns in this file are:
        MJD
        R1	<-- ignore (I think; doesn't matter much
	    R2	which of these two you use for this, I say)
        R2 err
        RA
        DEC
        X
        Y
        flag
        file
        field
        chip
        
        MJD's are relative to 2009-1-1 (54832)
    """
    
    names = ["mjd", "mag", "mag_err", "ra", "dec", "flags", "field", "ccd"]
    usecols = [0, 2, 3, 4, 5, 8, 10, 11]
    dtypes = [float, float, float, float, float, float, int, int]
    
    return np.genfromtxt(filename, usecols=usecols, dtype=zip(names, dtypes)).view(np.recarray)

if __name__ == "__main__":
    
    session.begin()
    logging.debug("Starting database load...")
    for file in glob.glob("data/merged_lc/*.txt"):
        lc = txtFileToRecarray(file)
        lc = lc[lc.flags == 0]
        
        filterid = [2] * len(lc.mjd)
        syserr = [0.] * len(lc.mjd)
        
        lightCurve = LightCurve()
        lightCurve.mjd = lc.mjd
        lightCurve.mag = lc.mag
        lightCurve.mag_err = lc.mag_err
        lightCurve.sys_err = syserr
        lightCurve.filter_id = filterid
        lightCurve.ra = lc.ra[0]
        lightCurve.dec = lc.dec[0]
        lightCurve.flags = [int(x) for x in lc.flags]
        lightCurve.field = lc.field
        lightCurve.ccd = lc.ccd
        session.add(lightCurve)
        
        if len(session.new) == 1000:
            session.commit()
            logging.debug("1000 light curves committed!")
            session.begin()
    
    logging.info("All light curves committed!")
    session.commit()
        
    
    