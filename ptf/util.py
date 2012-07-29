# coding: utf-8
""" General utilities for the PTF project """

# Standard library
import os, sys

# Third-party
import numpy as np
import matplotlib.pyplot as plt

# Project
from ptf.db.DatabaseConnection import *
import ptf.simulation.util as simu
from ptf import PTFLightCurve

parameter_to_label = {"j" : "J", "k" : "K", "sigma_mu" : r"$\sigma/\mu$", "eta" : r"$\eta$", "delta_chi_squared" : r"$\Delta \chi^2$"}

def txt_file_light_curve_to_recarray(filename):
    """ Load Marcel / Nick's text file light curves into a Numpy recarray.
    
        All columns in this file are:
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
    
    names = ["mjd", "mag", "mag_err", "ra", "dec", "flag", "fieldid", "ccd"]
    usecols = [0, 2, 3, 4, 5, 8, 10, 11]
    dtypes = [float, float, float, float, float, float, int, int]
    
    return np.genfromtxt(filename, usecols=usecols, dtype=zip(names, dtypes)).view(np.recarray)