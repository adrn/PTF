# coding: utf-8
""" General utilities for the PTF project """

# Standard library
import os, sys
import re
import logging

# Third-party
import numpy as np

# Convert my names for variability indices to the PDB names
pdb_index_name = dict(eta="vonNeumannRatio", \
                      j="stetsonJ", \
                      k="stetsonK", \
                      delta_chi_squared="chiSQ", \
                      sigma_mu=["magRMS","referenceMag"])

# This code will find the root directory of the project
_pattr = re.compile("(.*)\/ptf")
try:
    matched_path = _pattr.search(os.getcwd()).groups()[0]
except AttributeError: # match not found, try __file__ instead
    matched_path = _pattr.search(__file__).groups()[0]

if os.path.basename(matched_path) == "ptf":
    project_root = matched_path
else:
    project_root = os.path.join(matched_path, "ptf")

def source_index_name_to_pdb_index(source, index_name):
    """ Given a source (a row from chip.sources) and an index name (e.g. eta),
        return the value of the statistic. This is particularly needed for a
        computed index like sigma_mu.
    """
    if index_name == "sigma_mu":
        return source[pdb_index_name[index_name][0]] / source[pdb_index_name[index_name][1]]
    else:
        return source[pdb_index_name[index_name]]

def index_to_label(index):
    # Convert variability index to a Latex label
    mapping = {"j" : "J", \
               "k" : "K", \
               "sigma_mu" : r"$\sigma/\mu$", \
               "eta" : r"$\eta$", \
               "delta_chi_squared" : r"$\Delta \chi^2$", \
               "con" : "Con", \
               "corr" : "Corr"}
    return mapping[index]

# Create logger for this module
logger = logging.getLogger("ptf")
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_logger(name):
    return logger

def cache_all_fields(txt_filename="PTF_AllFields_ID.txt", 
                     npy_filename="all_fields.npy",
                     path="data/cache", overwrite=False):
    """ Convert the text file containing all PTF Field ID's and 
        positons to a numpy .npy save file.

        It is much faster to load the .npy file than to generate an array on
        the fly from the text file. The text file can be found here:
            http://astro.caltech.edu/~eran/PTF_AllFields_ID.txt
    """
    txt_filename = os.path.join(project_root, path, txt_filename)
    npy_filename = os.path.join(project_root, path, npy_filename)

    if not os.path.exists(txt_filename):
        url = "http://www.astro.caltech.edu/~eran/PTF_AllFields_ID.txt"
        logger.error("Download the txt file from: {0} and save to {1}."
                     .format(url, txt_filename))

    if os.path.exists(npy_filename) and overwrite:
        os.remove(npy_filename)

    if not os.path.exists(npy_filename):
        all_fields = np.genfromtxt(txt_filename, \
                                   skiprows=4, \
                                   usecols=[0,1,2,5,6,7], \
                                   dtype=[("id", int), \
                                          ("ra", float), ("dec", float),\
                                          ("gal_lon", float), ("gal_lat", float),\
                                          ("Eb_v", float)]).view(np.recarray)

        np.save(npy_filename, all_fields)

def richards_qso(sdss_colors):
    if sdss_colors == None:
        return False

    if (sdss_colors["g"]-sdss_colors["r"]) > -0.2 and \
       (sdss_colors["g"]-sdss_colors["r"]) < 0.9 and \
       (sdss_colors["r"]-sdss_colors["i"]) > -0.2 and \
       (sdss_colors["r"]-sdss_colors["i"]) > 0.6 and \
       (sdss_colors["i"]-sdss_colors["z"]) > -0.15 and \
       (sdss_colors["i"]-sdss_colors["z"]) > 0.5 and \
       17 < sdss_colors["i"] < 19.1:
        return True
    else:
        return False
