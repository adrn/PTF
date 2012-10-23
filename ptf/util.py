# coding: utf-8
""" General utilities for the PTF project """

# Standard library
import os, sys
import logging

# Convert my names for variability indices to the PDB names
pdb_index_name = dict(eta="vonNeumannRatio", \
                      j="stetsonJ", \
                      k="stetsonK", \
                      delta_chi_squared="chiSQ", \
                      sigma_mu=["magRMS","referenceMag"])

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

def convert_all_fields_txt(txt_filename="ptf_allfields.txt", npy_filename="all_fields.npy"):
    """ Convert the text file containing all PTF Field ID's and positons to a 
        numpy .npy save file.
        
        It is much faster to load the .npy file than to generate an array on 
        the fly from the text file. The text file can be found here:
            http://astro.caltech.edu/~eran/PTF_AllFields_ID.txt
    """
    all_fields = np.genfromtxt(txt_filename, \
                           skiprows=4, \
                           usecols=[0,1,2,5,6,7], \
                           dtype=[("id", int), ("ra", float), ("dec", float), ("gal_lon", float), ("gal_lat", float), ("Eb_v", float)]).\
                           view(np.recarray)

    np.save(npy_filename, all_fields)
    
    return True

def richards_qso(sdss_colors):
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
    