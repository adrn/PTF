# coding: utf-8
""" General utilities for the PTF project """

# Standard library
import os, sys

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
