# coding: utf-8

""" ...explain... """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import tables
import numpy as np
import astropy.units as u

# Project
from ptf.globals import all_fields

# Create logger
logger = logging.getLogger(__name__)

lcBaseDir = '/scr2/ptf/variable/matches/'
indexFile = os.path.join(lcBaseDir, 'index.npy')
index = np.load(indexFile)

np.random.seed(142)
idx = np.random.randint(len(all_fields))

sys.exit(0)
filterID = 2
fieldID = all_fields[idx][0]
chipID = 1

matchFile = os.path.join(lcBaseDir, 'match_{:02d}_{:06d}_{:02d}.pytable'\
                          .format(filterID, fieldID, chipID))
chip = tables.openFile(matchFile)

sourceTable = chip.getNode('/filter{:02d}/field{:06d}/chip{:02d}/sources'\
                           .format(filterID, fieldID, chipID))

print(len(sourceTable))