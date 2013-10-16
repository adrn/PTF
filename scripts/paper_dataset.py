# coding: utf-8
from __future__ import division

""" Select 100000 random light curves from PTF to publish. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import random
import cPickle as pickle

# Third-party
import numpy as np

# Project
import ptf.db.photometric_database as pdb
from ptf.globals import all_fields, data_path

np.random.seed(42)
Nfields = len(all_fields)
Nlightcurves = 100000
Nselected = 0

ccd_ids = range(13)
ccd_ids.pop(3)

field_ccd = dict()
while Nselected < Nlightcurves:
    field_row = all_fields[np.random.randint(Nfields)]
    field_id = field_row['id']
    
    field = pdb.Field(field_id, 'R')
    if len(field.ccds) == 0:
        continue

    ccd = random.choice(field.ccds.values())

    try:
        field_ccd[field.id]
    except KeyError:
        field_ccd[field.id] = dict()

    try:
        field_ccd[field_id][ccd.id]
    except KeyError:
        field_ccd[field_id][ccd.id] = 0

    field_ccd[field.id][ccd.id] += 1
    Nselected += 1

with open(os.path.join(data_path, "publish_field_ccds.pickle"), 'w') as f:
    pickle.dump(field_ccd, f)
