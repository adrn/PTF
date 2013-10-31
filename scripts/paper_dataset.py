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
from astropy.io.misc import fnunpickle

# Project
import ptf.db.photometric_database as pdb
from ptf.globals import all_fields, data_path

np.random.seed(42)
dict_file = os.path.join(data_path, "publish_field_ccds.pickle")

def make_lookup_dict(Nlightcurves=10000):
    Nfields = len(all_fields)
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

    with open(dict_file, 'w') as f:
        pickle.dump(field_ccd, f)
   
def random_light_curves(ccd, N):
    chip = ccd.read()
    sources = chip.sources.readWhere("ngoodobs > 10")
    np.random.shuffle(sources)

    lcs = []
    for source in sources:
        lc = ccd.light_curve(source["matchedSourceID"], clean=True, barebones=True)
        if len(lc) > 10:
            lcs.append(lc)

        if len(lcs) >= N:
            return lcs

    return []

def store_light_curves():
    field_ccd = fnunpickle(dict_file)

    all_lightcurves = []
    for ii,field_id in enumerate(sorted(field_ccd.keys())):
        this_file = os.path.join(data_path, "publish_light_curves", "{0}.fits".format(ii % 100))
        if os.path.exists(this_file):
            continue

        field = pdb.Field(field_id, 'R')

        for ccd_id in sorted(field_ccd[field_id].keys()):
            num = field_ccd[field_id][ccd_id]
            ccd = field.ccds[ccd_id]

            lcs = random_light_curves(ccd, num)
            all_lightcurves += lcs

        if ii > 0 and (ii % 100) == 0:
            #write to file

            data = None
            for lc in all_lightcurves:
                n = len(lc.mjd)
                f = [lc.field_id]*n
                c = [lc.ccd_id]*n
                s = [lc.source_id]*n
                
                if data is None:
                    data = np.array([f, c, s, lc.mjd, lc.mag, lc.error])
                else:
                    data = np.hstack((data, np.array([f, c, s, lc.mjd, lc.mag, lc.error])))

            print(data.shape)
            sys.exit(0)

            all_lightcurves = []


if __name__ == "__main__":
    if not os.path.exists(dict_file):
        make_lookup_dict(Nlightcurves=10000)

    store_light_curves()
