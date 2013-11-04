# coding: utf-8
from __future__ import division

""" Select 100000 random light curves from PTF to publish. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import random
import cPickle as pickle
from collections import defaultdict

# Third-party
import numpy as np
from astropy.io.misc import fnunpickle
from astropy.io import fits

# Project
import ptf.db.photometric_database as pdb
from ptf.globals import all_fields, data_path

np.random.seed(42)

cache = defaultdict(dict)
def get_lcs(Nlightcurves=1000, seed=42):
    Nfields = len(all_fields)

    ccd_ids = range(13)
    ccd_ids.pop(3)

    np.random.seed(seed)
    lcs = []
    while len(lcs) < Nlightcurves:
        field_row = all_fields[np.random.randint(Nfields)]
        field_id = field_row['id']

        field = pdb.Field(field_id, 'R')

        if len(field.ccds) == 0:
            continue

        ccd = random.choice(field.ccds.values())
        if not cache[field_id].has_key(ccd.id):
            chip = ccd.read()
            sources = chip.sources.readWhere("ngoodobs > 10")
            cache[field_id][ccd.id] = sources

        np.random.shuffle(cache[field_id][ccd.id])

        for source in cache[field_id][ccd.id]:
            lc = ccd.light_curve(source["matchedSourceID"], clean=True, barebones=True)
            if len(lc) > 10:
                lcs.append(lc)
                break

    return lcs

def store_light_curves(lcs):
    data = None
    for lc in lcs:
        n = len(lc.mjd)
        f = [lc.field_id]*n
        c = [lc.ccd_id]*n
        s = [lc.source_id]*n

        if data is None:
            data = np.array([f, c, s, lc.mjd, lc.mag, lc.error])
        else:
            data = np.hstack((data, np.array([f, c, s, lc.mjd, lc.mag, lc.error])))
    
    dtype = [("field", int), ("ccd", int), ("id", int), \
             ("mjd", float), ("R_mag", float), ("mag_err", float)]
    
    return np.array(data, dtype=dtype).view(np.recarray)

if __name__ == "__main__":       
    path = "/home/aprice-whelan/new/ptf/data/paper_dataset"

    for seed in range(10, 25):
        fn = os.path.join(path, "lc_{0}.fits".format(seed))
        if os.path.exists(fn): continue
        lcs = get_lcs(100, seed=seed)
        print(seed, len(lcs))
        data = store_light_curves(lcs)
        hdu = fits.BinTableHDU(data)
        hdu.writeto(fn)

        del data, lcs
