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

def get_lcs(field, N_per_field=5, seed=42):
    np.random.random(seed)

    cache = dict()
    lcs = []
    for ii in range(N_per_field):
        ccd = random.choice(field.ccds.values)

        if not cache.has_key(ccd.id):
            chip = ccd.read()
            sources = chip.sources.readWhere("ngoodobs > 10")
            cache[ccd.id] = sources

        sources = cache[ccd.id]
        np.random.shuffle(sources)

        for source in sources:
            lc = ccd.light_curve(source["matchedSourceID"], clean=True,
                                 barebones=True)
            if len(lc) > 10:
                lcs.append(lc)
                break

    field.close()

    return lcs

def lcsToArray(lcs):
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

    all_fields = all_fields[all_fields["dec"] > -30.]
    path = "/home/aprice-whelan/new/ptf/data/paper_dataset"

    ii = 0
    subdiv = 10
    Nfields = len(all_fields)
    field_subset = all_fields[ii*Nfields//subdiv:ii*Nfields//subdiv+Nfields//subdiv]

    fn = os.path.join(path, "{0}.fits".format(ii))

    for f in field_subset:
        field = pdb.Field(f["id"], "R")
        if len(field.ccds) == 0:
            continue

        lcs = get_lcs
        arr = lcsToArray(lcs)

        try:
            np.hstack(data, arr)
        except NameError:
            data = arr

        if len(data) >= 1000:
            hdu = fits.BinTableHDU(data)
            hdu.writeto(fn)
            del data

