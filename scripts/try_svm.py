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
from ptf.lightcurve import SimulatedLightCurve
from ptf.analyze import compute_variability_indices
from ptf.util import pdb_index_name
pdb_index_name['con'] = 'con'

# Create logger
logger = logging.getLogger(__name__)

lcBaseDir = '/scr2/ptf/variable/matches/'
indexFile = os.path.join(lcBaseDir, 'index.npy')
index = np.load(indexFile)

np.random.seed(42)

Ntrials = 100
Nsources = 1000
features = ['con', 'eta', 'j', 'k', 'sigma_mu']
Nfeatures = len(features)

filterID = 2
fieldID = 110001
chipID = 1

matchFile = os.path.join(lcBaseDir, 'match_{:02d}_{:06d}_{:02d}.pytable'\
                          .format(filterID, fieldID, chipID))
chip = tables.openFile(matchFile)
sourceData = chip.getNode('/filter{:02d}/field{:06d}/chip{:02d}/sourcedata'\
                         .format(filterID, fieldID, chipID))
sources = chip.getNode('/filter{:02d}/field{:06d}/chip{:02d}/sources'\
                       .format(filterID, fieldID, chipID))

goodSources = sources.readWhere("ngoodobs > 10")
sourceIDs = goodSources["matchedSourceID"]
all_data = np.zeros((len(sourceIDs),Nfeatures))

for ii,f in enumerate(features):
    if f == "sigma_mu":
        pdb_f1,pdb_f2 = pdb_index_name[f]
        all_data[:,ii] = goodSources[pdb_f1] / goodSources[pdb_f2]
    else:
        pdb_f = pdb_index_name[f]
        all_data[:,ii] = goodSources[pdb_f]

np.save("/home/aprice-whelan/tmp/all_data.npy", all_data)

random_sourceIDs = sourceIDs[np.random.randint(len(sourceIDs), size=Nsources)]

training_data = np.zeros((Nsources,Ntrials,Nfeatures))
for ii,sourceID in enumerate(random_sourceIDs):
    d = sourceData.readWhere("matchedSourceID == {0}".format(sourceID))
    mjd = d['mjd']
    mag = d['mag']
    err = d['magErr']

    for trial in range(Ntrials):
        lc = SimulatedLightCurve(mjd=mjd, mag=mag, error=err)
        lc.add_microlensing_event()
        stats = compute_variability_indices(lc, indices=features)
        training_data[ii, trial, :] = np.array([stats[x] for x in features])

np.save("/home/aprice-whelan/tmp/training_data.npy", training_data)

chip.close()
