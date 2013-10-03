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

np.random.seed(42)

Ntrials = 100
Nsources = 100
features = ['con', 'vonNeumannRatio', 'stetsonJ', 'stetsonK']
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

sourceIDs = sources.readWhere("ngoodobs > 10")["matchedSourceID"]
random_sourceIDs = sourceIDs[np.random.randint(len(sourceIDs), size=Nsources)]

training_data = np.zeros((Ntrials*Nsources,Nfeatures))
for sourceID in random_sourceIDs:
    d = sourceData.readWhere("matchedSourceID == {0}".format(sourceID))
    
    for trial in range(Ntrials):
        
        print(d.dtype.names)
        break
    break

chip.close()
