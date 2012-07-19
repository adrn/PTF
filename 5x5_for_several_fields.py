# coding: utf-8
from __future__ import division

""" This module will choose several fields, and produce csv files with columns for the 
    variability indices of all sources on the fields. These csv files will then be loaded
    into D3 for an interactive grid of plots like this: 
        http://mbostock.github.com/d3/talk/20111116/iris-splom.html
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging

# Third-party
import numpy as np

# Project
import candidate_pipeline
import ptf.photometricdatabase as pdb

# Randomly sample this many points from the total number ~100,000
SAMPLE_SIZE = 2000

CSV_PATH = os.path.join(os.getcwd(), "data", "js")
if not os.path.exists(CSV_PATH):
    os.mkdir(CSV_PATH)

field_ids = [100400, 100038, 3696]
fields = [pdb.PTFField(field_id) for field_id in field_ids]

for field in fields:
    filename = "field{:06d}.csv".format(field.id)
    if os.path.exists(os.path.join(CSV_PATH,filename)): continue
    
    j,k,eta,sigma_mu = [],[],[],[]
    
    for ccdid,ccd in field.ccds.items():
        ccd_file = ccd.read()
        
        subset = ccd_file.sources.readWhere("vonNeumannRatio > 0.0")
        w = np.isfinite(subset["stetsonJ"]) & np.isfinite(subset["stetsonK"]) & np.isfinite(subset["vonNeumannRatio"])
        
        j += list(subset["stetsonJ"][w])
        k += list(subset["stetsonK"][w])
        eta += list(subset["vonNeumannRatio"][w])
        sigma_mu += list(subset["medianAbsDev"][w] / subset["meanMag"][w])
    
    j = np.ravel(j)
    k = np.ravel(k)
    eta = np.ravel(eta)
    sigma_mu = np.ravel(sigma_mu)
    
    subset_idx = np.random.randint(0, len(j), size=SAMPLE_SIZE)
    species_arr = np.zeros(5, dtype=[('species',"|S1"),('j',float),('k',float),('eta',float),('sigma_mu',float)])
    species_arr["species"] = np.array(["d"]*SAMPLE_SIZE)
    species_arr["j"] = j[subset_idx]
    species_arr["k"] = k[subset_idx]
    species_arr["eta"] = eta[subset_idx]
    species_arr["sigma_mu"] = sigma_mu[subset_idx]
    
    header = "species,j,k,eta,sigma_mu"
    np.savetxt(os.path.join(CSV_PATH,filename), species_arr, fmt="%s %f %f %f %f", delimiter=",", header=header, comments="")
