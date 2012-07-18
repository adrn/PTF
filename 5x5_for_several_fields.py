# coding: utf-8
from __future__ import division

""" This module will choose several fields, and produce csv files with columns for the 
    variability indices of all sources on the fields. These csv files will then be loaded
    into D3 for an interactive grid of plots like this: 
        http://mbostock.github.com/d3/talk/20111116/iris-splom.html
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import logging

# Third-party
import numpy as np

# Project
import candidate_pipeline
import ptf.photometricdatabase as pdb

CSV_PATH = os.path.join(os.getcwd(), "data", "js")
if not os.path.exists(CSV_PATH):
    os.mkdir(CSV_PATH)

field_ids = [2471]
fields = [pdb.PTFField(field_id) for field_id in field_ids]

for field in fields:
    for ccdid,ccd in field.ccds.items():
        ccd_file = ccd.read()
        var_statistics = ccd_file.sources.col(["stetsonJ", "stetsonK", "vonNeumannRatio", "medianAbsDev", "meanMag"])
        

