# coding: utf-8
from __future__ import division, print_function

""" TODO: docstring """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import sys

# Third-party
import numpy as np

# Project
import ptf.globals as pg
import ptf.util as pu
logger = pu.get_logger("num_simulations.py")
import ptf.db.photometric_database as pdb
import ptf.variability_indices as vi
from ptf.simulation import simulate_light_curves_compute_indices

def read_light_curves_from_field(field, ccd=None, N_per_ccd=0, clean=True, randomize=False):
    """ Read some number of light curves from a given PTF field

        Parameters
        ----------
        field : int, pdb.Field
            The PTF field
        ccd : int, pdb.CCD (optional)
            The PTF CCD to grab light curves from. If None, uses all CCDs
        N_per_ccd : int
            The number of light curves to grab PER CCD. 0 means all.
        clean : bool
            Clean the light curves and require > min_number_good_observations.
        randomize : bool
            Grab random light curves vs. consecutive light curves.

    """

    field = pdb.Field(field, "R")
    if ccd != None:
        ccds = [pdb.CCD(ccd, field, field.filter)]
    else:
        ccds = field.ccds.values()

    all_light_curves = []
    for ccd in ccds:
        ccd_light_curves = []
        chip = ccd.read()

        sources = chip.sources.readWhere("(ngoodobs > {})".format(pg.min_number_of_good_observations))
        if randomize:
            np.random.shuffle(sources)

        for source in sources:
            light_curve = ccd.light_curve(source["matchedSourceID"], barebones=True, clean=True) # clean applies a quality cut to the data
            light_curve.db_indices = [pu.source_index_name_to_pdb_index(source,index) for index in indices]

            if light_curve == None or len(light_curve) < pg.min_number_of_good_observations:
                # If the light curve is not found, or has too few observations, skip this source_id
                continue

            ccd_light_curves.append(light_curve)

            if N_per_ccd != 0 and len(ccd_light_curves) >= N_per_ccd: break
        ccd.close()

        all_light_curves += ccd_light_curves

    return all_light_curves

# Configuration parameters
indices = ["eta"]
field = pdb.Field(3376, "R")
num_trials = 25
N_per_ccd = 1000

logfile = open(os.path.join(os.path.split(pg._base_path)[0], "tests", "num_simulations_{0}.log".format(field.id)), "w")

# Start the test
all_light_curves = read_light_curves_from_field(field, N_per_ccd=N_per_ccd)
print("Read in {0} light curves from Field {1} on {2} CCDs".format(len(all_light_curves), field.id, len(field.ccds)), file=logfile)

all_eta_lower_criteria = dict()
for subsample_size in [10, 100, 1000, 10000]:
    all_eta_lower_criteria[subsample_size] = dict()
    for number_of_simulations_per_light_curve in [10, 100]:
        all_eta_lower_criteria[subsample_size][number_of_simulations_per_light_curve] = []
        for trial in range(num_trials):
            print("Trial: {}, subsample size: {}".format(trial, subsample_size))
            lc_idx = np.random.randint(len(all_light_curves), size=subsample_size)
            light_curves = [all_light_curves[idx] for idx in lc_idx]
            simulated_statistics = simulate_light_curves_compute_indices(light_curves, \
                                                                         num=number_of_simulations_per_light_curve, \
                                                                         indices=indices)

            var_indices = {"db" : np.array([tuple(lc.db_indices) for lc in light_curves], dtype=[(index,float) for index in indices]), \
                           "simulated" : simulated_statistics}

            selection_criteria = vi.compute_selection_criteria(var_indices, indices=indices, fpr=0.01)
            all_eta_lower_criteria[subsample_size][number_of_simulations_per_light_curve].append(selection_criteria["eta"]["lower"])

print("\n\n\n", file=logfile)
for subsample_size in [10, 100, 1000, 10000]:
    for number_of_simulations_per_light_curve in [10, 100]:
        log_etas = all_eta_lower_criteria[subsample_size][number_of_simulations_per_light_curve]
        print("Num. light curves: {0}, Num. simulations: {1}, σ: {2}, μ: {3}".format(subsample_size, number_of_simulations_per_light_curve, np.mean(log_etas), np.std(log_etas)), file=logfile)

logfile.close()