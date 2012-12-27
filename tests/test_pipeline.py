# coding: utf-8
from __future__ import division

""" Tests for scripts/candidate_pipeline.py """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import logging
import warnings
warnings.simplefilter("ignore")

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from apwlib.globals import greenText

# Project
import ptf.db.photometric_database as pdb
import ptf.db.mongodb as mongo
import ptf.analyze as pa
import ptf.variability_indices as vi
from ptf.globals import min_number_of_good_observations
from ptf.util import get_logger, source_index_name_to_pdb_index, richards_qso
logger = get_logger(__name__)

def plot_lc(lc):
    plt.clf()
    ax = lc.plot()

    try:
        x = np.linspace(lc.mjd.min(),lc.mjd.max(),1000)
        mag = pa.microlensing_model(lc.features, x)
        ax.plot(x, mag, 'r-')
    except KeyError:
        logger.debug("Not plotting microlensing fit.")
        pass

    plt.savefig("plots/tests/{0}_{1}_{2}.png".format(lc.field_id,lc.ccd_id, lc.source_id))

def test_slice_peak():
    ''' Test the slicing around peak '''
    light_curve = pdb.get_light_curve(100024, 11, 2693, clean=True) # bad data

    ml_chisq = 1E6
    ml_params = None
    for ii in range(10):
        params = pa.fit_microlensing_event(light_curve)
        new_chisq = params["result"].chisqr

        if new_chisq < ml_chisq:
            ml_chisq = new_chisq
            ml_params = params

    import matplotlib.pyplot as plt

    for ii in [1,2,3]:
        ax = plt.subplot(3,1,ii)
        sliced_lc = light_curve.slice_mjd(ml_params["t0"].value-ii*ml_params["tE"].value, ml_params["t0"].value+ii*ml_params["tE"].value)
        sliced_lc.plot(ax)
    plt.savefig("plots/test_slice_peak_bad_data.png")

    # Now do with a simulated event
    from ptf.lightcurve import SimulatedLightCurve
    light_curve = SimulatedLightCurve(mjd=light_curve.mjd, mag=15, error=0.1)
    light_curve.add_microlensing_event(u0=0.1)

    ml_chisq = 1E6
    ml_params = None
    for ii in range(10):
        params = pa.fit_microlensing_event(light_curve)
        new_chisq = params["result"].chisqr

        if new_chisq < ml_chisq:
            ml_chisq = new_chisq
            ml_params = params

    plt.clf()
    for ii in [1,2,3]:
        ax = plt.subplot(3,1,ii)
        sliced_lc = light_curve.slice_mjd(ml_params["t0"].value-ii*ml_params["tE"].value, ml_params["t0"].value+ii*ml_params["tE"].value)
        sliced_lc.plot(ax)
    plt.savefig("plots/test_slice_peak_sim.png")

def test_slice_peak2():
    ''' Test the slicing around peak '''
    light_curve = pdb.get_light_curve(100024, 11, 2693, clean=True) # bad data

    ml_chisq = 1E6
    ml_params = None
    for ii in range(10):
        params = pa.fit_microlensing_event(light_curve)
        new_chisq = params["result"].chisqr

        if new_chisq < ml_chisq:
            ml_chisq = new_chisq
            ml_params = params

    import matplotlib.pyplot as plt

    ax = plt.subplot(211)
    sliced_lc = light_curve.slice_mjd(ml_params["t0"].value-ml_params["tE"].value, ml_params["t0"].value+ml_params["tE"].value)
    print len(sliced_lc)
    sliced_lc.plot(ax)

    ax2 = plt.subplot(212)
    sliced_ml_params = pa.fit_microlensing_event(sliced_lc)
    new_sliced_light_curve = pa.fit_subtract_microlensing(sliced_lc, fit_data=sliced_ml_params)
    new_sliced_light_curve.plot(ax2)
    ax2.set_title(r"Med. Err: {0}, $\sigma$: {1}".format(np.median(sliced_lc.error), np.std(new_sliced_light_curve.mag)))

    plt.savefig("plots/test_slice_peak2_bad_data.png")

    # Now do with a simulated event
    from ptf.lightcurve import SimulatedLightCurve
    light_curve = SimulatedLightCurve(mjd=light_curve.mjd, mag=15, error=0.1)
    light_curve.add_microlensing_event(u0=0.1, t0=55600, tE=40)

    ml_chisq = 1E6
    ml_params = None
    for ii in range(10):
        params = pa.fit_microlensing_event(light_curve)
        new_chisq = params["result"].chisqr

        if new_chisq < ml_chisq:
            ml_chisq = new_chisq
            ml_params = params

    plt.clf()
    ax = plt.subplot(211)
    sliced_lc = light_curve.slice_mjd(ml_params["t0"].value-ml_params["tE"].value, ml_params["t0"].value+ml_params["tE"].value)
    print len(sliced_lc)
    sliced_lc.plot(ax)

    ax2 = plt.subplot(212)
    sliced_ml_params = pa.fit_microlensing_event(sliced_lc)
    new_sliced_light_curve = pa.fit_subtract_microlensing(sliced_lc, fit_data=sliced_ml_params)
    new_sliced_light_curve.plot(ax2)
    ax2.set_title(r"Med. Err: {0}, $\sigma$: {1}".format(np.median(sliced_lc.error), np.std(new_sliced_light_curve.mag)))

    plt.savefig("plots/test_slice_peak2_sim.png")


def test_iscandidate(plot=False):
    ''' Use test light curves to test selection:
        - Periodic
        - Bad data
        - Various simulated events
        - Flat light curve
        - Transients (SN, Nova, etc.)
    '''

    np.random.seed(10)

    logger.setLevel(logging.DEBUG)
    from ptf.lightcurve import SimulatedLightCurve
    import ptf.db.mongodb as mongo

    db = mongo.PTFConnection()

    logger.info("---------------------------------------------------")
    logger.info(greenText("Periodic light curves"))
    logger.info("---------------------------------------------------")

    # Periodic light curves
    periodics = [(4588, 7, 13227), (4588, 2, 15432), (4588, 9, 17195), (2562, 10, 28317), (4721, 8, 11979), (4162, 2, 14360)]

    for field_id, ccd_id, source_id in periodics:
        periodic_light_curve = pdb.get_light_curve(field_id, ccd_id, source_id, clean=True)
        periodic_light_curve.indices = pa.compute_variability_indices(periodic_light_curve, indices=["eta", "delta_chi_squared", "j", "k", "sigma_mu"])
        assert pa.iscandidate(periodic_light_curve, lower_eta_cut=10**db.fields.find_one({"_id" : field_id}, {"selection_criteria" : 1})["selection_criteria"]["eta"]) in ["subcandidate" , False]
        if plot: plot_lc(periodic_light_curve)

    logger.info("---------------------------------------------------")
    logger.info(greenText("Bad light curves"))
    logger.info("---------------------------------------------------")

    # Bad data
    bads = [(3756, 0, 14281), (1983, 10, 1580)]

    for field_id, ccd_id, source_id in bads:
        bad_light_curve = pdb.get_light_curve(field_id, ccd_id, source_id, clean=True)
        bad_light_curve.indices = pa.compute_variability_indices(bad_light_curve, indices=["eta", "delta_chi_squared", "j", "k", "sigma_mu"])
        assert not pa.iscandidate(bad_light_curve, lower_eta_cut=10**db.fields.find_one({"_id" : field_id}, {"selection_criteria" : 1})["selection_criteria"]["eta"])
        if plot: plot_lc(bad_light_curve)

    logger.info("---------------------------------------------------")
    logger.info(greenText("Simulated light curves"))
    logger.info("---------------------------------------------------")

    # Simulated light curves
    for field_id,mjd in [(4721,periodic_light_curve.mjd)]:
        for err in [0.01, 0.05, 0.1]:
            logger.debug("field: {0}, err: {1}".format(field_id,err))
            light_curve = SimulatedLightCurve(mjd=mjd, mag=15, error=[err])
            light_curve.indices = pa.compute_variability_indices(light_curve, indices=["eta", "delta_chi_squared", "j", "k", "sigma_mu"])
            assert not pa.iscandidate(light_curve, lower_eta_cut=10**db.fields.find_one({"_id" : field_id}, {"selection_criteria" : 1})["selection_criteria"]["eta"])

            light_curve.add_microlensing_event(u0=np.random.uniform(0.2, 0.8), t0=light_curve.mjd[int(len(light_curve)/2)], tE=light_curve.baseline/8.)
            light_curve.indices = pa.compute_variability_indices(light_curve, indices=["eta", "delta_chi_squared", "j", "k", "sigma_mu"])
            if plot:
                plt.clf()
                light_curve.plot()
                plt.savefig("plots/tests/{0}_{1}.png".format(field_id,err))
            assert pa.iscandidate(light_curve, lower_eta_cut=10**db.fields.find_one({"_id" : field_id}, {"selection_criteria" : 1})["selection_criteria"]["eta"])

    logger.info("---------------------------------------------------")
    logger.info(greenText("Transient light curves"))
    logger.info("---------------------------------------------------")

    # Transients (SN, Novae)
    transients = [(4564, 0, 4703), (4914, 6, 9673), (100041, 1, 4855), (100082, 5, 7447), (4721, 8, 3208), (4445, 7, 11458),\
                  (100003, 6, 10741), (100001, 10, 5466), (4789, 6, 11457), (2263, 0, 3214), (4077, 8, 15293), (4330, 10, 6648), \
                  (4913, 7, 13436), (100090, 7, 2070), (4338, 2, 10330), (5171, 0, 885)]

    for field_id, ccd_id, source_id in transients:
        transient_light_curve = pdb.get_light_curve(field_id, ccd_id, source_id, clean=True)
        logger.debug(transient_light_curve)
        transient_light_curve.indices = pa.compute_variability_indices(transient_light_curve, indices=["eta", "delta_chi_squared", "j", "k", "sigma_mu"])
        assert pa.iscandidate(transient_light_curve, lower_eta_cut=10**db.fields.find_one({"_id" : field_id}, {"selection_criteria" : 1})["selection_criteria"]["eta"])
        if plot: plot_lc(transient_light_curve)

if __name__ == "__main__":
    #test_slice_peak()
    #test_slice_peak2()
    test_iscandidate(True)