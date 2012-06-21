# -*- coding: utf-8 -*-

""" Search David Levitan's database for microlensing events. 
    
    This script gets a list of exposures from the LSD, and selects all PTF fields
    with more than 100 observations. Then, for each field selected, we select out
    all sources with η < 2σ and saves this into a large ?? file for further analysis.

"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard Library
import os, sys
import glob
from optparse import OptionParser
import logging

import matplotlib
matplotlib.use("Agg")

# Third-party dependencies
import numpy as np
try:
    import tables
except:
    logging.warning("You should run this on Navtara with David Levitan's installation of Python here:\n\n \t/scr4/dlevitan/sw/epd-7.1-1-rh5-x86_64/bin/\n")
import matplotlib.pyplot as plt

# PTF dependences
from ptf.ptflightcurve import PTFLightCurve

db_path = "/scr4/dlevitan/matches"

def compute_eta_2sigma(chip):
    """ Given a pytables object from DL's photometric database for one field/ccd,
        compute the 2σ levels for the von Neumann ratio to select out candidate
        microlensing events.
        
        e.g. if the file is 
            dbFile = tables.openFile('matches/match_02_100083_10.pytable')
        you should pass this object:
            dbFile.root.filter02.field100083.chip10
    """
    etas = chip.sources.col("vonNeumannRatio")
    mu = np.mean(etas[etas > 0.0])
    sigma = np.std(etas[etas > 0.0])
    return (mu - 2.*sigma, mu + 2.*sigma)

def select_candidates_from_chip(chip):
    """ Given a pytables object from DL's photometric database for one field/ccd,
        select out candidate microlensing events
        
        e.g. if the file is 
            dbFile = tables.openFile('matches/match_02_100083_10.pytable')
        you should pass this object:
            dbFile.root.filter02.field100083.chip10
    """
    sigma_cut = compute_eta_2sigma(chip)
    candidate_sources = chip.sources.readWhere("(vonNeumannRatio < {}) & (vonNeumannRatio > 0) & (ngoodobs > 50)".format(sigma_cut[0]))
    idx = np.in1d(chip.sourcedata.col("matchedSourceID"), candidate_sources["matchedSourceID"])    
    return chip.sourcedata[idx]

def candidate_light_curves_from_field(fieldid, filename=None):
    """ Given a field id, return photometric information for candidate
        microlensing events selected on their von Neumann ratios.
    """
    
    if os.path.exists(filename): return
    
    field_candidates = None 
    for ccdid in range(12):
        file = "match_02_{:06d}_{:02d}.pytable".format(fieldid, ccdid)
        ccd_filename = os.path.join(db_path, file)
        
        if not os.path.exists(ccd_filename):
            logging.debug("Field: {}, CCD: {} table not found..skipping...".format(fieldid, ccdid))
            continue
        
        dbFile = tables.openFile(ccd_filename)
        chip = getattr(getattr(dbFile.root.filter02, "field{:06d}".format(fieldid)), "chip{:02d}".format(ccdid))
        candidates = select_candidates_from_chip(chip)
        
        if field_candidates == None:
            field_candidates = candidates
        else:
            field_candidates = np.hstack((field_candidates, candidates))
        
        dbFile.close()
    
    if filename == None:
        return field_candidates
    else:
        np.save(filename, field_candidates)

def find_all_candidates(path="data/candidates"):
    """ Search all fields that have been observed enough times for candidate microlensing
        events. Then select out these light curves and save them to .npy files.
    """
    f = pf.open("data/exposureData.fits")
    data = f[1].data
    fields = np.unique(data["field_id"])
    
    for fieldid in fields:
        one_field = data[data["field_id"] == fieldid]
        one_filter = one_field[one_field["filter_id"] == 2]
        
        nums = []
        for ccdid in range(12):
            one_ccd = one_filter[one_filter["ccd_id"] == ccdid]
            nums.append(len(one_ccd))
        
        num_observations = max(nums)
        
        if num_observations > 100:
            logging.debug("Field: {}".format(fieldid))
            array_filename = os.path.join(path, "{:06d}.npy".format(fieldid))
            try:
                candidate_light_curves_from_field(fieldid, array_filename)
            except NameError:
                logging.warning("No non-zero eta values...seems fishy")

def analyze_candidates_from_file(filename):
    """ Given a .npy file (e.g. created with candidate_light_curves_from_field()), analyze
        the selected candidate microlensing event light curves.
        
        Info per object:
        ('a_world', 'absphotzp', 'alpha_j2000', 'b_world', 'background', 'class_star', 
        'delta_j2000', 'errx2_image', 'errxy_image', 'erry2_image', 'flux_auto', 'flux_radius_1', 
        'flux_radius_2', 'flux_radius_3', 'flux_radius_4', 'flux_radius_5', 'fluxerr_auto', 
        'fwhm_image', 'ipacFlags', 'isoarea_world', 'kron_radius', 'mag', 'magErr', 'mag_aper_1', 
        'mag_aper_2', 'mag_aper_3', 'mag_aper_4', 'mag_aper_5', 'mag_auto', 'mag_iso', 'mag_petro', 
        'magerr_aper_1', 'magerr_aper_2', 'magerr_aper_3', 'magerr_aper_4', 'magerr_aper_5', 
        'magerr_auto', 'magerr_iso', 'magerr_petro', 'matchedSourceID', 'mjd', 'mu_max', 
        'mu_threshold', 'petro_radius', 'pid', 'relPhotFlags', 'sextractorFlags', 'sid', 
        'theta_j2000', 'threshold', 'x', 'x2_image', 'x_image', 'xpeak_image', 'y', 'y2_image', 
        'y_image', 'ypeak_image', 'z')
    
        Interesting ones:
        903
        1519
        1940
        2124
        2350
        2999
        3233
        3668
        4267
        4512
        5829
        9358
        
    
    """
    data = np.load(filename)
    scanned_lc_file = "data/candidates/scanned_light_curves.txt"
    plot_filename = "data/candidates/{}.png"
    
    # Remove nan or inf values from data
    data = data[np.isfinite(data["mag"])]
    data = data[np.isfinite(data["mjd"])]
    data = data[np.isfinite(data["magErr"])]
    
    # Remove points with ridiculous error bars / bad values
    data = data[data["magErr"] < 0.5]
    data = data[data["mag"] > 13.5]
    data = data[data["mag"] < 22]
    
    fig = plt.figure()
    scanned_light_curves = np.genfromtxt(scanned_lc_file)
    for data_objid in np.unique(data["matchedSourceID"]):
        logging.debug("objid: {}".format(data_objid))
        if data_objid in scanned_light_curves:
            logging.debug("objid in scanned_light_curves.txt. skipping...")
            continue
        if os.path.exists(plot_filename.format(data_objid)):
            logging.debug("Plot exists for objid. skipping...")
            continue
        
        one_object = data[data["matchedSourceID"] == data_objid]
        
        if len(one_object) < 50: continue
        
        light_curve = PTFLightCurve(mjd=one_object["mjd"].astype(float), mag=one_object["mag"].astype(float), error=one_object["magErr"].astype(float))
        J,K,eta,delta_chi_squared = light_curve.variability_index(indices=["j", "k", "eta", "delta_chi_squared"])
        
        if J >= 1000 and eta < 1.0:
            #logging.debug("Saving objid {}; J={}, K={}".format(objid, J, K))
            print "Saving objid {}; J={}, K={}, eta={}, delta_chi_squared={}".format(data_objid, J, K, eta, delta_chi_squared)
            
            #fig = plt.figure()
            fig.suptitle("objid {}; J={}, K={}, eta={}, delta_chi_squared={}".format(data_objid, J, K, eta, delta_chi_squared))
            ax = fig.add_subplot(211)
            light_curve.plot(ax)
            
            ax2 = fig.add_subplot(212)
            sparse_period_data = light_curve.aovFindPeaks(min_period=0.05, max_period=100.0, subsample=0.1, finetune=0.01)
            #long_period_data = light_curve.aovFindPeaks(min_period=1.0, max_period=100.0, subsample=0.5, finetune=0.05)
            ax2.plot(sparse_period_data["period"], -sparse_period_data["periodogram"], 'k-')
            #ax2.plot(long_period_data["period"], -long_period_data["periodogram"], 'k-')
            
            #plt.show()
            #del fig
            #sys.exit(0)
            
            fig.savefig(plot_filename.format(data_objid))
            fig.clf()
        
        else:
            with open(scanned_lc_file, "a+") as f:
                f.write("{}\n".format(data_objid))
            scanned_light_curves = np.genfromtxt(scanned_lc_file)


def save_objid_to_txt(objid, path="data/candidates"):
    """ Given an objid, save that light curve to a text file, e.g.
        
        # mjd mag mag_error
        55123.124 14.6 0.052
        etc... ... ...
        
    """
    for npy_file in glob.glob(os.path.join(path, "*.npy")):
        data = np.load(npy_file)
        obj_data = data[data["matchedSourceID"] == objid]
        if len(obj_data) == 0: 
            continue
        else:
            light_curve = PTFLightCurve(mjd=obj_data["mjd"], mag=obj_data["mag"], error=obj_data["magErr"])
            light_curve.savetxt(os.path.join(path, "lc_{}.txt".format(objid)))
            break

if __name__ == "__main__":
    import pyfits as pf
    
    parser = OptionParser(description="")
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_option("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! Shhh! (default = False)")
    parser.add_option("--find", action="store_true", dest="find", default=False,
                    help="Find candidate events -- must be run on navtara!")
    parser.add_option("--analyze", action="store_true", dest="analyze", default=False,
                    help="Analyze!")
    parser.add_option("--path", dest="path", type=str, default="data/candidates",
                    help="Path to the .npy files.")
    parser.add_option("--objid", dest="objid", type=int, default=None,
                    help="Specific objid")
    
    (options, args) = parser.parse_args()
    if options.verbose: logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    elif options.quiet: logging.basicConfig(level=logging.ERROR, format='%(message)s')
    else: logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    if options.find:
        find_all_candidates(path=options.path)
    
    if options.analyze:
        for file in glob.glob(os.path.join(options.path, "*.npy")):
            analyze_candidates_from_file(file)