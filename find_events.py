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
import re

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
import ptf.photometricdatabase as pdb

def compute_eta_2sigma(chip, plot=False):
    """ Given a pytables object from DL's photometric database for one field/ccd,
        compute the 2σ levels for the von Neumann ratio to select out candidate
        microlensing events.
        
        e.g. if the file is 
            dbFile = tables.openFile('matches/match_02_100083_10.pytable')
        you should pass this object:
            dbFile.root.filter02.field100083.chip10
    """
    etas = chip.sources.readWhere("vonNeumannRatio > 0")["vonNeumannRatio"]
    
    mu = np.mean(etas)
    sigma = np.std(etas)
    
    if plot:
        bins = np.logspace(np.log10(min(etas)), np.log10(max(etas)), 50)
        plt.hist(etas, bins=bins)
        plt.axvline(mu - 2.*sigma, color='r', ls='--')
        plt.show()
    
    return (mu - 2.*sigma, mu + 2.*sigma)

def select_candidates_from_chip(chip, ngoodobs=25, plot=False):
    """ Given a pytables object from DL's photometric database for one field/ccd,
        select out candidate microlensing events
        
        e.g. if the file is 
            dbFile = tables.openFile('matches/match_02_100083_10.pytable')
        you should pass this object:
            dbFile.root.filter02.field100083.chip10
    """
    sigma_cut = compute_eta_2sigma(chip, plot=plot)
    if np.isfinite(sigma_cut[0]):  
        candidate_sources = chip.sources.readWhere("(vonNeumannRatio < {}) & \
                                                    (vonNeumannRatio > 0) & \
                                                    (ngoodobs > {})".format(sigma_cut[0], ngoodobs))
        
        print "Number initially selected: {}".format(len(candidate_sources))
        
        idx = np.in1d(chip.sourcedata.col("matchedSourceID"), candidate_sources["matchedSourceID"])
    else:
        idx = np.array([])
    return chip.sourcedata[idx]

def cut_bad_data(data):
    """ Given a recarray-like array or table, clip out any bad data points """
    
    #data = data[(data["relPhotFlags"] & 5949) == 0]
    #print "Number after flag cut: {}".format(len(np.unique(data["matchedSourceID"])))
    data = data[(data["sextractorFlags"] < 8) == 0]
    print "Number after sextractor flag cut: {}".format(len(np.unique(data["matchedSourceID"])))
    data = data[(data["sextractorFlags"] & 1) == 0]
    print "Number after flag cut2: {}".format(len(np.unique(data["matchedSourceID"])))
    
    # Remove nan or inf values from data
    data = data[np.isfinite(data["mag"])]
    data = data[np.isfinite(data["mjd"])]
    data = data[np.isfinite(data["magErr"])]
    
    # Remove points with ridiculous error bars / bad values
    data = data[data["magErr"] < 0.5]
    data = data[data["mag"] > 13.5]
    data = data[data["mag"] < 22]
    
    print "Final number of light curves: {}".format(len(np.unique(data["matchedSourceID"])))
    
    return data
    
def save_candidates(candidate_data_path="data/candidates"):
    # Select fields that have been observed more than 25 times
    fields = pdb.select_most_observed_fields(minimum_num_observations=25, limit=0)
    
    # Only select out light curves from the field with more than 25 good observations
    #   that also meet our selection criteria
    candidate_data_file = "field{:06d}_ccd{:02d}.npy"
    for field in fields:
        print "Field: {}".format(field.id)
        for ccdid,ccd in field.ccds.items():
            # If the candidate data file already exists, skip processing this field
            if os.path.exists(os.path.join(candidate_data_path, candidate_data_file.format(field.id, ccdid))): 
                continue
                
            candidates = []
            print "\tCCD: {}".format(ccdid)
                        
            chip = ccd.read()
            candidate_data = select_candidates_from_chip(chip, plot=False)
            candidate_data = cut_bad_data(candidate_data)
            ccd.close()
            
            for source_id in np.unique(candidate_data["matchedSourceID"]):
                source_data = candidate_data[candidate_data["matchedSourceID"]==source_id]
                num_obs = len(source_data)
                print "Source: {}, Num Obs: {}".format(source_id, num_obs)
                
                if num_obs > 10: 
                    candidates.append(source_data)
        
            if len(candidates) > 0:
                np.save(os.path.join(candidate_data_path,candidate_data_file.format(field.id, ccdid)), np.hstack(tuple(candidates)))

def analyze_candidates_from_file(filename, light_curve_path="data/candidates/light_curves"):
    """ Given a .npy file, analyze the selected candidate microlensing event light curves.
        
        Each object must have (at minimum) columns "mjd", "mag", and "magErr"
        
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
    
    """
    data = np.load(filename)
    lc_filename = "data/candidates/light_curves/field{}_ccd{}_id{}.npy"
    
    pattr = re.compile("field(\d+)_ccd(\d+).npy")
    try:
        field_id, ccd_id = map(int, pattr.search(os.path.basename(filename)).groups())
    except AttributeError:
        logging.error("File not found.")
        return
    
    logging.debug("Field: {}, CCD: {}".format(field_id, ccd_id))
    
    fig = plt.figure()
    for source_id in np.unique(data["matchedSourceID"]):
        logging.debug("objid: {}".format(source_id))
        
        if os.path.exists(lc_filename.format(field_id, ccd_id, source_id)):
            logging.debug("Light curve file exists for objid")
            continue
            
        one_object = data[data["matchedSourceID"] == source_id]
        
        if len(one_object) < 10: continue
        
        light_curve = PTFLightCurve(mjd=one_object["mjd"].astype(float), mag=one_object["mag"].astype(float), error=one_object["magErr"].astype(float))
        #J, K, eta, delta_chi_squared = light_curve.variability_index(indices=["j", "k", "eta", "delta_chi_squared"])
        J = light_curve.variability_index(indices=["j"])[0]
        
        if J >= 100:
            np.save(lc_filename.format(field_id, ccd_id, source_id), one_object)

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
        #find_all_candidates(path=options.path)
        save_candidates(candidate_data_path=options.path)
    
    if options.analyze:
        for file in glob.glob(os.path.join(options.path, "*.npy")):
            analyze_candidates_from_file(file)