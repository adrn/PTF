# Standard library
import os, sys
import argparse
import re
import logging
import cPickle as pickle

# Third party
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sqlalchemy import func

# Project
import ptf.simulation.util as simu
#import ptf.db.util as dbu
from ptf.db.DatabaseConnection import *

""" TODO:
    - Point size rescaling isn't working -- check that code to see why some are HUUUUGGGEE
"""

def plot_five_by_five(varIndicesNoEvent, varIndicesWithEvent, plot_prefix="plots", scale_point_size=None):
    """ Generate a plot of each variabilty index vs. each other on a 5x5 grid """
    
    if not os.path.exists(plot_prefix): os.mkdir(plot_prefix)
    
    kk = 0
    params = ["sigma_to_mu", "Con", "eta", "J", "K"]
    for ii in range(len(params)):
        if params[ii] == "J":
            bins = np.arange(0, 2500, 100)
        else:
            bins = 50
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(varIndicesNoEvent[params[ii]], color='b', alpha=0.4, bins=bins, normed=True)
        ax.hist(varIndicesWithEvent[params[ii]], color='r', alpha=0.4, bins=bins, normed=True)
        fig.savefig("plots/{0}_hist.png".format(params[ii]))
        
        # Plot-fu to make pretty figures
        for jj in range(len(params)):
            if ii == jj: continue
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            
            #ax1.plot(varIndicesNoEvent[params[ii]], varIndicesNoEvent[params[jj]], 'b.', alpha=0.4)
            #ax1.loglog(varIndicesNoEvent[params[ii]], varIndicesNoEvent[params[jj]], 'b.', alpha=0.2)
            
            #ax2.plot(varIndicesWithEvent[params[ii]], varIndicesWithEvent[params[jj]], 'r.', alpha=0.4)
            #ax2.loglog(varIndicesWithEvent[params[ii]], varIndicesWithEvent[params[jj]], 'r.', alpha=0.2)
            
            if scale_point_size != None:
                ax1.scatter(varIndicesNoEvent[params[ii]], varIndicesNoEvent[params[jj]], c='b', alpha=0.2, s=scale_point_size)
                ax2.scatter(varIndicesWithEvent[params[ii]], varIndicesWithEvent[params[jj]], c='r', alpha=0.2, s=scale_point_size)
            else:
                ax1.loglog(varIndicesNoEvent[params[ii]], varIndicesNoEvent[params[jj]], 'b.', alpha=0.2)
                ax2.loglog(varIndicesWithEvent[params[ii]], varIndicesWithEvent[params[jj]], 'r.', alpha=0.2)
                
            xmin1,xmax1 = ax1.get_xlim()
            xmin2,xmax2 = ax2.get_xlim()
            ax1.set_xlim(min(xmin1, xmin2), max(xmax1, xmax2))
            ax2.set_xlim(min(xmin1, xmin2), max(xmax1, xmax2))
            
            ymin1,ymax1 = ax1.get_ylim()
            ymin2,ymax2 = ax2.get_ylim()
            ax1.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
            ax2.set_ylim(min(ymin1, ymin2), max(ymax1, ymax2))
            
            ax1.set_xlabel(params[ii])
            ax2.set_xlabel(params[ii])
            ax1.set_ylabel(params[jj])
            fig.savefig(os.path.join(plot_prefix, "{0}_vs_{1}.png".format(params[ii], params[jj])))

# ===========================================================================================

def simulation1(number_of_light_curves, number_per_light_curve, number_of_points_range=(50,60), plot=False):
    """ Compute variability indices for simulated light curves
    
        Default values should be something like
            number_of_light_curves = 10
            number_per_light_curve = 1000
    """
    
    offsetNumber = 0
    q = 100
    light_curves = []
    
    logging.debug("Selecting light curves from database...")
    # Select light curves with enough data points (as specified with number_of_points_range)
    light_curves = session.query(LightCurve).filter(func.array_length(LightCurve.mjd, 1) > number_of_points_range[0])\
                                            .filter(func.array_length(LightCurve.mjd, 1) < number_of_points_range[1])\
                                            .limit(number_of_light_curves).all()
    
    if len(light_curves) < number_of_light_curves: 
        logging.warn("Only able to select {0} light curves with more than {1} but less than {2} observations.".format(len(light_curves), *number_of_points_range))
        yesOrNo = raw_input("Is that ok? [y]/n:")
        if yesOrNo == "y":
            number_of_light_curves = len(light_curves)
        else:
            sys.exit(0)
    else: 
        logging.info("Selected {0} light curves with more than {1} but less than {2} observations.".format(len(light_curves), *number_of_points_range))
    
    variabilityIndices = [lc.variability_indices.all_tuple for lc in light_curves]
    varIndicesNoEvent = np.array(variabilityIndices, dtype=[("sigma_to_mu",float), ("Con",float), ("eta",float), ("J",float), ("K",float)]).view(np.recarray)
    logging.debug("Done with light curves with no events...")
    
    # For each light curve, compute the variability indices AFTER adding a microlensing event
    variabilityIndices = []
    pt_size = []
    for jj in range(number_of_light_curves):
        if light_curves[jj].filter_id == None: 
            print "?????"
            continue
        
        logging.debug("Computing indices for light curve {0}".format(light_curves[jj].objid))
        for ii in range(number_per_light_curve):
            lightCurve = simu.PTFLightCurve(light_curves[jj].Rmjd, light_curves[jj].Rmag, light_curves[jj].Rerror)
            lightCurve.addMicrolensingEvent()
            pt_size.append(5./lightCurve.u0)
            
            try:
                variabilityIndices.append(simu.computeVariabilityIndices(lightCurve, tuple=True))
            except TypeError:
                logging.debug("Continuum fit failed!")
                continue
    
    varIndicesWithEvent = np.array(variabilityIndices, dtype=[("sigma_to_mu",float), ("Con",float), ("eta",float), ("J",float), ("K",float)]).view(np.recarray)
    logging.debug("Done with light curves with events...")
    
    if plot:
        logging.debug("Plotting...")
        plot_five_by_five(varIndicesNoEvent, varIndicesWithEvent, plot_prefix="plots/{0}-{1}".format(*number_of_points_range), scale_point_size=pt_size)
    
    return varIndicesNoEvent, varIndicesWithEvent
    
def detection_efficiency(number_of_light_curves=10000, baseline=365, pattern="uniform", overwrite=False, **kwargs):
    """ The goal of this simulation is to determine the detection efficiency
        as a function of number of observations. For now, we assume a baseline
        of one year and vary the sampling pattern.
        
        pattern can be: 
            - 'uniform'
            - 'random'
            - 'clumpy'
        
        [TODO]: The code should have two modes
            1) Generate entirely simulated light curves
            2) Draw light curves from database instead
            
        [TODO]: The code should loop over the following
            - Get light curve (see above)
                - Add simulated microlensing event
                - Try to find event
                - Add to some array the baseline, number of observations, sigma(squared successive difference)
        
        *Also see TODO's below*
        
    """
    # Write the data to a pickle for faster plotting
    if kwargs.has_key("clumps"):
        clumpList = list(kwargs['clumps'])
    else:
        clumpList = [0]
    
    for numClumps in clumpList:
        if kwargs.has_key("clumps"):
            dataFile = "data/detectionEfficiencies_{0}_{1}clumps.pickle".format(pattern, numClumps)
        else:
            dataFile = "data/detectionEfficiencies_{0}.pickle".format(pattern)
    
        # If the overwrite keyword is thrown, delete the data file before writing to it
        if overwrite: 
            try:
                os.remove(dataFile)
            except OSError:
                pass
        
        if not os.path.exists(dataFile):
            
            detection_fraction = []
            number_of_observations = 2**np.arange(4,12,0.5)
            
            for obs_num in number_of_observations:
                
                # Uniform sampling over the baseline for each light curve
                if pattern == "uniform":
                    mjd = np.linspace(0., baseline, obs_num)
                
                number_of_detections = 0
                for ii in range(number_of_light_curves):
                    # Random sampling over the baseline
                    if pattern == "random":
                        mjd = np.random.random(obs_num)*baseline
                    elif pattern == "clumpy":
                        #raise ValueError("If pattern is 'clumpy', you must also specify the number of clumps with the 'clumps' keyword.")
                            
                        mjd = list(np.zeros(obs_num, dtype=float))
                        total_len = 0
                        for c in range(numClumps):
                            clump_t = np.random.uniform(0., baseline)
                            clump_sigma = np.random.uniform(2., 50.)
                            
                            if c == numClumps-1:
                                len1 = obs_num - total_len
                            else:
                                len1 = int(obs_num/numClumps)
                                total_len += len1
                                
                            mjd += list(np.random.normal(clump_t, clump_sigma, size=len1))
                        
                            #plt.clf()
                            #plt.hist(mjd)#, [1]*len(mjd), 'r.', ls='none')
                            #plt.xlim(0, 365)
                            #plt.show()
                        
                        mjd = np.array(mjd)
                        
                    error = np.fabs(np.random.normal(0.15, 0.05, size=len(mjd)))
                    lightCurve = simu.SimulatedLightCurve(mjd=mjd, error=error, outliers=True)
                    lightCurve.addMicrolensingEvent()
                    
                    # Can we measure a microlensing event at the correct position?
                    continuumMag, continuumSigma = simu.estimateContinuum(mjd, lightCurve.mag, error)
                    clusters = simu.findClusters(lightCurve.mag, continuumMag, continuumSigma, num_points_per_cluster=3, num_sigma=2.5)
                    
                    for cluster in clusters:
                        # If one of the detected clusters is close to the known position of the event, 
                        #   call this a detection
                        if np.fabs(np.mean(mjd[cluster]) - lightCurve.t0)/lightCurve.t0 <= 0.1:
                            number_of_detections += 1
                            break
    
                detection_fraction.append(number_of_detections / float(number_of_light_curves))
            
            # Pack into a numpy recarray for easier handling
            data = np.array(zip(number_of_observations, detection_fraction), dtype=[("observations",float),("detection_fractions",float)]).view(np.recarray)
            
            f = open(dataFile, "w")
            pickle.dump(data, f)
            f.close()
        
    f = open(dataFile, "r")
    data = pickle.load(f)
    f.close()
    
    return data

def plot_detection_efficiency(data):
    """ Given a recarray with 'observations' and 'detection_fractions',
        plot the detection efficiency curve
    """
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data.observations, data.detection_fractions, 'ko', ls='none')
    ax.set_xlabel("Baseline")
    ax.set_ylabel("Number of Observations")
    
    fig.savefig("plots/detection_efficiency.pdf")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                        help="Be chatty!")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                        help="Be quiet!")
    parser.add_argument("-r", "--range", type=str, dest="range", default="(45,55)",
        				help="Accepted range of number of observations")
    parser.add_argument("-l", "--number-of-light-curves", type=int, dest="num_light_curves", default=1000,
        				help="Number of light curves to select from the databse")
    parser.add_argument("-s", "--number-of-simulations", type=int, dest="num_simulations", default=100,
        				help="Number of simulations per light curve")
    parser.add_argument("-d", "--detection-efficiency", action="store_true", dest="detection_eff", default=False,
                        help="Run the detection efficiency simulation")
    
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.WARN)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if args.detection_eff:
        detection_efficiency()
    else:
        pattr = re.compile("^[\(|\s]{0,1}([0-9]+)[\,|\-|\s]([0-9]+)[\)|\s]{0,1}")
        try:
            num_observations_range = map(int, pattr.match(args.range.strip()).groups())
        except ValueError:
            raise ValueError("Invalid --range input, must be of the form 10-20 or (10,20)")
                
        simulation1(args.num_light_curves, args.num_simulations, num_observations_range)