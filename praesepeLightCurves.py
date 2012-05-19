# coding: utf-8
"""
    Load Marcel's Prasaepe light curves into my local Postgres database
    
    TODO: ** Nick's suggestions **
        
        1/ As you say, remove points with high errors. Note the error estimates are photon-noise only, 
            no systematics (which are the dominant errors in any errant data points), but at least this
            will get rid of some data taken  during cloudy periods or when the moon is bright.
        2/ Look for correlated noise in the dataset -- search for and remove epochs which have many 
            stars with significant deviations from their median magnitude, all happening at the same time.
        3/ If there are multiple data points taken during a night, down-weight sources which have only 
            one point that's a significant deviation. This is the main supernova false-positive rejection 
            system, but it may not work with the event lengths you're looking for.
        4/ Look at each target's environment -- for example, if there are lots of nearby (< 10 arcsec) 
            sources that could create confusion / blending, the photometry is more likely to have occasional 
            problems than if it's an isolated star. Similarly for targets near the edge of the chips or on 
            bad pixels (I can send you bad pixel masks if you like; that's already taken into account 
            pretty well in my pipeline though).
        
    I think there should be a way to define a microlensing "interest factor" that reduces all 
    of our variability indices down to one number. We could similarly define such a factor
    for eclipses, or periodic stars down the road, but the idea is to be able to select out
    light curves on one number. 
    
    TODO : Make 3D version of 5 by 5 plot in case there are multiple peaks that are getting washed out
            when plotting all the distributions as 2-d w/ opacity.
        
"""

# Standard library
import os, sys, glob
import copy
import logging

# Third-party
from scipy.stats import scoreatpercentile
import sqlalchemy
from sqlalchemy import func
import numpy as np
import apwlib.geometry as g
import apwlib.convert as c
import matplotlib
#matplotlib.use("WxAgg")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project
from ptf.db.DatabaseConnection import *
import ptf.simulation.util as simu

ALL_INDICES = ["j", "k", "con", "sigma_mu", "eta", "b", "f"]

import math

def hsv2rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    #r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return (r, g, b)

def txt_file_light_curve_to_recarray(filename):
    """ All columns in this file are:
        MJD
        R1	<-- ignore (I think; doesn't matter much
	    R2	which of these two you use for this, I say)
        R2 err
        RA
        DEC
        X
        Y
        flag
        file
        field
        chip
        
        MJD's are relative to 2009-1-1 (54832)
    """
    
    names = ["mjd", "mag", "mag_err", "ra", "dec", "flag", "fieldid", "ccd"]
    usecols = [0, 2, 3, 4, 5, 8, 10, 11]
    dtypes = [float, float, float, float, float, float, int, int]
    
    return np.genfromtxt(filename, usecols=usecols, dtype=zip(names, dtypes)).view(np.recarray)

def load_txt_file_light_curves_into_database():
    """ Load the text file light curves (sent by Marcel, merged-lc.tar.gz)
        into my local postgres database on deimos.
    """
    session.begin()
    logging.debug("Starting database load...")
    
    filenames = sorted(glob.glob("data/lc_merged*.txt"))
    numFilesLeft = len(filenames)
    for file in filenames:
        objid = int(file.split("_")[2].split(".")[0])
        try:
            dbLC = session.query(LightCurve).filter(LightCurve.objid == objid).one()
        except sqlalchemy.orm.exc.NoResultFound:
            lc = txt_file_light_curve_to_recarray(file)
            lc = lc[lc.flag == 0]
            
            filterid = [2] * len(lc.mjd)
            syserr = [0.] * len(lc.mjd)

            lightCurve = LightCurve()
            lightCurve.objid = objid
            lightCurve.mjd = lc.mjd
            lightCurve.mag = lc.mag
            lightCurve.mag_error = lc.mag_err
            lightCurve.sys_error = syserr
            lightCurve.filter_id = filterid
            lightCurve.ra = lc.ra[0]
            lightCurve.dec = lc.dec[0]
            lightCurve.flags = [int(x) for x in lc.flag]
            lightCurve.field = lc.fieldid
            lightCurve.ccd = lc.ccd
            session.add(lightCurve)
        
        if len(session.new) == 1000:
            session.commit()
            logging.debug("1000 light curves committed!")
            session.begin()
            numFilesLeft -= 1000
            print "{0} files left".format(numFilesLeft)
    
    logging.info("All light curves committed!")
    session.commit()

def compute_indices():
    """ For all of the Praesepe light curves, first check to see that more than
        50% of the data points in a given light curve are good. If not, ignore that
        light curve.
        
        TODO: When Nick emails me back, implement his suggestion for how to select out "good" 
                data points.
    """
    
    # Select out 1000 light curves at a time
    for ii in range(session.query(LightCurve).filter(LightCurve.objid < 100000).count()/1000+1):
        num_bad = 0.0
        for lightCurve in session.query(LightCurve).filter(LightCurve.objid < 100000).order_by(LightCurve.objid).offset(ii*1000).limit(1000).all():
            try:
                # Remove old variability indices
                var = session.query(VariabilityIndices).join(LightCurve).filter(LightCurve.objid == lightCurve.objid).one()
                session.delete(var)
            except sqlalchemy.orm.exc.NoResultFound:
                pass
            
            # Only select points where the error is less than the acceptable percent error
            #   based on a visual fit to the plot Nick sent on 4/30/12
            if 15 <= np.median(lightCurve.amag) <= 19:
                percent_acceptable = 0.01 * 10**( (np.median(lightCurve.amag) - 15.) / 3.5 )
            elif np.median(lightCurve.amag) < 15:
                percent_acceptable = 0.01
            else:
                percent_acceptable = 0.5
            
            idx = (lightCurve.error / lightCurve.amag) < percent_acceptable
            
            """if float(sum(idx)) < len(lightCurve.error):
                print "{0} points rejected".format(len(lightCurve.error)-float(sum(idx)))
                print percent_acceptable, np.median(lightCurve.amag), max(lightCurve.error)
                print lightCurve.error / lightCurve.amag
                print 
            """
            
            # Check that less than half of the data points are bad
            if float(sum(idx)) / len(lightCurve.error) <= 0.5:
                logging.debug("Bad light curve -- ignore=True")
                num_bad += 1
                lightCurve.ignore = True
                continue
            
            lc = simu.PTFLightCurve(lightCurve.amjd[idx], lightCurve.amag[idx], lightCurve.error[idx])
            try:
                # Returns a dictionary with keys = the names of the indices
                var_indices = simu.compute_variability_indices(lc)
            except NameError:
                continue
            
            logging.debug("Good light curve -- ignore=False")
            lightCurve.ignore = False
            variabilityIndices = VariabilityIndices()
            for key in var_indices.keys():
                setattr(variabilityIndices, key, var_indices[key])
                
            variabilityIndices.light_curve = lightCurve
            session.add(variabilityIndices)
            
        logging.info("Fraction of good light curves: {}".format(1-num_bad/1000))
        session.flush()
        
    session.flush()

# THIS FUNCTION NEEDS AN OVERHAUL
def load_and_match_txt_coordinates(file, vi_names):    
    raDecs = np.genfromtxt(file, delimiter=",")
    
    matchedTargets = []
    for raDec in raDecs:
        ra = raDec[0]
        dec = raDec[1]
        try:
            varIdx = session.query(VariabilityIndices)\
                        .join(LightCurve)\
                        .filter(LightCurve.objid < 100000)\
                        .filter(func.q3c_radial_query(LightCurve.ra, LightCurve.dec, ra, dec, 5./3600))\
                        .one()
                        
            matchedTargets.append(tuple([getattr(varIdx, idx) for idx in vi_names]))
        except sqlalchemy.orm.exc.NoResultFound:
            pass
            
    return np.array(matchedTargets, dtype=zip(vi_names, [float]*len(vi_names))).view(np.recarray)

def variability_indices_to_recarray(vi_list, vi_names):
    """ Convert a list of VariabilityIndices database objects into
        a numpy recarray
    """
    arr = np.array([tuple([getattr(vi_item, name) for name in vi_names]) for vi_item in vi_list], \
                   dtype=zip(vi_names, [float]*len(vi_names))).view(np.recarray)
    return arr

def linear_rescale(x):
    x = np.array(x)
    return (x-x.min()) / (x.max() - x.min())

# =====================================================================================
#   These next functions will be my "pipeline" for downweighting or ignoring light
#      curves with many bad points
#
def high_error_weight(light_curve):
    """ Given a LightCurve object, throw away messy data points.
    
        For brighter sources, e.g. magnitude 15 or so, this amounts
        to removing points with > 1% errors, but for fainter sources,
        we allow for up to 5% error.
        
    """
    # Only select points where the error is less than the acceptable percent error
    #   based on a visual fit to the plot Nick sent on 4/30/12
    median_magnitude = np.median(light_curve.mag)
    if median_magnitude < 15:
        percent_acceptable = 0.01
    else:
        percent_acceptable = 0.01 * 10**( (median_magnitude - 15.) / 7.153 )
    
    percent_error = (light_curve.error / np.array(light_curve.mag))
    idx = percent_error > percent_acceptable
    
    weights = np.ones(len(light_curve.mag), dtype=float)
    weights[idx] = percent_acceptable / percent_error[idx]
    
    return weights

def high_error_test():
    # Time the function
    import time
    lcs = session.query(LightCurve).filter(LightCurve.objid < 100000).limit(1000).all()
    
    times = []
    for lc in lcs:
        #print sum(high_error_weight(lc)), len(lc.mag)
        a = time.time()
        weights = high_error_weight(lc)
        times.append(time.time() - a)
        
        if sum(weights) < len(lc.mag):
            logging.debug("high_error_weight: {}".format(weights[weights != 1]))
    
    print "high_error_weight() average time: {}".format(np.mean(times))
    
def correlated_noise_weight(light_curve):
    """ Find all nearby light curves and see if they have correlated errors.
        If they do, downweight those points in the weight vector. It's very
        unlikely that two nearby sources will be variable with the same
        period, thus producing correlated outliers.
    """
    # Get all sources within 1 arcmin
    nearby_sources = session.query(LightCurve).filter(func.q3c_radial_query(LightCurve.ra, LightCurve.dec, \
                                                                            light_curve.ra, light_curve.dec, 0.0167)).all()
    
    light_curve_mag_array = np.array(light_curve.mag)
    light_curve_mjd_array = np.array(light_curve.mjd)
    mjd_set = set(light_curve.mjd)
    for nearby_light_curve in nearby_sources:
        if nearby_light_curve.objid == light_curve.objid: continue
        
        nearby_light_curve_mag_array = np.array(nearby_light_curve.mag)
        nearby_light_curve_mjd_array = np.array(nearby_light_curve.mjd)
        common_mjd = np.array(list(mjd_set.intersection(set(nearby_light_curve.mjd))))
        
        light_curve_idx = np.in1d(light_curve_mjd_array, common_mjd)
        nearby_light_curve_idx = np.in1d(nearby_light_curve_mjd_array, common_mjd)
        
        # *** These are the two things to correlate!
        a = light_curve_mag_array[light_curve_idx]
        v = nearby_light_curve_mag_array[nearby_light_curve_idx]
        
        plt.clf()
        plt.subplot(211)
        plt.plot(common_mjd, a, 'ro')
        plt.plot(common_mjd, v, 'bo')
        plt.subplot(212)
        plt.plot(common_mjd, np.correlate(a, v, "same"), 'k.')
        plt.show()
    
    weights = np.ones(len(light_curve.mag), dtype=float)
    
    return weights

def correlated_noise_test():
    # Time the function
    import time
    lcs = session.query(LightCurve).filter(LightCurve.objid < 100000).limit(1000).all()
    
    times = []
    for lc in lcs:
        a = time.time()
        weights = correlated_noise_weight(lc)
        times.append(time.time() - a)
        
        if sum(weights) < len(lc.mag):
            logging.debug("correlated_noise_weight: {}".format(weights[weights != 1]))
        
        break
    
    print "correlated_noise_weight() average time: {}".format(np.mean(times))
    
def single_outlier_weight(light_curve):
    """ For multiple observations taken over one night, downweight any single
        point outliers. For examlple, if 3 observations are made over one night,
        and one of them is a huge outlier, downweight it!
    """
    pass
    
def target_environment_weight(light_curve):
    """ Consider a target's environment. If there are many nearby sources,
        e.g. <10 arcsec, downweight the entire light curve because it is 
        more prone to blending and confusion in the photometry pipeline.
    """
    pass

def compute_weight_vector(light_curve):
    """ Call the whole pipeline to produce a weight vector for computing
        the new WEIGHTED J and K variability indices
    """
    pass
    

# =====================================================================================
#   Plotting stuff
#

class VISeries:
    
    def __init__(self, x, y, **kwargs):
        self.x = x
        self.y = y
        self.kwargs = kwargs

class VIAxis:
    
    def __init__(self, x_axis_parameter, y_axis_parameter, plot_function="plot"):
        self.x_axis_parameter = x_axis_parameter
        self.y_axis_parameter = y_axis_parameter
        self.plot_function = plot_function
        
        self.series = dict()
        self.series_plot_return = dict()
        
    def add_series(self, x, y=None, name=None, **kwargs):
        viSeries = VISeries(x, y, **kwargs)
        
        if name:
            self.series[name] = viSeries
        else:
            self.series[len(self.series)] = viSeries
        
class VIFigure:

    def __init__(self, filename, variability_index_names):
        self.filename = filename
        
        # This dictionary controls which parameter goes where in the plot grid
        self._subplot_map = dict(zip(variability_index_names, range(len(variability_index_names))))
        
        self.figure, self.subplot_array = plt.subplots(len(self._subplot_map.keys()), \
                                                       len(self._subplot_map.keys()), \
                                                       figsize=(25,25))
        
        self.figure.subplots_adjust(hspace=0.1, wspace=0.1)
        self.scatter_dict = dict()
        self.vi_axis_list = list()
        
        #plt.setp([a.get_xticklabels() for a in self.figure.axes], visible=False)
        #plt.setp([a.get_yticklabels() for a in self.figure.axes], visible=False)
        
    def add_subplot(self, vi_axis, legend=False):
        colIdx = self._subplot_map[vi_axis.x_axis_parameter]
        rowIdx = self._subplot_map[vi_axis.y_axis_parameter]
        thisSubplot = self.subplot_array[rowIdx, colIdx]
        
        # HACK
        if legend:
            # Stole the below from http://matplotlib.sourceforge.net/api/axes_api.html
            h,l = self.subplot_array[0,1].get_legend_handles_labels()
            thisSubplot.legend(h,l)
        
        if vi_axis.plot_function == "loglog":
            xlabel = "log({0})".format(vi_axis.x_axis_parameter)
            ylabel = "log({0})".format(vi_axis.y_axis_parameter)
        elif vi_axis.plot_function == "plot":
            xlabel = vi_axis.x_axis_parameter
            ylabel = vi_axis.y_axis_parameter
        else:
            xlabel = ""
            ylabel = ""
        
        plt.setp([thisSubplot.get_xticklabels()], visible=False)
        plt.setp([thisSubplot.get_yticklabels()], visible=False)
        
        if colIdx == 0:
            # Put ylabel to left of axis (default)
            thisSubplot.set_ylabel(ylabel, size="xx-large")
            thisSubplot.yaxis.set_ticks_position('left')
            plt.setp([thisSubplot.get_yticklabels()], visible=True)
            
        elif colIdx == (len(self._subplot_map.keys())-1):
            # Put ylabel to right of axis
            thisSubplot.set_ylabel(ylabel, size="xx-large")
            thisSubplot.yaxis.set_label_position('right')
            
            thisSubplot.yaxis.set_ticks_position('right') 
            plt.setp([thisSubplot.get_yticklabels()], visible=True)
        
        if rowIdx == 0:
            # Put xlabel above axis
            thisSubplot.set_xlabel(xlabel, size="xx-large")
            thisSubplot.xaxis.set_label_position('top')
            thisSubplot.xaxis.set_ticks_position('top')
            plt.setp([thisSubplot.get_xticklabels()], visible=True)
            plt.setp([thisSubplot.get_yticklabels()], visible=False)
            
        elif rowIdx == (len(self._subplot_map.keys())-1):
            # Put xlabel below axis (default)
            thisSubplot.set_xlabel(xlabel, size="xx-large")
            
            thisSubplot.xaxis.set_ticks_position('bottom')
            plt.setp([thisSubplot.get_xticklabels()], visible=True)
        
        if vi_axis.plot_function == "loglog":
            thisSubplot.set_xscale("log")
            thisSubplot.set_yscale("log")
        
        for name, viSeries in vi_axis.series.items():
            
            if vi_axis.plot_function == "hist":
                #plt.setp([thisSubplot.get_xticklabels()], visible=True)
                #thisSubplot.hist(viSeries.x, log=True, bins=np.linspace(scoreatpercentile(viSeries.x, 5), scoreatpercentile(viSeries.x, 95), 1000))
                #thisSubplot.set_xscale("log")
                thisSubplot.set_frame_on(False)
                thisSubplot.get_yaxis().set_visible(False)
                thisSubplot.get_xaxis().set_visible(False)
                break
            
            if viSeries.kwargs.has_key("colors"):
                viSeries.kwargs["c"] = viSeries.kwargs["colors"]
                del viSeries.kwargs["colors"]
            
            # HACK / TODO: absolute value of loglog plot data...not the best thing to do
            if vi_axis.plot_function == "loglog":
                vi_axis.series_plot_return[name] = thisSubplot.scatter(np.fabs(viSeries.x), np.fabs(viSeries.y), **viSeries.kwargs)
            else:
                vi_axis.series_plot_return[name] = thisSubplot.scatter(viSeries.x, viSeries.y, **viSeries.kwargs)
        
        """
        if vi_axis.plot_function == "hist":
            try:
                print self.subplot_array[rowIdx+1,colIdx].get_xlim()
            except:
                pass
            thisSubplot.set_xlim(self.subplot_array[rowIdx-1,colIdx].get_xlim())
        """
        
        self.vi_axis_list.append(vi_axis)
    
    def add_colorbar(self, name):
        """ Add a colorbar to the figure for all subplots """
        cax = self.figure.add_axes([0.235, 0.05, 0.6, 0.03])
        ax = self.vi_axis_list[1].series_plot_return[name]
        self.figure.colorbar(ax, cax, orientation="horizontal")
        
    def save(self):
        self.subplot_array[0,0].set_xlim(self.subplot_array[1,0].get_xlim())
        self.subplot_array[1,1].set_xlim(self.subplot_array[2,0].get_xlim())
        self.subplot_array[2,2].set_xlim(self.subplot_array[3,0].get_xlim())
        self.subplot_array[3,3].set_xlim(self.subplot_array[0,3].get_ylim())
        self.figure.savefig(self.filename)

def plot_indices(indices, filename=None, number=1E6):    
    varIndices = session.query(VariabilityIndices)\
                        .join(LightCurve)\
                        .filter(LightCurve.ignore == False)\
                        .filter(LightCurve.objid < 100000)\
                        .limit(number).all()
    
    if not filename:
        filename = "plots/praesepe_{}.png".format("".join(indices))
    
    varIndicesArray = variability_indices_to_recarray(varIndices, indices)
    
    # This code finds any known rotators from Agueros et al.
    knownRotators = load_and_match_txt_coordinates("data/praesepe_rotators.txt", indices)    
    kraus2007Members = load_and_match_txt_coordinates("data/praesepe_krauss2007_members.txt", indices)
    rrLyrae = load_and_match_txt_coordinates("data/praesepe_rrlyrae.txt", indices)
    eclipsing = load_and_match_txt_coordinates("data/praesepe_eclipsing.txt", indices)
    wUma = load_and_match_txt_coordinates("data/praesepe_wuma.txt", indices)
    
    viFigure = VIFigure(filename, indices)
    for ii,yParameter in enumerate(indices):
        for jj,xParameter in enumerate(indices):            
            if ii > jj:
                # Bottom triangular section of plots: Linear plots
                viAxis = VIAxis(xParameter, yParameter, plot_function="plot")
            elif ii < jj:
                # Top triangular section: Log plots
                viAxis = VIAxis(xParameter, yParameter, plot_function="loglog")
            else:
                viAxis = VIAxis(xParameter, yParameter, plot_function="hist")
            
            viAxis.add_series(varIndicesArray[xParameter], varIndicesArray[yParameter], color="k", marker=".", alpha=0.3, label="All Praesepe Field Stars")
            viAxis.add_series(kraus2007Members[xParameter], kraus2007Members[yParameter], color="g", marker="v", alpha=0.8, s=20, label=u"Kraus 2007 Member Catalog")
            viAxis.add_series(knownRotators[xParameter], knownRotators[yParameter], color="r", marker="*", alpha=0.8, s=20, label=u"Agüeros et al. 2011 rotators")
            viAxis.add_series(rrLyrae[xParameter], rrLyrae[yParameter], color="c", marker="^", s=20, alpha=0.8, label=u"Agüeros et al. 2011 RR Lyrae")
            viAxis.add_series(eclipsing[xParameter], eclipsing[yParameter], color="m", marker="^", s=20, alpha=0.8, label=u"Agüeros et al. 2011 Eclipsing Binary")
            #viAxis.add_series(wUma[xParameter], wUma[yParameter], color="y", marker="^", s=20, alpha=0.8, label=u"Agüeros et al. 2011 W Uma")
            
            if ii == jj:
                # Along diagonal
                viAxis.add_series(varIndicesArray[xParameter], varIndicesArray[xParameter], color="k", alpha=0.5)
            
            # Add whatever other series here
            # viAxis.add_series(varIndices[xParameter], varIndices[yParameter], color="r", marker=".", alpha=0.3)
            if ii == jj == (len(indices)-1):
                viFigure.add_subplot(viAxis, legend=True)
            else:
                viFigure.add_subplot(viAxis)
                
    viFigure.save()

def plot_indices_3d(indices, filename="plots/praesepe_var_indices_3d.png"):
    varIndices = session.query(VariabilityIndices)\
                        .join(LightCurve)\
                        .filter(LightCurve.ignore == False)\
                        .filter(LightCurve.objid < 100000)\
                        .limit(10000)\
                        .all()    
    
    cm = matplotlib.cm.get_cmap('Spectral')
    vi_array = variability_indices_to_recarray(varIndices, indices)
    
    fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    ax = fig.add_subplot(111)
    
    j = vi_array.j + abs(vi_array.j.min()) + 1
    k = vi_array.k
    
    hist, xedges, yedges = np.histogram2d(np.log10(j), np.log10(k), bins=1000)
    xbins = 0.5 * (xedges[:-1] + xedges[1:])
    ybins = 0.5 * (yedges[:-1] + yedges[1:])
    
    ax.contour(xbins, ybins, hist.T, 4, colors='k')
    
    #plt.hexbin(, vi_array.k, gridsize=200, cmap=cm, xscale="log", yscale="log")
    plt.show()
    #plt.savefig(filename)

def plotInterestingVariables():
    lightCurves = session.query(LightCurve).join(VariabilityIndices).\
                        filter(VariabilityIndices.eta < 1).\
                        filter(LightCurve.ignore==False).\
                        filter(VariabilityIndices.j > 100).\
                        filter(LightCurve.objid<100000).all()
    
    rrLyrae = [g.RADec((g.RA.fromDegrees(x[0]), g.Dec.fromDegrees(x[1]))) for x in np.genfromtxt("data/praesepe_rrlyrae.txt", delimiter=",")]
    eclipsing = [g.RADec((g.RA.fromDegrees(x[0]), g.Dec.fromDegrees(x[1]))) for x in np.genfromtxt("data/praesepe_eclipsing.txt", delimiter=",")]
    wUma = [g.RADec((g.RA.fromDegrees(x[0]), g.Dec.fromDegrees(x[1]))) for x in np.genfromtxt("data/praesepe_wuma.txt", delimiter=",")]
    knownRotators = [g.RADec((g.RA.fromDegrees(x[0]), g.Dec.fromDegrees(x[1]))) for x in np.genfromtxt("data/praesepe_rotators.txt", delimiter=",")]
    
    for lc in lightCurves:
        lcRADec = g.RADec((g.RA.fromDegrees(lc.ra), g.Dec.fromDegrees(lc.dec)))
        
        title = ""
        
        for star in rrLyrae:
            if lcRADec.subtends(star).degrees < 5./3600:
                title = "RR Lyrae"
        
        for star in eclipsing:
            if lcRADec.subtends(star).degrees < 5./3600:
                title = "Eclipsing"
        
        for star in wUma:
            if lcRADec.subtends(star).degrees < 5./3600:
                title = "W Uma"
        
        for star in knownRotators:
            if lcRADec.subtends(star).degrees < 5./3600:
                title = "Rotator"
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        lc.plot(ax, error_cut=0.05)
        fig.savefig("plots/praesepe/{}.png".format(lc.objid))

def single_field_add_microlensing(indices, field, num_light_curves=100000, color_by=None):
    """ For all light curves in a single PTF field around Praesepe, produce a 5x5 plot for 
        all light curves and then a random sample of all objects with random microlensing
        event parameters added.
        
        color_by controls which parameter to color the points on. 
    """
    
    # Select any light curve with at least one observation on the given field 
    lightCurves = session.query(LightCurve).\
                          filter(LightCurve.objid < 100000).\
                          filter(LightCurve.ignore == False).\
                          filter(sqlalchemy.literal(field) == func.any(LightCurve.field)).\
                          limit(num_light_curves).\
                          all()
    
    logging.info("Light curves loaded from database")
    
    # Now we loop through all light curves selected above and only select data points with 
    #   nonzero error that were taken in the given PTF field
    var_indices = []
    var_indices_with_event = []
    color_by_parameter = []
    
    for lightCurve in lightCurves:
        # Only select points where the error is less than 0.1 mag and only for the specified field
        this_field_idx = (lightCurve.afield == field) & (lightCurve.error != 0)

        if len(lightCurve.amjd[this_field_idx]) <= 25:
            logging.debug("Light curve doesn't have enough data points on this field")
            continue
        
        logging.debug("Light curve selected")

        # Create a PTFLightCurve object so we can easily compute the variability indices
        ptf_light_curve = simu.PTFLightCurve(lightCurve.amjd[this_field_idx], lightCurve.amag[this_field_idx], lightCurve.error[this_field_idx])
        var_indices.append(simu.compute_variability_indices(ptf_light_curve, indices=indices))
        
        # For a random sample of 10% of the light curves, add 100 different microlensing events to 
        #   their light curves and recompute the variability indices. Then
        if np.random.uniform() <= 0.1:
            for ii in range(100):
                # Copy the PTFLightCurve object, add a microlensing event, and recompute the variability indices
                lc = copy.copy(ptf_light_curve)
                lc.addMicrolensingEvent()
                var_indices_with_event.append(simu.compute_variability_indices(lc, indices=indices))
                
                if color_by: color_by_parameter.append(getattr(lc, color_by))
    
    if not lightCurves or not var_indices:
        logging.info("No light curves selected.")
        return
    
    var_indices_array = np.array(var_indices, dtype=zip(indices, [float]*len(indices))).view(np.recarray)
    var_indices_with_event_array = np.array(var_indices_with_event, dtype=zip(indices, [float]*len(indices))).view(np.recarray)
    color_by_parameter_array = np.array(color_by_parameter)
    
    if color_by: figure_name = "plots/praesepe_field{0}_coloredby{1}.png".format(field, color_by)
    else: figure_name = "plots/praesepe_field{0}.png".format(field)
        
    viFigure = VIFigure(figure_name, indices)
    cm = matplotlib.cm.get_cmap('Spectral')
    for ii,yParameter in enumerate(indices):
        for jj,xParameter in enumerate(indices):
            if ii > jj:
                # Bottom triangular section of plots: Linear plots
                viAxis = VIAxis(xParameter, yParameter, plot_function="plot")
            elif ii < jj:
                # Top triangular section: Log plots
                viAxis = VIAxis(xParameter, yParameter, plot_function="loglog")
            else:
                viAxis = VIAxis(xParameter, yParameter, plot_function="hist")
            
            #viAxis.add_series(var_indices_with_event_array[xParameter], var_indices_with_event_array[yParameter], \
            #                  color=var_indices_with_event_colors, marker="o", alpha=0.3, linestyle="none", label="1000 Random Light Curves\nw/ Artificial Microlensing Events")
            
            if color_by:
                # TODO: Change vmin and vmax to be scoreatpercentile?
                vmin = min(color_by_parameter_array)
                vmax = max(color_by_parameter_array)
                viAxis.add_series(var_indices_with_event_array[xParameter], var_indices_with_event_array[yParameter], \
                              name="to_color", c=color_by_parameter_array, alpha=0.2, vmin=vmin, vmax=vmax, cmap=cm, edgecolors='none')
            else:
                viAxis.add_series(var_indices_with_event_array[xParameter], var_indices_with_event_array[yParameter], \
                              color='k', alpha=0.25, marker=".")
            
            viAxis.add_series(var_indices_array[xParameter], var_indices_array[yParameter], \
                              color="k", marker=".", alpha=0.25, label="All Light Curves in Field {0}".format(field))
            
            if ii == jj:
                # Along diagonal
                # Skip histograms for now..
                viAxis.add_series(var_indices_array[xParameter], color="k", alpha=0.3)
            
            # Add whatever other series here
            # viAxis.add_series(varIndices[xParameter], varIndices[yParameter], color="r", marker=".", alpha=0.3)
            if ii == jj == 4:
                viFigure.add_subplot(viAxis, legend=True)
            else:
                viFigure.add_subplot(viAxis)
    
    viFigure.add_colorbar(name="to_color")
    viFigure.save()
    

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_argument("--compute-indices", action="store_true", dest="compute_indices", default=False,
                    help="Run compute_indices()")
    parser.add_argument("--indices", nargs='+', type=str, dest="indices", default=[],
                    help="Store the specified list of indices")
    parser.add_argument("--plot", action="store_true", dest="plot", default=False,
                    help="Run plot_indices()")
    parser.add_argument("--field", type=int, dest="field", default=0,
                    help="Store the specified field and run single_field_add_microlensing()")    
    parser.add_argument("--test", action="store_true", dest="test", default=False,
                    help="Run in test mode")
    parser.add_argument("--number", dest="number", default=1000000,
                    help="Number of light curves to use")
    parser.add_argument("--color-by", dest="color_by", type=str, default=None,
                    help="Color the plot by microlensing event parameter")
    parser.add_argument("--seed", dest="seed", type=int, default=None,
                    help="Seed the random number generator.")
    
    args = parser.parse_args()
    
    if args.seed:
        np.random.seed(args.seed)
    
    if args.test:
        #high_error_test()
        correlated_noise_test()
        
        print
        print "Test mode complete."
        print "-"*30
        sys.exit(0)
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if args.compute_indices:
        compute_indices()
    
    if args.plot:
        plot_indices(args.indices, number=args.number)
    
    if args.field != 0:
        single_field_add_microlensing(args.indices, args.field, args.number, args.color_by)
        
    #plotIndices("plots/praesepe.png")
    #reComputeIndices()
    #plotInterestingVariables()
    
    #single_field_add_microlensing(110004)