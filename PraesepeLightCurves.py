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
        
"""

# Standard library
import copy
import glob
import logging
import math
import os
import re
import sys

# Third-party
import apwlib.astrodatetime as astrodatetime
import apwlib.geometry as g
import apwlib.convert as c
import apwlib.plot as p
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyfits as pf
from scipy.stats import scoreatpercentile
import sqlalchemy
from sqlalchemy import func

# Project
from ptf.db.DatabaseConnection import *
import ptf.simulation.util as simu

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
        
        2012-05-21 TODO: Implement Nick's suggestions for how to select out "good" data points!
        
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
    return (x - x.min()) / (x.max() - x.min())

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
        
    def save(self, file=None):
        self.subplot_array[0,0].set_xlim(self.subplot_array[1,0].get_xlim())
        self.subplot_array[1,1].set_xlim(self.subplot_array[2,0].get_xlim())
        self.subplot_array[2,2].set_xlim(self.subplot_array[3,0].get_xlim())
        self.subplot_array[3,3].set_xlim(self.subplot_array[0,3].get_ylim())
        
        if file:
            self.figure.savefig(file)
        else:
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
        ptf_light_curve = simu.SimulatedLightCurve(lightCurve.amjd[this_field_idx], mag=lightCurve.amag[this_field_idx], error=lightCurve.error[this_field_idx])
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
                              color='k', alpha=0.2, marker=".")
                              
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
    viFigure2 = copy.copy(viFigure)
    viFigure.save()
    
    if color_by:
        for ax in viFigure.vi_axis_list:
            try:
                ax.series_plot_return["to_color"].set_alpha(0.0)
            except KeyError:
                pass
        
        viFigure2.save("plots/praesepe_field{0}.png".format(field))
    
# NEEDS CLEANING
def plot_delta_chi_squared_vs_mean_mag(number=1E5):
    """ Plot the variability index Delta Chi-squared vs. the mean magnitude of it's
        associated light curve.
    """
    avg_mag = [x[0] for x in session.query(func.array_avg(LightCurve.mag)).filter(LightCurve.objid < 100000).order_by(LightCurve.objid).limit(number).all()]
    delta_chi_squared = [x[0] for x in session.query(VariabilityIndices.delta_chi_squared).join(LightCurve).filter(LightCurve.objid < 100000).order_by(LightCurve.objid).limit(number).all()]
    
    data = np.loadtxt("data/mean_mag_delta_chisquared.txt", delimiter=",")
    plt.scatter(data[:,0], data[:,1], alpha=0.4, facecolor="r", edgecolor="none")
    plt.scatter(avg_mag, delta_chi_squared, alpha=0.2, facecolor="k", edgecolor="none")
    plt.yscale("log")
    plt.xlabel("Average Magnitude", size="xx-large")
    plt.ylabel(r"$\chi_{linear}^2-\chi_{ML}^2$", size="xx-large")
    plt.axhline(2.*np.std(delta_chi_squared), label=r"$2\sigma$", color="r", ls="--")
    plt.axhline(-2.*np.std(delta_chi_squared), color="r", ls="--")
    plt.axhline(3.*np.std(delta_chi_squared), label=r"$3\sigma$", color="g", ls="--")
    plt.axhline(-3.*np.std(delta_chi_squared), color="g", ls="--")
    plt.legend()
    plt.savefig("plots/meanmag_vs_deltachisquared.png")

def test_delta_chi_squared(number=1E5):
    import scipy.optimize as so
    
    light_curves = session.query(LightCurve).filter(LightCurve.objid < 100000).limit(number).all()
    
    for lc in light_curves:
        light_curve = simu.PTFLightCurve.fromDBLightCurve(lc)
        light_curve.addMicrolensingEvent()
        
        linear_fit_params, lin_num = so.leastsq(simu.linear_error_function, x0=(np.median(light_curve.amag), 0.), args=(light_curve.amjd, light_curve.amag, light_curve.error))
        microlensing_fit_params, ml_num = so.leastsq(simu.microlensing_error_function, x0=(np.median(light_curve.amag), light_curve.amjd[light_curve.amag.argmin()], 10., -25.), args=(light_curve.amjd, light_curve.amag, light_curve.error), maxfev=10000)
        
        linear_chisq = np.sum(simu.linear_error_function(linear_fit_params, \
                                                    light_curve.amjd, \
                                                    light_curve.amag, \
                                                    light_curve.error)**2)# / len(linear_fit_params)
        
        microlensing_chisq = np.sum(simu.microlensing_error_function(microlensing_fit_params, \
                                                                light_curve.amjd, \
                                                                light_curve.amag, \
                                                                light_curve.error)**2)# / len(microlensing_fit_params)
        #if (linear_chisq - microlensing_chisq) > 100:
        if True:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            light_curve.plot(ax)
            
            x0=(np.median(light_curve.amag), light_curve.amjd[light_curve.amag.argmin()], 10., -25.)
            t = light_curve.amjd
            ax.plot(t, simu.linear_model(linear_fit_params, t), "r-", label="linear fit")
            ax.plot(t, simu.microlensing_model(microlensing_fit_params, t), "b--", label="microlensing fit")
            ax.plot(t, simu.microlensing_model(x0, t), "g--", label="initial guess")
            ax.legend()
            ax.set_title("ML: {}_{}, Linear: {}_{}".format(microlensing_chisq, ml_num, linear_chisq, lin_num))
            plt.show()

# NEEDS CLEANING
def plot_delta_chi_squared_outliers(num_sigma=2.5):
    delta_chi_squared = [x[0] for x in session.query(VariabilityIndices.delta_chi_squared).join(LightCurve).filter(LightCurve.objid < 100000).all()]
    light_curves = session.query(LightCurve).join(VariabilityIndices)\
                                            .filter(VariabilityIndices.delta_chi_squared > num_sigma*np.std(delta_chi_squared))\
                                            .filter(func.array_avg(LightCurve.mag) < 19)\
                                            .filter(LightCurve.objid < 100000).all()
    
    for light_curve in light_curves:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        light_curve.plot(ax)
        plt.savefig("plots/praesepe_interesting/{}.png".format(light_curve.objid))
        del fig, ax


def parse_logs(filename):
    """ Parse a PTF observation log to grab mjd and seeing """
    f = open(filename)
    text = f.readlines()
    
    mjd_epoch = astrodatetime.astrodatetime(2009, 1, 1,0,0,0, tzinfo=astrodatetime.gmt).mjd
    
    pattr = re.compile("^\s*11000\d\s+([0-9\:]+)\s+object\s+[A-Za-z]\s+[0-9\.]+\s+([0-9\.]+).*PTF(2010\d\d\d\d)")
    
    marcel_jds = []
    seeings = []
    for line in text:
        try:
            time, seeing, date = pattr.search(line).groups()
        except AttributeError:
            continue
        
        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:])
        
        hour = g.Angle.fromHours(time).hours
        MarcelJD = astrodatetime.astrodatetime(year, month, day, *map(int,c.hoursToHMS(hour)), tzinfo=astrodatetime.gmt).mjd - mjd_epoch
        
        marcel_jds.append(round(MarcelJD, 3))
        seeings.append(float(seeing)/2.35)
    
    return np.array(zip(marcel_jds, seeings), dtype=[("mjd", float), ("seeing", float)]).view(np.recarray)
        
# NEEDS CLEANING
def coadd_light_curves(num=1000):
    """ -> Create an MJD vector for ALL mjd values in DB
    	-> Every time there is a data point for a given MJD, increment weight
	    -> Sum and divide!
    """
    
    #filter(func.array_avg(LightCurve.mag) <= 17).\
    
    plt.clf()
    fields = [110001, 110002, 110003, 110004]
    
    for ii, field in enumerate(fields):
        lcs = session.query(LightCurve).\
                       filter(LightCurve.objid < 100000).\
                       filter(sqlalchemy.literal(field) == func.all(LightCurve.field)).\
                       order_by(LightCurve.objid).limit(num).all()
        
        all_mjds = set([])
        for lc in lcs:
            all_mjds = set(lc.mjd).union(all_mjds)
        
        all_mjds = [str(x) for x in all_mjds]
        coadd_dict = dict(zip(all_mjds, [0.]*len(all_mjds)))
        scatter_dict = dict(zip(all_mjds, []*len(all_mjds)))
        weight_dict = dict(zip(all_mjds, [0.]*len(all_mjds)))
        
        for lc in lcs:
            med_mag = np.median(lc.mag)
            mjd_to_mag = dict(zip([str(x) for x in lc.mjd], lc.mag))
            
            for mjd in all_mjds:
                if mjd not in mjd_to_mag.keys(): continue
                #coadd_dict[mjd] += (mjd_to_mag[mjd] - med_mag)**2 / med_mag**2
                coadd_dict[mjd] += (mjd_to_mag[mjd] / med_mag)
                try:
                    scatter_dict[mjd].append(mjd_to_mag[mjd] / med_mag)
                except:
                    scatter_dict[mjd] = [mjd_to_mag[mjd] / med_mag]
                    
                weight_dict[mjd] += 1.
        
        mjds = []
        mags = []
        variances = []
        
        for mjd, mag in coadd_dict.items():
            mjds.append(float(mjd))
            mags.append(mag / weight_dict[mjd])
            variances.append(np.var(scatter_dict[mjd]))
        
        plt.clf()
        plt.figure(figsize=(15,15))
        plt.subplot(311)
        plt.title("Field: {}".format(field))
        plt.plot(mjds, np.array(mags)-1.0, 'ko', alpha=0.6)
        plt.ylabel("Normalized coadd")
        plt.xlim(390, 510)
        plt.ylim(-0.0031, 0.0031)
        plt.gca().xaxis.set_ticks([])
        
        plt.subplot(312)
        plt.plot(mjds, variances, 'bo', alpha=0.6)
        plt.xlim(390, 510)
        plt.ylim(0.00002, 0.00016)
        plt.gca().xaxis.set_ticks([])
        plt.ylabel("Variance per obs.")
        
        plt.subplot(313)
        obs_log = parse_logs("data/all_{}.txt".format(field))
        plt.plot(obs_log.mjd, obs_log.seeing, "r+")
        plt.xlabel("MJD - Epoch [Days]")
        plt.ylabel("Seeing")
        plt.xlim(390, 510)
        
        plt.subplots_adjust(hspace=0.1, wspace=0.0)
        plt.savefig("plots/praesepe_coadd_{}.png".format(field))
        
        """
        plt.subplot(2,2,ii+1)
        plt.title("Field: {}".format(field))
        plt.plot(mjds, mags, 'ko', alpha=0.75)
        plt.xlim(390, 510)
        #plt.ylim(0., 0.0025)
        
        obs_log = parse_logs("data/all_{}.txt".format(field))
        plt.plot(obs_log.mjd, obs_log.seeing, "r+")
        
        if ii == 2 or ii == 3:
            plt.xlabel("MJD - Epoch [Days]")
        else:
            plt.gca().xaxis.set_ticks([])
            
        if ii == 0 or ii == 2:
            plt.ylabel("RMS Error per source per exposure")
        else:
            plt.gca().yaxis.set_ticks([])
        
    plt.subplots_adjust(hspace=0.1, wspace=0.0)
        """
    
# ===========================================================================================
# ===========================================================================================

def detection_efficiency_worker(lc_tuple):
    mjd, mag, error = lc_tuple
    return simu.compute_delta_chi_squared(lc_tuple)

def delta_chi_squared_detection_efficiency(num_light_curves=1E5, num_events=1000):
    """ I used these queries to get the minimum and maximum mjd values for all of
        the Praesepe observations:
            
            SELECT min(min_mjds) FROM (
                    SELECT min(unnested_mjd) AS min_mjds FROM (
                            SELECT unnest(mjd) AS unnested_mjd FROM light_curve WHERE objid < 100000
                    ) foo
            ) bar;
            
            SELECT max(max_mjds) FROM (
                    SELECT max(unnested_mjd) AS max_mjds FROM (
                            SELECT unnest(mjd) AS unnested_mjd FROM light_curve WHERE objid < 100000
                    ) foo
            ) bar;
        
        Min = 397.223
        Max = 501.212
        Baseline = 103.989 days
       
        I'm going to use the actual Praesepe light curves here, add a microlensing event
        some time between the min and max mjd, and see how we do detecting it using the
        delta chi-squared matched filter approach.
        
        Parameters
        ----------
        num_events : int
            The number of events to add to each light curve.
    """
    num_light_curves = 10000
    num_events = 100
    
    import time
    a = time.time()
    
    data = []    
    for xx in range(num_light_curves/1000):
        print xx
        light_curves = session.query(LightCurve).filter(LightCurve.objid < 100000).order_by(LightCurve.objid).offset(xx*1000).limit(1000).all()
        
        for light_curve in light_curves:
            ptf_light_curve = simu.PTFLightCurve.fromDBLightCurve(light_curve)
            
            for ii in range(num_events):
                sim_light_curve = copy.copy(ptf_light_curve)
                
                t0 = np.random.uniform(397.223, 501.212)
                sim_light_curve.addMicrolensingEvent(t0=t0)
                data.append((np.mean(ptf_light_curve.mjd), detection_efficiency_worker((sim_light_curve.mjd, sim_light_curve.mag, sim_light_curve.error))))

    print "took {} seconds for {} events added to {} light curves".format(time.time()-a, num_events, num_light_curves)
    data = np.array(data, dtype=[("mean_mag", float), ("delta_chisquared", float)]).view(np.recarray)
    np.savetxt("data/mean_mag_delta_chisquared.txt", data, delimiter=",", fmt="%.3f")

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
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
    parser.add_argument("--number", dest="number", default=1E5,
                    help="Number of light curves to use")
    parser.add_argument("--color-by", dest="color_by", type=str, default=None,
                    help="Color the plot by microlensing event parameter")
    parser.add_argument("--seed", dest="seed", type=int, default=None,
                    help="Seed the random number generator.")
    parser.add_argument("--plot-delta-chisquared", dest="plot_deltachisq", default=False, action="store_true",
                    help="Run plot_delta_chi_squared_vs_median_mag()")
    parser.add_argument("--detection-efficiency", dest="simulate_detection_efficiency", default=False, action="store_true",
                    help="Run a detection efficiency simulation.")
    
    args = parser.parse_args()
    
    if args.seed:
        np.random.seed(args.seed)
    
    if args.test:
        #high_error_test()
        #correlated_noise_test()
        test_delta_chi_squared(args.number)
        
        print
        print "Test mode complete."
        print "-"*30
        sys.exit(0)
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    elif args.quiet:
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if args.compute_indices:
        compute_indices()
    
    if args.plot:
        plot_indices(args.indices, number=args.number)
    
    if args.field != 0:
        single_field_add_microlensing(args.indices, args.field, args.number, args.color_by)
    
    if args.plot_deltachisq:
        plot_delta_chi_squared_vs_median_mag(args.number)
    
    if args.simulate_detection_efficiency:
        delta_chi_squared_detection_efficiency()