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
import os, sys, glob
import copy
import logging

# Third-party
import sqlalchemy
from sqlalchemy import func
import numpy as np
import apwlib.geometry as g
import apwlib.convert as c
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project
from ptf.db.DatabaseConnection import *
import ptf.simulation.util as simu

ALL_INDICES = ["j", "k", "con", "sigma_mu", "eta", "b", "f"]

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
                var_indices = simu.computeVariabilityIndices(lc)
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
    pass

def correlated_noise_weight(light_curve):
    """ Find all nearby light curves and see if they have correlated errors.
        If they do, downweight those points in the weight vector.
    """
    pass
    
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
        
        self.series = []
        
    def add_series(self, x, y=None, **kwargs):
        viSeries = VISeries(x, y, **kwargs)
        self.series.append(viSeries)
        
class VIFigure:

    def __init__(self, filename, variability_index_names):
        self.filename = filename
        
        # This dictionary controls which parameter goes where in the plot grid
        self._subplot_map = dict(zip(variability_index_names, range(len(variability_index_names))))
        
        self.figure, self.subplot_array = plt.subplots(len(self._subplot_map.keys()), \
                                                       len(self._subplot_map.keys()), \
                                                       figsize=(25,25))
        self.figure.subplots_adjust(hspace=0, wspace=0)
        plt.setp([a.get_xticklabels() for a in self.figure.axes], visible=False)
        plt.setp([a.get_yticklabels() for a in self.figure.axes], visible=False)
        
    def addSubplot(self, vi_axis, legend=False):
        rowIdx = self._subplot_map[vi_axis.x_axis_parameter]
        colIdx = self._subplot_map[vi_axis.y_axis_parameter]
        thisSubplot = self.subplot_array[colIdx, rowIdx]
        
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
        
        if colIdx == 0:
            # Put xlabel above axis
            thisSubplot.set_xlabel(xlabel)
            thisSubplot.xaxis.set_label_position('top')
        elif colIdx == (len(self._subplot_map.keys())-1):
            # Put xlabel below axis (default)
            thisSubplot.set_xlabel(xlabel)
        
        if rowIdx == 0:
            # Put ylabel to left of axis (default)
            thisSubplot.set_ylabel(ylabel)
            pass
        elif rowIdx == (len(self._subplot_map.keys())-1):
            # Put ylabel to right of axis
            thisSubplot.set_ylabel(ylabel)
            thisSubplot.yaxis.set_label_position('right')
        
        # We have negative and 0 values for some indices, so this figures
        #   out how to offset that stuff based on the FIRST item in the
        #   series array e.g. the one on the bottom layer
        """viSeries = vi_axis.series[0]
        xOffset = min(viSeries.x)
        if xOffset == 0: xOffset = 1
        elif xOffset > 0: xOffset = 0
        elif xOffset < 0: xOffset = np.log(xOffset * -1)
        
        yOffset = min(viSeries.y)
        if yOffset == 0: yOffset = 1
        elif yOffset > 0: yOffset = 0
        elif yOffset < 0: yOffset = np.log(yOffset * -1)
        
        print vi_axis.x_axis_parameter, vi_axis.y_axis_parameter
        print xOffset, yOffset"""
        
        for viSeries in vi_axis.series:
            if vi_axis.plot_function == "loglog":                
                #thisSubplot.loglog(viSeries.x + xOffset, viSeries.y + yOffset, **viSeries.kwargs)
                thisSubplot.loglog(np.fabs(viSeries.x), np.fabs(viSeries.y), **viSeries.kwargs)
            
            elif vi_axis.plot_function == "hist":
                #bins = np.logspace(min(viSeries.x), max(viSeries.x), 100)
                thisSubplot.set_frame_on(False)
                #thisSubplot.set_xscale('log')
                thisSubplot.get_yaxis().set_visible(False)
                thisSubplot.get_xaxis().set_visible(False)
                #thisSubplot.hist(viSeries.x, bins=50, **viSeries.kwargs)
                
            elif vi_axis.plot_function == "plot":
                thisSubplot.plot(viSeries.x, viSeries.y, **viSeries.kwargs)
            
            else:
                raise ValueError("Invalid plot_function! You specified: {}".format(vi_axis.plot_function))
    
    def save(self):
        self.figure.savefig(self.filename, facecolor="#efefef")

def plot_indices(indices, filename="plots/praesepe_var_indices.png"):    
    varIndices = session.query(VariabilityIndices)\
                        .join(LightCurve)\
                        .filter(LightCurve.ignore == False)\
                        .filter(LightCurve.objid < 100000).all()

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
            
            viAxis.add_series(varIndicesArray[xParameter], varIndicesArray[yParameter], color="k", marker=".", alpha=0.3, linestyle="none", label="All Praesepe Field Stars")
            viAxis.add_series(kraus2007Members[xParameter], kraus2007Members[yParameter], color="g", marker="v", alpha=0.8, markersize=8, linestyle="none", label=u"Kraus 2007 Member Catalog")
            viAxis.add_series(knownRotators[xParameter], knownRotators[yParameter], color="r", marker="*", alpha=0.8, markersize=10, linestyle="none", label=u"Agüeros et al. 2011 rotators")
            viAxis.add_series(rrLyrae[xParameter], rrLyrae[yParameter], color="c", marker="^", markersize=10, alpha=0.8, linestyle="none", label=u"Agüeros et al. 2011 RR Lyrae")
            viAxis.add_series(eclipsing[xParameter], eclipsing[yParameter], color="m", marker="^", markersize=10, alpha=0.8, linestyle="none", label=u"Agüeros et al. 2011 Eclipsing Binary")
            viAxis.add_series(wUma[xParameter], wUma[yParameter], color="y", marker="^", markersize=10, alpha=0.8, linestyle="none", label=u"Agüeros et al. 2011 W Uma")
            
            if ii == jj:
                # Along diagonal
                # Skip histograms for now..
                viAxis.add_series(varIndicesArray[xParameter], color="k", alpha=0.3)
            
            # Add whatever other series here
            # viAxis.add_series(varIndices[xParameter], varIndices[yParameter], color="r", marker=".", alpha=0.3)
            if ii == jj == (len(indices)-1):
                viFigure.addSubplot(viAxis, legend=True)
            else:
                viFigure.addSubplot(viAxis)
                
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

def single_field_add_microlensing(field, num_microlensing_light_curves=10000):
    """ For all light curves in a single PTF field around Praesepe, produce a 5x5 plot for 
        all light curves and then a random sample of all objects with random microlensing
        event parameters added.
    """

    lightCurves = session.query(LightCurve).\
                          filter(LightCurve.objid < 100000).\
                          filter(LightCurve.ignore == False).\
                          filter(sqlalchemy.literal(field) == func.any(LightCurve.field)).\
                          all()
    
    logging.info("Light curves loaded from database")
    
    var_indices = []
    var_indices_with_event = []
    
    for lightCurve in lightCurves:
        # Only select points where the error is less than 0.1 mag and only for the specified field
        idx = (lightCurve.error < 0.1) & (lightCurve.afield == field) & (lightCurve.error != 0)

        if len(lightCurve.amjd[idx]) <= 10:
            logging.debug("Light curve doesn't have enough data points for this field")
            continue
        
        logging.debug("Light curve selected")

        ptf_light_curve = simu.PTFLightCurve(lightCurve.amjd[idx], lightCurve.amag[idx], lightCurve.error[idx])
        var_indices.append(simu.computeVariabilityIndices(ptf_light_curve, tuple=True))
        
        # For 10% of the light curves, add 100 different microlensing events to their light curves
        #   and recompute the variability indices
        if np.random.uniform() <= 0.1:
            for ii in range(100):
                lc = copy.copy(ptf_light_curve)
                lc.addMicrolensingEvent()
                var_indices_with_event.append(simu.computeVariabilityIndices(lc, tuple=True))
    
    var_indices_array = np.array(var_indices, dtype=zip(NAMES, [float]*len(NAMES))).view(np.recarray)
    var_indices_with_event_array = np.array(var_indices_with_event, dtype=zip(NAMES, [float]*len(NAMES))).view(np.recarray)
    
    viFigure = VIFigure("plots/praesepe_field{0}.png".format(field))
    for ii,yParameter in enumerate(NAMES):
        for jj,xParameter in enumerate(NAMES):            
            if ii > jj:
                # Bottom triangular section of plots: Linear plots
                viAxis = VIAxis(xParameter, yParameter, plot_function="plot")
            elif ii < jj:
                # Top triangular section: Log plots
                viAxis = VIAxis(xParameter, yParameter, plot_function="loglog")
            else:
                viAxis = VIAxis(xParameter, yParameter, plot_function="hist")
            
            viAxis.add_series(var_indices_with_event_array[xParameter], var_indices_with_event_array[yParameter], \
                              color="r", marker=".", alpha=0.3, linestyle="none", label="1000 Random Light Curves\nw/ Artificial Microlensing Events")
            viAxis.add_series(var_indices_array[xParameter], var_indices_array[yParameter], \
                              color="k", marker=".", alpha=0.3, linestyle="none", label="All Light Curves in Field {0}".format(field))
            
            if ii == jj:
                # Along diagonal
                # Skip histograms for now..
                viAxis.add_series(var_indices_array[xParameter], color="k", alpha=0.3)
            
            # Add whatever other series here
            # viAxis.add_series(varIndices[xParameter], varIndices[yParameter], color="r", marker=".", alpha=0.3)
            if ii == jj == 4:
                viFigure.addSubplot(viAxis, legend=True)
            else:
                viFigure.addSubplot(viAxis)
                
    viFigure.save()

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty (default = False)")
    parser.add_argument("--compute-indices", action="store_true", dest="compute_indices", default=False,
                    help="Run compute_indices()")
    parser.add_argument("--plot-indices", nargs='+', type=str, dest="plot_indices", default=[],
                    help="Run plot_indices() with the specified list of indices")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    if args.compute_indices:
        compute_indices()
    
    if len(args.plot_indices) > 0:
        plot_indices(args.plot_indices)
    
    #plotIndices("plots/praesepe.png")
    #reComputeIndices()
    #plotInterestingVariables()
    
    #single_field_add_microlensing(110004)