# coding: utf-8
"""
    Load Marcel's Prasaepe light curves into my local Postgres database
    ** Table schema at bottom! **
"""

# Standard library
import os, sys, glob
from argparse import ArgumentParser
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

NAMES = ["sigma_mu", "con", "eta", "j", "k"]

def txtFileToRecarray(filename):
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

def loadData():
    session.begin()
    logging.debug("Starting database load...")
    
    filenames = sorted(glob.glob("data/lc_merged*.txt"))
    numFilesLeft = len(filenames)
    for file in filenames:
        objid = int(file.split("_")[2].split(".")[0])
        try:
            dbLC = session.query(LightCurve).filter(LightCurve.objid == objid).one()
        except sqlalchemy.orm.exc.NoResultFound:
            lc = txtFileToRecarray(file)
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

def computeIndices():
    lightCurves = session.query(LightCurve).filter(LightCurve.objid < 100000).all()
    for lightCurve in lightCurves:
        if lightCurve.variability_indices == None:
            
            lc = simu.PTFLightCurve(lightCurve.mjd, lightCurve.mag, lightCurve.error)
            try:
                var_indices = simu.computeVariabilityIndices(lc)
            except NameError:
                continue
            
            variabilityIndices = VariabilityIndices()
            variabilityIndices.sigma_mu = var_indices["sigma_mu"]
            variabilityIndices.con = var_indices["con"]
            variabilityIndices.eta = var_indices["eta"]
            variabilityIndices.j = var_indices["J"]
            variabilityIndices.k = var_indices["K"]
            variabilityIndices.light_curve = lightCurve
            session.add(variabilityIndices)
            
        if len(session.new) == 1000:
            session.flush()
    session.flush()

def reComputeIndices():
    lightCurves = session.query(LightCurve).filter(LightCurve.objid < 100000).all()
    for lightCurve in lightCurves:
        if lightCurve.variability_indices == None:
            
            lc = simu.PTFLightCurve(lightCurve.mjd, lightCurve.mag, lightCurve.error)
            try:
                var_indices = simu.computeVariabilityIndices(lc)
            except NameError:
                continue
            
            variabilityIndices = VariabilityIndices()
            variabilityIndices.sigma_mu = var_indices["sigma_mu"]
            variabilityIndices.con = var_indices["con"]
            variabilityIndices.eta = var_indices["eta"]
            variabilityIndices.j = var_indices["J"]
            variabilityIndices.k = var_indices["K"]
            variabilityIndices.light_curve = lightCurve
            session.add(variabilityIndices)
            
        if len(session.new) == 1000:
            session.flush()
    session.flush()

def loadAndMatchTxtCoordinates(file, dtype=None):    
    raDecs = np.genfromtxt(file, delimiter=",")
    
    matchedTargets = []
    for raDec in raDecs:
        ra = raDec[0]
        dec = raDec[1]
        try:
            matchedTargets.append(session.query(VariabilityIndices.sigma_mu, VariabilityIndices.con, VariabilityIndices.eta, VariabilityIndices.j, VariabilityIndices.k)\
                        .join(LightCurve)\
                        .filter(LightCurve.objid < 100000)\
                        .filter(func.q3c_radial_query(LightCurve.ra, LightCurve.dec, ra, dec, 5./3600))\
                        .one())
        except sqlalchemy.orm.exc.NoResultFound:
            pass
            
    return np.array(matchedTargets, dtype=zip(NAMES, [float]*len(NAMES))).view(np.recarray)

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
    # This dictionary controls which parameter goes where in the plot grid
    _subplot_map = dict(zip(NAMES, range(5)))
    #{"sigma_mu" : 0, "con" : 1, "eta" : 2, "j" : 3, "k" : 4}
    
    def __init__(self, filename):
        self.filename = filename
        self.figure, self.subplot_array = plt.subplots(5, 5, figsize=(25,25))
        
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
            thisSubplot.set_xlabel("log({0})".format(vi_axis.x_axis_parameter))
            thisSubplot.set_ylabel("log({0})".format(vi_axis.y_axis_parameter))
        else:
            thisSubplot.set_xlabel(vi_axis.x_axis_parameter)
            thisSubplot.set_ylabel(vi_axis.y_axis_parameter)
        
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
        self.figure.savefig(self.filename)

def plotIndices():    
    # ** IF YOU CHANGE EITHER OF THESE LINES, YOU MUST ALSO CHANGE NAMES ABOVE! **
    varIndices = session.query(VariabilityIndices.sigma_mu, VariabilityIndices.con, VariabilityIndices.eta, VariabilityIndices.j, VariabilityIndices.k)\
                        .join(LightCurve)\
                        .filter(LightCurve.objid < 100000).all()
    varIndicesArray = np.array(varIndices, dtype=zip(NAMES, [float]*len(NAMES))).view(np.recarray)
    
    
    # This code finds any known rotators from Agueros et al.
    knownRotators = loadAndMatchTxtCoordinates("data/praesepe_rotators.txt")
    #wang1995Members = loadAndMatchTxtCoordinates("data/praesepe_wang1995_members.txt")
    kraus2007Members = loadAndMatchTxtCoordinates("data/praesepe_krauss2007_members.txt")
    rrLyrae = loadAndMatchTxtCoordinates("data/praesepe_rrlyrae.txt")
    eclipsing = loadAndMatchTxtCoordinates("data/praesepe_eclipsing.txt")
    wUma = loadAndMatchTxtCoordinates("data/praesepe_wuma.txt")
    
    viFigure = VIFigure("plots/praesepe_5x5.png")
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
            
            viAxis.add_series(varIndicesArray[xParameter], varIndicesArray[yParameter], color="k", marker=".", alpha=0.3, linestyle="none", label="All Praesepe Field Stars")
            #viAxis.add_series(wang1995Members[xParameter], wang1995Members[yParameter], color="g", marker="v", alpha=0.8, markersize=8, linestyle="none", label=u"Wang 1995 Member Catalog")
            viAxis.add_series(kraus2007Members[xParameter], kraus2007Members[yParameter], color="g", marker="v", alpha=0.8, markersize=8, linestyle="none", label=u"Krauss 2007 Member Catalog")
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
            if ii == jj == 4:
                viFigure.addSubplot(viAxis, legend=True)
            else:
                viFigure.addSubplot(viAxis)
                
    viFigure.save()

def _plotHighJLightCurves():
    
    lightCurves = session.query(LightCurve)\
                        .join(VariabilityIndices)\
                        .filter(LightCurve.objid < 100000)\
                        .filter(VariabilityIndices.j > 1000)\
                        .all()
    
    for lightCurve in lightCurves:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        lightCurve.plot(ax)
        fig.savefig("plots/praesepe/{0}.png".format(lightCurve.objid))

if __name__ == "__main__":
    #plotIndices()
    _plotHighJLightCurves()
    
    