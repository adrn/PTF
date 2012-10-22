# Standard library
import os, sys
from argparse import ArgumentParser
import logging
import cPickle as pickle

# Third party
import numpy as np
from sqlalchemy import func
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import apwlib.geometry as g

# Project
from ptf.db.DatabaseConnection import *
import simulations

def sampling_figure(prefix="plots"):
    matplotlib.rcParams["axes.labelsize"] = 40
    matplotlib.rcParams["xtick.labelsize"] = 26
    
    xmin = 0.
    radecs = []
    level = 0.
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    for ii in range(100):
        lightCurves = session.query(LightCurve).filter(func.array_length(LightCurve.mjd, 1) > 100).limit(100).all()
        
        if level >= 10: break
        for lc in lightCurves:
            if (max(lc.mjd) - min(lc.mjd)) > 365:
                next_lc = False
                for radec in radecs:
                    try:
                        sep = g.subtends(lc.ra, lc.dec, radec[0], radec[1], units="degrees")
                    except ValueError:
                        next_lc = True
                    if sep < 5.:
                        next_lc = True
                
                if next_lc: continue
                else: radecs.append((lc.ra, lc.dec))
                    
                
                #ax.errorbar(lc.Rmjd, lc.Rmag, lc.Rerror, ls='none', marker='o', c='k', ecolor='0.7', capsize=0)
                ax.plot(lc.Rmjd, [level]*len(lc.Rmjd), marker='o', ms=20.0, c='k', ls='none', alpha=0.3)
                
                if xmin == 0:
                    xmin = min(lc.Rmjd)
                    ax.set_xlim(xmin, xmin+365)
                
                level += 1
    
    ymin, ymax = ax.get_ylim()
    ydel = ymax-ymin
    ax.set_ylim(ymin-ydel/10., ymax+ydel/10.)
    ax.set_xlabel("MJD")
    ax.set_yticklabels([])
    plt.savefig(os.path.join(prefix, "ptf_sampling_figure.pdf"))

def indices_figure(prefix="plots", overwrite=False):    
    dataFile = "data/indices_figure.pickle"
    
    if overwrite:
        os.remove(dataFile)
    
    if not os.path.exists(dataFile):
        varIndicesNoEvent, varIndicesWithEvent = simulations.simulation1(1000, 100, (100, 500))
        print len(varIndicesNoEvent)
        
        f = open(dataFile, "w")
        pickle.dump((varIndicesNoEvent, varIndicesWithEvent), f)
        f.close()
    
    f = open(dataFile, "r")
    varIndicesNoEvent, varIndicesWithEvent = pickle.load(f)
    f.close()
    
    matplotlib.rcParams["axes.titlesize"] = 40
    matplotlib.rcParams["axes.labelsize"] = 40
    matplotlib.rcParams["xtick.labelsize"] = 26
    matplotlib.rcParams["ytick.labelsize"] = 26
    
    fig = plt.figure(figsize=(20,25))
    fig.subplots_adjust(wspace=0.1, hspace=0.1, left=0.1)
    
    ax1 = fig.add_subplot(221)
    ax1.set_ylabel("Con")
    ax1.set_title("No Event")
    ax1.plot(varIndicesNoEvent["J"], varIndicesNoEvent["Con"], c='b', marker=".", ls='none', alpha=0.2)
    
    ax2 = fig.add_subplot(222)
    ax2.set_title("With Event")
    ax2.plot(varIndicesWithEvent["J"], varIndicesWithEvent["Con"], c='r', marker=".", ls='none', alpha=0.2)
    
    ax3 = fig.add_subplot(223)
    ax3.set_xlabel("J")
    ax3.set_ylabel("K")
    ax3.plot(varIndicesNoEvent["J"], varIndicesNoEvent["K"], c='b', marker=".", ls='none', alpha=0.2)
    
    ax4 = fig.add_subplot(224)
    ax4.set_xlabel("J")
    ax4.plot(varIndicesWithEvent["J"], varIndicesWithEvent["K"], c='r', marker=".", ls='none', alpha=0.2)
    
    xmin = min(min(varIndicesWithEvent["J"]), min(varIndicesNoEvent["J"]))
    xmax = max(max(varIndicesWithEvent["J"]), max(varIndicesNoEvent["J"]))
    xdel = (xmax-xmin)
    
    ymin_Con = min(min(varIndicesNoEvent["Con"]), min(varIndicesWithEvent["Con"]))
    ymax_Con = max(max(varIndicesNoEvent["Con"]), max(varIndicesWithEvent["Con"]))
    ydel_Con = ymax_Con-ymin_Con
    
    ymin_K = min(min(varIndicesNoEvent["K"]), min(varIndicesWithEvent["K"]))
    ymax_K = max(max(varIndicesNoEvent["K"]), max(varIndicesWithEvent["K"]))
    ydel_K = ymax_K-ymin_K
    
    ax1.set_xlim(xmin-xdel/10., xmax+xdel/10.)
    ax2.set_xlim(xmin-xdel/10., xmax+xdel/10.)
    ax3.set_xlim(xmin-xdel/10., xmax+xdel/10.)
    ax4.set_xlim(xmin-xdel/10., xmax+xdel/10.)
    
    ax1.set_ylim(ymin_Con-ydel_Con/10., ymax_Con+ydel_Con/10.)
    ax2.set_ylim(ymin_Con-ydel_Con/10., ymax_Con+ydel_Con/10.)
    
    ax3.set_ylim(ymin_K-ydel_K/10., ymax_K+ydel_K/10.)
    ax4.set_ylim(ymin_K-ydel_K/10., ymax_K+ydel_K/10.)
    
    # Remove extraneous tick labels
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax4.set_yticklabels([])
    
    plt.savefig(os.path.join(prefix,"J_vs_K_Con_figure.png"), dpi=300)

def detection_efficiency_figure(prefix="plots"):
    matplotlib.rcParams["axes.titlesize"] = 40
    matplotlib.rcParams["axes.labelsize"] = 40
    matplotlib.rcParams["xtick.labelsize"] = 26
    matplotlib.rcParams["ytick.labelsize"] = 26
    matplotlib.rcParams["legend.fontsize"] = 32
    
    uniformData = simulations.detection_efficiency(10000, pattern="uniform")
    randomData = simulations.detection_efficiency(10000, pattern="random")
    clumpyData = simulations.detection_efficiency(10000, pattern="clumpy")
    
    plt.figure(figsize=(20,20))
    plt.semilogx(uniformData.observations, uniformData.detection_fractions, 'k+', ls='none', label="Uniform Observations", ms=20.)
    plt.semilogx(uniformData.observations, uniformData.detection_fractions, 'k-', alpha=0.4)
    plt.semilogx(randomData.observations, randomData.detection_fractions, 'ro', ls='none', label="Randomized Observations", ms=10.)
    plt.semilogx(randomData.observations, randomData.detection_fractions, 'r-', alpha=0.4)
    plt.semilogx(clumpyData.observations, clumpyData.detection_fractions, 'g*', ls='none', label="Clumpy Observations", ms=20.)
    plt.semilogx(clumpyData.observations, clumpyData.detection_fractions, 'g-', alpha=0.4)
    plt.xlabel(r"Number of Observations")
    plt.ylabel("Detection Fraction")
    plt.legend(loc=5)
    plt.savefig(os.path.join(prefix,"detection_efficiency_figure.pdf"))

def bad_data_figure(prefix="plots"):
    matplotlib.rcParams["axes.titlesize"] = 40
    matplotlib.rcParams["axes.labelsize"] = 40
    matplotlib.rcParams["xtick.labelsize"] = 20
    matplotlib.rcParams["ytick.labelsize"] = 20
    
    lightCurve1 = session.query(LightCurve).filter(LightCurve.objid == 14688560553413090141).one()
    lightCurve2 = session.query(LightCurve).filter(LightCurve.objid == 14688560553413090134).one()
    
    print "Distance between objects: {0}\"".format(g.subtends(lightCurve1.ra, lightCurve1.dec, lightCurve2.ra, lightCurve2.dec, units="degrees")*3600.)
    
    medMag1 = np.median(lightCurve1.mag)
    medMag2 = np.median(lightCurve2.mag)
    
    fig = plt.figure(figsize=(20,25))
    fig.subplots_adjust(hspace=0.2, left=0.1)
    ax1 = fig.add_subplot(211)
    ax1.errorbar(lightCurve1.mjd, lightCurve1.mag, lightCurve1.error, ls='none', marker='o', ecolor='0.7', capsize=0)
    ax2 = fig.add_subplot(212)
    ax2.errorbar(lightCurve2.mjd, lightCurve2.mag, lightCurve2.error, ls='none', marker='o', ecolor='0.7', capsize=0)
    
    ax1.set_ylim(medMag1-0.5,medMag1+0.5)
    ax1.set_ylim(ax1.get_ylim()[::-1])
    ax1.set_xlim(55000, 56000)
    ax1.set_ylabel(r"$R$")
    ax1.set_xticklabels([])
    ax1.set_title("objid: {0}".format(lightCurve1.objid))
    
    ax2.set_ylim(medMag2-0.5,medMag2+0.5)
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_xlim(55000, 56000)
    ax2.set_ylabel(r"$R$")
    ax2.set_xlabel("MJD")
    ax2.set_title("objid: {0}".format(lightCurve2.objid))
    
    plt.savefig(os.path.join(prefix, "bad_data_figure.pdf"))
    
    return
    
    # Code below is to search for light curves that are similar
    while True:
        lightCurve1 = session.query(LightCurve).filter(func.array_length(LightCurve.mjd, 1) > 200).order_by(func.random()).limit(1).one()
        lightCurve2 = session.query(LightCurve).filter(func.array_length(LightCurve.mjd, 1) > 200).\
                                                filter(func.q3c_radial_query(LightCurve.ra, LightCurve.dec, lightCurve1.ra, lightCurve1.dec, 60./3600.)).\
                                                order_by(func.random()).limit(1).one()
        
        medMag1 = np.median(lightCurve1.mag)
        medMag2 = np.median(lightCurve2.mag)
        if medMag1 > 17 or medMag2 > 17: continue
        
        fig = plt.figure(figsize=(20,25))
        fig.subplots_adjust(hspace=0.3, left=0.1)
        ax1 = fig.add_subplot(211)
        ax1.errorbar(lightCurve1.mjd, lightCurve1.mag, lightCurve1.error, ls='none', marker='o', ecolor='0.7', capsize=0)
        ax2 = fig.add_subplot(212)
        ax2.errorbar(lightCurve2.mjd, lightCurve2.mag, lightCurve2.error, ls='none', marker='o', ecolor='0.7', capsize=0)
        
        ax1.set_ylim(medMag1-0.5,medMag1+0.5)
        ax1.set_ylim(ax1.get_ylim()[::-1])
        ax1.set_xlim(55000, 56000)
        ax1.set_ylabel(r"$R$")
        ax1.set_xticklabels([])
        ax1.set_title("objid: {0}".format(lightCurve1.objid))
        
        ax2.set_ylim(medMag2-0.5,medMag2+0.5)
        ax2.set_ylim(ax2.get_ylim()[::-1])
        ax2.set_xlim(55000, 56000)
        ax2.set_ylabel(r"$R$")
        ax2.set_xlabel("MJD")
        ax2.set_title("objid: {0}".format(lightCurve2.objid))
        
        plt.show()
    
if __name__ == "__main__":
    #sampling_figure()
    #indices_figure()
    detection_efficiency_figure()
    #bad_data_figure()