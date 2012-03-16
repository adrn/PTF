# Standard library
import os, sys
from argparse import ArgumentParser
import logging

# Third party
import numpy as np
from sqlalchemy import func
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import apwlib.geometry as g

# Project
from ptf.db.DatabaseConnection import *
import simulations

def sampling_figure(prefix="plots"):
    matplotlib.rcParams["axes.labelsize"] = "xx-large"
    matplotlib.rcParams["xtick.labelsize"] = "x-large"
    matplotlib.rcParams["ytick.color"] = "w"
    
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
    plt.savefig(os.path.join(prefix, "ptf_sampling_plot.pdf"))

def indices_figure(prefix="plots"):
    matplotlib.rcParams["axes.titlesize"] = "xx-large"
    matplotlib.rcParams["axes.labelsize"] = "xx-large"
    matplotlib.rcParams["xtick.labelsize"] = "x-large"
    matplotlib.rcParams["ytick.labelsize"] = "x-large"
    
    varIndicesNoEvent, varIndicesWithEvent = simulations.simulation1(1000, 100, (100, 500))
    print varIndicesNoEvent.dtype.names
    print varIndicesWithEvent.dtype.names
    
    fig = plt.figure(figsize=(15,15))
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
    
    plt.savefig(os.path.join(prefix,"J_vs_K_Con.pdf"))


if __name__ == "__main__":
    sampling_figure()
    indices_figure()