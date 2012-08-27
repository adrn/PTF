# coding: utf-8
from __future__ import division

""" This module contains routines for generating the variability indices figures, with and 
    without simulated events
"""

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os, sys
import logging
import time
import cPickle as pickle
import json

# Third-party
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
from apwlib.globals import greenText, redText

# Project
import ptf.photometricdatabase as pdb
import detectionefficiency as de
import ptf.analyze.analyze as analyze
from ptf.globals import index_to_label

# Create logger for this module
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

def get_var_indices(field, light_curves_per_ccd, events_per_light_curve, indices, overwrite=True):
    """ This function will create a len(indices) x len(indices) plot grid and plot
        distributions of all of the variability indices
    """

    limiting_mags = [14.3, 21]
    
    file_base = "field{:06d}_Nperccd{}_Nevents{}".format(field.id, light_curves_per_ccd, events_per_light_curve) + ".{ext}"
    pickle_filename = os.path.join("data", "var_indices", file_base.format(ext="pickle"))
    plot_filename = os.path.join("plots", "var_indices", file_base.format(ext="png"))
    
    if not os.path.exists(os.path.dirname(pickle_filename)):
        os.mkdir(os.path.dirname(pickle_filename))
    
    if not os.path.exists(os.path.dirname(plot_filename)):
        os.mkdir(os.path.dirname(plot_filename))

    if os.path.exists(pickle_filename) and overwrite:
        logger.debug("Data file exists, but you want to overwrite it!")
        os.remove(pickle_filename)
        logger.debug("Data file deleted...")

    # If the cache pickle file doesn't exist, generate the data
    if not os.path.exists(pickle_filename):
        logger.info("Data file {} not found. Generating data...".format(pickle_filename))
        
        # Conditions for reading from the 'sources' table
        #   - Only select sources with enough good observations (>25)
        wheres = ["(ngoodobs > 25)"]
        
        for ccd in field.ccds.values():
            logger.info(greenText("Starting with CCD {}".format(ccd.id)))
            
            # Get the chip object for this CCD
            chip = ccd.read()
            
            for ii, limiting_mag in enumerate(limiting_mags[:-1]):
                # Define bin edges for selection on reference magnitude
                limiting_mag1 = limiting_mag
                limiting_mag2 = limiting_mags[ii+1]
                mag_key = (limiting_mag1,limiting_mag2)
                logger.info("\tMagnitude range: {:.2f} - {:.2f}".format(limiting_mag1, limiting_mag2))
                    
                read_wheres = wheres + ["(referenceMag >= {:.3f})".format(limiting_mag1)]
                read_wheres += ["(referenceMag < {:.3f})".format(limiting_mag2)]
                
                # Read information from the 'sources' table
                sources = chip.sources.readWhere(" & ".join(read_wheres))
                #source_ids = sources["matchedSourceID"]
                source_idxs = range(len(sources))
                
                # Randomly shuffle the sources
                np.random.shuffle(source_idxs)
                logger.info("\t\tSelected {} source ids".format(len(sources)))
                
                dtype = zip(indices + ["tE", "u0", "m", "event_added"], [float]*len(indices) + [float, float, float, bool])
                count = 0
                good_source_ids = []
                for source_idx in source_idxs:
                    source = sources[source_idx]
                    source_id = source["matchedSourceID"]
                    
                    logger.debug("\t\t\tSource ID: {}".format(source_id))
                    light_curve = ccd.light_curve(source_id, clean=True, barebones=True)
                    
                    # After quality cut, if light curve has less than 25 observations, skip it!
                    if len(light_curve.mjd) < 25:
                        continue
                    
                    these_var_indices = np.array([analyze.compute_variability_indices(light_curve, indices, return_tuple=True) + (None, None, np.median(light_curve.mag), False)], dtype=dtype)
                    try:
                        var_indices = np.hstack((var_indices, these_var_indices))
                    except NameError:
                        var_indices = these_var_indices
                    
                    these_indices = de.simulate_events_compute_indices(light_curve, events_per_light_curve=events_per_light_curve, indices=indices)
                    try:
                        var_indices_with_events = np.hstack((var_indices_with_events, these_indices))
                    except NameError:
                        var_indices_with_events = these_indices
                    
                    good_source_ids.append(source_id)
                    count += 1
                    if count >= light_curves_per_ccd and light_curves_per_ccd != 0: break
                
                if len(good_source_ids) == 0:
                    logger.error("No good sources selected from this CCD for mag range {:.2f}-{:.2f}!".format(limiting_mag1, limiting_mag2))
                    continue
                
                logger.info("\t\t{} good light curves selected".format(count))
                
            ccd.close()
        
        with open(pickle_filename, "w") as f:
            pickle.dump((var_indices, var_indices_with_events), f)
            
    else:
        logger.info("Data file {} already exists".format(pickle_filename))
    
    logger.debug("\t\tReading in data file...")
    f = open(pickle_filename, "r")
    var_indices, var_indices_with_events = pickle.load(f)
    f.close()
    
    return var_indices, var_indices_with_events





#########
class Encoder(json.JSONEncoder):
    def default(self, obj):           
        if obj == False:
            return 0
        else:
            return 1

        
        return json.JSONEncoder.default(self, obj)

def var_indices_to_json(var_indices, var_indices_with_events, filename=None):
    """ Converts variability indices with and without simulated events into a JSON structure to be
        fed into Crossfilter:
            https://github.com/square/crossfilter/wiki/API-Reference
    """
    
    var_indices["u0"][np.logical_not(np.isfinite(var_indices["u0"]))] = 0.0
    var_indices["tE"][np.logical_not(np.isfinite(var_indices["tE"]))] = 0.0
    var_indices_with_events["tE"][np.logical_not(np.isfinite(var_indices_with_events["tE"]))] = 0.0
    var_indices_with_events["u0"][np.logical_not(np.isfinite(var_indices_with_events["u0"]))] = 0.0
    
    json_blob = dict()
    json_blob["var_indices"] = [dict(zip(row.dtype.names, row)) for row in var_indices]
    json_blob["var_indices_with_events"] = [dict(zip(row.dtype.names, row)) for row in var_indices_with_events]
    
    blob = json.dumps(json_blob, cls=Encoder)
    
    if filename != None:
        f = open(filename, "wb")
        f.write(blob)
        f.close()
        
        return
    else:
        return blob
##############

import scipy.optimize as so
def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

class VIFigure(object):
    """ This class represents a variability index (VI) figure where we plot the distributions
        of the variability indices against each other.
    """
    
    def _validate_variability_indices(self, variability_indices):
        """ Validate any variability indices given to this object """
        
        # Validate input
        for var_idx in self.indices:
            try:
                variability_indices[var_idx]
            except ValueError: # e.g. field not found
                print redText("Variability statistic '{}' not found in the data you provided.".format(var_idx))
                raise
        
        return True
    
    def __init__(self, indices, **figure_kwargs):
        """ Given a numpy structured array containing variability index values, 
            initialize a new variability index figure
            
            Parameters
            ----------
            indices : iterable
                Probably a list of strings, e.g. ['eta', 'j', 'k'] etc.
            scatter: bool
                Create a scatter plot instead of a contour plot
            
            figure_kwargs are any keyword arguments that will get passed through to the matplotlib figure function.
        """
        
        self.indices = indices
        
        self.figure, self._mpl_axes = plt.subplots(len(self.indices), len(self.indices), **figure_kwargs)
        self._scatter_data_bounds = dict()
        
        # Here I create a dictionary that makes it easier to keep track of where the indices are 
        #   plotted, so I don't have to remember numeric values. To access the axes objects, you
        #   use the format: vifigure.axes[x axis name][y axis name], for example if I wanted the 
        #   subplot with 'eta' on the x-axis and 'con' on the y-axis, I would do:
        #       vifigure.axes["eta"]["con"]
        self.axes = dict()
        for ii, x_idx in enumerate(self.indices):
            self.axes[x_idx] = dict()

            for jj, y_idx in enumerate(self.indices):
                if jj < ii: 
                    self._mpl_axes[jj, ii].set_visible(False)
                    continue
                    
                self.axes[x_idx][y_idx] = self._mpl_axes[jj, ii]
    
    def scatter(self, variability_indices, **kwargs):
        """ Add scatter plots to each subplot with the supplied array of variability indices
        
            Parameters
            ---------
            variability_indices : numpy structured array
                This must be a numpy structured array with at minimum the following columns:
                    - one column for each index in 'indices',
                    - m (the reference magnitude of the source),
                    - u0 (the impact parameter of any microlensing event),
                    - tE (the microlensing event timescale)
        """
        self._validate_variability_indices(variability_indices)
        
        alpha = kwargs.pop("alpha", 0.1)
        marker = kwargs.pop("marker", ".")
        markersize = kwargs.pop("markersize", 2)
        color = kwargs.pop("color", "#222222")
        
        # Now we will do the initial plotting of the points
        for x_idx in self.axes.keys():
            for y_idx in self.axes[x_idx].keys():
                this_axis = self.axes[x_idx][y_idx]
                if x_idx == y_idx: 
                    # Make a histogram
                    x_var = variability_indices[x_idx]
                    neg_x_var = x_var[x_var < 0.0]
                    pos_x_var = x_var[x_var > 0.0]
                    log_x_var = np.log10(pos_x_var)
                    
                    this_axis.hist(log_x_var, bins=100, color=color)
                    
                else:
                    x_var = variability_indices[x_idx]
                    y_var = variability_indices[y_idx]
                    
                    neg_x_var = x_var[x_var < 0.0]
                    pos_x_var = x_var[(x_var > 0.0) & (y_var > 0.0)]
                    log_x_var = np.log10(pos_x_var)
                    
                    neg_y_var = y_var[y_var < 0.0]
                    pos_y_var = y_var[(x_var > 0.0) & (y_var > 0.0)]
                    log_y_var = np.log10(pos_y_var)
                    
                    this_axis.scatter(log_x_var, log_y_var, zorder=-1, alpha=alpha, marker=marker, color=color, s=markersize)
                    self._scatter_data_bounds[this_axis] = (min(log_x_var), max(log_x_var), min(log_y_var), max(log_y_var))
    
    def contour(self, variability_indices, nbins=100, nlevels=2, **kwargs):
        """ Add contour plots to each subplot with the supplied array of variability indices
        
            Parameters
            ---------
            variability_indices : numpy structured array
                This must be a numpy structured array with at minimum the following columns:
                    - one column for each index in 'indices',
                    - m (the reference magnitude of the source),
                    - u0 (the impact parameter of any microlensing event),
                    - tE (the microlensing event timescale)
        """ 
        self._validate_variability_indices(variability_indices)
        
        if nlevels > 3: raise ValueError("nlevels must be <=3")
        
        alpha = kwargs.pop("alpha", 0.75)
        colors = kwargs.pop("colors", ["#d44a4c", "#91e685", "#459bd0"])
        linestyles = kwargs.pop("linestyles", ["-", "-", "-"])
        
        # Now we will do the initial plotting of the points
        for x_idx in self.axes.keys():
            for y_idx in self.axes[x_idx].keys():
                if x_idx == y_idx: continue
                this_axis = self.axes[x_idx][y_idx]
                
                x_var = variability_indices[x_idx]
                y_var = variability_indices[y_idx]
                
                neg_x_var = x_var[x_var < 0.0]
                pos_x_var = x_var[(x_var > 0.0) & (y_var > 0.0)]
                log_x_var = np.log10(pos_x_var)
                
                neg_y_var = y_var[y_var < 0.0]
                pos_y_var = y_var[(x_var > 0.0) & (y_var > 0.0)]
                log_y_var = np.log10(pos_y_var)
                
                H, xedges, yedges = np.histogram2d(log_x_var, log_y_var, bins=(nbins,nbins), normed=True)
                x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins))
                y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins,1))
                pdf = (H*(x_bin_sizes*y_bin_sizes))
                
                level1 = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68))
                level2 = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95))
                level3 = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.99))
                levels = [level1,level2,level3][:nlevels]
                
                X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])
                #Z = np.log10(H.T+1.)
                Z = pdf.T
                #print levels
                #sys.exit(0)
                
                contour = this_axis.contour(X, Y, Z, levels=levels, origin="lower", colors=colors, alpha=alpha)
                for ii, c in enumerate(contour.collections): 
                    c.set_linestyle(linestyles[ii])
        
    def beautify(self):
        """ 1) Align the x-axis for axes in a column, y-axis for rows
            2) Get rid of axis labels for all but the outside subplots
            3) Adjust the subplot spacing
            4) Add the names of the indices
        """
        
        for ii,row_idx in enumerate(self.indices):
            for jj,col_idx in enumerate(self.indices):
                
                plt.setp(self._mpl_axes[ii,jj].get_xticklabels(), visible=False)
                plt.setp(self._mpl_axes[ii,jj].get_yticklabels(), visible=False)
                
                if ii == len(self.indices)-1:
                    plt.setp(self._mpl_axes[ii,jj].get_xticklabels(), visible=True)
                    self._mpl_axes[ii,jj].set_xlabel(r"$\log$"+index_to_label[col_idx], fontsize=28)
                
                if jj == 0 and ii > 0:
                    plt.setp(self._mpl_axes[ii,jj].get_yticklabels(), visible=True)
                    self._mpl_axes[ii,jj].set_ylabel(r"$\log$"+index_to_label[row_idx], fontsize=28)
                    
        
        self.align_axis_limits()
        self.figure.subplots_adjust(wspace=0.1, hspace=0.1)
    
    def align_axis_limits(self):
        """ This function will go through and make sure each column shares an x axis and 
            each row shares a y axis.
            
        """
        
        pad = 0.1
        for row in range(len(self.indices)):
            xlims = []
            for col in range(len(self.indices)):
                ax = self._mpl_axes[col,row]
                try:
                    xlims.append(self._scatter_data_bounds[ax][:2])
                except: 
                    pass
            
            xlims = np.array(xlims, dtype=[("l", float), ("r", float)])
            for col in range(len(self.indices)):
                try:
                    this_min = xlims["l"].min()
                    this_max = xlims["r"].min()
                    delta = np.fabs(this_max-this_min)
                    self._mpl_axes[col,row].set_xlim(this_min-pad*delta, this_max+pad*delta)
                except: 
                    pass
        
        for col in range(len(self.indices)):
            ylims = []
            for row in range(len(self.indices)):
                ax = self._mpl_axes[col,row]
                try:
                    ylims.append(self._scatter_data_bounds[ax][2:])
                except: 
                    pass
            
            ylims = np.array(ylims, dtype=[("b", float), ("t", float)])
            for row in range(len(self.indices)):
                if row == col: continue
                try:
                    this_min = xlims["b"].min()
                    this_max = xlims["t"].min()
                    delta = np.fabs(this_max-this_min)
                    self._mpl_axes[col,row].set_ylim(this_min-pad*delta, this_max+pad*delta)
                except: 
                    pass
        
                
    
    def save(self, filename):
        """ Save the figure to the specified filename """
        self.figure.savefig(filename, bbox_inches="tight")

if __name__ == "__main__":
    
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite any existing / cached files")
    parser.add_argument("-N", "--N", dest="N", default=100, type=int,
                    help="The number of microlensing events to be added to each light curve")
    parser.add_argument("-f", "--field-id", dest="field_id", default=None, type=int,
                    help="The PTF field ID to run on")
    parser.add_argument("--limit", dest="limit", default=None, type=int,
                    help="The number of light curves to select from each CCD in a field")
    parser.add_argument("--plot", dest="plot", action="store_true", default=False,
                    help="Plot and save the figure")
    parser.add_argument("--indices", dest="indices", nargs="+", type=str, default=["eta","delta_chi_squared", "con","j","k","sigma_mu"],
                    help="Specify the variability indices to compute")
    parser.add_argument("--mag", dest="limiting_mag", nargs="+", type=float, default=[None, None],
                    help="Specify the magnitude bin edges, e.g. 6 bin edges specifies 5 bins.")
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    np.random.seed(42)
    field = pdb.Field(args.field_id, filter="R")
    
    var_indices, var_indices_with_events = get_var_indices(field, indices=args.indices, \
                                               events_per_light_curve=args.N,
                                               light_curves_per_ccd=args.limit,
                                               overwrite=args.overwrite)
    
    if args.plot:
        vifigure = VIFigure(indices=args.indices, figsize=(22,22))
        
        vifigure.scatter(var_indices_with_events, alpha=0.1)
        vifigure.contour(var_indices, nbins=50)
        vifigure.beautify()
        plot_path = os.path.join("plots", "var_indices")
        vifigure.save(os.path.join(plot_path, "field{}_Nperccd{}_Nevents{}.pdf".format(field.id, args.limit, args.N)))