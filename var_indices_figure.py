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

class VIFigure(object):
    """ This class represents a variability index (VI) figure where we plot the distributions
        of the variability indices against each other.
    """
    
    def __init__(self, variability_indices, indices, scatter=True, **figure_kwargs):
        """ Given a numpy structured array containing variability index values, 
            initialize a new variability index figure
            
            Parameters
            ----------
            variability_indices : numpy structured array
                This must be a numpy structured array with at minimum the following columns:
                    - one column for each index in 'indices',
                    - m (the reference magnitude of the source),
                    - u0 (the impact parameter of any microlensing event),
                    - tE (the microlensing event timescale)
            indices : iterable
                Probably a list of strings, e.g. ['eta', 'j', 'k'] etc.
            scatter: bool
                Make a scatter plot. If False, it will make a contour plot instead.
            
            figure_kwargs are any keyword arguments that will get passed through to the matplotlib figure function.
        """
        
        # Validate input
        for var_idx in indices:
            try:
                variability_indices[var_idx]
            except ValueError: # e.g. field not found
                print redText("Variability statistic '{}' not found in the data you provided.".format(var_idx))
                raise
        
        self.variability_indices = variability_indices
        self.indices = indices
        
        self.figure, self._mpl_axes = plt.subplots(len(self.indices), len(self.indices), **figure_kwargs)
        
        # Here I create a dictionary that makes it easier to keep track of where the indices are 
        #   plotted, so I don't have to remember numeric values. To access the axes objects, you
        #   use the format: vifigure.axes[x axis name][y axis name], for example if I wanted the 
        #   subplot with 'eta' on the x-axis and 'con' on the y-axis, I would do:
        #       vifigure.axes["eta"]["con"]
        self.axes = dict()
        for ii, x_idx in enumerate(self.indices):
            self.axes[x_idx] = dict()
            
            for jj, y_idx in enumerate(self.indices):
                self.axes[x_idx][y_idx] = self._mpl_axes[jj, ii]
        
        # Now we will do the initial plotting of the points
        for x_idx in self.indices:
            for y_idx in self.indices:
                this_axis = self.axes[x_idx][y_idx]
                
                if scatter:
                    this_axis.scatter(self.variability_indices[x_idx], self.variability_indices[y_idx])
                else:
                    H, xedges, yedges = np.histogram2d(self.variability_indices[x_idx], self.variability_indices[y_idx], bins=(100,100))
                    levels = (1E4, 1E3, 1E2, 20)
                    this_axis.countour(H, levels)

    def color_points_by(self, param):
        """ Color the points by the specified parameter (can be u0, tE, or m) """
        # USE ax.collections[0].set_color() to change color of points in scatter plot
        
    def reset_color(self):
        """ Reset the color of the points to black """
    
    def save(self, filename):
        """ Save the figure to the specified filename """



def make_var_indices_plots(var_indices, var_indices_with_events, indices, filename_base):
    """ """
    
    num_indices = len(indices)
    fig1, axes1 = plt.subplots(num_indices, num_indices, figsize=(20,20))
    fig2, axes2 = plt.subplots(num_indices, num_indices, figsize=(20,20))
    
    var_indices["con"] += 1
    var_indices_with_events["con"] += 1
    
    for ii, row_index in enumerate(indices):
        for jj, col_index in enumerate(indices):
            ax_without = axes1[ii, jj]
            ax_with = axes2[ii, jj]
            
            if ii < jj: 
                ax_without.set_visible(False) 
                ax_with.set_visible(False) 
                continue
            
            if col_index in ["eta", "con", "k"]:
                xscale = "log"
            else:
                xscale = "symlog"
            
            if row_index in ["eta", "con", "k"]:
                yscale = "log"
            else:
                yscale = "symlog"
            
            if ii != jj:
                ax_without.plot(var_indices[col_index], var_indices[row_index], color='black', marker='.', linestyle="none", alpha=0.1, ms=3)
                ax_without.set_xscale(xscale)
                ax_without.set_yscale(yscale)
                
                ax_with.plot(var_indices_with_events[col_index], var_indices_with_events[row_index], color='black', marker='.', linestyle="none", alpha=0.1, ms=3)
                ax_with.set_xscale(xscale)
                ax_with.set_yscale(yscale)
                
                xlims_without = (min(var_indices[col_index]), max(var_indices[col_index]))
                xlims_with = (min(var_indices_with_events[col_index]), max(var_indices_with_events[col_index]))
                new_xmin = min(xlims_without[0], xlims_with[0])
                if new_xmin == 0: new_xmin = -0.01
                new_xmax = max(xlims_without[1], xlims_with[1])
                new_xlim = (new_xmin - 0.2*np.fabs(new_xmin), new_xmax + 0.2*np.fabs(new_xmax))
                
                ax_without.set_xlim(new_xlim)
                ax_with.set_xlim(new_xlim)
                
                #ylims_without = ax_without.get_ylim()
                #ylims_with = ax_with.get_ylim()
                ylims_without = (min(var_indices[row_index]), max(var_indices[row_index]))
                ylims_with = (min(var_indices_with_events[row_index]), max(var_indices_with_events[row_index]))
                new_ymin = min(ylims_without[0], ylims_with[0])
                if new_ymin == 0: new_ymin = -0.01
                new_ymax = max(ylims_without[1], ylims_with[1])
                new_ylim = (new_ymin - 0.2*np.fabs(new_ymin), new_ymax + 0.2*np.fabs(new_ymax))
                
                ax_without.set_ylim(new_ylim)
                ax_with.set_ylim(new_ylim)
                
            else:
                log_idxs_without = np.log10(var_indices[col_index])
                log_idxs_with = np.log10(var_indices_with_events[col_index])
                
                bins_without = np.logspace(log_idxs_without.min(), log_idxs_without.max(), 100)
                bins_with = np.logspace(log_idxs_with.min(), log_idxs_with.max(), 100)
                
                #ax_without.hist(var_indices[col_index], color='black', alpha=0.5, bins=bins_without, log=True) 
                #ax_with.hist(var_indices_with_events[col_index], color='black', alpha=0.5, bins=bins_with, log=True) 
                #ax_without.hist(var_indices[col_index], color='black', alpha=0.5, bins=100, log=True) 
                #ax_with.hist(var_indices_with_events[col_index], color='black', alpha=0.5, bins=100, log=True) 
                ax_without.set_xscale("symlog")
                ax_with.set_xscale("symlog")
            
            plt.setp(ax_without.get_xticklabels(), visible=False)
            plt.setp(ax_without.get_yticklabels(), visible=False)
            
            plt.setp(ax_with.get_xticklabels(), visible=False)
            plt.setp(ax_with.get_yticklabels(), visible=False)
            
            if ii == num_indices-1:
                plt.setp(ax_without.get_xticklabels(), visible=True)
                ax_without.set_xlabel(col_index)
                plt.setp(ax_with.get_xticklabels(), visible=True)
                ax_with.set_xlabel(col_index)
            
            if jj == 0:
                plt.setp(ax_without.get_yticklabels(), visible=True)
                ax_without.set_ylabel(row_index)
                plt.setp(ax_with.get_yticklabels(), visible=True)
                ax_with.set_ylabel(row_index)
    
    fig1.subplots_adjust(hspace=0.1, wspace=0.1)
    fig2.subplots_adjust(hspace=0.1, wspace=0.1)
    
    fig1.savefig(filename_base + "_without.png")
    fig2.savefig(filename_base + "_with.png")


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
    parser.add_argument("--test", dest="test", action="store_true", default=False,
                    help="Run tests")
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
    
    vifigure = VIFigure(var_indices, indices=args.indices, scatter=False, figsize=(15,15))
    plt.show()