# coding: utf-8
from __future__ import division

""" This module contains utilities for computing PTF's survey coverage, both in time and position. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import re
import glob
import json
import math

# Third-party
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tables
import apwlib.geometry as g

# Project
import ptf.db.photometricdatabase as pdb
from ptf.globals import camera_size_degrees, camera_size_radius, all_fields
import ptf.coverage as pc
import ptf.util as pu
logger = pu.get_logger("survey_coverage")

class CoveragePlot(object):
    
    def __init__(self, projection=None, scaler=None, color_map=None, **fig_kw):
        """ Create a PTF Survey Coverage plot. Any extra parameters passed in at instantiation 
            are passed through to the matplotlib "figure()" function.
            
            Parameters
            ----------
            projection : str
                A valid Matplotlib projection string, to be passed in to add_subplot()
            scaler : matplotlib.colors.scaler
                Can pass in a color scaler
        """
        self.figure = plt.figure(**fig_kw)
        self.projection = projection
        
        if projection == None:
            self.axis = self.figure.add_subplot(111)
        else:
            self.axis = self.figure.add_subplot(111, projection=projection)
        
        self._scaler = scaler
        self.color_map = color_map
            
        self.fields = []
        self._field_kwargs = []
    
    def add_field(self, field, **kwargs):
        """ """
        self.fields.append(field)
        self._field_kwargs.append(kwargs)

    def _make_scaler(self):
        """ """
        if self._scaler == None:
            #create the scaler on the fly
            num_exposures = np.array([field.number_of_exposures for field in self.fields if field.id != 101001])
            return mc.LogNorm(vmin=num_exposures.min(), vmax=num_exposures.max(), clip=True)
        else:
            return self._scaler
    
    def _make_plot(self):
        """ """
        self.axis.clear()
        scaler = self._make_scaler()
        
        logger.debug("{} fields on this coverage plot".format(len(self.fields)))
        
        sorted_field_indices = np.argsort([f.number_of_exposures for f in self.fields])
        for idx in sorted_field_indices:
            # Sort the fields by number of observations, so the fields with more observations appear on top
            field = self.fields[idx]
            kwargs = self._field_kwargs[idx]
            
            # Compute the size of the field in the horizontal and vertical directions
            try:
                xsize = camera_size_degrees[0] / np.cos(field.dec.radians)
                ysize = camera_size_degrees[1]
            except AttributeError:
                # The field has no 'dec'
                continue
            
            # Compute the coordinates of the top left corner of the rectangle
            rec_x1 = ((field.ra.degrees + (camera_size_degrees[0] / np.cos(field.dec.radians)) / 2.) % 360.) * -1 + 180 # degrees
            rec_y1 = field.dec.degrees - camera_size_degrees[1] / 2. # degrees
            
            if kwargs.has_key("alpha"):
                alpha = kwargs["alpha"]
                del kwargs["alpha"]
            else:
                alpha = None
            
            if kwargs.has_key("color"):
                color = kwargs["color"]
                del kwargs["color"]
            else:
                # TODO: use scaler to get the color of this field
                if self.color_map == None:
                    color = (0.,0.,0.,)
                    if alpha == None:
                        alpha = scaler(field.number_of_exposures)*0.5 + 0.1
                else:
                    self.color_map(scaler(field.number_of_exposures))
            
            if alpha == None:
                alpha = 0.2
            
            rec = Rectangle((np.radians(rec_x1), np.radians(rec_y1)), \
                            -np.radians(xsize), np.radians(ysize), \
                            color=color, alpha=alpha, **kwargs)
                
            rec.set_edgecolor("none")
            self.axis.add_patch(rec)
        
        if self.projection in ["aitoff", "hammer", "mollweide"]:
            self.axis.set_xticklabels([330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30])
        
        if self.projection == "rectilinear":
            self.axis.set_xlim(-3.14159, 3.14159)
            self.axis.set_ylim(-3.14159/2., 3.14159/2)
        
        self.axis.legend(bbox_to_anchor=(0.8, 0.17), ncol=3, fancybox=True, shadow=True)
    
    def show(self):
        self._make_plot()
        plt.show()
    
    def save(self, filename, title=""):
        self._make_plot()
        self.axis.set_title(title)
        self.figure.savefig(filename, bbox_inches="tight")

def fields_to_coverage_plot(fields):
    """ """
    cov_plot = CoveragePlot(projection="aitoff", figsize=(18,22), facecolor="w", edgecolor="none")
            
    k,r,b = True,True,True
    for field in fields:
        kwargs = dict()
        if field.number_of_exposures >= 100:
            kwargs["color"] = (136/255., 86/255., 167/255.)
            kwargs["alpha"] = 0.8
            if r and field.id > 1000: 
                kwargs["label"] = r"$\geq$100 observations"
                r = False
        elif field.number_of_exposures >= 25:
            kwargs["color"] = (158/255., 188/255., 218/255.)
            kwargs["alpha"] = 0.85
            if b and field.id > 1000:
                kwargs["label"] = r"$\geq$25 observations"
                b = False
        else:
            kwargs["color"] = (0.8,0.8,0.8)
            kwargs["alpha"] = 0.75
            if k and field.id > 1000:
                kwargs["label"] = "<25 observations"
                k = False
            
        cov_plot.add_field(field, **kwargs)
    
    return cov_plot

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose", default=False,
                    help="Be chatty! (default = False)")
    parser.add_argument("-q", "--quiet", action="store_true", dest="quiet", default=False,
                    help="Be quiet! (default = False)")
    parser.add_argument("-o", "--overwrite", action="store_true", dest="overwrite", default=False,
                    help="Overwrite any existing / cached files")
    parser.add_argument("--print-info", action="store_true", dest="print_info", default=False,
                    help="Print out information about the fields and number of exposures.")
    parser.add_argument("--plot", action="store_true", dest="plot", default=False,
                    help="Create survey coverage plots (R-band and g-band)")
    parser.add_argument("--dump-json", action="store_true", dest="dump_json", default=False,
                    help="Dump a JSON FeatureCollection for the interactive survey coverage plot")
    parser.add_argument("--file", dest="json_file", default="data/survey_coverage/ptf_fields.json",
                    help="Dump a JSON FeatureCollection to the specified file")
                    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)
    
    R_info = pc.SurveyInfo(filter="R", overwrite=args.overwrite)
    g_info = pc.SurveyInfo(filter="g", overwrite=args.overwrite)
    
    if args.print_info:    
        num_R10_fields = len(R_info.fields(10))
        num_R100_fields = len(R_info.fields(100))
        logger.info("Number of R-band fields >10  observations:\t\t{} = {:.2f} sq. deg.".format(num_R10_fields, num_R10_fields*7.26))
        logger.info("Number of R-band fields >100 observations:\t\t{} = {:.2f} sq. deg.".format(num_R100_fields, num_R100_fields*7.26))
        
        num_g10_fields = len(g_info.fields(10))
        num_g100_fields = len(g_info.fields(100))
        logger.info("Number of g-band fields >10  observations:\t\t{} = {:.2f} sq. deg.".format(num_g10_fields, num_g10_fields*7.26))
        logger.info("Number of g-band fields >100 observations:\t\t{} = {:.2f} sq. deg.".format(num_g100_fields, num_g100_fields*7.26))
    
    if args.plot:
        # Make survey coverage plots
        R_cov_plot = fields_to_coverage_plot(R_info.fields(1))
        R_cov_plot.save("plots/survey_coverage/R_coverage.pdf", title="PTF R-band survey coverage")
        R_cov_plot.save("plots/survey_coverage/R_coverage.png", title="PTF R-band survey coverage")
        
        g_cov_plot = fields_to_coverage_plot(g_info.fields(1))
        g_cov_plot.save("plots/survey_coverage/g_coverage.pdf", title="PTF g-band survey coverage")
        g_cov_plot.save("plots/survey_coverage/g_coverage.png", title="PTF g-band survey coverage")
    
    if args.dump_json:
        # dump json file for survey coverage site
        print R_info.timestamp
        pc.field_list_to_json(R_info.fields(1), filename=args.json_file)
    
