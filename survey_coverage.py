# coding: utf-8
from __future__ import division

""" This module contains utilities for computing PTF's survey coverage, both in time and position. """

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import datetime
import re
import glob
import cPickle as pickle
import logging
import time, datetime
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
import ptf.photometricdatabase as pdb
from ptf.globals import camera_size_degrees
import ptf.globals as pg

# Create logger for this module
logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(name)s / %(levelname)s / %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

all_fields = np.load(os.path.join(os.path.split(pg._base_path)[0], "data", "all_fields.npy"))

class SurveyInfo(object):
    """ Some example use cases for this object:
        
        TODO: Fill this in!
    
    """
    def __init__(self, filter, overwrite=False):
        """ Convenience class for getting information about the PTF survey
            
            Parameters
            ----------
            filter : int, str, ptf.photometricdatabase.Filter
                Any parseable, unambiguous filter representation.
            overwrite : bool (optional)
                Overwrite any cache files if they already exist
        """
        
        if not isinstance(filter, pdb.Filter):
            self.filter = pdb.Filter(filter)
        
        cache_filename = os.path.join(os.path.split(pg._base_path)[0], "data", "survey_coverage", "fields_observations_{}.npy".format(str(self.filter)))
        self._fields_exposures = get_fields_exposures(self.filter, filename=cache_filename, overwrite=overwrite)
        self.timestamp = datetime.datetime.strptime(time.ctime(os.path.getmtime(cache_filename)), "%a %b %d %H:%M:%S %Y")
        
    def fields(self, min_num_observations):
        """ Return a list of fields with more than the above number of observations """
        
        rows = self._fields_exposures[self._fields_exposures["num_exposures"] >= min_num_observations]
        fields = [pdb.Field(row["field"], self.filter, number_of_exposures=row["num_exposures"]) for row in rows]
        return [f for f in fields if f.ra != None]

def get_fields_exposures(filter, filename=None, overwrite=False):
    """ Given a filter, go to the PTF photometric database and get information about all PTF
        fields for that filter, and the number of good exposures per field.
        
        Parameters
        ----------
        filter : ptf.photometricdatabase.Filter
            Must be a Filter object (see the above module)
        filename : str (optional)
            The filename to store this data to.
        overwrite : bool (optional)
            Overwrite 'filename' if it already exists.
        
    """
    
    if not isinstance(filter, pdb.Filter):
        raise ValueError("Filter must be a valid Filter() object!")
    
    if filename == None:
        filename = os.path.join("data", "survey_coverage", "fields_observations_{}.npy".format(str(filter)))
    
    if os.path.exists(filename) and overwrite:
        logger.debug("Data file already exists, but you want to overwrite it")
        os.remove(filename)
        logger.debug("File {} deleted".format(filename))
    elif os.path.exists(filename) and not overwrite:
        logger.info("Data file already exists: {}".format(filename))
    
    if not os.path.exists(filename):
        logger.info("Data file doesn't exist -- it could take some time to create it!")
        
        fields = []
        exposures = []
        
        pattr = re.compile(".*match_(\d+)_(\d+)_(\d+)")
        for match_filename in glob.glob("/scr4/dlevitan/matches/match_{:02d}_*.pytable".format(filter.id)):
            logger.debug("Reading file: {}".format(match_filename))
            
            filter_id, field_id, ccd_id = map(int, pattr.search(match_filename).groups())
            
            if field_id in fields:
                continue
    
            try:
                file = tables.openFile(match_filename)
                chip = getattr(getattr(getattr(file.root, "filter{:02d}".format(filter_id)), "field{:06d}".format(field_id)), "chip{:02d}".format(ccd_id))
            except:
                continue
            
            fields.append(field_id)
            exposures.append(len(chip.exposures))
            
            file.close()
        
        fields_exposures = np.array(zip(fields, exposures), dtype=[("field", int), ("num_exposures", int)])
        logger.debug("Saving file {}".format(filename))
        np.save(filename, fields_exposures)
    
    fields_exposures = np.load(filename)
    logger.debug("Data file loaded!")
    
    return fields_exposures

def field_to_feature(field):
    """ Converts a PTF Field object into a 'feature' object to be stuffed into JSON for the 
        interactive survey coverage viewer.
        
        Parameters
        ----------
        field : ptf.photometricdatabase.Field
            Must be a PTF Field object. See above module for details
    """
    feature = dict(type="Feature", id=str(field.id))
    properties = dict(name=str(field.id))
    geometry = dict(type="Polygon")
    
    ra = field.ra.degrees-180.0
    dec = field.dec.degrees
    
    ra_offset = camera_size_degrees[0]/math.cos(math.radians(dec))/2.
    dec_offset = camera_size_degrees[1]/2.
    
    min_ra = ra - ra_offset
    max_ra = ra + ra_offset
    
    min_dec = dec - dec_offset
    max_dec = dec + dec_offset
    
    coordinates = [[ [min_ra, min_dec], [max_ra, min_dec], [max_ra, max_dec], [min_ra, max_dec], [min_ra, min_dec] ]]
    geometry["coordinates"] = coordinates
    feature["geometry"] = geometry
    feature["properties"] = properties

    return feature

def field_list_to_json(fields, filename=None):
    """ Given a list of fields, create a FeatureCollection JSON file to use with the 
        interactive survey coverage viewer.
        
        The final structure should look like this, where "name" is the field id,
        and coordinates are a list of the coordinates of the 4 corners of the field.
        
            {"type" : "FeatureCollection",
                "features": [
                    {"type" : "Feature",
                     "properties" : { "name" : "2471"},
                                      "geometry": { "type" : "Polygon",
                                                    "coordinates" : [[ [100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0] ]]
                                                  },
                     "id":"2471"}, etc.
                            ]
            }      
        
        Parameters
        ----------
        fields : list, iterable
            Must be a list of PTF Field objects. See: ptf.photometricdatabase.Field
        filename : str (optional)
            If filename is specified, will save the JSON to the file. Otherwise, return the JSON
    """
    
    final_dict = dict(type="FeatureCollection", features=[])
    
    # Minimum and maximum number of observations for all fields
    min_obs = 1
    
    # There is one crazy outlier: the Orion field 101001, so I remove that for calculating the max...
    num_exposures = np.array([field.number_of_exposures for field in fields if field.id != 101001])
    max_obs = num_exposures.max()
    
    # Create a matplotlib lognorm scaler between these values
    scaler = matplotlib.colors.LogNorm(vmin=min_obs, vmax=max_obs)
    
    for field in fields:
        if field.number_of_exposures < 1:
            logger.debug("Skipping field {}".format(field))
            continue
        
        try:
            field.ra
            field.dec
        except AttributeError:
            this_field = all_fields[all_fields["id"] == field.id]
            if len(this_field) != 1: 
                logger.warning("Field {} is weird".format(field))
                continue
                
            field.ra = g.RA.fromDegrees(this_field["ra"][0])
            field.dec = g.Dec.fromDegrees(this_field["dec"][0])
        
        if field.dec.degrees < -40:
            logger.warning("Field {} is weird, dec < -40".format(field))
            continue
        
        feature = field_to_feature(field)
        
        # Determine color of field
        #rgb = cm.autumn(scaler(field.number_of_exposures))
        rgb = cm.gist_heat(scaler(field.number_of_exposures))
        feature["properties"]["color"] = mc.rgb2hex(rgb)
        feature["properties"]["alpha"] = scaler(field.number_of_exposures)*0.75 + 0.05
        feature["properties"]["number_of_observations"] = str(field.number_of_exposures)
        feature["properties"]["ra"] = "{:.5f}".format(field.ra.degrees)
        feature["properties"]["dec"] = "{:.5f}".format(field.dec.degrees)
        
        final_dict["features"].append(feature)
    
    blob = json.dumps(final_dict)
    
    if filename != None:
        f = open(filename, "wb")
        f.write(blob)
        f.close()
        
        return
    else:
        return blob

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

def get_overlapping_fields(ra, dec, fields=None, filter="R", size=1.):
    """ Given a position and a region size (in degrees), get all fields
        that overlap that region.
    """
    R = size + pg.camera_size_radius
    
    if not isinstance(ra, g.Angle):
        if isinstance(ra, str):
            ra = g.RA(ra)
        else:
            ra = g.RA.fromDegrees(ra)
    
    if not isinstance(dec, g.Angle):
        # Assume dec is degrees
        dec = g.Dec(dec)
    
    if fields == None:
        s_info = SurveyInfo(filter=filter)
        fields = s_info.fields(1)
    
    matched_fields = []
    for field in fields:
        dist = g.subtends_degrees(ra.degrees, dec.degrees, field.ra.degrees, field.dec.degrees)
        if dist <= R:
            matched_fields.append(field)
    
    return matched_fields

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
    
    R_info = SurveyInfo(filter="R", overwrite=args.overwrite)
    g_info = SurveyInfo(filter="g", overwrite=args.overwrite)
    
    if args.print_info:    
        num_R25_fields = len(R_info.fields(25))
        num_R100_fields = len(R_info.fields(100))
        logger.info("Number of R-band fields >25  observations:\t\t{} = {:.2f} sq. deg.".format(num_R25_fields, num_R25_fields*7.26))
        logger.info("Number of R-band fields >100 observations:\t\t{} = {:.2f} sq. deg.".format(num_R100_fields, num_R100_fields*7.26))
        
        num_g25_fields = len(g_info.fields(25))
        num_g100_fields = len(g_info.fields(100))
        logger.info("Number of g-band fields >25  observations:\t\t{} = {:.2f} sq. deg.".format(num_g25_fields, num_g25_fields*7.26))
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
        field_list_to_json(R_info.fields(1), filename=args.json_file)
    