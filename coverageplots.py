# coding: utf-8
"""
    Various functions for plotting PTF's sky coverage and number of observations.
    
    TODO: Make a 2D plot on a grid (with arbitrary units) -- size of the squares correspond
            to the baseline, color corresponds to the number of observations
            
"""

# Standard library
import sys, os
import copy
import cPickle as pickle
import json
import math

# Third-party
import apwlib.convert as c
import apwlib.geometry as g

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mc

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pyfits as pf
from scipy.stats import scoreatpercentile

# PTF
from ptf.globals import camera_size_degrees

def observations_per_field(field_data, filename="data/field_observations.npy"):
    """ Determine the number of observations per field, given an array of 
        field data. Returns a dictionary, keyed by PTF Field IDs
    """
    
    if not os.path.exists(filename):
        unq_field_ids = np.unique(field_data.field_id)
        
        field_observations = []
        for field_id in unq_field_ids:
            this_field = field_data[field_data["field_id"] == field_id]
            
            obs_nums = []
            for ccd_id in np.unique(this_field["ccd_id"]):
                obs_nums.append(sum(this_field["ccd_id"] == ccd_id))
            
            field_observations.append((field_id,max(obs_nums)))
        
        arr = np.array(field_observations, dtype=[("field_id", int), ("observations", int)])    
        np.save(filename, arr)
    else:
        arr = np.load(filename)
    
    return arr

def field_exposure_data_to_feature(field_data):
    """ Converts an array of exposure data for a single field into a
        'feature' object to be stuffed into JSON.
    """
    feature = dict(type="Feature", id=str(field_data["id"][0]))
    properties = dict(name=str(field_data["id"][0]))
    geometry = dict(type="Polygon")
    
    ra = field_data["ra"][0]-180.0
    dec = field_data["dec"][0]
    
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

def fits_to_json(exposure_data_file):
    """ Converts a FITS file containing exposure information into a JSON file
        with a list of Features to be displayed by D3's GeoJson plugin.
        
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
    """
    
    final_dict = dict(type="FeatureCollection", features=[])
    
    fields = np.load("data/all_fields.npy")
    raw_field_data = pf.open(exposure_data_file)[1].data
    obs_num_per_field = observations_per_field(raw_field_data)
    
    # Minimum and maximum number of observations for all fields
    min_obs = 1
    # There is one crazy outlier: the Orion field, so I remove that for calculating the max...
    obs_copy = copy.copy(obs_num_per_field["observations"])
    obs_copy[obs_num_per_field["observations"].argmax()] = 0.0
    max_obs = obs_copy.max()
    
    # Create a matplotlib lognorm scaler between these values
    scaler = matplotlib.colors.LogNorm(vmin=min_obs, vmax=max_obs)
    
    ptf_fields = []
    for row in obs_num_per_field:
        field_id = row["field_id"]
        this_field = fields[fields["id"] == field_id]
        if len(this_field) == 0 or field_id <= 642:
            print "ERROR", field_id
            continue
        
        if this_field["dec"][0] < -40:
            print "WTF", field_id
        
        feature = field_exposure_data_to_feature(this_field)
        
        # Determine color of field
        rgb = cm.autumn(scaler(row["observations"]))
        feature["properties"]["color"] = mc.rgb2hex(rgb)
        feature["properties"]["alpha"] = scaler(row["observations"])*0.5 + 0.2
        feature["properties"]["number_of_observations"] = str(row["observations"])
        feature["properties"]["ra"] = str(this_field["ra"][0])
        feature["properties"]["dec"] = str(this_field["dec"][0])
        
        final_dict["features"].append(feature)
    
    return json.dumps(final_dict)
















# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# OLD CODE!
# =~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

class CCD(object):
    
    def __init__(self, table_data):
        self.id = table_data.ccd_id[0]
        self.ra = np.mean(table_data.ra)
        self.dec = np.mean(table_data.dec)
        self.mjd = table_data.mjd
        self._table_data = table_data
        
    @property
    def number_of_observations(self):
        return len(self.mjd)
    
    @property
    def baseline(self):
        return max(self.mjd) - min(self.mjd)

class Field(object):
    """ Abstract Field class """
    
    def __init__(self, table_data, width=None, height=None):
        """ Create a field object given a recarray / pyfits rows for a Field """
        self.ccds = {}
        
        for ccd_id in np.unique(table_data.ccd_id):
            ccd = CCD(table_data[table_data.ccd_id == ccd_id])
            self.ccds[ccd_id] = ccd
        
        self.id = table_data.field_id[0]
        
    @property
    def number_of_observations(self):
        """ Get the total number of observations of this field """
        
        per_ccd_observations = []
        for ccd in self.ccds.values():
            per_ccd_observations.append(ccd.number_of_observations)
        
        return max(per_ccd_observations)
    
    @property
    def baseline(self):
        """ Get the total baseline of observations of this field """
        
        per_ccd_baseline = []
        for ccd in self.ccds.values():
            per_ccd_baseline.append(ccd.baseline)
        
        return max(per_ccd_baseline)
    
    @property
    def ra(self):
        # TODO: Lookup field ra/dec from table
        return g.RA.fromDegrees(np.mean([ccd.ra for ccd in self.ccds.values()]))
        
    @property
    def dec(self):
        # TODO: Lookup field ra/dec from table
        return g.Dec.fromDegrees(np.mean([ccd.dec for ccd in self.ccds.values()]))
    
    @property
    def name(self):
        if not self._name and self.id:
            return str(self.id)
        elif not self._name and not self.id:
            raise ValueError("No name or id set!")
        else:
            return self._name
    
    @name.setter
    def name(self, val):
        self._name = str(val)
    
class PTFField(Field):
    """ Represents a PTF field """
        
    def __init__(self, table_data):
        super(PTFField, self).__init__(table_data, width=camera_size_degrees[0], height=camera_size_degrees[1])

#class OGLEField(Field):
#    """ Represents an OGLEIV field """
#        
#    def __init__(self, ra, dec, id=None):
#        super(OGLEField, self).__init__(ra, dec, id=id, width=ogle_camera_size[0], height=ogle_camera_size[1])

class PTFCoveragePlot(object):
    """ Given a list of PTFField objects, create a coverage plot """
    def __init__(self, figsize=None, projection=None):
        if figsize:
            self.figure = plt.figure(figsize=figsize)
        else:
            self.figure = plt.figure()
        
        if projection:
            self.axis = self.figure.add_subplot(111, projection=projection)
        else:
            self.axis = self.figure.add_subplot(111)
        
        self.axis.set_xticklabels([330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30])
        self.title = self.axis.set_title("Comparing survey coverage in equatorial coordinates")
        self.title.set_y(1.07)
        
    def addFields(self, fields, label=None, color_by_observations=False, size_by_baseline=False, **kwargs):
        """ Add a list of Field() objects to be plotted as Rectangle patches """
        
        if color_by_observations:
            num_obs = [x.number_of_observations for x in fields]
            maxNum = scoreatpercentile(num_obs, 99) #max(num_obs)+1
            #scaler = matplotlib.colors.LogNorm(0.9999, maxNum)
            scaler = matplotlib.colors.Normalize(0, maxNum)
        
            sorted_field_indices = np.argsort(num_obs)
        else:
            sorted_field_indices = range(len(fields))
        
        if size_by_baseline:
            baselines = np.array([x.baseline for x in fields])[sorted_field_indices]
            baseline_scaler = matplotlib.colors.Normalize(vmin=baselines.min(), vmax=baselines.max())
            
        for ii,field_idx in enumerate(sorted_field_indices):
            field = fields[field_idx]
            
            if size_by_baseline:
                factor = baseline_scaler(baselines[ii])
                xsize = camera_size_degrees[0] / np.cos(field.dec.radians) * factor
                ysize = camera_size_degrees[1] * factor
                
                rec_x1 = ((field.ra.degrees + xsize / 2.) % 360.) * -1 + 180 # degrees
                rec_y1 = field.dec.degrees - ysize / 2. # degrees
            else:
                xsize = camera_size_degrees[0] / np.cos(field.dec.radians)
                ysize = camera_size_degrees[1]
                rec_x1 = ((field.ra.degrees + (camera_size_degrees[0] / np.cos(field.dec.radians)) / 2.) % 360.) * -1 + 180 # degrees
                rec_y1 = field.dec.degrees - camera_size_degrees[1] / 2. # degrees
            
            if color_by_observations:
                col = ((scaler(field.number_of_observations)-1.)*-1.)*0.5
                
                if col < 0: col = 0
                
                if ii == 0 and label:
                    rec = Rectangle((np.radians(rec_x1), np.radians(rec_y1)), \
                                    -np.radians(xsize), np.radians(ysize), \
                                    color='{:0.2f}'.format(col), label=label, alpha=0.7)
                else:
                    rec = Rectangle((np.radians(rec_x1), np.radians(rec_y1)), \
                                    -np.radians(xsize), np.radians(ysize), \
                                    color='{:0.2f}'.format(col), alpha=0.7)
            else:
                if ii == 0 and label:
                    rec = Rectangle((np.radians(rec_x1), np.radians(rec_y1)), \
                                    -np.radians(xsize), np.radians(ysize),\
                                    label=label, **kwargs)
                else:
                    rec = Rectangle((np.radians(rec_x1), np.radians(rec_y1)), \
                                    -np.radians(xsize), np.radians(ysize),\
                                    **kwargs)
            self.axis.add_patch(rec)
        
    def addLegend(self):
        self.legend = self.axis.legend(bbox_to_anchor=(0.75, -0.05), ncol=3, fancybox=True, shadow=True)

def new_2d_plot():
    """ This is still in 'development,' but the idea is to make a 2D plot
        to visualize the PTF coverage, baselines, and number of observations.
        
        The size of the rectangles will represent the baseline of observation,
        and the opacity will represent the number of observations.
    
    """
    
    if not os.path.exists("data/fields.pickle"):
        raw_field_data = pf.open("data/exposureData.fits")[1].data
        unq_field_ids = np.unique(raw_field_data.field_id)
        
        ptf_fields = []
        for field_id in unq_field_ids:
            one_field_data = raw_field_data[raw_field_data.field_id == field_id]
            if len(one_field_data) < 1: continue
            
            ptf_fields.append(PTFField(one_field_data))
        
        f = open("data/fields.pickle", "w")
        pickle.dump(ptf_fields, f)
        f.close()
    
    f = open("data/fields.pickle", "r")
    ptf_fields = pickle.load(f)
    f.close()
    
    coverage_plot = PTFCoveragePlot(figsize=(25,25), projection="aitoff")
    coverage_plot.addFields(ptf_fields, label="PTF", color_by_observations=True, size_by_baseline=True)
    
    coverage_plot.figure.savefig("plots/ptf_coverage_new.png")
    
    return 

if False:
    # PTF:
    raw_field_data = pf.open("data/exposureData.fits")[1].data
    unq_field_ids = np.unique(raw_field_data.field_id)
    
    ptf_fields = []
    for field_id in unq_field_ids:
        one_field_data = raw_field_data[raw_field_data.field_id == field_id]
        #mean_ra = np.mean(one_field_data.ra) / 15.
        #mean_dec = np.mean(one_field_data.dec)
        #observations = len(one_field_data) / len(np.unique(one_field_data.ccd_id))
        
        #ptf_fields.append(PTFField(mean_ra, mean_dec, id=field_id, number_of_observations=observations))
        ptf_fields.append(PTFField(one_field_data))
    
    # OGLE:
    high_cadence = np.genfromtxt("data/ogle4_common.txt", names=["ra","dec","l","b"], usecols=[6,7,8,9]).view(np.recarray)
    low_cadence = np.genfromtxt("data/ogle4_less_frequent.txt", names=["ra","dec","l","b"], usecols=[6,7,8,9]).view(np.recarray)
    
    ogle_high_cadence_fields = []
    for row in high_cadence: ogle_high_cadence_fields.append(OGLEField(row["ra"], row["dec"]))
    
    ogle_low_cadence_fields = []
    for row in low_cadence: ogle_low_cadence_fields.append(OGLEField(row["ra"], row["dec"]))
    
    coverage_plot = PTFCoveragePlot(figsize=(25,25), projection="aitoff")
    coverage_plot.addFields(ptf_fields, label="PTF", color_by_observations=True)
    coverage_plot.addFields(ogle_low_cadence_fields, label="OGLE-IV - low cadence", color="b", alpha=0.15)
    coverage_plot.addFields(ogle_high_cadence_fields, label="OGLE-IV - high cadence", color="r", alpha=0.15)
    
    coverage_plot.addLegend()
    
    #plt.show()
    coverage_plot.figure.savefig("plots/ptf_coverage.png")