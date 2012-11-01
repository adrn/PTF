import os
import numpy as np

# PTF Stats
pix_scale = 1.01 #arcsec / pixel
camera_size = (12000., 8000.) #pixels
ccd_size = (2048, 4096) # x, y

camera_size_degrees = (camera_size[0]*pix_scale/3600., camera_size[1]*pix_scale/3600.)
camera_size_radius = ((camera_size_degrees[0]/2.)**2 + (camera_size_degrees[1]/2.)**2)**0.5

# OGLE IV field sizes
ogle_camera_size = (1.225,0.9) #degrees

config = dict()
_base_path = os.path.split(__file__)[0]

# Need this for mongodb config
with open(os.path.join(_base_path, "config"), "r") as f:
    for line in f.readlines():
        key,val = line.split()
        try:
            config[key] = int(val)
        except ValueError:
            config[key] = val

all_fields = np.load(os.path.join(os.path.split(_base_path)[0], "data", "all_fields.npy"))

# Configuration stuff:
min_number_of_good_observations = 10
ccd_edge_cutoff = 15 # pixels