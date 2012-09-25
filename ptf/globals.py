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

# Convert variability index to a Latex label
index_to_label = {"j" : "J", "k" : "K", "sigma_mu" : r"$\sigma/\mu$", "eta" : r"$\eta$", "delta_chi_squared" : r"$\Delta \chi^2$", "con" : "Con", "corr" : "Corr"}

config = dict()

#config_filename = os.path.
_base_path = os.path.split(__file__)[0]

with open(os.path.join(_base_path, "config"), "r") as f:
    for line in f.readlines():
        key,val = line.split()
        try:
            config[key] = int(val)
        except ValueError:
            config[key] = val

# Convert my names for variability indices to the PDB names
pdb_index_name = dict(eta="vonNeumannRatio", j="stetsonJ", k="stetsonK", delta_chi_squared="chiSQ", sigma_mu=["magRMS","referenceMag"])

def source_index_name_to_pdb_index(source, index_name):
    """ Given a source (a row from chip.sources) and an index name (e.g. eta),
        return the value of the statistic. This is particularly needed for a
        computed index like sigma_mu.
    """
    if index_name == "sigma_mu":
        return source[pdb_index_name[index_name][0]] / source[pdb_index_name[index_name][1]]            
    else:
        return source[pdb_index_name[index_name]]

all_fields = np.load(os.path.join(os.path.split(_base_path)[0], "data", "all_fields.npy"))