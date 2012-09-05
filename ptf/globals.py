# PTF Stats
pix_scale = 1.01 #arcsec / pixel
camera_size = (12000., 8000.) #pixels
ccd_size = (2048, 4096) # x, y

camera_size_degrees = (camera_size[0]*pix_scale/3600., camera_size[1]*pix_scale/3600.)

# OGLE IV field sizes
ogle_camera_size = (1.225,0.9) #degrees

# Convert variability index to a Latex label
index_to_label = {"j" : "J", "k" : "K", "sigma_mu" : r"$\sigma/\mu$", "eta" : r"$\eta$", "delta_chi_squared" : r"$\Delta \chi^2$", "con" : "Con", "corr" : "Corr"}

config = dict()

with open("config", "r") as f:
    for line in f.readlines():
        key,val = line.split()
        config[key] = val
        