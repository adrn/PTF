# Test cadence stuff..

from DatabaseConnection import *
import matplotlib.pyplot as plt
import numpy as np

fields = np.unique(Session.query(LightCurve.field).all())

derp = dict()
derp["25-50"] = []
derp["50-100"] = []
derp["100-150"] = []
derp["150+"] = []

for field in fields.T[0]:
    lcs = Session.query(LightCurve).filter(LightCurve.field == field).all()
    for lc in lcs:
        lmjd = len(lc.mjd)
        
        if 25 <= lmjd < 50:
            derp["25-50"].append(np.median(lc.amjd[1:]-lc.amjd[:-1]))
        elif 50 <= lmjd < 100:
            derp["50-100"].append(np.median(lc.amjd[1:]-lc.amjd[:-1]))
        elif 100 <= lmjd < 150:
            derp["100-150"].append(np.median(lc.amjd[1:]-lc.amjd[:-1]))
        elif 150 <= lmjd:
            derp["150+"].append(np.median(lc.amjd[1:]-lc.amjd[:-1]))
        else: 
            continue
    

fig = plt.figure()
for ii,key in enumerate(derp.keys()):
    ax = fig.add_subplot(2,2,ii+1)
    ax.hist(derp[key], bins=100)
    ax.set_title(key)
    ax.set_xlabel("Days")

plt.show()