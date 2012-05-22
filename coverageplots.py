#!/usr/bin/env python
import sys
import cPickle as pickle
import numpy as np
import apwlib.convert as c
import apwlib.geometry as g
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ra = 1.225
dec = 0.9

OGLE = False

from numpy import cos, sin, arcsin, arccos, radians, degrees

def raDecToGalactic(ra, dec):
    b = degrees(arcsin(sin(dec.radians)*cos(radians(62.6)) - cos(dec.radians)*sin(radians(ra.degrees - 282.25))*sin(radians(62.6))))
    l = degrees(arcsin((1. / cos(b)) * ( sin(dec.radians)*sin(radians(62.6)) + cos(dec.radians)*sin(radians(ra.degrees - 282.25))*cos(radians(62.6)) ))) + 33.
    
    return l, b

f = open("fieldDict.pickle")
fieldDict = pickle.load(f)
f.close()

cadences = []
baselineLengths = []
numObservations = []

fields = fieldDict.keys()
for fieldid in fields:
    w = (fieldDict[fieldid].ccdid == 1)
    numMJDs = len(fieldDict[fieldid][w].mjd)
    sortedMJD = np.sort(fieldDict[fieldid][w].mjd)
    
    if fieldid == 101001:
        numObservations.append(0)
        cadences.append(0)
        baselineLengths.append(0)
        continue
    
    medianCadence = np.median(sortedMJD[1:] - sortedMJD[:-1])
    
    if numMJDs > 1:
        cadences.append(medianCadence)
        baselineLengths.append(sortedMJD.max()-sortedMJD.min())
    else:
        cadences.append(0)
        baselineLengths.append(0)

    numObservations.append(numMJDs)

fig = plt.figure()
ax = fig.add_subplot(111, projection="aitoff")
ax.grid()

maxNum = max(numObservations)
maxBaseline = max(baselineLengths)

baselines = np.array(baselineLengths)
wavelengths = (((baselines-min(baselines))/max(baselines)-min(baselines))*400. + 380.)

for ii, fieldid in enumerate(fields):
    if numObservations[ii] < 1:
        continue
        
    data = fieldDict[fieldid][fieldDict[fieldid].ccdid == 1]
    
    #centerX, centerY = -((np.mean(data.ra)-5.*3.5/12.)-180.), np.mean(data.dec)-2.31/4.
    ra = g.RA.fromDegrees(np.mean(data.ra)-5.*3.5/12.)
    dec = g.Dec.fromDegrees(np.mean(data.dec)-2.31/4.)
    centerX, centerY = raDecToGalactic(ra, dec)
    #centerX -= 180.0
    
    rec_x1, rec_y1 = centerX + (3.5/np.cos(np.radians(centerY)))/2, centerY - 2.31/2. #?? or -?
    rec = Rectangle((np.radians(rec_x1), np.radians(rec_y1)), \
        -np.radians(3.5/np.cos(np.radians(centerY))), np.radians(2.31), color='{:0.2f}'.format(1.-(numObservations[ii]/float(maxNum))**(1/3.)))
    ax.add_patch(rec)
    
# If you want to add OGLE to the plot..
if OGLE:
    ogle3 = np.genfromtxt("ogle3_fields.txt", names=True, delimiter=",", dtype=[("name",np.str_,6), ("ra",np.str_, 10), ("dec",np.str_, 10)]).view(np.recarray)
    derp = True
    for ra,dec in zip(ogle3.ra, ogle3.dec):
        ra = g.RA.fromHours(ra).degrees
        dec = g.Dec.fromDegrees(dec).degrees
        print ra,dec
        
        centerX, centerY = -(ra - 180.0), dec
        
        rec_x1, rec_y1 = centerX + ((36./60)/np.cos(np.radians(centerY)))/2, centerY - (36./60)/2. #?? or -?
        
        if derp:
            rec = Rectangle((np.radians(rec_x1), np.radians(rec_y1)), \
                -np.radians((36./60)/np.cos(np.radians(centerY))), np.radians(36./60), color='r', alpha=0.3, label="OGLE-III")
        else:
            rec = Rectangle((np.radians(rec_x1), np.radians(rec_y1)), \
                -np.radians((36./60)/np.cos(np.radians(centerY))), np.radians(36./60), color='r', alpha=0.3)
        ax.add_patch(rec)
        derp = False
    
#ax.plot([-np.radians(86.42)], [np.radians(-29.)], 'ro', ms=10.)
ax.set_xticklabels([330, 300, 270, 240, 210, 180, 150, 120, 90, 60, 30])
ax.legend()
#fig.suptitle("Bluer = shorter baseline, Redder = longer baseline -- Darker = more observations, Lighter = fewer observations")

plt.show()
#plt.savefig("t.pdf")
