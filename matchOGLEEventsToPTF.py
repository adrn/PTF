import numpy as np
import apwlib.geometry as g

import lsd
import lsd.bounds as lb
db = lsd.DB("/scr4/bsesar")

from db.DatabaseConnection import *

data = np.genfromtxt("data/ogleEvents.txt", usecols=[3,4,6], dtype=[("ra", "|S11"), ("dec", "|S11"), ("date", "|S13")], skiprows=1).view(np.recarray)

allBounds = []
bounds_t = lb.intervalset((40000, 60000))
for row in data:
    ra = g.RA.fromHours(row["ra"])
    dec = g.Dec.fromDegrees(row["dec"])
    radius = 5./3600.
    bounds_xy = lb.beam(ra, dec, radius)
    allBounds.append((bounds_xy, bounds_t))

results = db.query("ptf_det.ra as ra, ptf_det.dec as dec, mjd, mag_abs/1000. as mag, magerr_abs/1000. as magErr, \
                apbsrms as sys_err, fid, obj_id, ptf_field, ccdid, flags, imaflags_iso \
                FROM ptf_exp, ptf_det, ptf_obj \
                WHERE ((ccdid == {0}) & (ptf_field == {1}) & (flags & 1) == 0) & ((imaflags_iso & 3797) == 0) & (flags < 8) & (apbsrms > 0))".format(ccdid, fieldid))\
            .fetch(bounds=[(bounds_xy, bounds_t)])

resultsArray = np.array(results, dtype=[('ra', np.float64), ('dec', np.float64), ('mjd', np.float64), ('mag', np.float64), ('mag_err', np.float64), \
        ('sys_err', np.float32), ('filter_id', np.uint8), ('obj_id', np.uint64), ('field_id', np.uint32), ('ccd_id', np.uint8), ('flags', np.uint16), ('imaflags_iso', np.uint16)])

f = open("data/ogle-ptf_match.pickle", "w")
pickle.dump(resultsArray.view(np.recarray), f)
f.close()