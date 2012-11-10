from __future__ import division
import sys
import time
import numpy as np
import ptf.db.photometric_database as pdb

field = pdb.Field(4588, "R")
ccd = field.ccds[0]
chip = ccd.read()
sourcedata = chip.sourcedata

def test_quality_cut(source_id):
    a = time.time()
    srcdata = pdb.quality_cut(sourcedata, source_id=source_id)
    return time.time() - a

def test_where(source_id):
    wheres = "(matchedSourceID == {})".format(source_id)

    a = time.time()
    data = [x.fetch_all_fields() for x in sourcedata.where(wheres)]
    data = np.array(data, dtype=sourcedata.dtype)
    data = data[(data["x_image"] > 15) & (data["x_image"] < 2033) & \
                        (data["y_image"] > 15) & (data["y_image"] < 4081) & \
                        (data["relPhotFlags"] < 4) & \
                        (data["mag"] > 14.3) & (data["mag"] < 21) & \
                        ((data["sextractorFlags"] & 251) == 0) & \
                        ((data["ipacFlags"] & 6077) == 0) & \
                        np.isfinite(data["mag"])]

    return time.time()-a

source_ids = np.array(chip.sources.readWhere("ngoodobs > 100")["matchedSourceID"])
np.random.shuffle(source_ids)
src1 = source_ids[::2]
src2 = source_ids[1::2]

quality_cut_times = []
comprehension_times = []

N = 100
for ii in range(N):
    comprehension_times.append(test_where(src1[ii]))
    quality_cut_times.append(test_quality_cut(src2[ii]))

print "List comprehension:", sum(comprehension_times)/N
print "Quality cut:", sum(quality_cut_times)/N