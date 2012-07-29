"""
cython -a aov.pyx
deimos:
    gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -L/usr/include/python2.7 -o aov.so aov.c
laptop:
    gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
    -I /usr/include/python2.7 \
    -L /usr/lib/python2.7 \
    -l python \
    -I/System/Library/Frameworks/Python.framework/Versions/2.7/Extras/lib/python/numpy/core/include \
    -o aov.so aov.c
python test_aov.py

"""

import numpy as np
import aov
import matplotlib.pyplot as plt

data = np.genfromtxt("lc_3936.txt", names=["mjd", "mag", "error"])
idx = np.argsort(data["mjd"])
t = data["mjd"][idx]
mag = data["mag"][idx]
sig = data["error"][idx]

Nbin = 20
minP = 0.1
maxP = 10.
subsample = 0.1
finetune = 0.01
Npeaks = 5
operiodogram = 1

T = t[-1] - t[0]
N = 5

Nperiod = 0
freq = 1. / minP
minfreq = 1. / maxP
freqstep = subsample / T

while freq >= minfreq:
    freq -= freqstep
    Nperiod += 1

i = 0
periods = np.zeros(Nperiod, dtype=float)

freq = 1. / minP
while freq >= minfreq:
    periods[i] = 1./freq
    freq -= freqstep
    i += 1

hist_size = 20

# Test the periodogram
periodogram = aov.aov_periodogram_asczerny(t, mag, Nperiod, periods, hist_size)

# Test the peak finder
findpeaks = aov.findPeaks_aov(t, mag, sig, 5, 0.1, 10., 0.1, 0.01, hist_size)

plt.plot(findpeaks["period"], -findpeaks["periodogram"])
for ii in range(len(findpeaks["peak_period"])):
    print findpeaks["peak_period"][ii]
    print findpeaks["peak_power"][ii]
    print findpeaks["peak_SNR"][ii]
    print "--"*5
    plt.axvline(findpeaks["peak_period"][ii], color='r', ls="--")
    plt.text(findpeaks["peak_period"][ii], 40-2.*findpeaks["peak_period"][ii], "SNR: {}".format(findpeaks["peak_SNR"][ii]))
plt.show()
