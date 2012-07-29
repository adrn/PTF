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
"""
from __future__ import division

import sys
import numpy as np
cimport numpy as np
cimport cython

cdef extern from "math.h":
    double fmod(double, double)
    double floor(double)
    double fmax(double, double)
    double fmin(double, double)
    double sqrt(double)
    int isnan(double)
    double fabs(double)
    
cdef double ERROR_SCORE
cdef double MINVAR
cdef long CTMIN
cdef long MAX_DOUBLE_CHECK_MULTIPLE
cdef long MAX_PERIOD_DIFF_MULTIPLE

ERROR_SCORE = 1000000.
MINVAR = 1.E-32
CTMIN = 5
MAX_DOUBLE_CHECK_MULTIPLE = 19
MAX_PERIOD_DIFF_MULTIPLE = 5

@cython.boundscheck(False) # turn of bounds-checking for entire function
def _mysort2dblint(long N, np.ndarray[double, ndim=1] _data1, np.ndarray[long, ndim=1] _data2):
    cdef int i, j
    cdef double v, t1
    cdef long t2, length
    
    cdef np.ndarray[double, ndim=1] data1
    cdef np.ndarray[long, ndim=1] data2
    
    length = len(_data1)
    
    data1 = np.zeros(length, dtype=float)
    data2 = np.zeros(length, dtype=int)
    
    for i in range(length):
        data1[i] = _data1[i]
        data2[i] = _data2[i]
    
    if N <= 1: 
        return (data1, data2, 0)
    
    v = data1[0]
    i = 0
    j = N
    while True:
        
        #while (++i < N ? data1[i] < v : 0) { }
        while i+1 < N:
            if data1[i] >= v:
                break
            i += 1
        
        #while(data1[--j] > v) { }
        j -= 1
        while data1[j] > v:
            j -= 1
                
        if (i >= j):
            break
        
        t1 = data1[i]
        data1[i] = data1[j]
        data1[j] = t1
        
        t2 = data2[i]
        data2[i] = data2[j]
        data2[j] = t2

    t1 = data1[i-1]
    data1[i-1] = data1[0]
    data1[0] = t1
    
    t2 = data2[i-1]
    data2[i-1] = data2[0]
    data2[0] = t2
    
    return (data1, data2, i)

def mysort2dblint(long N, np.ndarray[double, ndim=1] data1, np.ndarray[long, ndim=1] data2):
    cdef long i, xx, yy
    
    data1, data2, i = _mysort2dblint(N, data1, data2)
    data1, data2, xx = _mysort2dblint(i-1, data1, data2)
    data1, data2, yy = _mysort2dblint(N-i, data1+i, data2+i)
    
    return data1, data2

@cython.boundscheck(False) # turn of bounds-checking for entire function
def normalize(np.ndarray[double, ndim=1] time, np.ndarray[double, ndim=1] mag, np.ndarray[double, ndim=1] sig):
    cdef double ave
    cdef double stddev
    cdef long size
    cdef long i, i1, n
    
    size = len(time)
    
    if size > 0:
        i = 0
        
        while isnan(mag[i]):
            i += 1
            
        ave = mag[i]
        n = 1
        i1 = i
        
        for i in range(i+1, size):
            if not isnan(mag[i]):
                ave += mag[i]
                n += 1
        
        ave /= float(n)
        stddev = (mag[i1] - ave)*(mag[i1] - ave)
        
        for i in range(i1+1, size):
            if not isnan(mag[i]):
                stddev += (mag[i] - ave)*(mag[i] - ave)
        
        stddev = sqrt(stddev / float(n))
        for i in range(size):
            mag[i] = (mag[i] - ave) / stddev
            sig[i] = sig[i] / stddev
    
    return time, mag, sig

# Determine whether two different periods are the same 
def isDifferentPeriods(double period1, double period2, double T):
    cdef long a, b
    cdef double period1mul
    
    if (T * fabs(period2 - period1) < period2*period1):
        return 0
    
    for a in range(1, MAX_PERIOD_DIFF_MULTIPLE):
        for b in range(a+1, MAX_PERIOD_DIFF_MULTIPLE+1):
            period1mul = period1 * b / a
    
            if (T * fabs(period2 - period1mul)) < (period2 * period1mul):
                return 0
    
    return 1

@cython.boundscheck(False) # turn of bounds-checking for entire function
def aov_periodogram_asczerny(np.ndarray[double, ndim=1] time, np.ndarray[double, ndim=1] mag, long Nperiod, np.ndarray[double, ndim=1] periods, long nbins):
    cdef np.ndarray[double, ndim=1] periodogram
    
    cdef int ii, ip, ifr
    cdef long size, ibin, nbc, ncov, iflex, MAXBIN, nobs
    cdef double af, vf, sav, fr, at, dbc, dbh
    
    cdef np.ndarray[long, ndim=1] ncnt
    cdef np.ndarray[long, ndim=1] ind
    cdef np.ndarray[double, ndim=1] f
    cdef np.ndarray[double, ndim=1] ph
    cdef np.ndarray[double, ndim=1] ave
    cdef np.ndarray[double, ndim=1] t
    
    periodogram = np.zeros(Nperiod, dtype=float)
    size = len(time)
    ncov = 1
    MAXBIN = (nbins + 1)*ncov
    nobs = size
    nbc = nbins * ncov
    dbc = float(nbc)
    
    ncnt = np.zeros((nbins+1), dtype=int)
    ind = np.zeros(size, dtype=int)
    f = np.zeros(size, dtype=float)
    ph = np.zeros(size, dtype=float)
    ave = np.zeros((nbins+1), dtype=float)
    t = np.zeros(size, dtype=float)
    
    iflex = 0
    at = 0.
    af = 0.
    vf = 0.
    
    for ii in range(nobs):
        af = af + mag[ii]
        at = at + time[ii]
    
    af = af / float(nobs)
    at = at / float(nobs)
    
    for ii in range(nobs):
        t[ii] = time[ii] - at
        sav = mag[ii] - af
        f[ii] = sav
        vf += sav*sav
    
    for ifr in range(Nperiod):
        fr = 1. / periods[ifr]
        
        for ip in range(2):
            for ii in range(nbc):
                ave[ii] = 0.
                ncnt[ii] = 0
            
            if ip == 0:
                for ii in range(nobs):
                    dph = t[ii]*fr
                    sav = float(dph - floor(dph))
                    ph[ii] = sav
                    ibin = int(floor(sav*dbc))
                    ave[ibin] += f[ii]
                    ncnt[ibin] += 1
            else:
                iflex += 1
                
                for ii in range(nobs):
                    ind[ii] = ii
                
                # TODO: Implement this in C!
                #ph,ind = mysort2dblint(nobs, ph, ind)
                xx = np.argsort(ph)
                ph = ph[xx]
                ind = ind[xx]
                
                for ii in range(nobs):
                    ibin = ii*nbc // nobs
                    ave[ibin] += f[ind[ii]]
                    ncnt[ibin] += + 1
            
            for ii in range(ncov):
                ncnt[ii+nbc] = ncnt[ii]
        
            ibin = 0
            for ii in range(ncov+nbc-1, -1, -1):
                ibin += ncnt[ii]
                ncnt[ii] = ibin
            
            for ii in range(nbc):
                ncnt[ii] = ncnt[ii] - ncnt[ii+ncov]
            
            for ii in range(nbc):
                if ncnt[ii] < CTMIN: break
            
            if ii >= nbc: break
            
        # Calculate A.O.V. statistics for a given frequency
        for ii in range(ncov):
            ave[ii+nbc] = ave[ii]        
        
        sav = 0.
        for ii in range(ncov+nbc-1, 0, -1):
            sav = sav + ave[ii]
            ave[ii] = sav
        
        for ii in range(nbc):
            ave[ii] = ave[ii] - ave[ii+ncov]
        
        sav = 0.
        for ii in range(nbc):
            sav += ave[ii]*ave[ii]/ncnt[ii]
        
        sav /= float(ncov)
        
        periodogram[ifr] = -sav / (nbins - 1) / fmax(vf-sav, MINVAR)*float(nobs-nbins)
    
    return periodogram

def testperiod_asczerny(np.ndarray[double, ndim=1] time, np.ndarray[double, ndim=1] mag, double period, long nbins):
    cdef long ii, ibin, ip, nbc, ncov
    cdef long ifr, iflex
    cdef double af, vf, sav,
    cdef double fr, at, dbc, dbh
    cdef long MAXBIN, nobs
    cdef double periodogram
    
    cdef np.ndarray[long, ndim=1] ncnt
    cdef np.ndarray[long, ndim=1] ind
    cdef np.ndarray[double, ndim=1] f
    cdef np.ndarray[double, ndim=1] ph
    cdef np.ndarray[double, ndim=1] ave
    cdef np.ndarray[double, ndim=1] t
    
    size = len(time)
    ncov = 1
    MAXBIN = (nbins + 1)*ncov
    nobs = size
    
    ncnt = np.zeros((nbins+1), dtype=int)
    ind = np.zeros(size, dtype=int)
    f = np.zeros(size, dtype=float)
    ph = np.zeros(size, dtype=float)
    ave = np.zeros((nbins+1), dtype=float)
    t = np.zeros(size, dtype=float)
    
    # set variables (incl. mean and variance)
    nbc = nbins * ncov
    dbc = float(nbc)
    
    # calculate totals and normalize variables
    iflex = 0
    at = af = vf = 0.
    
    for ii in range(nobs):
        af += mag[ii]
        at += time[ii]
    
    af /= float(nobs)
    at /= float(nobs)
    
    for ii in range(nobs):
        t[ii] = time[ii] - at
        sav = mag[ii] - af
        f[ii] = sav
        vf += sav*sav
    
    fr = 1. / period
    
    # up to two passes over all frequencies
    for ip in range(2):
        for ii in range(nbc):
            ave[ii] = 0.
            ncnt[ii] = 0
        
        if ip == 0:
            for ii in range(nobs):
                dph = t[ii]*fr
                sav = float(dph - floor(dph))
                ph[ii] = sav
                ibin = int(floor(sav*dbc))
                ave[ibin] += f[ii]
                ncnt[ibin] += 1
        else:
            iflex += 1
            
            for ii in range(nobs):
                ind[ii] = ii
            
            # TODO: Implement this in C!
            #ph,ind = mysort2dblint(nobs, ph, ind)
            xx = np.argsort(ph)
            ph = ph[xx]
            ind = ind[xx]
            
            for ii in range(nobs):
                ibin = ii*nbc // nobs
                ave[ibin] += f[ind[ii]]
                ncnt[ibin] += + 1
        
        for ii in range(ncov):
            ncnt[ii+nbc] = ncnt[ii]
    
        ibin = 0
        for ii in range(ncov+nbc-1, -1, -1):
            ibin += ncnt[ii]
            ncnt[ii] = ibin
        
        for ii in range(nbc):
            ncnt[ii] = ncnt[ii] - ncnt[ii+ncov]
        
        for ii in range(nbc):
            if ncnt[ii] < CTMIN: break
        
        if ii >= nbc: break
        
    # Calculate A.O.V. statistics for a given frequency
    for ii in range(ncov):
        ave[ii+nbc] = ave[ii]        
    
    sav = 0.
    for ii in range(ncov+nbc-1, 0, -1):
        sav = sav + ave[ii]
        ave[ii] = sav
    
    for ii in range(nbc):
        ave[ii] = ave[ii] - ave[ii+ncov]
    
    sav = 0.
    for ii in range(nbc):
        sav += ave[ii]*ave[ii]/ncnt[ii]
    
    sav /= float(ncov)
    
    periodogram = -sav / (nbins - 1) / fmax(vf-sav, MINVAR)*float(nobs-nbins)
    
    return periodogram

def findPeaks_aov(np.ndarray[double, ndim=1] time, np.ndarray[double, ndim=1] mag, np.ndarray[double, ndim=1] sig, long Npeaks, double minp, double maxp, double subsample, double finetune, long Nbin):
    # Return peakperiods, peakvalues, peakSNR
    
    cdef double T, freq, minfreq, freqstep, Sum, Sumsqr, aveper, stdper, aveaov, stddevaov, testperiod, a_, b_
    cdef long N, m_eff, Nperiod, i, k, a, b, Ngood, nclippedthis, nclippedlast, clipiter, foundsofar, ismultiple, abest, bbest
    
    cdef np.ndarray[double, ndim=1] _time
    cdef np.ndarray[double, ndim=1] _mag
    cdef np.ndarray[double, ndim=1] _sig
    cdef np.ndarray[double, ndim=1] periods
    cdef np.ndarray[double, ndim=1] periodogram
    cdef np.ndarray[double, ndim=1] periods_copy
    cdef np.ndarray[double, ndim=1] periodogram_copy
    cdef np.ndarray[double, ndim=1] perpeaks
    cdef np.ndarray[double, ndim=1] aovpeaks
    cdef np.ndarray[double, ndim=1] aovSNR
    cdef np.ndarray[long, ndim=1] ix
    
    perpeaks = np.zeros(Npeaks, dtype=float)
    aovpeaks = np.zeros(Npeaks, dtype=float)
    
    N = len(time)
    T = time[N-1] - time[0]
    
    _time, _mag, _sig = normalize(time, mag, sig)
    
    # Initialize the periodogram
    Nperiod = 0
    freq = 1. / minp
    minfreq = 1. / maxp
    freqstep = subsample / T
    
    while (freq >= minfreq):
        freq -= freqstep
        Nperiod += 1
    
    periods = np.zeros(Nperiod, dtype=float)
    
    # Set the estimate for the effective number of independent frequency samples
    m_eff = int((1./minp - 1/maxp) * T)
    if subsample > 1.:
        m_eff = int(float(m_eff) / subsample)
    
    freq = 1. / minp
    minfreq = 1. / maxp
    
    i = 0
    while freq >= minfreq:
        periods[i] = 1./freq
        freq -= freqstep
        i += 1
    
    periodogram = aov_periodogram_asczerny(_time, _mag, Nperiod, periods, Nbin)
    
    Sum = 0.
    Sumsqr = 0.
    Ngood = 0
    for i in range(Nperiod):
        if (periodogram[i] < ERROR_SCORE) and (periodogram[i]*0.0 == 0.0):
            Sum += periodogram[i]
            Sumsqr += (periodogram[i]*periodogram[i])
            Ngood += 1
    
    Sum /= float(Ngood)
    Sumsqr /= float(Ngood)
    
    aveper = Sum
    stdper = sqrt(Sumsqr - Sum*Sum)
    
    # Sigma clip the light curve? I'm inclined to ignore this...
    #nclippedthis = 0
    #while nclippedthis > nclippedlast:
    #   ...
    aveaov = aveper
    stddevaov = stdper
    
    #print "aveper:", aveper
    #print "stdper:", stdper
    #sys.exit(0)
        
    # Make copy of the periodogram to return later
    periods_copy = np.zeros(Nperiod, dtype=float)
    periodogram_copy = np.zeros(Nperiod, dtype=float)
    
    for i in range(Nperiod):
        periods_copy[i] = periods[i]
        periodogram_copy[i] = periodogram[i]
    
    # Replace the periodogram with only points that are local minima
    lastpoint = periodogram[0] - 1.
    
    i = 0
    k = 0
    while k < (Nperiod-1):
        
        if (periodogram[k] < lastpoint) and (periodogram[k] < periodogram[k+1]):
            lastpoint = periodogram[k]
            periodogram[i] = periodogram[k]
            periods[i] = periods[k]
            i += 1
        else:
            lastpoint = periodogram[k]
        
        k += 1 
    
    if periodogram[k] < lastpoint:
        periodogram[i] = periodogram[k]
        periods[i] = periods[k]
        i += 1
    
    Nperiod = i
    
    # Search through the periodogram to identify the best Npeaks periods
    foundsofar = 0
    i = 0
    
    while (foundsofar < Npeaks) and (i < Nperiod):
        if (periodogram[i] < ERROR_SCORE) and (periodogram[i]*0.0 == 0.0):
            test = 1
            for j in range(foundsofar):
                if not isDifferentPeriods(periods[i], perpeaks[j], T):
                    if periodogram[i] < aovpeaks[j]:
                        perpeaks[j] = periods[i]
                        aovpeaks[j] = periodogram[i]
                    test = 0
                    break
            
            if test:
                perpeaks[foundsofar] = periods[i]
                aovpeaks[foundsofar] = periodogram[i]
                foundsofar += 1
        
        i += 1
    
    if foundsofar < Npeaks:
        for k in range(foundsofar, Npeaks):
            perpeaks[k] = ERROR_SCORE + 1.
            aovpeaks[k] = ERROR_SCORE + 1.
    
    ix = np.argsort(aovpeaks)
    perpeaks = perpeaks[ix]
    aovpeaks = aovpeaks[ix]
    
    minbest = aovpeaks[Npeaks-1]
    
    while i < Nperiod:
        if (periodogram[i] < ERROR_SCORE) and (periodogram[i]*0.0 == 0.0):
            if periodogram[i] < minbest:
                test = 1
                
                for j in range(Npeaks):
                    if not isDifferentPeriods(periods[i], perpeaks[j],T):
                        
                        if periodogram[i] < aovpeaks[j]:
                            aovpeaks[j] = periodogram[i]
                            perpeaks[j] = periods[i]
                            ix = np.argsort(aovpeaks)
                            aovpeaks = aovpeaks[ix]
                            perpeaks = perpeaks[ix]
                            minbest = aovpeaks[Npeaks - 1]
                        test = 0
                        break
                
                if test:
                    perpeaks[Npeaks - 1] = periods[i]
                    aovpeaks[Npeaks - 1] = periodogram[i]
                    ix = np.argsort(aovpeaks)
                    aovpeaks = aovpeaks[ix]
                    perpeaks = perpeaks[ix]
                    minbest = aovpeaks[Npeaks - 1];        
        
        i += 1
    
    # Now do the high-resolution period scan on the peaks
    smallfreqstep = finetune / T
    for j in range(Npeaks):
        freq = fmin( (1./perpeaks[j]) + freqstep, (1./minp) )
        minfreq = fmax( (1./perpeaks[j]) - freqstep, (1./maxp) )
        
        while freq >= minfreq:
            testperiod = 1./freq
            # TODO APW: Shiii... gotta implement this
            score = testperiod_asczerny(time, mag, testperiod, Nbin)
            
            if (score < ERROR_SCORE) and (score*0.0 == 0.0):
                if score < aovpeaks[j]:
                    aovpeaks[j] = score
                    perpeaks[j] = testperiod
            
            freq -= smallfreqstep
    
        bestscore = aovpeaks[j]
        ismultiple = 0
        
        for a in range(1, MAX_DOUBLE_CHECK_MULTIPLE+1):
            for b in range(1, MAX_DOUBLE_CHECK_MULTIPLE+1):
                if a != b:
                    testperiod = perpeaks[j] * a / b
                    if (testperiod > minp) and (testperiod < maxp):
                        # TODO APW: Shiii... gotta implement this
                        score = testperiod_asczerny(time, mag, testperiod, Nbin)
                        
                        if (score < ERROR_SCORE) and (score*0.0 == 0.0):
                            if score < bestscore:
                                ismultiple = 1
                                abest = a
                                bbest = b
                                bestscore = score
        
        if ismultiple:
            perpeaks[j] = perpeaks[j] * abest / bbest
            aovpeaks[j] = bestscore
    
    ix = np.argsort(aovpeaks)
    aovpeaks = aovpeaks[ix]
    perpeaks = perpeaks[ix]
    
    aovSNR = np.zeros(Npeaks, dtype=float)
    
    for j in range(Npeaks):
        if aovpeaks[j] < ERROR_SCORE:
            aovSNR[j] = (aveper - aovpeaks[j]) / stdper
            aovpeaks[j] = -aovpeaks[j]
            a_ = 0.5*(float(Nbin - 1))
            b_ = 0.5*(float(N - Nbin))
        else:
            aovpeaks[j] = -ERROR_SCORE - 1.
            perpeaks[j] = 1.0
    
    return {"peak_period" : perpeaks,\
            "peak_power" : aovpeaks,\
            "period" : periods_copy,\
            "periodogram" : periodogram_copy,\
            "peak_SNR" : aovSNR}
    
    #return (perpeaks, aovpeaks, periods_copy, periodogram_copy)