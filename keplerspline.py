#! /usr/bin/env python
"""
The idea is based on Andrew Vanderburgs Keplerspline.pro
IDL module that iteratively fits 4th-orther B-splines
to the data to remove trends. 

Breakpoints for each campaign come from A.V.

the `knots` utility program defines inner knots
and is based on D. Spergel's bspline_bkpts.pro 
modified and simplified for SciPy
"""

import numpy as np
from astropy.io import fits
#import glob
#from scipy import signal
#from scipy import stats
#from scipy import interpolate
from scipy.interpolate import LSQUnivariateSpline

#import scipy.fftpack as fft
#import matplotlib.pyplot as plt
#import os
import fitsio


#here some changes
def get_knots(x, dt = None, npts = None, k=4,verbose=False):
    """
    determines the inner knots for a spline
    the satisfy the Shoenberg-Whiteney conditions
    """

    # if there is an empty list, return it and fail
    if len(x)<1:
        return x, 'fail'
    
    # Get the range in x
    x.sort() # sort x from low to high
    x_range = x[-1] - x[0]

    ##########################################################
    ## Get evenly spaced knots                               #
    ## knots must be internal to the                         #
    ## abcissa. We first generate                            # 
    ## a list evenly spaced on [min(x) + dt/2,max(x) - dt/2) #
    ##########################################################
    
    # if dt is given, use it
    if dt is not None:
        t = np.arange(x[0]+ dt/2.,x[-1]-dt/2.,dt)
	# if dt not given & npts is, divide
    elif npts is not None: 
        npts = int(npts)
        dt = x_range/(npts - 1.) # find dt
        t = np.arange(x[0]+ dt/2.,x[-1]-dt/2.,dt)
    # Default to 11 knots
    else:
        npts = 11
        dt = dt = x_range/(npts - 1.)
        print('Defaulting to %i knots. dt = %0.2f'%(npts,dt))
        t = np.arange(x[0]+ dt/2.,x[-1]-dt/2.,dt)
    
    ## Check Shoenberg-Whiteney conditions
    fmode = True,None
    
    ## Check condition again after 
    ## removing offending knots
    while fmode[0]:
        fmode = check_knots(x,t,k,verbose=verbose) # Schoenberg-Whitney conditions
        if not isinstance(fmode,bool):
            if verbose: print 'delete'
            if fmode[1]=='sw':
                t = np.delete(t,fmode[1])
            
    
    return t,fmode

def check_knots(x,t,k,verbose=True):
    m = len(x)
    #oldt = t.copy()
    t = np.concatenate(([x[0]]*(k+1), t, [x[-1]]*(k+1)))
    n = len(t)
    # condition 1
    if not np.all(t[k+1:n-k]-t[k:n-k-1] > 0):
        if verbose: print 'Failed condition 1 (t[k+1:n-k]-t[k:n-k-1]>0)'
        return False,'f1'
    if  not (k+1 <= n-k-1 <= m):
        # >2k+2 and < m-(k+1) point
        if verbose: print 'Failed condition 2 (too few points for order)'
        return False,'f2'
    if not np.all(t[1:]-t[:-1] >= 0):
        # monitonically increasing
        if verbose: print 'Failed condition 3a (monotonic abscissa)'
        return False,'f3'
    if not np.all(t[n-k-1:-1] <= t[n-k:]):
        # monitonically increasing
        if verbose: print 'Failed condition 3b (monotonic abscissa)'
        return False,'f3'
    
    # Schoenberg-Whitney Condition
    # i.e., there must be data between
    # every knot. 
    # implementation 1
    arr = []
    for j in xrange(n-k-1):
        arr.append(np.any((t[j] <= x[j:]) & (x[j:] <= t[j+k+1])))
    if not np.all(arr):
        return 'sw',np.where(~np.asarray(arr))[0]
    else:
        return True, None
    
def find_data_gaps(time):
    diff = np.diff(time)
    delta = np.median(np.diff(time))
    madev = np.median(np.abs(diff-delta))
    breaks = np.where(diff > 3*madev)
    return breaks  
    
def get_breaks(cadenceno, campaign = None, time=None):
    """
    if no campaign number is given it checks 
    to see if cadenceno is a K2 LC fits file
    
    For speed, preferably just pass cadenceno and campaign
    """
    if campaign is None:
        try:
            campaign = cadenceno[0].header['CAMPAIGN']
            cadenceno = cadenceno['BESTAPER'].data['CADENCENO']
        except:
            pass
    
    if campaign==3:
        breakp1 = np.where(cadenceno >= 100174)[0][0]
        breakp2 = np.where(cadenceno >= 101801)[0][0]
        breakp = [breakp1, breakp2]
    elif campaign==4:
        breakp1 = np.where(cadenceno >= 104222)[0][0]
        breakp2 = np.where(cadenceno >= 105854)[0][0]
        breakp = [breakp1, breakp2]
    elif campaign==5:
        breakp = np.where(cadenceno >= 109374)[0][0]
    elif campaign==6:
        breakp1 = np.where(cadenceno >= 111550)[0][0]
        breakp2 = np.where(cadenceno >= 113482)[0][0]
        breakp = [breakp1, breakp2]
    elif campaign==7:
        breakp = np.where(cadenceno > 117870)[0][0]
    elif campaign==8:
        breakp1 = np.where(cadenceno >= 120881)[0][0]
        breakp2 = np.where(cadenceno >= 121345)[0][0]
        breakp3 = np.where(cadenceno >= 121824)[0][0]
        breakp = [breakp1, breakp2, breakp3]
    elif campaign==10:
        breakp = find_data_gaps(time)
    else:
        breakp = []
            
    return breakp


def piecewise_spline(time, fcor, cadenceno, campaign = 0, mask = None, verbose=False,\
                    breakpoints = None, delta = None, return_knots = False, k = 4):
    """
    returns the piecewise spline fit for every x (or time) value
                    
    breakpoints are the **indices** of the break in the data
    
    time, fcor, and candenceno must have the same size/indexing
    for breakpoints to work
    
    if you must mask, pass the mask with the keyword, otherwise
    you will get unexpected results and offsets
    """
    
    if breakpoints is None:
        breakpoints = get_breaks(cadenceno, campaign, time=time)
    
    if mask is None:
        mask = np.full(cadenceno.shape,True,dtype=np.bool)
    
    condlist = []
    spl = np.asarray([])
    
    breakpoints = np.append(0,breakpoints).astype(int) #first segment starts at 0
    
    # create condlist defining interval start
    for breakpoint in breakpoints:
        condlist.append(cadenceno >= cadenceno[breakpoint])
    
    # isolate each interval with XOR        
    for i,c in enumerate(condlist[:-1]):
        condlist[i] = c ^ condlist[i+1]
    
    # Build up the spline array
    for cond in condlist:
        x = time[cond & mask]
        y = fcor[cond & mask]
        kn,fail_mode = get_knots(x, delta,verbose=verbose)
        #print len(kn),len(x)
        if not isinstance(fail_mode,bool):
            x = time[cond]
            y = fcor[cond]
            kn,_ = get_knots(x, delta,verbose=verbose)
            spl_part = LSQUnivariateSpline(time[cond],fcor[cond],t=kn,k=4)
        else:
            spl_part = LSQUnivariateSpline(x, y, t=kn, k=4 ) #eval spline
        spl = np.append(spl,spl_part(time[cond]))
    return spl.ravel()


def single_spline(time, fcor, mask = None, \
                     delta = None, return_knots = False, k = 4):
    """
    if you must mask, pass the mask with the keyword, otherwise
    you will get unexpected results and offsets
    """
    
    if mask is None:
        mask = np.full(time.shape,True,dtype=np.bool)
    
    x = time[mask]
    y = fcor[mask]
    srt = np.argsort(x)
    x = x[srt]
    y = y[srt]
    kn,fmode = get_knots(x, delta)
    #print x.min(),kn.min(),kn.max(),x.max()
    spl= LSQUnivariateSpline(x, y, t=kn, k=k ) #eval spline
    spl = spl(time)
    
    if return_knots:
        return spl.ravel(),kn
    else:
        return spl.ravel()
                
def get_k2_data(k2dataset):
    if isinstance(k2dataset,str):
        try:
            data = fitsio.FITS(k2dataset)
            t = data[1]['T'].read()
            f = data[1]['FCOR'].read()
            cadenceno = data[1]['CADENCENO'].read()
            campaign = data[0].read_header()['CAMPAIGN']
            mag = data[0].read_header()['KEPMAG']
            data.close()
        except:
            print 'Problem'
            data = fits.open(k2dataset)
            t = data[1].data['T']
            f = data[1].data['FCOR']
            campaign = data[0].header['CAMPAIGN']
            cadenceno = data[1].data['CADENCENO']
            mag = data[0].header['KEPMAG']
    else:
        t = k2dataset['BESTAPER'].data['T']
        f = k2dataset['BESTAPER'].data['FCOR']
        cadenceno = k2dataset['BESTAPER'].data['CADENCENO']
        campaign = k2dataset[0].header['CAMPAIGN']
        mag = k2dataset[0].header['KEPMAG']
    
    return t,f, cadenceno, campaign, mag

def detrend_iter_single(t,f,delta, k=4,low=3,high=3):
    '''
     My iterative detrending algorithm, based on the concept in Vandenberg & Johnson 2015
     borrowed clipping portion from scipy.stats.sigmaclip
     with substantial modifications. 
    '''
    clip = 1
    c = np.asarray(f).ravel()
    mask = np.full(c.shape,True,dtype=np.bool)
    i = 0
    while clip:
        i+=1
        c_trend = single_spline(t, c, mask=mask, k=k, delta=delta)
        c_detrend = (c - c_trend + 1)[mask] #get masked detreneded lighcurve
        c_mean = c_detrend.mean()
        c_std = c_detrend.std()
        size = c_detrend.size
        critlower = c_mean - c_std*low
        critupper = c_mean + c_std*high
        newmask = (c_detrend >= critlower) & (c_detrend <= critupper)
        mask[mask] = newmask
        clip = size - c[mask].size
        #print i,clip, np.sum(mask), len(t)
        #plt.plot(x[mask],c_trend[newmask])
    
    return t, c_trend, mask

    
    
def detrend_iter(k2dataset, delta, k = 4, low = 3, high = 3, cutboth=False,verbose=False):
    # My iterative detrending algorithm, based on the concept in Vandenberg & Johnson 2015
    # borrowed clipping portion from scipy.stats.sigmaclip
    # with substantial modifications. 
    
    if len(k2dataset) > 4:
        t = k2dataset['BESTAPER'].data['T']
        f = k2dataset['BESTAPER'].data['FCOR']
        cadenceno = k2dataset['BESTAPER'].data['CADENCENO']
        campaign = k2dataset[0].header['CAMPAIGN']
    else:
        t,f,cadenceno,campaign = k2dataset
    
    clip = 1
    c = np.asarray(f).ravel()
    mask = np.full(c.shape,True,dtype=np.bool)
    outmask = np.full(c.shape,True,dtype=np.bool)
    if isinstance(low,tuple):
        low, masklow = low
        if np.isinf(masklow):
            None
    i = 0
    while clip:
        i+=1
        c_trend = piecewise_spline(t, c, cadenceno, campaign=campaign, mask=mask, k=k, delta=delta,verbose=verbose)
        c_detrend = (c - c_trend + 1)[mask] #get masked detreneded lighcurve
        c_mean = c_detrend.mean()
        c_std = c_detrend.std()
        size = c_detrend.size
        critlower = c_mean - c_std*high
        critupper = c_mean + c_std*high
        newmask = (c_detrend >= critlower) & (c_detrend <= critupper)
        outmask[mask] = c_detrend <= critupper
        mask[mask] = newmask
        clip = size - c[mask].size
        #print i,clip, np.sum(mask), len(t)
        #plt.plot(x[mask],c_trend[newmask])
    
    if cutboth:
        outmask = mask
    return t, c_trend, outmask
