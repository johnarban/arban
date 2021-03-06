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
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline,interp1d
from scipy.special import erf,erfc,betainc,binom
from scipy.signal import medfilt


# check if fitsio is installed
try:
    import fitsio
    nofitsio = False
except ImportError:
    nofitsio = True

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


#here some changes
def get_knots(x, dt = None, npts = None, k=4,verbose=False):
    """
    determines the inner knots for a spline
    the satisfy the Shoenberg-Whiteney conditions
    """

    # if there is an empty list, return it and fail
    n = len(x)
    if n<1:
        return x, (True,)
    
    # Get the range in x
    x = np.array(x)
    x.sort() # sort x from low to high
    x_range = x[-1] - x[0]
    

    ##########################################################
    ## Get evenly spaced knots                               #
    ## knots must be internal to the                         #
    ## abcissa. We first generate                            # 
    ## a list evenly spaced on [min(x) + dt/2,max(x) - dt/2) #
    ## OLD         #t = np.arange(x[0]+ dt/2.,x[-1]-dt/2.,dt)
    ##########################################################
    
    # if dt is given, use it
    if dt is not None:
        npts = int(x_range / dt) + 1
        tempdt = x_range/(npts - 1.)
        if npts < 2: npts = 2
        t = np.arange(npts,dtype=float) * tempdt + x[0]
	# if dt not given & npts is, divide
    elif npts is not None: 
        npts = int(npts)
        tempdt = x_range/(npts - 1.)
        t = np.arange(npts,dtype=float) * tempdt + x[0]
    else:
        npts = 11
        tempdt = x_range/(npts - 1.)
        print(('Defaulting to %i knots. dt = %0.2f'%(npts,dt)))
        t = np.arange(npts,dtype=float) * tempdt + x[0]
        
    if np.nanmin(x) < np.min(t):
        t[np.argmin(t)] = np.nanmin(x)
    if np.nanmax(x) > np.max(t):
        t[np.argmax(t)] = np.nanmax(x)
    
    
    t = t[(t>np.min(x)) & (t<np.max(x))] # LSQUnivariateSpline internally adds boundary knots 
    # https://github.com/scipy/scipy/issues/5916#issuecomment-191346579
    
    ## Check Shoenberg-Whiteney conditions
    ## set fmode to True so that it actually starts 
    ## checking. Keep checking until fmode[0] is False,
    ## indicating that knots satisfy SW conditions
    fmode = True, None 

    ## Check condition again after 
    ## removing offending knots
    while fmode[0]:
        if verbose and not fmode[0]: print("Checking Schoenberg-Whitney")
        #fmode contains bool for if failed, and list of where it fails
        fmode = check_knots(x,t,k,verbose=verbose) # Schoenberg-Whitney conditions
        if fmode[0]=='sw':
            if verbose:
                print('Deleting %s knots'%len(fmode[1]))
            t = np.delete(t,fmode[1])
            fmode=True,None # set to recheck SW conditions
        elif fmode[1]=='f3':
            t = np.unique(t) # sort and recheck SW conditions
        elif fmode[1]=='f2':
            return t,(True,None) # Let us know it failed
        elif fmode[1]=='f1':
            return None, (True, None) # Let us know if failed

    return t,fmode

def check_knots(x,t,k,verbose=True):
    '''
    returns bool,fmode or 'sw', [indices where it fails]
    bool: Did it fail SW conditions
    fmode: f1 - fails a uniqueness conditions
           f2 - too few points
           f3 - not monotonic increasing
           sw - fails SW conditions
    '''
    m = len(x)
    #oldt = t.copy()
    t = np.concatenate(([x[0]]*(k+1), t, [x[-1]]*(k+1)))
    n = len(t)
    # condition 1
    if not np.all(t[k+1:n-k]-t[k:n-k-1] > 0):
        if verbose: print('Failed condition 1 (t[k+1:n-k]-t[k:n-k-1]>0)')
        return True,'f1'
    if  not (k+1 <= n-k-1 <= m):
        # >2k+2 and < m-(k+1) point
        if verbose: print('Failed condition 2 (too few points for order)')
        return True,'f2'
    if not np.all(t[1:]-t[:-1] >= 0):
        # monitonically increasing
        if verbose: print('Failed condition 3a (monotonic abscissa)')
        return True,'f3'
    if not np.all(t[n-k-1:-1] <= t[n-k:]):
        # monitonically increasing
        if verbose: print('Failed condition 3b (monotonic abscissa)')
        return True,'f3'
    
    # Schoenberg-Whitney Condition
    # i.e., there must be data between
    # every knot. 
    # implementation 1
    arr = []
    for j in range(n-k-1):
        arr.append(np.any((t[j] <= x[j:]) & (x[j:] <= t[j+k+1])))
    if not np.all(arr):
        if verbose: print("Failed Schoenberg-Whitney")
        return 'sw',np.where(~np.asarray(arr))[0]
    
    return False, None
    
def find_data_gaps(time, delta = 1.5):
    '''
    return np.where(np.diff(time) > delta).tolist()
    delta = 1.5 [default]
    '''
    if time is None:
        return []
    else:
        return np.where(np.diff(time) > delta)[0].tolist()

def where(condition):
    wh = np.where(condition)[0]
    if len(wh) > 0:
        return [wh[0]-1]
    else:
        return wh

    
def get_breaks(cadenceno, campaign = None, time=None, dt = 1.5):
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

    breakp = [0]
    if campaign==3:
        breakp.extend(where(cadenceno > 100174))
        breakp.extend(where(cadenceno > 101801))
        
    elif campaign==4:
        breakp.extend(where(cadenceno > 104222))
        breakp.extend(where(cadenceno > 105854))
        
    elif campaign==5:
        breakp.extend(where(cadenceno > 109374))
        
    elif campaign==6:
        breakp.extend(where(cadenceno > 111550))
        breakp.extend(where(cadenceno > 113482))
        
    elif campaign==7:
        breakp.extend(where(cadenceno > 117870))
        
    elif campaign==8:
        breakp.extend(where(cadenceno > 120881))
        breakp.extend(where(cadenceno > 121345))
        breakp.extend(where(cadenceno > 121824))
        
    #elif campaign==102:
    #    breakp = find_data_gaps(time)
    elif campaign==111: # bad breakpoints for c11
        breakp.extend(where(cadenceno > 134742))
        
    elif campaign==12:
        breakp.extend(where(cadenceno > 137332))
        
    elif campaign==13:
        breakp.extend(where(cadenceno > 141356))
        breakp.extend(where(cadenceno > 143095))
        
    elif campaign==14:
        breakp.extend(where(cadenceno > 145715))
        breakp.extend(where(cadenceno > 147539))
        
    else:
        breakp.extend([])

    breakp.extend(find_data_gaps(time, delta = dt))

    return np.unique(breakp).tolist()


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
    
    if mask is None:
        mask = np.full(cadenceno.shape,True,dtype=np.bool)
    
    if breakpoints is None:
        breakpoints = get_breaks(cadenceno, campaign, time=time, dt = delta)
    
    condlist = []
    spl = np.asarray([])
    
    #breakpoints = np.append(0,breakpoints).astype(int) #first segment starts at 0
    
    # create condlist defining interval start
    for i,breakpoint in enumerate(breakpoints):
        if i < len(breakpoints)-1:
            condlist.append((cadenceno >= cadenceno[breakpoint]) & (cadenceno < cadenceno[breakpoints[i+1]]))
        else:
            condlist.append((cadenceno >= cadenceno[breakpoint]))
    
    ## isolate each interval with XOR        
    #for i,c in enumerate(condlist[:-1]):
    #    condlist[i] = c ^ condlist[i+1]
    
    # Build up the spline array
    for cond in condlist:
        x = time[cond & mask]
        y = fcor[cond & mask]
        kn,fail_mode = get_knots(x, delta,verbose=verbose,k=k)
        if fail_mode[0]:
            if verbose: print('Couldn\'t find knots. Using LSQUnivariate Spline w/o mask')
            x = time[cond]
            y = fcor[cond]
            kn,_ = get_knots(x, delta,verbose=verbose)
            spl_part = LSQUnivariateSpline(time[cond & mask],fcor[cond & mask], t=kn, k=k)
        else:
            spl_part = LSQUnivariateSpline(x, y, t=kn, k=k ) #eval spline
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
            if not nofitsio:
                data = fitsio.FITS(k2dataset)
                t = data[1]['T'].read()
                f = data[1]['FCOR'].read()
                firing = data[1]['MOVING'].read()
                cadenceno = data[1]['CADENCENO'].read()
                campaign = data[0].read_header()['CAMPAIGN']
                mag = data[0].read_header()['KEPMAG']
                data.close()

            else:
                data = fits.open(k2dataset)
                t = data[1].data['T']
                f = data[1].data['FCOR']
                firing = data[1].data['MOVING']
                campaign = data[0].header['CAMPAIGN']
                cadenceno = data[1].data['CADENCENO']
                mag = data[0].header['KEPMAG']
        except:
            print('Problem')
            data = fits.open(k2dataset)
            t = data[1].data['T']
            f = data[1].data['FCOR']
            firing = data[1].data['MOVING']
            campaign = data[0].header['CAMPAIGN']
            cadenceno = data[1].data['CADENCENO']
            mag = data[0].header['KEPMAG']
    else:
        t = k2dataset['BESTAPER'].data['T']
        f = k2dataset['BESTAPER'].data['FCOR']
        firing = k2dataset['BESTAPER'].data['FIRING']
        cadenceno = k2dataset['BESTAPER'].data['CADENCENO']
        campaign = k2dataset[0].header['CAMPAIGN']
        mag = k2dataset[0].header['KEPMAG']
    g = firing == 0
#    import pdb;pdb.set_trace()
    return t[g],f[g], cadenceno[g], campaign, mag

def pmpoints(m,n,ns):
    '''
    m: number of points found
    n: number of trials
    ns: n*sigma for prob of finding a point
    '''
    m = m // 1 # np.floor(m)
    valid = np.ones_like(m)
    #np.sqrt(2) = 1.4142135623730951
    prob = erfc(ns/1.4142135623730951)
    #np.log(2) = 0.69314718055994529
    p = -(n*0.69314718055994529) + np.log(binom(n,m)) 
    p += (-m + n)*np.log(2 - prob) 
    p += m*np.log(prob)
    valid[(m < 0) | (m >n)] = 0
    return np.exp(p) * valid

def cmpoints(m,n,ns):
    m = m // 1 # np.floor(m)
    valid = np.ones_like(m)
    p = np.log(betainc(-m + n,1 + m,1 - erfc(ns/np.sqrt(2))/2.))
    valid[(m<0)] = 0.
    p[m>=n] = 1.
    return np.exp(p) * valid

def pgtnsigma(n):
    return 0.5 * erfc(n / np.sqrt(2.))

def mad(x,axis=None):
    return 1.4826*np.nanmedian(np.abs(x-np.nanmedian(x,axis=axis)),axis=axis)

def std1(x):
    '''
    get full standard deviation
    for top/bottom half of lc
    '''
    if np.mean(x-1) < 0:
        return mad(np.hstack([x-1,np.abs(x-1)]))
    else:
        return mad(np.hstack([x-1,-np.abs(x-1)]))

def statmask(f,sl=1e-5,sigma=3,perpoint=False):
    mxu=np.where(f>=1)[0]
    gfluxu = f[mxu]
    Mu = np.arange(0,len(gfluxu))[::-1]
    srtu = np.argsort(np.argsort(gfluxu))
    Mu = Mu[srtu]
    snrsu = np.abs(gfluxu-1) / std1(gfluxu)
    if perpoint:
        pu = 1-cmpoints(Mu,len(gfluxu),snrsu)
    else:
        pu = 1-cmpoints(Mu,len(gfluxu),sigma)

    mxd=np.where(f<1)[0]
    gfluxd = f[mxd]
    Md = np.arange(0,len(gfluxd))
    srtd = np.argsort(np.argsort(gfluxd))
    Md = Md[srtd]
    snrsd = np.abs(gfluxd-1) / std1(gfluxd)
    if perpoint:
        pd = 1-cmpoints(Md,len(gfluxd),snrsd)
    else:
        pd = 1-cmpoints(Md,len(gfluxd),sigma)

    mx = np.hstack([mxu,mxd])
    M = np.hstack([Mu,Md])
    gflux = np.hstack([gfluxu,gfluxd])
    srt = np.hstack([srtu,srtd])
    snrs = np.hstack([snrsu,snrsd])
    p = np.hstack([pu,pd])
    
    return (p<sl)[np.argsort(mx)]


def detrend_iter_single(t,f,delta, k=4,low=3,high=3,cutboth=False):
    '''
     My iterative detrending algorithm, based on the concept in Vandenberg & Johnson 2015
     borrowed clipping portion from scipy.stats.sigmaclip
     with substantial modifications. 
    '''
    clip = 1
    c = np.asarray(f).ravel()
    #mask = np.full(c.shape,True,dtype=np.bool)
    mask = np.isfinite(c)
    outmask = np.copy(mask) #np.full(c.shape,True, dtype=np.bool)
    i = 0
    while clip:
        i+=1
        c_trend = single_spline(t, c, mask=mask, k=k, delta=delta)
        c_detrend = (c - c_trend + 1)[mask] #get masked detreneded lighcurve
        c_mean = np.median(c_detrend) # use median for mean
        c_std = 1.4826 * np.median(np.abs(c_detrend - c_mean))# use mad # c_detrend.std()
        size = c_detrend.size
        critlower = c_mean - c_std*low
        critupper = c_mean + c_std*high
        newmask = (c_detrend >= critlower) & (c_detrend <= critupper)
        outmask[mask] = c_detrend <= critupper
        mask[mask] = newmask
        clip = size - c[mask].size
        print(i,clip, np.sum(mask), len(t))
        #plt.plot(x[mask],c_trend[newmask])
        if i > 50:
            clip = 0
    if cutboth:
        outmask = mask
    return t, c_trend, outmask

    
    
def detrend_iter(k2dataset, delta, delta_start=1.25, k = 4, low = 3, high = 3, cutboth=False,verbose=False,maxiter=50,sigma=5,sl=1e-5):
    # My iterative detrending algorithm, based on the concept in Vandenberg & Johnson 2015
    # borrowed clipping portion from scipy.stats.sigmaclip
    # with substantial modifications. 
    
    if len(k2dataset) > 5:
        t = k2dataset['BESTAPER'].data['T']
        f = k2dataset['BESTAPER'].data['FCOR']
        cadenceno = k2dataset['BESTAPER'].data['CADENCENO']
        campaign = k2dataset[0].header['CAMPAIGN']
    else:
        t,f,cadenceno,campaign,mag = k2dataset
    
    c = np.asarray(f).ravel()
    mask = np.full(c.shape,True,dtype=np.bool)
    outmask = np.full(c.shape,True,dtype=np.bool)
    if isinstance(low,tuple):
        low, masklow = low
        if np.isinf(masklow):
            None
    clip = 1
    clipped = len(t) - np.sum(mask)
    M = np.arange(len(c))
    i = 0
    if True:
        for dt in [delta_start,delta]:
            i = 0
            clip = 1
            while clip:
                i+=1
                c_trend = piecewise_spline(t, c, cadenceno, campaign=campaign, mask=mask, k=k, delta=dt,verbose=verbose)
                c_detrend = (c - c_trend + 1)[mask] #get masked detreneded lighcurve
                c_mean = c_detrend.mean()
                c_std = mad(c_detrend)
                size = c_detrend.size
                critlower = c_mean - c_std*low
                critupper = c_mean + c_std*high
                newmask = (c_detrend >= critlower) & (c_detrend <= critupper)

                outmask[mask] = c_detrend <= critupper
                mask[mask] = newmask
                clip = size - c[mask].size
                if verbose: print('iter: %i dt: %0.2f num clipped: %i num good: %i  num orig: %i clipped low: %i clipped high: %i'%(i,dt,clip, np.sum(mask), len(t),np.sum(c_detrend < critlower),np.sum(c_detrend > critupper)))
                if (i == maxiter):
                    clip = 0
        
        if cutboth:
            outmask = mask
        
        return t, c_trend, outmask  & statmask(c - c_trend + 1,sigma=sigma,sl=sl), dt