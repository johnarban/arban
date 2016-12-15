import glob
import os,sys
import fitsio
import numpy as np
from astropy.io import fits
import keplerspline as bs
import pdb
from astropy.stats import LombScargle,sigma_clipped_stats
from scipy import stats,signal,optimize
from scipy.interpolate import LSQUnivariateSpline
#import matplotlib.pyplot as plt

def regrid(x,y,diff=30.):
    #diff = np.min(np.diff(x))
    diff = diff/60./24.
    new_x = np.arange(x.min(),x.max(),diff)
    new_y = np.interp(new_x,x,y)
    gaps = np.where(np.diff(x) > .25)[0]
    for gap in gaps:
        bad = (new_x > x[gap]) & (new_x < x[gap+1])
        new_y[bad] = 1 # assign data mean 
    
    return diff,new_x,new_y

def scargle_slow(t,c):
    """
    ripped straight out of IDL scargle
    """
    time = t-t[0]
    n0 = len(time)
    horne = -6.363 + 1.93*n0 + 0.00098*n0**2
    nfreq = int(horne)
    fmin = 1./max(time)
    fmax = n0 / (2 * max(time))
    
    om = 2*np.pi * (fmin+(fmax-fmin)*np.arange(nfreq)/(nfreq -1.))
    
    cn = c - np.mean(c)
    
    px = np.zeros((nfreq,),dtype=float)
    for i in range(nfreq):
        tau = np.arctan(np.sum(np.sin(2*om[i]*time))/np.sum(np.cos(2*om[i]*time)))
        tau = tau/(2*om[i])
        
        co = np.cos(om[i]*(time-tau))
        si = np.sin(om[i]*(time-tau))
        
        px[i] = 0.5*(np.sum(cn*co)**2 / np.sum(co**2) + np.sum(cn*si)**2 / np.sum(si**2))
    var = np.mean(cn)
    if var != 0.:
        var = px/var
    nu = om/(2*np.pi)
    period = 1./nu
                
    return px,period
 
def lsfreq(t,fmin=None,fmax=None,freq=None):
    time = t-t[0]
    tmax = max(time)	
    n0 = len(t)
    horne = -6.363 + 1.93*n0 + 0.00098*n0**2
    nfreq = int(horne)
    if fmin is None:
        fmin = 1./tmax
    if fmax is None:
        fmax = n0 / (2 * tmax)
    
    if freq is None:
        nu = (fmin+(fmax-fmin)*np.arange(nfreq)/(nfreq -1.))
    else:
        nu=freq

    return nu

def scargle_fast(t,c,fmin=None,fmax=None,freq=None,norm='psd'):
    """
    Lomb-Scargle periodogram using 
    Horne specs for frequencies
    """
    time = t-t[0]
    n0 = len(time)
    horne = -6.363 + 1.93*n0 + 0.00098*n0**2
    nfreq = int(horne)
    if fmin is None:
        fmin = 1./max(time)
    if fmax is None:
        fmax = n0 / (2 * max(time))
    
    if freq is None:
        nu = (fmin+(fmax-fmin)*np.arange(nfreq)/(nfreq -1.))
    else:
        nu=freq
    
    
    px = LombScargle(time,c - np.mean(c)).power(nu,method='fast',normalization=norm)
                    
    return px,nu


def period_analysis(t, f, mask = None, dft = True, scargle = True, scipy = False, phase = False, fmin = None, fmax = None,lsfreq=None):
    
    if mask is None:
        mask = np.full(t.shape,True,dtype=np.bool)
    else:
        mask = np.asarray(mask, dtype=np.bool)
    
    ret = ()
    if dft:
        dnu, new_t, new_f = regrid(t[mask],f[mask]) #grid to 30 minutes
        fft = np.abs(np.fft.fft(new_f))
        fftfreq = np.fft.fftfreq(len(new_f),d=dnu) # cycles/second
        if not phase:
            ret = ret + (fft,fftfreq)
        else:
            ret = ret + (fft,np.angle(fft),fftfreq)

    if scargle:
        lmscrgl,lmsfreq = scargle_fast(t[mask],f[mask],fmin=fmin,fmax=fmax,freq=lsfreq)
        ret = ret + (lmscrgl, lmsfreq)
    
    if scipy:
        fprdgrm,prdgrm = signal.periodogram(new_f,fs=1./dnu)
        ret = ret + (prdgrm,fprdfrm)
            
    return ret
        
################################
################################
### Folding the light-curve ####
################################
################################    
def ppp(fft, freq,angle=None):
    pos = freq > 0
    fft = fft[pos]
    freq = freq[pos]
    peak = np.where(fft == max(fft))[0][0]
    period = 1./freq[peak]
    if angle is None:
        return peak,period, None
    else:
        angle = angle[pos]
        phase = angle[peak]
        return peak, period, phase

def t2phi(time,period,phase=None):
    if phase is None:
        return (time/period % 1) - 0.5
    else:
        return ((time + period*phase/np.pi)/period % 1 ) - 0.5

def bindata(x, y, mode = 'mean', return_std = False, nbins=30):
    N, binex = np.histogram(x, bins=nbins)
    binx = (binex[1:] + binex[:-1])/2.
    mean = stats.binned_statistic(x,y,bins=binex,statistic=mode)[0]
    ## shift bins to center minimum
    min = np.where(mean == np.nanmin(mean))[0]
    phase_offset = binx[min]
    if return_std:
        std = stats.binned_statistic(x,y,bins=binex,statistic=np.std)[0]    
        return binx, mean, phase_offset, std
    else:
        return binx, mean, phase_offset


def var_period(t, f, period,nbins=30):

    
    def func(P):
        #Ps.append(P)
        p = t2phi(t,P)
        srt = np.argsort(p)
        p = p[srt]
        fp = f[srt]
        kn = np.linspace(-0.5,0.5,25)[1:-1]
        try:
            spl = LSQUnivariateSpline(p, fp, t=kn, k=1 )
        except:
            try:
                kn = np.linspace(-0.5,0.5,50)[1:-1]
                spl = LSQUnivariateSpline(p, fp, t=kn, k=1 )
            except:
                return np.inf
        #std.append(np.std(fp - spl(p)))
        return np.std(fp - spl(p))**2
    
    res = optimize.minimize(func,[period],method='Powell')#,bounds=[(period*0.4,period*2.4)])
    #Ps = np.logspace(np.log10(1./24.),np.log10(100./24.),30000)
    #std = map(func,Ps)
    #print (period - res.x)*24*60
    return res.x




















