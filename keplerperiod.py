# import glob
import os
import sys
# import fitsio
import numpy as np
# from astropy.io import fits
# import keplerspline as bs
# import pdb
from astropy.stats import LombScargle  # ,sigma_clipped_stats
from scipy import stats, signal, optimize
from scipy.interpolate import LSQUnivariateSpline
# import matplotlib.pyplot as plt
# import peakutils as peakutil
#import bls

#> sys.platform
#> win32 or linus2 or macosx


def peak_detect(y, thres=0.3, min_dist=1):
    '''Peak detection routine.
    Finds the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks.
    Parameters
    ----------
    y : ndarray
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.
    Returns
    -------
    ndarray
        Array containing the indexes of the peaks that were detected

    stolen from https://pypi.python.org/pypi/PeakUtils :)
    '''
    thres *= np.max(y) - np.min(y)
    print(('Threshold: %0.3f' % thres))
    # find the peaks by using the first order difference
    dy = np.diff(y)
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                   & (np.hstack([0., dy]) > 0.)
                   & (y > thres))[0]

    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
           if not rem[peak]:
               sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
               rem[sl] = True
               rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks


def regrid(x, y, diff=None, gap_width=.25):
    """
    inputs x, y
    optional inputs diff, gap_width

    returns: spacing, new_x, new_y
    """
    if diff is None:
        diff = np.nanmedian(np.diff(x))
    else:
        diff = diff/60./24.

    new_x = np.arange(x.min(), x.max(), diff)
    new_y = np.interp(new_x, x, y)
    gaps = np.where(np.diff(x) > gap_width)[0]
    for gap in gaps:
        bad = (new_x > x[gap]) & (new_x < x[gap+1])
        new_y[bad] = 1  # assign data mean

    return diff, new_x, new_y


def scargle_slow(t, c):
    """
    ripped straight out of IDL scargle
    """
    time = t-t[0]
    n0 = len(time)
    horne = -6.363 + 1.93*n0 + 0.00098*n0**2
    nfreq = int(horne)
    fmin = 1./max(time)
    fmax = n0 / (2 * max(time))

    om = 2*np.pi * (fmin+(fmax-fmin)*np.arange(nfreq)/(nfreq - 1.))

    cn = c - np.mean(c)

    px = np.zeros((nfreq,), dtype=float)
    for i in range(nfreq):
        tau = np.arctan(
           np.sum(np.sin(2*om[i]*time))/np.sum(np.cos(2*om[i]*time)))
        tau = tau/(2*om[i])

        co = np.cos(om[i]*(time-tau))
        si = np.sin(om[i]*(time-tau))

        px[i] = 0.5*(np.sum(cn*co)**2 / np.sum(co**2) +
                   np.sum(cn*si)**2 / np.sum(si**2))
    var = np.mean(cn)
    if var != 0.:
        var = px/var
    nu = om/(2*np.pi)
    period = 1./nu

    return px, period


def horne(n0):
    # number of frequencies via horne algorithm
    # wrt to number of data points
    return int(-6.363 + 1.93*n0 + 0.00098*n0**2)


def clear_windows(nu, freq, width, aliases=True):
    if aliases:
        return nu[~((nu % freq) <= width)]
    else:
        return nu[~((nu <= freq + width) & (nu <= freq + width))]


def lsfreq(t, fmin=None, fmax=None, freq=None):
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
        nu = (fmin+(fmax-fmin)*np.arange(nfreq)/(nfreq - 1.))
    else:
        nu = freq

    # nu = clear_windows(nu, 47.2e-6 * 86400., .25e-6 * 86400.,aliases=True)

    return nu


def scargle_fast(t, c, fmin=None, fmax=None, freq=None, norm='psd', window=False):
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
        # fmax is 2/<dt> or ~ nyquist
        fmax = n0 / (2 * max(time))

    if freq is None:
        nu = (fmin+(fmax-fmin)*np.arange(nfreq)/(nfreq - 1.))
    else:
        nu = freq

    # nu = clear_windows(nu, 47.2e-6 * 86400., .25e-6 * 86400.,aliases=True)

    if window:
        px = LombScargle(time, c, fit_mean=False, center_data=False).power(
           nu, method='fast', normalization=norm)
    else:
        px = LombScargle(time, c, fit_mean=True).power(
           nu, method='fast', normalization=norm)

    return px, nu


def box_least_squares(time, flux, pmin, pmax):
    u = np.empty_like(flux)
    v = np.empty_like(flux)
    # nf = 338
    nf = horne(len(time))
    df = 0.0007224
    nb = 50
    qmi = 0.01
    qma = 0.5
    fmax = 1/pmin
    fmin = 1/pmax
    df = (fmax - fmin)/nf
    nu = fmin + np.arange(nf)*df
    # nf = (fmax-fmin)/df //1
    results = bls.eebls(time, flux, u, v, nf, fmin, df, nb, qmi, qma)
    return results, nu


def period_analysis(t, f, mask=None, dft=False, scargle=False, nft=False, bls=False, fmin=None, fmax=None, lfreq=None, window=False):
    '''
    period_analysis(t, f, mask = None,
                      dft = True, scargle = True, nft = False
                      fmin = None, fmax = None, lsfreq=None,
                      window=False)

    get periodograms for a specific dataset
    t : time step
    f:  flux
    mask: which data points to keep [True] or skip [False]
    scargle: default True, return fast LombScargle with psd normalization
    fmin : frequency min
    fmax : frequency min
    lsfreq : pass a list of frequencies for LS or nufft
    window : find the window function

    returns periodogram, frequency for each element
    '''
    if mask is None:
        mask = np.full(t.shape, True, dtype=np.bool)
    else:
        mask = np.asarray(mask, dtype=np.bool)

    ret = ()

    if dft:
        # print 'DFT'
        dnu, new_t, new_f = regrid(t[mask], f[mask])  # grid to 30 minutes
        fft = np.abs(np.fft.fft(new_f))
        fftfreq = np.fft.fftfreq(len(new_f), d=dnu)  # cycles/second
        ret = ret + (fft, fftfreq, None, None, 'DFT')

    if scargle:
        # print 'L-S'
        ls, ls_freq = scargle_fast(
           t[mask], f[mask], fmin=fmin, fmax=fmax, freq=lfreq, window=False)
        ret = ret + (ls, ls_freq, None, None, 'LS')

    if nft:
        # print 'NUFFT'
        freq = lsfreq(t, fmin=fmin, fmax=fmax, freq=lfreq)
        nfft, nfreq = nufft_j(t[mask], f[mask], freq=freq)
        ret = ret + (nfft, nfreq, None, None, 'NUFFT')

    if bls:
        # print 'BLS'
        results, bls_freq = box_least_squares(
           t[mask], f[mask], pmin=1/fmax, pmax=1/fmin)
        power, best_period, best_power, depth, q, in1, in2 = results
        ret = ret + (power, bls_freq, best_period, best_power, 'BLS')

    return ret


################################
################################
### Folding the light-curve ####
################################
################################
def ppp(fft, freq, angle=None, use_peaks=False, thresh=0.5, min_dist=0.):
    if not use_peaks:
        pos = freq > 0
        fft = fft[pos]
        freq = freq[pos]
        peak = np.where(fft == max(fft))[0][0]
        if fft[peak] > thresh * (fft.max()-fft.min()):
           peak = peak
        else:
           return np.nan,np.nan,None
    else:
        peaks = peak_detect(fft,thres=thresh,min_dist=min_dist)
        if peaks.size == 0:
           return np.nan,np.nan,None
        peak = peaks[0]
        # peak = np.sort(peaks[np.argsort(fft[peaks])[::-1][:3]])[0]
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
    if not all(np.isnan(x)):
        N, binex = np.histogram(x, bins=nbins)
        binx = (binex[1:] + binex[:-1])/2.
        mean = stats.binned_statistic(x,y,bins=binex,statistic=mode)[0]
        count = stats.binned_statistic(x,y,bins=binex,statistic='count')[0]
        # shift bins to center minimum
        imin = np.where(mean == np.nanmin(mean))[0]
        phase_offset = binx[imin]
        if return_std:
           std = stats.binned_statistic(x,y,bins=binex,statistic=np.std)[0]/(np.sqrt(count)-1)
           return binx, mean, phase_offset, std
        else:
           return binx, mean, phase_offset
    else:
        if return_std:
           return np.linspace(-.5,.5,nbins),np.ones((nbins,)),np.nan,np.nan
        else:
           return np.linspace(-.5,.5,nbins),np.ones((nbins,)),np.nan




def var_period(t, f, period,nbins=60):

    def func(P):
        # Ps.append(P)
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
        # std.append(np.std(fp - spl(p)))
        return np.std(fp - spl(p))**2

    res = optimize.minimize(func,[period],method='Nelder-Mead')#,bounds=[(period*0.4,period*2.4)])
    # Ps = np.logspace(np.log10(1./24.),np.log10(100./24.),30000)
    # std = map(func,Ps)
    # print (period - res.x)*24*60
    return res.x


#==================================================
#==================================================
#   NUFFT programs #
# if True:

import nufft
def freq_grid(t,fmin=None,fmax=None,oversamp=10.,pmin=None,pmax=None):
    if pmax is not None:
        if pmax==pmin:
           pmax=10*pmax
        fmin=1./pmax
    if pmin is not None:
        if pmax==pmin:
           pmin=.1*pmin
        fmax = 1./pmin

    dt = t.max()-t.min()
    nyquist = 2./dt
    df = nyquist/oversamp
    Nf = 1 + int(np.round((fmax-fmin)/df))
    return fmin + df * np.arange(Nf)





def nufft_j(x, y, freq = None, period_max=1., period_min=.5/24, window=False, oversamp=10.):
    """
    nufft_j(x, y, period_max=1.,
      period_min=.5/24, window=False, oversamp=10.):

    Basic STFT algorithm
    for evenly sampled data
    """
    srt = np.argsort(x)
    x = x[srt] # get sorted x, y arrays
    y = y[srt]

    if freq is None:
      # Get a good frequency sampling, based on scargle in IDL
      # freq = LombScargle(x,y).autofrequency()
      #    minimum_frequency=1./period_max,maximum_frequency=1./period_min)
      freq = freq_grid(x,fmin=1./period_max,fmax=1./period_min,oversamp=oversamp)
      # create array to hold fft results

    fft = np.zeros_like(freq)


    if window:
        np.absolute(nufft.nufft3(x,y/y,freq*np.pi*2),out=fft)
    else:
        np.absolute(nufft.nufft3(x,y-np.nanmean(y),freq*np.pi*2),out=fft)


    return fft,freq



def PDM(x,y,periods):
    p = x[:,np.newaxis]/period % 1
    return None












