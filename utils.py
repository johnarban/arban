# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:11:33 2016

@author: johnlewisiii
"""
import matplotlib.colors as colors
import emcee
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate, interpolate, signal, special, stats
from weighted import quantile

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))

__filtertable__ = Table.read(os.path.join(
    __location__, 'FilterSpecs.tsv'), format='ascii')

#############################
#############################
####  Plotting commands  ####
#############################
#############################

# Set uniform plot options


def set_plot_opts(serif_fonts=True):

    if serif_fonts:
        mpl.rcParams['mathtext.fontset'] = 'stix'
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.size'] = 12
    return None


def get_cax(ax=None, size=3):
    if ax is None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="%f%%" % (size*1.), pad=0.05)
    # plt.sca(ax)
    return cax


# Plot the KDE for a set of x,y values. No weighting code modified from
# http://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
def kdeplot(xp, yp, filled=False, ax=None, grid=None, bw=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    rvs = np.append(xp.reshape((xp.shape[0], 1)),
                    yp.reshape((yp.shape[0], 1)),
                    axis=1)

    kde = stats.kde.gaussian_kde(rvs.T)
    #kde.covariance_factor = lambda: 0.3
    # kde._compute_covariance()
    kde.set_bandwidth(bw)

    if grid is None:
        # Regular grid to evaluate kde upon
        x_flat = np.r_[rvs[:, 0].min():rvs[:, 0].max():256j]
        y_flat = np.r_[rvs[:, 1].min():rvs[:, 1].max():256j]
    else:
        x_flat = np.r_[0:grid[0]:complex(0, grid[0])]
        y_flat = np.r_[0:grid[1]:complex(0, grid[1])]
    x, y = np.meshgrid(x_flat, y_flat)
    grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

    z = kde(grid_coords.T)
    z = z.reshape(x.shape[0], x.shape[1])
    if filled:
        cont = ax.contourf
    else:
        cont = ax.contour
    cs = cont(x_flat, y_flat, z, *args, **kwargs)
    return cs

#############################
#############################
# Convenience math functions
#############################
#############################


def freq_grid(t, fmin=None, fmax=None, oversamp=10., pmin=None, pmax=None):
    '''
    freq_grid(t,fmin=None,fmax=None,oversamp=10.,pmin=None,pmax=None)
    Generate a 1D list of frequences over a certain range
    '''
    if pmax is not None:
        if pmax == pmin:
            pmax = 10*pmax
        fmin = 1./pmax
    if pmin is not None:
        if pmax == pmin:
            pmin = .1*pmin
        fmax = 1./pmin

    dt = t.max()-t.min()
    nyquist = 2./dt
    df = nyquist/oversamp
    Nf = 1 + int(np.round((fmax-fmin)/df))
    return fmin + df * np.arange(Nf)


def sigconf1d(n):
    '''
    calculate the percentile corresponding to n*sigma
    for a 1D gaussian
    '''
    cdf = (1/2.)*(1+special.erf(n/np.sqrt(2)))
    return (1-cdf)*100, 100 * cdf, 100*special.erf(n/np.sqrt(2))


# In[ Convert tables to arrays]
def table_to_array(table):
    arr = [list(t) for t in table]
    return np.asarray(arr)


def t2a(table):
    return table_to_array(table)

# In[Discrete Colorbar]


def discrete_cmap(colormap, N_colors):
    print('Not doing anything yet')
    return None

# In[WCS axis labels]


def wcsaxis(wcs, N=6, ax=None, fmt='%0.2f', use_axes=False):
    if ax is None:
        ax = plt.gca()
    xlim = ax.axes.get_xlim()
    ylim = ax.axes.get_ylim()
    try:
        wcs = WCS(wcs)
    except:
        None
    hdr = wcs.to_header()
    naxis = wcs.naxis  # naxis
    naxis1 = wcs._naxis1  # naxis1
    naxis2 = wcs._naxis2  # naxis2
    #crpix1 = hdr['CRPIX1']
    #crpix2 = hdr['CRPIX2']
    #crval1 = hdr['CRVAL1']
    #crval2 = hdr['CRVAL2']
    # try:
    #    cdelt1 = wcs['CDELT1']
    #    cdelt2 = wcs['CDELT2']
    # except:
    #    cdelt1 = wcs['CD1_1']
    #    cdelt2 = wcs['CD2_2']

    if not use_axes:
        xoffset = (naxis1/N)/5
        x = np.linspace(xoffset, naxis1-xoffset, N)
        if naxis >= 2:
            yoffset = (naxis2/N)/5
            y = np.linspace(yoffset, naxis2-yoffset, N)
    else:
        x = ax.get_xticks()
        if naxis >= 2:
            y = ax.get_yticks()

    if naxis == 1:
        x_tick = wcs.all_pix2world(x, 0)
    elif naxis == 2:
        coord = list(zip(x, y))
        x_tick, y_tick = wcs.all_pix2world(coord, 0).T
    elif naxis > 2:
        c = [x, y]
        for i in range(naxis-2):
            c.append([0]*N)
        coord = list(zip(*c))
        ticks = wcs.all_pix2world(coord, 0)
        x_tick, y_tick = np.asarray(ticks)[:, :2].T

    plt.xticks(x, [fmt % i for i in x_tick])
    plt.yticks(y, [fmt % i for i in y_tick])

    if hdr['CTYPE1'][0].lower() == 'g':
        plt.xlabel('Galactic Longitude (l)')
        plt.ylabel('Galactic Latitude (b)')
    else:
        plt.xlabel('Right Ascension (J2000)')
        plt.ylabel('Declination (J2000)')

    ax.axes.set_xlim(xlim[0], xlim[1])
    ax.axes.set_ylim(ylim[0], ylim[1])
    return ax


# In[ writefits]
def writefits(filename, data, header=None, wcs=None, clobber=True):
    if header is None:
        if wcs is not None:
            header = wcs
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(filename, overwrite=clobber)
    return hdu


def grid_data(x, y, z, nxy=(512, 512), interp='linear', plot=False,
              cmap='Greys', levels=None, sigmas=None, filled=False):
    '''
    stick x,y,z data on a grid and return
    XX, YY, ZZ
    '''
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    nx, ny = nxy
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    xi, yi = np.meshgrid(xi, yi)

    zi = interpolate.griddata((x, y), z, (xi, yi), method=interp)

    if plot:
        if levels is None:
            if sigmas is None:
                sigmas = np.arange(0.5, 3.1, 0.5)
            else:
                sigmas = np.atleast_1d(sigmas)
            levels = (1.0 - np.exp(-0.5 * sigmas ** 2))
        ax = plt.gca()
        if filled:
            cont = ax.contourf
        else:
            cont = ax.contour
        cont(xi, yi, zi/np.max(zi[np.isfinite(zi)]), cmap=cmap, levels=levels)

    return xi, yi, zi


##########################################
##########################################
# A general utility to convert fluxes
# and magnitudes.
def convert_flux(mag=None, emag=None, filt=None, return_wavelength=False):
    """"Return flux for a given magnitude/filter combo

    Input:
    mag -- the input magnitude. either a number or numpy array
    filter -- either filter zeropoint or filer name
    """

    if mag is None or filt is None:
        print('List of filters and filter properties')
        __filtertable__.pprint(max_lines=len(__filtertable__)+3)
        return None

    if not isinstance(filt, float):
        tab = __filtertable__
        tab['fname'] = [s.lower() for s in tab['fname']]
        if not filt.lower() in tab['fname']:
            print('Filter %s not found' % filt.lower())
            print('Please select one of the following')
            print(tab['fname'].data)
            filt = eval(
                input('Include quotes in answer (example (\'johnsonK\')): '))

        f0 = tab['F0_Jy'][np.where(filt.lower() == tab['fname'])][0]
    else:
        f0 = filt

    flux = f0 * 10.**(-mag/2.5)

    if emag is not None:
        eflux = 1.08574 * emag * flux
        if return_wavelength:
            return flux, eflux, tab['Wavelength'][np.where(filt.lower() == tab['fname'])]
        else:
            return flux, eflux
    else:
        if return_wavelength:
            return flux, tab['Wavelength'][np.where(filt.lower() == tab['fname'])][0]
        else:
            return flux


# ================================================================== #
#
#  Function copied from schmidt_funcs to make them generally available
#

def rot_matrix(theta):
    '''
    rot_matrix(theta)
    2D rotation matrix for theta in radians
    returns numpy matrix
    '''
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])


def rectangle(c, w, h, angle=0, center=True):
    '''
    create rotated rectangle
    for input into PIL ImageDraw.polygon
    to make a rectangle polygon mask

    Rectagle is created and rotated with center
    at zero, and then translated to center position

    accepts centers
    Default : center
    options for center: tl, tr, bl, br
    '''
    cx, cy = c
    # define initial polygon irrespective of center
    x = -w / 2., +w / 2., +w / 2., -w / 2.
    y = +h / 2., +h / 2., -h / 2., -h / 2.
    # correct center if starting from corner
    if center is not True:
        if center[0] == 'b':
            # y = tuple([i + h/2. for i in y])
            cy = cy + h / 2.
        else:
            # y = tuple([i - h/2. for i in y])
            cy = cy - h / 2.
        if center[1] == 'l':
            # x = tuple([i + w/2 for i in x])
            cx = cx + w / 2.
        else:
            # x = tuple([i - w/2 for i in x])
            cx = cx - w / 2.

    R = rot_matrix(angle * np.pi / 180.)
    c = []

    for i in range(4):
        xr, yr = np.dot(R, np.asarray([x[i], y[i]])).A.ravel()
        # coord switch to match ordering of FITs dimensions
        c.append((cx + xr, cy + yr))
    # print (cx,cy)
    return c


def comp(arr):
    '''
    returns the compressed version
    of the input array if it is a
    numpy MaskedArray
    '''
    try:
        return arr.compressed()
    except:
        return arr


def mavg(arr, n=2, mode='valid'):
    '''
    returns the moving average of an array.
    returned array is shorter by (n-1)
    '''
    if len(arr) > 400:
        return signal.fftconvolve(arr, [1. / float(n)] * n, mode=mode)
    else:
        return signal.convolve(arr, [1. / float(n)] * n, mode=mode)


def mgeo(arr, n=2):
    '''
    Returns array of lenth len(arr) - (n-1)

    # # written by me
    # # slower for short loops
    # # faster for n ~ len(arr) and large arr
    a = []
    for i in xrange(len(arr)-(n-1)):
        a.append(stats.gmean(arr[i:n+i]))

    # # Original method# #
    # # written by me ... ~10x faster for short arrays
    b = np.array([np.roll(np.pad(arr,(0,n),mode='constant',constant_values=1),i)
              for i in xrange(n)])
    return np.product(b,axis=0)[n-1:-n]**(1./float(n))
    '''
    a = []
    for i in range(len(arr) - (n - 1)):
        a.append(stats.gmean(arr[i:n + i]))

    return np.asarray(a)


def avg(arr, n=2):
    '''
    NOT a general averaging function
    return bin centers (lin and log)
    '''
    diff = np.diff(arr)
    # 2nd derivative of linear bin is 0
    if np.allclose(diff, diff[::-1]):
        return mavg(arr, n=n)
    else:
        return np.power(10., mavg(np.log10(arr), n=n))
        # return mgeo(arr, n=n) # equivalent methods, only easier


def shift_bins(arr, phase=0, nonneg=False):
    # assume original bins are nonneg
    if phase != 0:
        diff = np.diff(arr)
        if np.allclose(diff, diff[::-1]):
            diff = diff[0]
            arr = arr + phase*diff
            #pre = arr[0] + phase*diff
            return arr
        else:
            arr = np.log10(arr)
            diff = np.diff(arr)[0]
            arr = arr + phase * diff
            return np.power(10., arr)
    else:
        return arr


def llspace(xmin, xmax, n=None, log=False, dx=None, dex=None):
    '''
    llspace(xmin, xmax, n = None, log = False, dx = None, dex = None)
    get values evenly spaced in linear or log spaced
    n [10] -- Optional -- number of steps
    log [false] : switch for log spacing
    dx : spacing for linear bins
    dex : spacing for log bins (in base 10)
    dx and dex override n
    '''
    xmin, xmax = float(xmin), float(xmax)
    nisNone = n is None
    dxisNone = dx is None
    dexisNone = dex is None
    if nisNone & dxisNone & dexisNone:
        print('Error: Defaulting to 10 linears steps')
        n = 10.
        nisNone = False

    # either user specifies log or gives dex and not dx
    log = log or (dxisNone and (not dexisNone))
    if log:
        if xmin == 0:
            print("log(0) is -inf. xmin must be > 0 for log spacing")
        xmin, xmax = np.log10(xmin), np.log10(xmax)
    # print nisNone, dxisNone, dexisNone, log # for debugging logic
    if not nisNone:  # this will make dex or dx if they are not specified
        if log and dexisNone:  # if want log but dex not given
            dex = (xmax - xmin) / n
            # print dex
        elif (not log) and dxisNone:  # else if want lin but dx not given
            dx = (xmax - xmin) / n  # takes floor
            print(dx)

    if log:
        # return np.power(10, np.linspace(xmin, xmax , (xmax - xmin)/dex + 1))
        return np.power(10, np.arange(xmin, xmax + dex, dex))
    else:
        # return np.linspace(xmin, xmax, (xmax-xmin)/dx + 1)
        return np.arange(xmin, xmax + dx, dx)


def nametoradec(name):
    '''
    Get names formatted as
    hhmmss.ss+ddmmss to Decimal Degree
    only works for dec > 0 (splits on +, not -)
    Will fix this eventually...
    '''
    if 'string' not in str(type(name)):
        rightascen = []
        declinatio = []
        for n in name:
            if '+' in n:
                ra, de = n.split('+')
                sign = ''
            elif '-' in n:
                ra, de = n.split('-')
                sign = '-'
            ra = ra[0:2] + ':' + ra[2:4] + ':' + ra[4:6] + '.' + ra[6:8]
            de = sign + de[0:2] + ':' + de[2:4] + ':' + de[4:6]
            coord = SkyCoord(ra, de, frame='icrs',
                             unit=('hourangle', 'degree'))
            rightascen.append(coord.ra.value)
            declinatio.append(coord.dec.value)
        return np.array(rightascen), np.array(declinatio)
    else:
        if '+' in n:
            ra, de = n.split('+')
            sign = ''
        elif '-' in n:
            ra, de = n.split('-')
            sign = '-'
        #ra, de = name.split('+')
        ra = ra[0:2] + ':' + ra[2:4] + ':' + ra[4:6] + '.' + ra[6:8]
        de = sign + de[0:2] + ':' + de[2:4] + ':' + de[4:6]
        coord = SkyCoord(ra, de, frame='icrs', unit=('hourangle', 'degree'))
        return np.array(coord.ra.value), np.array(coord.dec.value)


def pdf(values, bins=None, range=None):
    '''
    ** Normalized differential area function. **
    (statistical) probability denisty function
    normalized so that the integral is 1
    and. The integral over a range is the
    probability of the value is within
    that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    '''
    if hasattr(bins, '__getitem__') and (range is None):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, x = np.histogram(values, bins=bins, range=range, density=False)
    # From the definition of Pr(x) = dF(x)/dx this
    # is the correct form. It returns the correct
    # probabilities when tested
    pdf = h / (np.sum(h, dtype=float) * np.diff(x))
    return pdf, avg(x)


def pdf2(values, bins=None, range=None):
    '''
    N * PDF(x)
    The ~ PDF normalized so that
    the integral is equal to the
    total amount of a quantity.
    The integral over a range is the
    total amount within that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    '''
    if hasattr(bins, '__getitem__') and (range is None):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    pdf, x = np.histogram(values, bins=bins, range=range, density=False)
    pdf = pdf.astype(float) / np.diff(x)
    return pdf, avg(x)


def edf(data, pdf=False):
    y = np.arange(len(data), dtype=float)
    x = np.sort(data).astype(float)
    return y, x


def cdf(values, bins):
    '''
    CDF(x)
    (statistical) cumulative distribution function
    Integral on [-inf, b] is the fraction below b.
    CDF is invariant to binning.
    This assumes you are using the entire range in the binning.
    Returns array of size len(bins)
    Plot versus bins[:-1]
    '''
    if hasattr(bins, '__getitem__'):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range,
                           density=False)  # returns int

    # cumulative fraction below bin_k
    c = np.cumsum(h / np.sum(h, dtype=float))
    # append 0 to beginning because P( X < min(x)) = 0
    return np.append(0, c), bins


def cdf2(values, bins):
    '''
    # # Exclusively for area_function which needs to be unnormalized
    (statistical) cumulative distribution function
    Value at b is total amount below b.
    CDF is invariante to binning

    Plot versus bins[:-1]
    Not normalized to 1
    '''
    if hasattr(bins, '__getitem__'):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range, density=False)
    c = np.cumsum(h).astype(float)
    return np.append(0., c), bins


def area_function(extmap, bins):
    '''
    Complimentary CDF for cdf2 (not normalized to 1)
    Value at b is total amount above b.
    '''
    c, bins = cdf2(extmap, bins)
    return c.max() - c, bins


def diff_area_function(extmap, bins, scale=1):
    '''
    See pdf2
    '''
    s, bins = area_function(extmap, bins)
    dsdx = -np.diff(s) / np.diff(bins)
    return dsdx*scale, avg(bins)


def log_diff_area_function(extmap, bins):
    '''
    See pdf2
    '''
    s, bins = diff_area_function(extmap, bins)
    g = s > 0
    dlnsdlnx = np.diff(np.log(s[g])) / np.diff(np.log(bins[g]))
    return dlnsdlnx, avg(bins[g])


def mass_function(values, bins, scale=1, aktomassd=183):
    '''
    M(>Ak), mass weighted complimentary cdf
    '''
    if hasattr(bins, '__getitem__'):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range,
                           density=False, weights=values*aktomassd*scale)
    c = np.cumsum(h).astype(float)
    return c.max() - c, bins


def linregress(X, Y, pass_through_origin=True):
    if pass_through_origin:
        A = np.array([X*0 + 0, X]).T
    else:
        A = np.array([X*0 + 1, X]).T
    B = Y
    coeff, _r, _rank, _s = np.linalg.lstsq(A, B)
    return coeff


def mad(X, stddev=True):
    if stddev:
        return 1.4826*np.nanmedian(np.abs(X-np.nanmedian(X)))
    else:
        return np.nanmedian(np.abs(X-np.nanmedian(X)))


def rms(X, axis=None):
    return np.sqrt(np.nanmean(X**2, axis=axis))


def wcs_to_grid(wcs, index=False, verbose=False):
    try:
        wcs = WCS(wcs)
    except:
        None

    wcs = wcs.dropaxis(2)
    if verbose:
        print(wcs)
    naxis = wcs.naxis
    naxis1 = wcs._naxis1  # naxis1
    naxis2 = wcs._naxis2  # naxis2
    x, y = np.arange(naxis1), np.arange(naxis2)
    if not index:
        # first FITS pixel is 1, numpy index is 0
        xc, _ = wcs.all_pix2world(x, x*0, 0)
        _, yc = wcs.all_pix2world(y*0, y, 0)
        coord_grid = np.meshgrid(xc, yc)
    else:
        coord_grid = np.meshgrid(x, y)

    return coord_grid


def gauss(x, a, mu, sig):
    return a * np.exp(- (x-mu)**2 / (2 * sig**2))


def forward_fill_nan(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def ffill_nan_3d(arr):
    ''' foward fill 3d arrays along first axis
    from: https://stackoverflow.com/a/41191127

    '''
    shape = arr.shape
    arr = arr.T.reshape(np.product(shape[1:]), shape[0])
    mask = np.isnan(arr)
    print('1,', arr.shape, mask.shape)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out.T.reshape(shape)


def little_emcee_fitter(x, y, model=None, yerr=None,
                        loglike=None, lnprior=None,
                        nwalkers=10, theta_init=None, use_lnf=False):
    '''
    ## sample call
    sampler,pos = little_emcee_fitter(x,y,
        theta_init=np.array(mfit.parameters),
        use_lnf=True)
    samples = sampler.chain[:,1000:,:].reshape((-1,sampler.dim))

    corner.corner(samples,show_titles=True,
            quantiles=[.16,.84],
            labels=["$m$", "$b$", "$\ln\,f$"])
    ---------------------------------------------

    Arguments:
        x {np.array} -- x values as numpy array
        y {np.array} -- y values as numpy array

    Keyword Arguments:
        model {function that is called model(x,theta)} -- (default: {linear model})
        yerr {yerr as numpy array} -- options (default: {.001 * range(y)})
        loglike {custom loglikelihood} -- (default: {chi^2})
        lnprior {custom logprior} --  (default: {no prior})
        nwalkers {number of walkers} -- (default: {10})
        theta_init {initial location of walkers} -- [required for operation] (default: {Noalne})
        use_lnf {use jitter term} -- (default: {False})

    Returns:
        sampler, pos -- returns sampler and intial walker positions
    '''

    if yerr is None:
        yerr = np.full_like(y, 0.001 * (np.nanmax(y) - np.nanmin(y)))

    if model is None:
        def model(x, theta):
            m, b = theta
            return m*x + b

    if loglike is None:
        def lnlike(theta, x, y, yerr, model=model):
            # print(theta)
            if use_lnf:
                lnf = theta[-1]
                theta = theta[:-1]
            ymodel = model(x, theta)
            if use_lnf:
                inv_sigma2 = 1.0/(yerr**2 + ymodel**2*np.exp(2*lnf))
            else:
                inv_sigma2 = 1.0/yerr**2
            return -0.5*(np.sum((y-ymodel)**2*inv_sigma2 - np.log(inv_sigma2)))

    if lnprior is None:
        def lnprior(theta):
            if np.all(np.isfinite(theta)):
                return 0.0
            return -np.inf

    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y, yerr)

    if use_lnf:
        theta_init = np.append(theta_init, 0)
    ndim = len(theta_init)
    pos = [theta_init + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    return sampler, pos


# TODO

# Make it scale properly
# How does matplotlib
# scaling work
def custom_cmap(colormaps, lower, upper, log=(0, 0)):
    '''
    colormaps : a list of N matplotlib colormap classes
    lower : the lower limits for each colormap: array or tuple
    upper : the upper limits for each colormap: array or tuple
    log   : Do you want to plot logscale. This will create
            a color map that is usable with LogNorm()
    '''
    if isinstance(log, tuple):
        for lg in log:
            if lg:
                upper = [np.log10(i/lower[0]) for i in upper]
                lower = [np.log10(i/lower[0]) for i in lower]
                norm = upper[-1:][0]
            else:
                lower = lower
                upper = upper
                norm = upper[-1:][0]
    elif log:
        upper = [np.log10(i/lower[0]) for i in upper]
        lower = [np.log10(i/lower[0]) for i in lower]
        norm = upper[-1:][0]
    else:
        lower = lower
        upper = upper
        norm = upper[-1:][0]

    cdict = {'red': [], 'green': [], 'blue': []}

    for color in ['red', 'green', 'blue']:
        for j, col in enumerate(colormaps):
            # print j,col.name,color
            x = [i[0] for i in col._segmentdata[color]]
            y1 = [i[1] for i in col._segmentdata[color]]
            y0 = [i[2] for i in col._segmentdata[color]]
            x = [(i-min(x))/(max(x)-min(x)) for i in x]
            x = [((i * (upper[j] - lower[j]))+lower[j])/norm for i in x]
            if (j == 0) & (x[0] != 0):
                x[:0], y1[:0], y0[:0] = [0], [y1[0]], [y0[0]]
            for i in range(len(x)):  # first x needs to be zero
                cdict[color].append((x[i], y1[i], y0[i]))

    return colors.LinearSegmentedColormap('my_cmap', cdict)


def plot_2dhist(X, Y, xlog=True, ylog=True, cmap=None, norm=mpl.colors.LogNorm(),
                vmin=None, vmax=None, bins=50, statistic=np.nanmean, statstd=np.nanstd,
                histbins=None, histrange=None, cmin=1, binbins=None, weighted_fit=True, ax=None,
                plot_bins=True, plot_fit=True):
    """[plot the 2d hist and x-binned version]

    Arguments:
        X {array} -- array of x-values
        Y {array} -- array of y-values

    Keyword Arguments:
        xlog {bool} -- use log of X (default: {True})
        ylog {bool} -- use log of Y (default: {True})
        cmap {[type]} -- cmap for histogram (default: {None})
        norm {[type]} -- normalization for histogram cmap (default: {mpl.colors.LogNorm()})
        vmin {number} -- min val for cmap (default: {None})
        vmax {number} -- max val for cmap (default: {None})
        bins {int} -- number of bins for hist2d (default: {50})
        statistic {function} -- statistic function (default: {np.nanmean})
        statstd {function} -- error stat function (default: {np.nanstd})
        histbins {[type]} -- bins for hisogram (default: {None})
        histrange {(xmin,xmax),(ymin,ymax)} -- range for histogram (default: {None})
        cmin {int} -- [description] (default: {1})
        binbins {[type]} -- [description] (default: {None})
        weighted_fit {bool} -- [description] (default: {True})
        ax {[type]} -- [description] (default: {None})
        plot_bins {bool} -- [description] (default: {True})
        plot_fit {bool} -- [description] (default: {True})

    Returns:
        [tuple] -- [x, y, p, ax]

    Notes:
    this uses mavg from this file. if it is not available, please change
    """

    if ax is None:
        ax = plt.gca()

    if xlog:
        x = np.log10(X)
    else:
        x = np.asarray(X)

    if ylog:
        y = np.log10(Y)
    else:
        y = np.asarray(Y)

    im = ax.hist2d(x, y, range=histrange, bins=bins, cmap=cmap,
                   cmin=cmin, norm=norm, vmin=vmin, vmax=vmax, zorder=1,)

    # bin the data

    if binbins is None:
        binbins = np.linspace(np.nanmin(x), np.nanmax(x), 10)

    st, be, _ = stats.binned_statistic(x, y, statistic=statistic, bins=binbins)
    est, be, _ = stats.binned_statistic(x, y, statistic=statstd, bins=binbins)
    cl = np.isfinite(st) & np.isfinite(est)
    if plot_bins:
        ax.errorbar(mavg(be)[cl], st[cl], yerr=est[cl], fmt='s',
                    color='r', label='binned data', lw=1.5, zorder=2)

    if weighted_fit:
        p = np.polyfit(mavg(be)[cl][1:], st[cl][1:], 1, w=1/est[cl][1:]**2)
    else:
        p = np.polyfit(mavg(be)[cl][1:], st[cl][1:], 1)
    funcname = 'Best fit: {m:0.5G}*x + {b:0.5G}'.format(m=p[0], b=p[1])
    if plot_fit:
        ax.plot([0, 64], np.polyval(p, [0, 64]),
                'dodgerblue', lw=1.5, label=funcname)

    ax.legend()

    return x, y, p, ax
