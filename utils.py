# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:11:33 2016

@author: johnlewisiii
"""
import math
import os
import statistics
import sys
from importlib import reload

import emcee
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.constants as const
import astropy.units as u
from astropy.wcs import WCS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import integrate, interpolate
from scipy import ndimage as nd
from scipy import signal, special, stats
from weighted import quantile


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

__filtertable__ = Table.read(os.path.join(__location__, "FilterSpecs.tsv"), format="ascii")


#############################
#############################
####  Plotting commands  ####
#############################
#############################

# Set uniform plot options


def set_plot_opts(serif_fonts=True):

    if serif_fonts:
        mpl.rcParams["mathtext.fontset"] = "stix"
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.size"] = 14
    return None


def get_cax(ax=None, size=3):
    if ax is None:
        ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="%f%%" % (size * 1.0), pad=0.05)
    plt.sca(ax)
    return cax


# Plot the KDE for a set of x,y values. No weighting code modified from
# http://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
def kdeplot(xp, yp, filled=False, ax=None, grid=None, bw=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    rvs = np.append(xp.reshape((xp.shape[0], 1)), yp.reshape((yp.shape[0], 1)), axis=1)

    kde = stats.kde.gaussian_kde(rvs.T)
    # kde.covariance_factor = lambda: 0.3
    # kde._compute_covariance()
    kde.set_bandwidth(bw)

    # Regular grid to evaluate kde upon
    if grid is None:
        x_flat = np.r_[rvs[:, 0].min(): rvs[:, 0].max(): 256j]
        y_flat = np.r_[rvs[:, 1].min(): rvs[:, 1].max(): 256j]
    else:
        x_flat = np.r_[0: grid[0]: complex(0, grid[0])]
        y_flat = np.r_[0: grid[1]: complex(0, grid[1])]
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


AtomicMass = {"H2": 2, "12CO": 12 + 16,
              "13CO": 13 + 18, "C18O": 12 + 18, "ISM": 2.33}


def thermal_v(T, mu=None, mol=None):
    """thermal_v(T,atomicmass)
    get thermal velocity for a temperature & molecular mass mu

    Arguments:
        T {[type]} -- [description]

    Keyword Arguments:
        atomicmass {int} -- [description] (default: {28})

    Returns:
        [type] -- [description]
    """
    if mu is None:
        if mol in AtomicMass.keys():
            mu = AtomicMass[mol]
        else:
            mu = 1

    return np.sqrt(const.k_B * T * u.Kelvin / (mu * const.m_p)).to("km/s").value


#############################
#############################
# Convenience math functions
#############################
#############################


def freq_grid(t, fmin=None, fmax=None, oversamp=10.0, pmin=None, pmax=None):
    """
    freq_grid(t,fmin=None,fmax=None,oversamp=10.,pmin=None,pmax=None)
    Generate a 1D list of frequences over a certain range
    [oversamp] * nyquist sampling
    """
    if pmax is not None:
        if pmax == pmin:
            pmax = 10 * pmax
        fmin = 1.0 / pmax
    if pmin is not None:
        if pmax == pmin:
            pmin = 0.1 * pmin
        fmax = 1.0 / pmin

    dt = t.max() - t.min()
    nyquist = 2.0 / dt
    df = nyquist / oversamp
    Nf = 1 + int(np.round((fmax - fmin) / df))
    return fmin + df * np.arange(Nf)


def sigconf1d(n):
    """
    calculate the percentile corresponding to n*sigma
    for a 1D gaussian
    """
    cdf = (1 / 2.0) * (1 + special.erf(n / np.sqrt(2)))
    return (1 - cdf) * 100, 100 * cdf, 100 * special.erf(n / np.sqrt(2))


def wcsaxis(wcs, N=6, ax=None, fmt="%0.2f", use_axes=False):
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
    # crpix1 = hdr['CRPIX1']
    # crpix2 = hdr['CRPIX2']
    # crval1 = hdr['CRVAL1']
    # crval2 = hdr['CRVAL2']
    # try:
    #    cdelt1 = wcs['CDELT1']
    #    cdelt2 = wcs['CDELT2']
    # except:
    #    cdelt1 = wcs['CD1_1']
    #    cdelt2 = wcs['CD2_2']

    if not use_axes:
        xoffset = (naxis1 / N) / 5
        x = np.linspace(xoffset, naxis1 - xoffset, N)
        if naxis >= 2:
            yoffset = (naxis2 / N) / 5
            y = np.linspace(yoffset, naxis2 - yoffset, N)
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
        for i in range(naxis - 2):
            c.append([0] * N)
        coord = list(zip(*c))
        ticks = wcs.all_pix2world(coord, 0)
        x_tick, y_tick = np.asarray(ticks)[:, :2].T

    plt.xticks(x, [fmt % i for i in x_tick])
    plt.yticks(y, [fmt % i for i in y_tick])

    if hdr["CTYPE1"][0].lower() == "g":
        plt.xlabel("Galactic Longitude (l)")
        plt.ylabel("Galactic Latitude (b)")
    else:
        plt.xlabel("Right Ascension (J2000)")
        plt.ylabel("Declination (J2000)")

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


def grid_data(x, y, z, nxy=(512, 512), interp="linear", plot=False, cmap="Greys", levels=None, sigmas=None, filled=False, ):
    """
    stick x,y,z data on a grid and return
    XX, YY, ZZ
    """
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
            levels = 1.0 - np.exp(-0.5 * sigmas ** 2)
        ax = plt.gca()
        if filled:
            cont = ax.contourf
        else:
            cont = ax.contour
        cont(xi, yi, zi / np.max(zi[np.isfinite(zi)]),
             cmap=cmap, levels=levels)

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
        print("List of filters and filter properties")
        __filtertable__.pprint(max_lines=len(__filtertable__) + 3)
        return None

    if not isinstance(filt, float):
        tab = __filtertable__
        tab["fname"] = [s.lower() for s in tab["fname"]]
        if not filt.lower() in tab["fname"]:
            print("Filter %s not found" % filt.lower())
            print("Please select one of the following")
            print(tab["fname"].data)
            filt = eval(input("Include quotes in answer (example ('johnsonK')): "))

        f0 = tab["F0_Jy"][np.where(filt.lower() == tab["fname"])][0]
    else:
        f0 = filt

    flux = f0 * 10.0 ** (-mag / 2.5)

    if emag is not None:
        eflux = 1.08574 * emag * flux
        if return_wavelength:
            return (flux, eflux, tab["Wavelength"][np.where(filt.lower() == tab["fname"])], )
        else:
            return flux, eflux
    else:
        if return_wavelength:
            return flux, tab["Wavelength"][np.where(filt.lower() == tab["fname"])][0]
        else:
            return flux


# ================================================================== #
#
#  Function copied from schmidt_funcs to make them generally available
#


def rot_matrix(theta):
    """
    rot_matrix(theta)
    2D rotation matrix for theta in radians
    returns numpy matrix
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])


def rectangle(c, w, h, angle=0, center=True):
    """
    create rotated rectangle
    for input into PIL ImageDraw.polygon
    to make a rectangle polygon mask

    Rectagle is created and rotated with center
    at zero, and then translated to center position

    accepts centers
    Default : center
    options for center: tl, tr, bl, br
    """
    cx, cy = c
    # define initial polygon irrespective of center
    x = -w / 2.0, +w / 2.0, +w / 2.0, -w / 2.0
    y = +h / 2.0, +h / 2.0, -h / 2.0, -h / 2.0
    # correct the center if starting from corner
    if center is not True:
        if center[0] == "b":
            # y = tuple([i + h/2. for i in y])
            cy = cy + h / 2.0
        else:
            # y = tuple([i - h/2. for i in y])
            cy = cy - h / 2.0
        if center[1] == "l":
            # x = tuple([i + w/2 for i in x])
            cx = cx + w / 2.0
        else:
            # x = tuple([i - w/2 for i in x])
            cx = cx - w / 2.0

    R = rot_matrix(angle * np.pi / 180.0)
    c = []

    for i in range(4):
        xr, yr = np.dot(R, np.asarray([x[i], y[i]])).A.ravel()
        # coord switch to match ordering of FITs dimensions
        c.append((cx + xr, cy + yr))
    # print (cx,cy)
    return c


def rectangle2(c, w, h, angle=0, center=True):
    """
    create rotated rectangle
    for input into PIL ImageDraw.polygon
    to make a rectangle polygon mask

    Rectagle is created and rotated with center
    at zero, and then translated to center position

    accepts centers
    Default : center
    options for center: tl, tr, bl, br
    """
    cx, cy = c
    # define initial polygon irrespective of center
    x = -w / 2.0, +w / 2.0, +w / 2.0, -w / 2.0
    y = +h / 2.0, +h / 2.0, -h / 2.0, -h / 2.0
    # correct center if starting from corner
    if center is not True:
        if center[0] == "b":
            # y = tuple([i + h/2. for i in y])
            cy = cy + h / 2.0
        else:
            # y = tuple([i - h/2. for i in y])
            cy = cy - h / 2.0
        if center[1] == "l":
            # x = tuple([i + w/2 for i in x])
            cx = cx + w / 2.0
        else:
            # x = tuple([i - w/2 for i in x])
            cx = cx - w / 2.0

    R = rot_matrix(angle * np.pi / 180.0)
    c = []

    for i in range(4):
        xr, yr = np.dot(R, np.asarray([x[i], y[i]])).A.ravel()
        # coord switch to match ordering of FITs dimensions
        c.append((cx + xr, cy + yr))
    # print (cx,cy)

    return np.array([c[0], c[1], c[2], c[3], c[0]]).T


def plot_rectangle(c, w, h, angle=0, center=True, ax=None, n=10, m="-", **plot_kwargs):
    if center is True:
        print("Hey, did you know this is built into matplotlib")
        print("Yeah, just do  ax.add_patch(plt.Rectangle(xy=(cx,cy),height=h, width=w, angle=deg))"
        )
        print("of course this one will work even if grid is not rectilinear and can use points"
        )
        print("defined w.r.t. a corner")
    if ax is None:
        ax = plt.gca()
    x, y = rectangle2(c, w, h, angle=angle, center=center)
    ax.plot(x, y, **plot_kwargs)
    n = n * 1j
    # interpolate each linear segment
    leg1 = np.r_[x[0]: x[1]: n], np.r_[y[0]: y[1]: n]
    leg2 = np.r_[x[1]: x[2]: n], np.r_[y[1]: y[2]: n]
    leg3 = np.r_[x[2]: x[3]: n], np.r_[y[2]: y[3]: n]
    leg4 = np.r_[x[3]: x[4]: n], np.r_[y[3]: y[4]: n]
    ax.plot(*leg1, m, *leg2, m, *leg3, m, *leg4, m, **plot_kwargs)
    return ax


def rolling_window(arr, window):
    """[summary]
    Arguments:
        arr {[numpy.ndarray]} -- N-d numpy array
        window {[int]} -- length of window
    Returns:
        out -- array s.t. np.mean(arr,axis=-1) gives the running mean along rows (or -1 axis of a)
            out.shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    """
    shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window, )  # the new shape (a.shape)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def comp(arr):
    """
    returns the compressed version
    of the input array if it is a
    numpy MaskedArray
    """
    try:
        return arr.compressed()
    except:
        return arr


def mavg(arr, n=2, axis=-1):
    """
    returns the moving average of an array.
    returned array is shorter by (n-1)
    applied along last axis by default
    """
    return np.mean(rolling_window(arr, n), axis=axis)


def weighted_std(x, w):
    SS = np.sum(w * (x - x.mean()) ** 2) / np.sum(w)
    quantile(x, w, 0.5)
    return np.sqrt(SS)


def weighted_mad(x, w, stddev=True):
    def median(arr, wei): return quantile(arr, wei, 0.5)
    if stddev:
        return 1.4826 * median(np.abs(x - median(x, w)), w)
    else:
        return median(np.abs(x - median(x, w)), w)


def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)


def _mavg(arr, n=2, mode="valid"):
    """
    returns the moving average of an array.
    returned array is shorter by (n-1)
    """
    # weigths = np.full((n,), 1 / n, dtype=float)
    if len(arr) > 400:
        return signal.fftconvolve(arr, [1.0 / float(n)] * n, mode=mode)
    else:
        return signal.convolve(arr, [1.0 / float(n)] * n, mode=mode)


def mgeo(arr, n=2, axis=-1):
    """
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
    """
    # a = []
    # for i in range(len(arr) - (n - 1)):
    #    a.append(stats.gmean(arr[i:n + i]))

    return stats.gmean(rolling_window(arr, n), axis=axis)


def avg(arr, n=2):
    """
    NOT a general averaging function
    return bin centers (lin and log)
    """
    diff = np.diff(arr)
    # 2nd derivative of linear bin is 0
    if np.allclose(diff, diff[::-1]):
        return mavg(arr, n=n)
    else:
        return np.power(10.0, mavg(np.log10(arr), n=n))
        # return mgeo(arr, n=n) # equivalent methods, only easier


def err_div(x, y, ex, ey):
    Q = x / y
    dQ = np.abs(Q) * np.sqrt((ex / x) ** 2 + (ey / y) ** 2)
    return Q, dQ


def shift_bins(arr, phase=0, nonneg=False):
    # assume original bins are nonneg
    if phase != 0:
        diff = np.diff(arr)
        if np.allclose(diff, diff[::-1]):
            diff = diff[0]
            arr = arr + phase * diff
            # pre = arr[0] + phase*diff
            return arr
        else:
            arr = np.log10(arr)
            diff = np.diff(arr)[0]
            arr = arr + phase * diff
            return np.power(10.0, arr)
    else:
        return arr


def llspace(xmin, xmax, n=None, log=False, dx=None, dex=None):
    """
    llspace(xmin, xmax, n = None, log = False, dx = None, dex = None)
    get values evenly spaced in linear or log spaced
    n [10] -- Optional -- number of steps
    log [false] : switch for log spacing
    dx : spacing for linear bins
    dex : spacing for log bins (in base 10)
    dx and dex override n
    """
    xmin, xmax = float(xmin), float(xmax)
    nisNone = n is None
    dxisNone = dx is None
    dexisNone = dex is None
    if nisNone & dxisNone & dexisNone:
        print("Error: Defaulting to 10 linears steps")
        n = 10.0
        nisNone = False

    # either user specifies log or gives dex and not dx
    log = log or (dxisNone and (not dexisNone))
    if log:
        if xmin == 0:
            print("log(0) is -inf. xmin must be > 0 for log spacing")
            return 0
        else:
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
    """
    Get names formatted as
    hhmmss.ss+ddmmss to Decimal Degree
    only works for dec > 0 (splits on +, not -)
    Will fix this eventually...
    """
    if "string" not in str(type(name)):
        rightascen = []
        declinatio = []
        for n in name:
            if "+" in n:
                ra, de = n.split("+")
                sign = ""
            elif "-" in n:
                ra, de = n.split("-")
                sign = "-"
            ra = ra[0:2] + ":" + ra[2:4] + ":" + ra[4:6] + "." + ra[6:8]
            de = sign + de[0:2] + ":" + de[2:4] + ":" + de[4:6]
            coord = SkyCoord(ra, de, frame="icrs",
                             unit=("hourangle", "degree"))
            rightascen.append(coord.ra.value)
            declinatio.append(coord.dec.value)
        return np.array(rightascen), np.array(declinatio)
    else:
        if "+" in n:
            ra, de = n.split("+")
            sign = ""
        elif "-" in n:
            ra, de = n.split("-")
            sign = "-"
        # ra, de = name.split('+')
        ra = ra[0:2] + ":" + ra[2:4] + ":" + ra[4:6] + "." + ra[6:8]
        de = sign + de[0:2] + ":" + de[2:4] + ":" + de[4:6]
        coord = SkyCoord(ra, de, frame="icrs", unit=("hourangle", "degree"))
        return np.array(coord.ra.value), np.array(coord.dec.value)


def pdf(values, bins=None, range=None):
    """
    ** Normalized differential area function. **
    (statistical) probability denisty function
    normalized so that the integral is 1
    and. The integral over a range is the
    probability of the value is within
    that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    """
    if hasattr(bins, "__getitem__") and (range is None):
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
    """
    N * PDF(x)
    The ~ PDF normalized so that
    the integral is equal to the
    total amount of a quantity.
    The integral over a range is the
    total amount within that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    """
    if hasattr(bins, "__getitem__") and (range is None):
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
    """
    CDF(x)
    (statistical) cumulative distribution function
    Integral on [-inf, b] is the fraction below b.
    CDF is invariant to binning.
    This assumes you are using the entire range in the binning.
    Returns array of size len(bins)
    Plot versus bins[:-1]
    """
    if hasattr(bins, "__getitem__"):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range,
                           density=False)  # returns int

    # cumulative fraction below bin_k
    c = np.cumsum(h / np.sum(h, dtype=float))
    # append 0 to beginning because P(X < min(x)) = 0
    return np.append(0, c), bins


def cdf2(values, bins):
    """
    # # Exclusively for area_function which needs to be unnormalized
    (statistical) cumulative distribution function
    Value at b is total amount below b.
    CDF is invariante to binning

    Plot versus bins[:-1]
    Not normalized to 1
    """
    if hasattr(bins, "__getitem__"):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range, density=False)
    c = np.cumsum(h).astype(float)
    return np.append(0.0, c), bins


def area_function(extmap, bins):
    """
    Complimentary CDF for cdf2 (not normalized to 1)
    Value at b is total amount above b.
    """
    c, bins = cdf2(extmap, bins)
    return c.max() - c, bins


def diff_area_function(extmap, bins, scale=1):
    """
    See pdf2
    """
    s, bins = area_function(extmap, bins)
    dsdx = -np.diff(s) / np.diff(bins)
    return dsdx * scale, avg(bins)


def log_diff_area_function(extmap, bins):
    """
    See pdf2
    """
    s, bins = diff_area_function(extmap, bins)
    g = s > 0
    dlnsdlnx = np.diff(np.log(s[g])) / np.diff(np.log(bins[g]))
    return dlnsdlnx, avg(bins[g])


def mass_function(values, bins, scale=1, aktomassd=183):
    """
    M(>Ak), mass weighted complimentary cdf
    """
    if hasattr(bins, "__getitem__"):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range, density=False, weights=values * aktomassd * scale, )
    c = np.cumsum(h).astype(float)
    return c.max() - c, bins


def polyregress(X, Y, order=1, thru_origin=False):
    g = np.isfinite(X + Y)
    X, Y = X[g], Y[g]

    rank = order + 1
    A = np.array([X ** i for i in range(rank)]).T
    if thru_origin:
        A[:, 0] = 0
    B = Y

    if np.linalg.det(np.dot(A.T, A)) != 0:
        coeff = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, B))
    else:
        coeff, _r, _rank, _s = np.linalg.lstsq(A, B, rcond=-1)

    # coeff = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, B))

    return coeff


def linregress(X, Y, thru_origin=False):
    return polyregress(X, Y, order=1, thru_origin=thru_origin)


def polyregress_bootstrap(X, Y, order=1, iterations=10, thru_origin=False, return_errs=False):

    g = np.isfinite(X + Y)

    X, Y = X[g], Y[g]

    rank = order + 1

    i = 0
    coeff = []
    while i < iterations:
        boot_samp = np.random.randint(0, len(X), size=len(X) // 2)
        x = X[boot_samp]
        B = Y[boot_samp]
        A = np.array([x ** i for i in range(rank)]).T
        if thru_origin:
            A[:, 0] = 0

        if np.linalg.det(np.dot(A.T, A)) != 0:
            c = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, B))
        else:
            c = np.linalg.lstsq(A, B, rcond=-1)[0]

        coeff.append(c)
        i += 1

    if return_errs:
        # correct error by sqrt(2) since using 1/2 the data per iteration
        return (np.nanmean(coeff, axis=0), np.std(coeff, axis=0) / np.sqrt(2), np.nanpercentile(coeff, [16, 50, 84], axis=0).T, )
    else:
        return np.array(coeff)


class PolyRegress(object):
    ###
    # borrowed covariance equations from
    # https://xavierbourretsicotte.github.io/stats_inference_2.html#Custom-Python-class
    # but did not use their weird definition for "R^2"
    # reference for R^2: https://en.wikipedia.org/wiki/Coefficient_of_determination
    ###
    def __init__(self, X, Y, P=1, fit=False, pass_through_origin=False):

        g = np.isfinite(X + Y)

        self.X = X[g]
        self.Y = Y[g]

        self.N = len(self.X)
        self.P = P + 1
        # A is the matrix for X
        self.A = np.array([self.X ** i for i in range(self.P)]).T
        if pass_through_origin:
            self.A[:, 0] = 0

        self.XX = np.dot(self.A.T, self.A)
        self.X_bar = np.mean(self.X, 0)
        self.y_bar = np.mean(self.Y)

        self.b, self.cov, self.err = None, None, None
        self.norm_resid = None
        self.y_hat = None
        self.R2, self.R2_a = None, None

        if fit:
            self.fit()

    def fit(self):
        if np.linalg.det(self.XX) != 0:
            # self.b = np.dot(np.dot(np.linalg.inv(self.XX),self.A.T),self.Y)
            self.b = np.linalg.solve(np.dot(self.A.T, self.A), np.dot(self.A.T, self.Y))
        else:
            self.b, *_ = np.linalg.lstsq(self.A, self.Y, rcond=-1,)
            # self.b = np.dot(np.dot(np.linalg.pinv(self.XX),self.A.T),self.Y)

        self.y_hat = np.dot(self.A, self.b)

        # Sum of squares
        SS_res = np.sum((self.Y - self.y_hat) ** 2)
        SS_tot = np.sum((self.Y - self.y_bar) ** 2)
        SS_exp = np.sum((self.y_hat - self.y_bar) ** 2)
        R2 = 1 - SS_res / SS_tot

        # R squared and adjusted R-squared
        self.R2 = R2  # Use more general definition SS_exp / SS_tot
        self.R2_a = (self.R2 * (self.N - 1) - self.P) / (self.N - self.P - 1)

        # Variances and standard error of coefficients
        self.norm_resid = SS_res / (self.N - self.P - 1)
        self.cov = self.norm_resid * np.linalg.pinv(self.XX)
        self.err = np.sqrt(np.diag(self.cov))
        return self.b, self.err


def linregress_ppv(x, y):
    """Where we perform linear regression
    for ppv cube against a 1D x vector

    Arguments:
        y {array (M,Ny,Nx)} -- 3-D array of data

    Returns:
        f -- best fit least squares solution for whole cube
    """
    g = np.isfinite(x + y)
    x = x[g]
    y = y[g]
    xbar = np.mean(x)
    ybar = np.mean(y, axis=0)
    m = np.sum((x - xbar)[:, np.newaxis, np.newaxis] * (y - ybar), axis=0) / (np.sum((x - xbar) ** 2, axis=0)
    )
    b = ybar - m * xbar
    f = m[np.newaxis, :, :] * x[:, np.newaxis,
                                np.newaxis] + b[np.newaxis, :, :]
    return f


def mad(X, stddev=True, axis=None):
    if stddev:
        return 1.4826 * np.nanmedian(np.abs(X - np.nanmedian(X, axis=axis)), axis=axis)
    else:
        return np.nanmedian(np.abs(X - np.nanmedian(X, axis=axis)), axis=axis)


def mean_mad(X, stddev=True, axis=None):
    if stddev:
        return 1.4826 * np.nanmedian(np.abs(X - np.nanmeam(X, axis=axis)), axis=axis)
    else:
        return np.nanmedian(np.abs(X - np.nanmean(X, axis=axis)), axis=axis)


def rms(X, axis=None):
    return np.sqrt(np.nanmean(X ** 2, axis=axis))


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
        xc, _ = wcs.all_pix2world(x, x * 0, 0)
        _, yc = wcs.all_pix2world(y * 0, y, 0)
        coord_grid = np.meshgrid(xc, yc)
    else:
        coord_grid = np.meshgrid(x, y)

    return coord_grid


def gauss(x, a, mu, sig):
    return a * np.exp(-((x - mu) ** 2) / (2 * sig ** 2))


def forward_fill_nan(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out


def ffill_nan_3d(arr):
    """ foward fill 3d arrays along first axis
    from: https://stackoverflow.com/a/41191127

    """
    shape = arr.shape
    arr = arr.T.reshape(np.product(shape[1:]), shape[0])
    mask = np.isnan(arr)
    print("1,", arr.shape, mask.shape)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:, None], idx]
    return out.T.reshape(shape)


def linear_emcee_fitter(x, y, yerr=None, fit_log=False, gauss_prior=False, nwalkers=10, theta_init=None, use_lnf=True, bounds=([-np.inf, np.inf], [-np.inf, np.inf]), ):
    """
    ## sample call
    sampler,pos = little_emcee_fitter(x,y, theta_init=np.array(mfit.parameters), use_lnf=True)
    samples = sampler.chain[:,1000:,:].reshape((-1,sampler.dim))

    corner.corner(samples,show_titles=True, quantiles=[.16,.84], labels=["$m$", "$b$", "$\ln\,f$"])
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
        use_lnf {use jitter term} -- (default: {True})

    Returns:
        sampler, pos -- returns sampler and intial walker positions
    """

    if fit_log:
        x, y = np.log(x), np.log(y)

    if yerr is None:
        yerr = np.full_like(y, 0.001 * (np.nanmax(y) - np.nanmin(y)))

    g = np.isfinite(x + y + 1 / yerr ** 2)
    x, y, yerr = x[g], y[g], yerr[g]

    bounds = np.sort(bounds, axis=1)

    def model(x, theta):
        return theta[0] * x + theta[1]

    theta_init, cov = np.polyfit(x, y, 1, cov=True)
    if use_lnf:
        theta_init = np.append(theta_init, 0)
        newcov = np.zeros((3, 3))
        newcov[0:2, 0:2] = cov
        newcov[2, 2] = 0.01
        cov = newcov
    ndim = len(theta_init)

    pos = np.random.multivariate_normal(theta_init, cov, size=nwalkers)

    def lnlike(theta, x, y, yerr, use_lnf=use_lnf):
        ymodel = model(x, theta)
        if use_lnf:
            inv_sigma2 = 1.0 / (yerr ** 2 + ymodel **
                                2 * np.exp(2 * theta[-1]))
        else:
            inv_sigma2 = 1.0 / yerr ** 2
        return -0.5 * (np.sum((y - ymodel) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    def lnprior(theta):
        if gauss_prior:
            return stats.multivariate_normal(theta[:-1], theta_init[:-1], cov)
        else:
            c1 = (theta[0] > bounds[0].min()) & (theta[0] < bounds[0].max())
            c2 = (theta[1] > bounds[1].min()) & (theta[0] < bounds[1].max())
            if c1 & c2 & np.all(np.isfinite(theta)):
                return 0.0
            return -np.inf

    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y, yerr, use_lnf=use_lnf)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    return sampler, pos


def plot_walkers(sampler, limits=None, bad=None):
    """
    sampler :  emcee Sampler class
    """

    if hasattr(sampler, "__getitem__"):
        chain = sampler
        ndim = chain.shape[-1]
    else:
        chain = sampler.chain
        ndim = sampler.ndim

    fig = plt.figure(figsize=(8 * ndim, 4 * ndim))

    for w, walk in enumerate(chain[:, limits:, :]):
        if bad is None:
            color = "k"
        elif bad[w]:
            color = "r"
        else:
            color = "k"
        for p, param in enumerate(walk.T):
            ax = plt.subplot(ndim, 1, p + 1)
            ax.plot(param, color, alpha=0.75, lw=0.75)
            # ax.set_ylim(param.min()*0.5,param.max()*1.5)
            # ax.semilogy()
    plt.tight_layout()
    return fig


# TODO

# Make it scale properly
# How does matplotlib
# scaling work


def custom_cmap(colormaps, lower, upper, log=(0, 0)):
    """
    colormaps : a list of N matplotlib colormap classes
    lower : the lower limits for each colormap: array or tuple
    upper : the upper limits for each colormap: array or tuple
    log   : Do you want to plot logscale. This will create
            a color map that is usable with LogNorm()
    """
    if isinstance(log, tuple):
        for lg in log:
            if lg:
                upper = [np.log10(i / lower[0]) for i in upper]
                lower = [np.log10(i / lower[0]) for i in lower]
                norm = upper[-1:][0]
            else:
                lower = lower
                upper = upper
                norm = upper[-1:][0]
    elif log:
        upper = [np.log10(i / lower[0]) for i in upper]
        lower = [np.log10(i / lower[0]) for i in lower]
        norm = upper[-1:][0]
    else:
        lower = lower
        upper = upper
        norm = upper[-1:][0]

    cdict = {"red": [], "green": [], "blue": []}

    for color in ["red", "green", "blue"]:
        for j, col in enumerate(colormaps):
            # print j,col.name,color
            x = [i[0] for i in col._segmentdata[color]]
            y1 = [i[1] for i in col._segmentdata[color]]
            y0 = [i[2] for i in col._segmentdata[color]]
            x = [(i - min(x)) / (max(x) - min(x)) for i in x]
            x = [((i * (upper[j] - lower[j])) + lower[j]) / norm for i in x]
            if (j == 0) & (x[0] != 0):
                x[:0], y1[:0], y0[:0] = [0], [y1[0]], [y0[0]]
            for i in range(len(x)):  # first x needs to be zero
                cdict[color].append((x[i], y1[i], y0[i]))

    return colors.LinearSegmentedColormap("my_cmap", cdict)


def split_cmap(split=0.5, vmin1=12, vmax1=18, vmax2=50, vstep=1, log1=False, cmapn="coolwarm"):

    vmin1 = vmin1
    vmax1 = vmax1
    vmin2 = vmax1
    vmax2 = vmax2
    vstep = vstep
    levels1 = np.arange(vmin1, vmax1 + vstep, vstep)
    levels2 = np.arange(vmin2, vmax2 + vstep, vstep)
    levels_pieces2 = np.hstack((levels1, levels2[1:]))
    ncols1 = len(levels1) - 1
    ncols2 = len(levels2) - 1
    ncols = ncols1 + ncols2
    split = split
    # Sample the right number of colours
    # from the right bits (between 0 &amp; 1) of the colormaps we want.
    if log1:
        cmap1 = mpl.cm.get_cmap(cmapn + "_r")
        cols1 = cmap1(np.logspace(np.log10(1 - split), 0, ncols1))[::-1]
    else:
        cmap1 = mpl.cm.get_cmap(cmapn)
        cols1 = cmap1(np.linspace(0.0, split, ncols1))

    cmap2 = mpl.cm.get_cmap(cmapn)
    cols2 = cmap2(np.logspace(np.log10(split), 0, ncols2))

    # Combine them and build a new colormap:
    allcols2 = np.vstack((cols1, cols2))
    return mpl.colors.LinearSegmentedColormap.from_list("piecewise2", allcols2)


def plot_2dhist(X, Y, xlog=True, ylog=True, cmap=None, norm=mpl.colors.LogNorm(), vmin=None, vmax=None, bins=50, statistic=np.nanmean, statstd=np.nanstd, histbins=None, histrange=None, cmin=1, binbins=None, weighted_fit=True, ax=None, plot_bins=True, plot_fit=True, ):
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

    _ = ax.hist2d(x, y, range=histrange, bins=histbins, cmap=cmap, cmin=cmin, norm=norm, vmin=vmin, vmax=vmax, zorder=1, )

    # bin the data

    if binbins is None:
        binbins = np.linspace(np.nanmin(x), np.nanmax(x), 10)

    st, be, _ = stats.binned_statistic(x, y, statistic=statistic, bins=binbins)
    est, be, _ = stats.binned_statistic(x, y, statistic=statstd, bins=binbins)
    cl = np.isfinite(st) & np.isfinite(est)
    if plot_bins:
        ax.errorbar(mavg(be)[cl], st[cl], yerr=est[cl], fmt="s", color="r", label="binned data", lw=1.5, zorder=2, )

    if weighted_fit:
        p = np.polyfit(mavg(be)[cl][1:], st[cl][1:], 1, w=1 / est[cl][1:] ** 2)
    else:
        p = np.polyfit(mavg(be)[cl][1:], st[cl][1:], 1)
    funcname = "Best fit: {m:0.5G}*x + {b:0.5G}".format(m=p[0], b=p[1])
    if plot_fit:
        ax.plot([0, 64], np.polyval(p, [0, 64]),
                "dodgerblue", lw=1.5, label=funcname)

    ax.legend()

    return x, y, p, ax


def data2rank(arr, clip=0, notadummy=True):
    # stolen from scipy.stats.rankdata
    # so just just that dummy
    arr = np.array(arr, copy=True)
    if notadummy:
        out = stats.rankdata(arr, method="dense")
        return out.reshape(arr.shape) / out.max()
    else:
        shape = arr.shape
        arr = arr.flatten()
        sort = np.argsort(arr)  # smallest to largest
        invsort = np.argsort(sort)  # get sorted array in to original order
        sorted_arr = arr[sort]
        uniqsort = np.r_[True, sorted_arr[1:] != sorted_arr[:-1]]
        order = uniqsort.cumsum()
        return order[invsort].reshape(shape)


def data2norm(H):
    H = np.array(H, copy=True)
    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.cumsum(Hflat)
    sm /= sm[-1]
    return sm[np.argsort(inds)].reshape(H.shape)


def extend_hist(H, X1, Y1, fill=0, padn=2):
    """ Extend the array for the sake of the contours at the plot edges.
        extracted from dfm's corner.hist2d and
        modified for different fill and padding

        H {m x n}- array to pad
        X1 {length n} - X indices to pad
        Y1 {length m}- Y indices to pad
        fill : 0 for histogram, 1 for percentiles

    Returns:
        [tuple] -- H2, X2, Y2
    """
    before = np.arange(-padn, 0, 1)
    after = np.arange(1, padn + 1, 1)
    X2 = np.concatenate([X1[0] + before * np.diff(X1[:2]), X1, X1[-1] +
         after * np.diff(X1[-2:]), ]
    )
    Y2 = np.concatenate([Y1[0] + before * np.diff(Y1[:2]), Y1, Y1[-1] +
         after * np.diff(Y1[-2:]), ]
    )

    padn = ((padn, padn), (padn, padn))
    H2 = np.pad(H, padn, mode="constant", constant_values=fill)

    return H2, X2, Y2


def hist2d(x, y, range=None, bins=20, smooth=False, clip=False):
    if bins is not None:
        xedges = np.histogram_bin_edges(x, bins=bins)
        yedges = np.histogram_bin_edges(y, bins=bins)
        bins = [xedges, yedges]
        range = None
    elif range is None:
        xedges = np.histogram_bin_edges(x, bins=bins)
        yedges = np.histogram_bin_edges(y, bins=bins)
        bins = [xedges, yedges]
        range = None
    else:
        range = list(map(np.sort, range))
    H, X, Y = np.histogram2d(x, y, bins=bins, range=range)

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    padn = np.max([2, int(smooth * 2 // 1)])
    H, X1, Y1 = extend_hist(H, X1, Y1, fill=0, padn=padn)

    if smooth:
        if clip:
            oldH = H == 0
        H = nd.gaussian_filter(H, smooth)

    sm = data2norm(H)

    return sm.T, X1, Y1


def clean_color(color, reverse=False):
    if isinstance(color, str):
        if color[-2:] == "_r":
            return color[:-2], True
        elif reverse is True:
            return color, True
        else:
            return color, False
    else:
        return color, reverse


def color_cmap(c, alpha=1, to_white=True, reverse=False):
    if to_white:
        end = (1, 1, 1, alpha)
    else:
        end = (0, 0, 0, alpha)

    color, reverse = clean_color(c, reverse=reverse)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("density_cmap", [color, end])
    if reverse:
        return cmap.reversed()
    else:
        return cmap


def contour_level_colors(cmap, levels, vmin=None, vmax=None, center=True):
    """get colors corresponding to those produced by contourf

    Arguments:
        cmap {string or cmap} -- colormap
        levels {list or array} -- desired levels

    Keyword Arguments:
        vmin {number} -- min value (default: {None})
        vmax {number} -- max value (default: {None})
        center {True} -- contourf uses center=True values.
                         False will produce a border effect (default: {True})

    Returns:
        [ndarray] -- [list of colors]
    """
    vmin = vmin or 0
    vmax = vmax or max(levels)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # offset = np.diff(levels)[0] * .5
    # colors = mpl.cm.get_cmap(cmap)(norm(levels-offset))
    levels = np.r_[0, levels]
    center_levels = 0.5 * (levels[1:] + levels[:-1])
    return mpl.cm.get_cmap(cmap)(norm(center_levels))


def stat_plot1d(x, ax=None, bins="auto", histtype="step", lw=2, **plot_kwargs):
    """
    really just a fall back for stat_plot2d
    if one of the paramters has no varaince
    Arguments:
        x {[type]} -- array
    """
    if ax is None:
        ax = plt.gca()

    ax.hist(x[np.isfinite(x)], bins="auto",
            histtype="step", lw=2, **plot_kwargs)
    return ax


def stat_plot2d(x, y, marker="k.", bins=20, range=None, smooth=0, xscale=None, yscale=None, plot_data=False, plot_contourf=False, plot_contour=False, plot_imshow=False, plot_binned=True, color=None, cmap=None, levels=None, mfc=None, mec=None, mew=None, ms=None, vmin=None, vmax=None, alpha=1, rasterized=True, linewidths=None, data_kwargs=None, contourf_kwargs=None, contour_kwargs=None, data_color=None, contour_color=None, default_color=None, binned_color=None, contourf_levels=None, contour_levels=None, lw=None, debug=False, zorder=0, ax=None, plot_datapoints=False):
    """
    based on hist2d dfm's corner.py
    but so much prettier and so many more options
    will eventually part of my own corner.py
    (of course most of the corner part will lifted
    directly from corner.py (with attribution of course :D
    )
    ## Look Has! Crappy Documentation!!! ##
    just know the kwargs give the most direct control
    they have precedence over the other keywords
    color precedence:
            color
            marker color (for data only)
            data_color (for data only, overrides marker)
            contour_color (contour only, overrides color)
            match (contour only, overrides both)
    """

    if ax is None:
        ax = plt.gca()

    if xscale == "log":
        x = np.log10(x)
    if yscale == "log":
        y = np.log10(y)

    if not (plot_data or plot_contour or plot_contourf or plot_datapoints):
        # give the user a decent default plot
        plot_data = True
        plot_contour = True
        bins = 'auto'
        smooth = 2

    if smooth is None:
        smooth = 0

    g = np.isfinite(x + y)
    x, y = np.asarray(x)[g], np.asarray(y)[g]

    if np.isclose(x.var(), 0) & np.isclose(y.var(), 0):
        print("Both variables have Variance=0. So no plot can be generated. Here is a plot to help"
        )
        print("First 10 (or less) elements of x", x[:10])
        print("First 10 (or less) elements of y", y[:10])
        ax.scatter(x, y)
        return 0
    elif np.isclose(x.var(), 0):
        print("Variable X has variance=0. Instead of making an ugly plot, here is a histogram of the remaining variable"
        )
        stat_plot1d(y)
        return 0
    elif np.isclose(y.var(), 0):
        print("Variable X has variance=0. Instead of making an ugly plot, here is a histogram of the remaining variable"
        )
        stat_plot1d(x)
        return 0

    if range is None:
        range = [[x.min(), x.max()], [y.min(), y.max()]]

    sm_unflat, X1, Y1 = hist2d(x, y, bins=bins, range=range, smooth=smooth)

    if xscale == "log":
        x = np.power(10, x)
        X1 = np.power(10, X1)
        ax.set_xscale("log")
    if yscale == "log":
        y = np.power(10, y)
        Y1 = np.power(10, Y1)
        ax.set_yscale("log")

    # Choose the default "sigma" contour levels.
    if levels is None:
        levels = 1.0 - np.exp(-0.5 * np.arange(0.5, 2.1, 0.5) ** 2)

    # ALL the plotting stuff

    if data_kwargs is None:
        data_kwargs = dict()
    if contour_kwargs is None:
        contour_kwargs = dict()
    if contourf_kwargs is None:
        contourf_kwargs = dict()

    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)

    if default_color is None:
        default_color = ax.plot([], [])[0].get_color()

    color_match = color == "match"
    data_match = data_color == "match"
    colors_not_set = (color is None) & (cmap is None)
    color_is_set = (color is not None) & (not color_match)
    cmap_is_set = cmap is not None

    reverse = False
    if isinstance(color, str):
        if color[-2:] == "_r":
            color, reverse = color[:-2], True
        else:
            color, reverse = color, False

    # MAKE SENSIBLE CHOICES WITH THE COLORS
    if debug:
        print("(1)", color, cmap)
    # we only need color to be set
    if colors_not_set:  # color not set and cmap not set
        color = default_color
        cmap = "viridis"
        cmap_is_set = True
        color_is_set = True
        if debug:
            print("(1a)", color, cmap, color_is_set, cmap_is_set)
    elif color_match & (not cmap_is_set):  # color is match and cmap not set
        color = default_color
        cmap = "viridis"
        color_is_set = True
        cmap_is_set = True
        if debug:
            print("(1b)", color, cmap, color_is_set, cmap_is_set)
    elif color_match & cmap_is_set:
        color = mpl.cm.get_cmap(cmap)(0.5)
        color_is_set = True
        if debug:
            print("(1c)", color, cmap, color_is_set, cmap_is_set)
    elif (not color_is_set) & cmap_is_set:
        color = default_color
        color_is_set = True
        if debug:
            print("(1d)", color, cmap, color_is_set, cmap_is_set)

    if debug:
        print("(2)", color, cmap, color_is_set, cmap_is_set)
    if data_match & colors_not_set:
        # warnings.warn("Used data_color='match' w/o setting color or cmap"+
        #              "Setting data_color to default color")
        data_match = False
        data_color = color
        if debug:
            print("2(a)", data_color)
    elif data_match & cmap_is_set:
        data_color = mpl.cm.get_cmap(cmap)(0.5)

        if debug:
            print("2(b)", data_color)
    elif data_match & color_is_set:
        data_color = color
        if debug:
            print("2(c)", data_color)
    elif data_color is None:
        data_color = color
        if debug:
            print("2(d)", data_color)

    if debug:
        print("2(e)", data_color)

    if debug:
        print("(3)", color, cmap, color_is_set, cmap_is_set)

    # only create linear colormap is cmap is not set
    if not cmap_is_set:
        if debug:
            print("making linear cmap")
        cmap = color_cmap(color, reverse=reverse)
        cmap_is_set = True

    if debug:
        print("(3)", color, cmap, color_is_set, cmap_is_set)

    def listornone(thing):
        if thing is None:
            return thing
        elif isinstance(thing, list):
            return thing
        else:
            return [thing]

    # color_match is for contours and data
    no_set_contour_color = contour_color is None
    kwargs_not_set = (contour_kwargs.get("cmap") is None) & (contour_kwargs.get("colors") is None
    )
    if kwargs_not_set:
        if (color_match & no_set_contour_color) | (contour_color == "match"):
            contour_kwargs["colors"] = contour_level_colors(cmap, levels)
        elif contour_kwargs.get("colors") is None:
            contour_kwargs["colors"] = listornone(contour_color) or listornone(color)

    if contour_kwargs.get("levels") is None:
        contour_kwargs["levels"] = np.array(levels)  # levels

    if contour_kwargs.get("linewidths") is None:
        if (linewidths is None) & (lw is None):
            pass
        else:
            lw = linewidths or lw
            contour_kwargs["linewidths"] = [
                i for i in np.asarray([lw]).flatten()]

    if contour_kwargs.get("alpha") is None:
        contour_kwargs["alpha"] = alpha

    if contourf_kwargs.get("levels") is None:
        contourf_kwargs["levels"] = np.hstack([[0], levels])  # close top contour

    if contourf_kwargs.get("alpha") is None:
        contourf_kwargs["alpha"] = alpha

    if (contourf_kwargs.get("cmap") is None) & (contourf_kwargs.get("colors") is None):
        contourf_kwargs["cmap"] = cmap

    if data_kwargs.get("color") is None:
        _, dmarker, dcolor = mpl.axes._base._process_plot_format(marker)
        if dcolor is None:
            if color_match | data_match:
                data_kwargs["color"] = data_color or color
                marker = dmarker
            else:
                data_kwargs["color"] = data_color or color

    if data_kwargs.get("mfc") is None:
        data_kwargs["mfc"] = mfc

    if data_kwargs.get("mec") is None:
        data_kwargs["mec"] = mec

    if data_kwargs.get("mew") is None:
        data_kwargs["mew"] = mew

    if data_kwargs.get("ms") is None:
        data_kwargs["ms"] = ms

    if data_kwargs.get("alpha") is None:
        data_kwargs["alpha"] = alpha

    # FINALLY GETTING TO THE PLOTS

    if plot_datapoints:
        p = ax.plot(x, y, marker, **data_kwargs, rasterized=rasterized, zorder=zorder + 1
        )
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    else:
        p = None

    # if vmin is None:
    #    vmin = 0
    # if vmax is None:
    #    vmax = levels[-1]

    if plot_contourf:
        cntrf = ax.contourf(X1, Y1, sm_unflat, **contourf_kwargs, vmin=vmin, vmax=vmax, zorder=zorder + 2
        )
    else:
        cntrf = None

    if plot_contour:
        cntr = ax.contour(X1, Y1, sm_unflat, **contour_kwargs, vmin=vmin, vmax=vmax, zorder=zorder + 3
        )
    else:
        cntr = None

    if plot_imshow:
        ax.imshow(sm_unflat, origin="lower", extent=[X1.min(), X1.max(), Y1.min(), Y1.max()], zorder=zorder + 4, )

    if plot_datapoints:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

    if plot_contour & plot_contourf:
        return ax, cntr, cntrf
    elif plot_contour:
        return ax, cntr
    elif plot_contourf:
        return ax, cntrf
    elif plot_datapoints:
        return ax, p
    else:
        return ax


# alias only used becuase of old code


def jhist2d(*args, **kwargs):
    return stat_plot2d(*args, **kwargs)


def standardize(X, mean=True, std=True):
    if mean:
        mean = np.nanmean(X)
    else:
        mean = 0
    if std:
        std = np.nanstd(X)
    else:
        std = 1

    return (X - mean) / std


def oplot_hist(X, bins=None, ylim=None, scale=0.5, ax=None):
    if ax is None:
        ax = plt.gca()
    if ylim is None:
        ylim = ax.get_ylim()
    if bins is None:
        bins = "auto"

    H, xedge = np.histogram(X, range=np.nanpercentile(X, [0, 100]), bins=bins, density=True
    )
    H = (H / H.max()) * (ylim[1] - ylim[0]) * scale + ylim[0]
    ax.step(mavg(xedge), H, where="mid",
            color="0.25", alpha=1, zorder=10, lw=1.5)
    return ax


def get_edges(x):
    diff = np.diff(x)
    if diff[0] == diff[-1]:
        dx = diff[0]
        return np.r_[x - dx / 2, x[-1] + dx / 2]
    else:
        dx = np.diff(np.log(x))[0]
        return np.exp(np.r_[np.log(x) - dx / 2, x[-1] + dx / 2])
