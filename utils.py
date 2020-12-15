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
import pandas as pd
from astropy import constants as constants
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from scipy import integrate, interpolate, ndimage, signal, special, stats
from weighted import quantile

nd = ndimage


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

__filtertable__ = Table.read(
    os.path.join(__location__, "FilterSpecs.tsv"), format="ascii"
)


def nice_pandas(format="{:3.3g}"):
    pd.set_option("display.float_format", lambda x: format.format(x))


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


def check_iterable(arr):
    return hasattr(arr, "__iter__")


def color_array(arr, alpha=1):
    img = np.zeros(arr.shape + (4,))
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            c = mpl.colors.to_rgb(arr[row, col])
            img[row, col, 0:3] = c
            img[row, col, 3] = alpha
    return img


def invert_color(ml, *args, **kwargs):
    rgb = mpl.colors.to_rgb(ml)
    hsv = mpl.colors.rgb_to_hsv(rgb)
    h, s, v = hsv
    h = 1 - h
    s = 1 - s
    v = 1 - v
    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))


def icol(*args, **kwargs):
    return invert_color(*args, **kwargs)


def to64(arr):
    # Convert numpy to 64-bit precision
    if hasattr(arr, "astype"):
        return arr.astype("float64")
    else:
        if hasattr(arr, "__iter__"):
            if isinstance(arr[0], u.quantity.Quantity):
                return u.quantity.Quantity(arr, dtype=np.float64)
        return np.float64(arr)


def get_xylim(ax=None):
    if ax is None:
        ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    return xlim, ylim


def set_xylim(xlim=None, ylim=None, ax=None, origin=None):
    """set xylims with tuples
    xlim: tuple of x axis limits
    ylim: tuple of y axis limits
    origin: sometimes you just want to change the origin
            so you can keep the axis limits the same
            but just change origin


    """
    if ax is None:
        ax = plt.gca()

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    if isinstance(xlim, tuple):
        xlim = list(xlim)
    if isinstance(ylim, tuple):
        ylim = list(ylim)
    if origin is not None:
        if origin is True:
            if ax.get_xaxis().get_scale()[:3] != "log":
                xlim[0] = 0
            if ax.get_yaxis().get_scale()[:3] != "log":
                ylim[0] = 0
        else:
            xlim[0] = origin[0]
            ylim[0] = origin[1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return tuple(xlim), tuple(ylim)


def _normalize_location_orientation(location, orientation):
    loc_settings = {
        "left": {
            "location": "left",
            "orientation": "vertical",
            "anchor": (1.0, 0.5),
            "panchor": (0.0, 0.5),
            "pad": 0.10,
        },
        "right": {
            "location": "right",
            "orientation": "vertical",
            "anchor": (0.0, 0.5),
            "panchor": (1.0, 0.5),
            "pad": 0.05,
        },
        "top": {
            "location": "top",
            "orientation": "horizontal",
            "anchor": (0.5, 0.0),
            "panchor": (0.5, 1.0),
            "pad": 0.05,
        },
        "bottom": {
            "location": "bottom",
            "orientation": "horizontal",
            "anchor": (0.5, 1.0),
            "panchor": (0.5, 0.0),
            "pad": 0.15,
        },
    }

    return loc_settings


def get_cax(ax=None, position="right", frac=0.03, pad=0.05):
    """get a colorbar axes of the same height as current axes
    position: "left" "right" ( vertical | )
              "top"  "bottom"  (horizontal --- )

    """
    if ax is None:
        ax = plt.gca()

    size = f"{frac*100}%"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size=size, pad=pad)
    plt.sca(ax)
    return cax


def colorbar(mappable=None, cax=None, ax=None, size=0.03, pad=0.05, **kw):
    """wrapper for pyplot.colorbar.

    """
    if ax is None:
        ax = plt.gca()

    if cax is None:
        cax = get_cax(ax=ax, frac=size, pad=pad)

    ret = plt.colorbar(mappable, cax=cax, ax=ax, **kw)
    return ret


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
        x_flat = np.r_[rvs[:, 0].min() : rvs[:, 0].max() : 256j]
        y_flat = np.r_[rvs[:, 1].min() : rvs[:, 1].max() : 256j]
    else:
        x_flat = np.r_[0 : grid[0] : complex(0, grid[0])]
        y_flat = np.r_[0 : grid[1] : complex(0, grid[1])]
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


AtomicMass = {"H2": 2, "12CO": 12 + 16, "13CO": 13 + 18, "C18O": 12 + 18, "ISM": 2.33}


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

    return np.sqrt(constants.k_B * T * u.Kelvin / (mu * constants.m_p)).to("km/s").value


def virial(sig, mass, r):
    s = 1.33 * (sig * (u.km / u.s)) ** 2
    r = r * u.pc
    m = constants.G * (mass * u.Msun)
    return (s * r / m).si.value


def numdens(mass, radius):
    """number density from mass/radius
    assuming spherical symmetry

    Parameters
    ----------
    mass : float
        in solar masses
    radius : float
        in parsecs

    Returns
    -------
    float
        in cm^-3
    """
    mass = mass * u.solMass
    radius = radius * u.pc
    volume = (4 / 3) * np.pi * (radius ** 3)
    dens = (mass / volume) / (2.33 * constants.m_p)
    return dens.to(u.cm ** -3).value


def jeansmass(temp, numdens, mu=2.33):  # 12.03388 msun T=n=1
    """
    temp in K
    numdens in cm^-3
    mu is mean molecular weight [default: 2.33, ISM w/ Helium corr]
    returns Mjeans in solar masses
    .5 * (5 * kb / G)^3 * (3/4π) * (1/2.33 mp)^4 * T^3 / n
    """
    mj = (5 * constants.k_B / (constants.G)) ** 3
    mj *= 3 / (4 * np.pi)
    mj *= (1 / (mu * constants.m_p)) ** 4
    mj = mj * (temp * u.K) ** 3
    mj = mj * (u.cm ** 3 / numdens)
    mj = mj ** 0.5
    return mj.to(u.solMass).value


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
    return (1 - cdf) * 100, 100 * cdf  # , 100 * special.erf(n / np.sqrt(2))

def nsigma(dim=1, n=1):
    """Generalized n-sigma relation

    Parameters
    ----------
    dim : float, optional
        dimensionality, by default 1
    n : float, optional
        N-sigma, by default 1

    Returns
    -------
    float
        the percential/100 corresponding the given sigma

    References:
        https://math.stackexchange.com/a/3668447
        https://mathworld.wolfram.com/RegularizedGammaFunction.html

    The generalized N-sigma relation for M dimensions is given
    by the Regularized Lower Incomplete Gamma Function -
        P(a,z) = γ(a,z)/Γ(a), where γ(a,z) is the lower incomplete gamma function
    The Incomplete Gamma Function is defined
        $\Gamma(a,z0,z1) = \int_z0^z1 t^{a-1} e^{-t} dt$

    For 1D: $Erf(n/sqrt(2)) = \Gamma(1,0,n^2/2)/\Gamma(1)$ gives the Percentile for n-sigma
    For 2D: 1 - exp(-m^2 /2) gives the Percentile for n-sigma
    P(m/2,n^2 / 2) generalizes this to m dimensions
    We need the regularized lower incomplete gamma, which is Gamma(a,z0,z1)/Gamma(a,0,inf)
    this is the incomp. reg. gamma func P(a,z) in

    If we want to think about this in terms of Mahalanobis distance
    Then, well, the Mahalanobis distance is distributed like
    a chi2-distribution with k = m degrees of freedom (assuming the
    eigenvectors are of the covariance matrix are all independent)
    So this covariance is also written as the
    SurvivalFunction(χ^2(k=m),x=n**2) where n = mahalanobis distance
    this would be written stats.chi2(m).cdf(n**2), but this is half
    the speed of using special.gammainc
    @astrojthe3
    """
    return special.gammainc(dim/2,n**2 /2)

def sort_bool(g, srt):
    isrt = np.argsort(srt)
    return srt[g[srt[isrt]]]


def wcsaxis(header, N=6, ax=None, fmt="%0.2f", use_axes=False):
    oldax = plt.gca()
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    xlim = ax.axes.get_xlim()
    ylim = ax.axes.get_ylim()

    wcs = WCS(header)

    naxis = header["NAXIS"]  # naxis
    naxis1 = header["NAXIS1"]  # naxis1
    naxis2 = header["NAXIS2"]  # naxis2
    # crpix1 = hdr['CRPIX1']
    # crpix2 = hdr['CRPIX2']
    # crval1 = hdr['CRVAL1']
    # crval2 = hdr['CRVAL2']
    # try:
    #    cdelt1 = wcs['CDELT1']
    #    cdelt2 = wcs['CDELT2']
    # except BaseException:
    #    cdelt1 = wcs['CD1_1']
    #    cdelt2 = wcs['CD2_2']

    if not use_axes:
        xoffset = ((xlim[1] - xlim[0]) / N) / 5
        x = np.linspace(xlim[0] + xoffset, xlim[1] - xoffset, N)
        if naxis >= 2:
            yoffset = ((ylim[1] - ylim[0]) / N) / 5
            y = np.linspace(ylim[0] + yoffset, ylim[1] - yoffset, N)
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

    if header["CTYPE1"][0].lower() == "g":
        plt.xlabel("Galactic Longitude (l)")
        plt.ylabel("Galactic Latitude (b)")
    else:
        plt.xlabel("Right Ascension (J2000)")
        plt.ylabel("Declination (J2000)")

    ax.axes.set_xlim(xlim[0], xlim[1])
    ax.axes.set_ylim(ylim[0], ylim[1])

    plt.sca(oldax)
    return ax


# In[ writefits]
def writefits(filename, data, header=None, wcs=None, clobber=True):
    if header is None:
        if wcs is not None:
            header = wcs
    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(filename, overwrite=clobber)
    return hdu


def grid_data(
    x,
    y,
    z,
    nxy=(512, 512),
    interp="linear",
    plot=False,
    cmap="Greys",
    levels=None,
    sigmas=None,
    filled=False,
):
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
        cont(xi, yi, zi / np.max(zi[np.isfinite(zi)]), cmap=cmap, levels=levels)

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
            return (
                flux,
                eflux,
                tab["Wavelength"][np.where(filt.lower() == tab["fname"])],
            )
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


def rot_mask(img, pivot=None, angle=0):
    ### https://stackoverflow.com/a/25459080/11594175
    if pivot is None:
        pivot = list(map(int, nd.center_of_mass(img)))[::-1]

    img = img * 1
    padX = [img.shape[1] - (pivot[0]), pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], "constant")
    imgR = nd.rotate(imgP, angle, reshape=False)
    imgC = imgR[padY[0] : -padY[1], padX[0] : -padX[1]]
    return imgC


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
    if False:  # center is True:
        print("Hey, did you know this is built into matplotlib")
        print(
            "Yeah, just do  ax.add_patch(plt.Rectangle(xy=(cx,cy),height=h, width=w, angle=deg))"
        )
        print(
            "of course this one will work even if grid is not rectilinear and can use points"
        )
        print("defined w.r.t. a corner")
    if ax is None:
        ax = plt.gca()
    x, y = rectangle2(c, w, h, angle=angle, center=center)
    ax.plot(x, y, **plot_kwargs)
    n = n * 1j
    # interpolate each linear segment
    leg1 = np.r_[x[0] : x[1] : n], np.r_[y[0] : y[1] : n]
    leg2 = np.r_[x[1] : x[2] : n], np.r_[y[1] : y[2] : n]
    leg3 = np.r_[x[2] : x[3] : n], np.r_[y[2] : y[3] : n]
    leg4 = np.r_[x[3] : x[4] : n], np.r_[y[3] : y[4] : n]
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
    shape = arr.shape[:-1] + (
        arr.shape[-1] - window + 1,
        window,
    )  # the new shape (a.shape)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def minmax(arr, axis=None):
    return np.nanmin(arr, axis=axis), np.nanmax(arr, axis=axis)


def comp(arr):
    """
    returns the compressed version
    of the input array if it is a
    numpy MaskedArray
    """
    try:
        return arr.compressed()
    except BaseException:
        return arr


def mavg(arr, n=2, axis=-1):
    """
    returns the moving average of an array.
    returned array is shorter by (n-1)
    applied along last axis by default
    """
    return np.mean(rolling_window(arr, n), axis=axis)


def weighted_generic_moment(x, k, w=None):
    x = np.asarray(x, dtype=np.float64)
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
    else:
        w = np.ones_like(x)

    return np.sum(x ** k * w) / np.sum(w)


def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)


def weighted_std(x, w):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    SS = np.sum(w * (x - weighted_mean(x, w)) ** 2) / np.sum(w)
    # quantile(x, w, 0.5)
    return np.sqrt(SS)


def weighted_percentile(x, w, percentile, p=0):
    k = np.isfinite(x + w)
    clean_x = np.asarray(x[k], dtype=np.float64)
    clean_w = np.asarray(w[k], dtype=np.float64)
    srt = np.argsort(clean_x)
    sorted_w = clean_w[srt]
    sorted_x = clean_x[srt]
    Sn = np.cumsum(sorted_w)
    Pn = (Sn - 0.5 * sorted_w) / Sn[-1]
    return np.interp(np.asarray(percentile) / 100.0, Pn, sorted_x)


def weighted_median(x, w):
    return weighted_percentile(x, w, 50)


def weighted_mad(x, w, stddev=True):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    if stddev:
        return 1.4826 * weighted_median(np.abs(x - weighted_median(x, w)), w)
    else:
        return weighted_median(np.abs(x - weighted_median(x, w)), w)


def sigmoid(x, a, a0, k, b):
    """
    return a * ( (1 + np.exp(-k*(x-a0)))**-1 - b)
    """
    # a, a0, k, b = p
    return a * ((1 + np.exp(-k * (x - a0))) ** -1 - b)


def mgeo(arr, n=2, axis=-1):
    """rolling geometric mean

    Arguments:
        arr {no.array} -- array

    Keyword Arguments:
        n {int} -- window size (default: {2})
        axis {int} -- axis to roll over (default: {-1})

    Returns:
        [type] -- [description]
    """
    return stats.gmean(rolling_window(arr, n), axis=axis)


def bin_center(arr, n=2):
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


def get_bin_edges(x, log=False):
    diff = np.diff(x)
    if not log:
        dx = diff[0]
        return np.r_[x - dx / 2, x[-1] + dx / 2]
    else:
        dx = np.diff(np.log(x))[0]
        return np.exp(np.r_[np.log(x) - dx / 2, x[-1] + dx / 2])


def get_bin_widths(x):
    return np.diff(get_bin_edges(x))


def err_div(x, y, ex, ey):
    Q = x / y
    dQ = np.abs(Q) * np.sqrt((ex / x) ** 2 + (ey / y) ** 2)
    return Q, dQ


def nside2resol(nside,):
    """nside2resol: get healpix resolution from Nsides

    Parameters
    ----------
    nside : int
        Nsides

    Returns
    -------
    Astropy Quantity [arcmin]
        resolution in arcmin
    """
    resol = 60 * (180 / np.pi) * np.sqrt(np.pi / 3) / nside
    return resol * u.arcmin


def lin_from_log(p, x, log10=False):
    """Convert linear function to exponential
    take log(y) = m log(x) + b  ----> y = exp(b) x^m

    Parameters
    ----------
    p : [type]
        [description]
    x : [type]
        [description]
    log10 : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    if log10:
        return 10 ** (p[0] * np.log10(x) + p[1])
    else:
        return np.exp(p[0] * np.log(x) + p[1])


def color_hue_shift(c, shift=1):
    c = mpl.colors.to_rgb(c)
    h, s, v = mpl.colors.rgb_to_hsv(c)
    h = h + shift % 1
    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))


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
            coord = SkyCoord(ra, de, frame="icrs", unit=("hourangle", "degree"))
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
    return pdf, bin_center(x)


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
    return pdf, bin_center(x)


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

    h, bins = np.histogram(values, bins=bins, range=range, density=False)  # returns int

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


def area_function(extmap, bins, scale=1):
    """
    Complimentary CDF for cdf2 (not normalized to 1)
    Value at b is total amount above b.
    """
    c, bins = cdf2(extmap, bins)
    return scale * (c.max() - c), bins


def diff_area_function(extmap, bins, scale=1):
    """
    See pdf2
    """
    s, bins = area_function(extmap, bins)
    dsdx = -np.diff(s) / np.diff(bins)
    return dsdx * scale, bin_center(bins)


def log_diff_area_function(extmap, bins):
    """
    See pdf2
    """
    s, bins = diff_area_function(extmap, bins)
    g = s > 0
    dlnsdlnx = np.diff(np.log(s[g])) / np.diff(np.log(bins[g]))
    return dlnsdlnx, bin_center(bins[g])


def mass_function(values, bins, scale=1, aktomassd=183):
    """
    M(>Ak), mass weighted complimentary cdf
    """
    if hasattr(bins, "__getitem__"):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    if scale != 1:
        aktomassd = scale

    h, bins = np.histogram(
        values,
        bins=bins,
        range=range,
        density=False,
        weights=values * aktomassd * scale,
    )
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


def linregress_ppv(x, y):
    """Where we perform linear regression
    for ppv cube against a 1D x vector

    Arguments:
        y {array (M,Ny,Nx)} -- 3-D array of data

    Returns:
        f -- best fit least squares solution for whole cube
    """
    # g = np.isfinite(x + y)
    # x = x[g]
    # y = y[g]
    xbar = np.mean(x)
    ybar = np.mean(y, axis=0)
    m = np.sum((x - xbar)[:, np.newaxis, np.newaxis] * (y - ybar), axis=0) / (
        np.sum((x - xbar) ** 2, axis=0)
    )
    b = ybar - m * xbar
    f = m[np.newaxis, :, :] * x[:, np.newaxis, np.newaxis] + b[np.newaxis, :, :]
    return f


def polyregress_bootstrap(
    X, Y, order=1, iterations=10, thru_origin=False, return_errs=False
):

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
        return (
            np.nanmean(coeff, axis=0),
            np.std(coeff, axis=0) / np.sqrt(2),
            np.nanpercentile(coeff, [16, 50, 84], axis=0).T,
        )
    else:
        return np.array(coeff)


def ortho_dist(x, y, m, b):
    """
    get the orthogonal distance
    from a point to a line
    """
    ortho_dist = (y - m * x - b) / np.sqrt(1 + m ** 2)
    return ortho_dist


from scipy.optimize import curve_fit


def curve_fit_line(x, y, yerr=None):
    def model(xdata, m, b):
        return m * xdata + b

    p0 = np.polyfit(x, y, 1)
    out = curve_fit(model, x, y, p0=p0, sigma=yerr)
    return out


class PolyRegress(object):
    ###
    # borrowed covariance equations from
    # https://xavierbourretsicotte.github.io/stats_inference_2.html#Custom-Python-class
    # but did not use their weird definition for "R^2"
    # reference for R^2: https://en.wikipedia.org/wiki/Coefficient_of_determination
    ###
    def __init__(
        self,
        X,
        Y,
        P=1,
        fit=False,
        pass_through_origin=False,
        log=False,
        dtype=np.float64,
    ):
        if log:
            X, Y = np.log10(X).ravel(), np.log10(Y).ravel()
        else:
            X, Y = np.array(X).ravel(), np.array(Y).ravel()
        self.log = log
        g = np.isfinite(X + Y)

        self.X = X[g].astype(dtype)
        self.Y = Y[g].astype(dtype)

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
        self.scatter = None
        self.norm_resid = None
        self.y_hat = None
        self.R2, self.R2_a = None, None

        if fit:
            self.fit()

    def __str__(self):
        if self.b is not None:
            b = self.b
            e = self.err
            s = self.scatter
            term = lambda i: f"({self.b[i]:0.3g}+/-{e[i]:0.3g}) * x^{i}"
            out1 = " + ".join([term(i) for i in range(self.P)])
            if not self.log:
                # out=f'm:{b[1]:0.3g}+/-{e[1]:0.3g}, b:{b[0]:0.3g}+/-{e[0]:0.3g}, scatter:{s:0.3g}'
                out = out1 + f" , scatter:{s:0.3g}"
            else:
                # out=f'm:{b[1]:0.3g}+/-{e[1]:0.3g}, b:{b[0]:0.3g}+/-{e[0]:0.3g}, scatter:{s:0.3g}'
                out = out1 + f" , scatter:{s:0.3g}"
                out = f"{out}\n+10^b:{10**b[0]:0.3g}"
        else:
            out = "\nERROR::Has not been run. run model.fit() now\n"
        return "\n" + out + "\n"

    def __repr__(self):
        out1 = f"\norder={self.P-1} regression"
        out2 = self.__str__()
        return out1 + out2

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

        self.residual = self.y_hat - self.Y

        # R squared and adjusted R-squared
        self.R2 = R2  # Use more general definition SS_exp / SS_tot
        self.R2_a = (self.R2 * (self.N - 1) - self.P) / (self.N - self.P - 1)

        # Variances and standard error of coefficients
        self.norm_resid = SS_res / (self.N - self.P - 1)
        self.cov = self.norm_resid * np.linalg.pinv(self.XX)
        self.err = np.sqrt(np.diag(self.cov))

        # ortho_dist = (self.Y - self.b[1] * self.X - self.b[0])/np.sqrt(1 + self.b[1]**2)
        # self.scatter = np.std(ortho_dist)/np.cos(np.arctan(self.b[1]))
        self.scatter = np.std(self.residual)

        if self.log:
            self.percent_scatter = np.mean(
                np.abs(10 ** self.y_hat - 10 ** self.Y) / 10 ** self.Y
            )
        else:
            self.percent_scatter = np.mean(self.residual / self.Y)

        return self.b, self.err

    def func(self, x):
        p = self.b[::-1]
        if self.log:
            return 10 ** np.polyval(p, np.log10(x))
        else:
            return np.polyval(p, x)

    def sample_covarariance(self, n=10000):
        return np.random.multivariate_normal(self.b, self.cov, n)

    def plot(
        self,
        ax=None,
        color="",
        data=False,
        marker="o",
        ms=5,
        mec="none",
        unlog=False,
        **kwargs,
    ):
        if ax is None:
            ax = plt.gca()

        if self.log or unlog:
            x = 10 ** self.X
            y = 10 ** self.Y
            xx = np.logspace(self.X.min(), self.X.max(), 30)
        else:
            x = self.X
            y = self.Y
            xx = np.linspace(self.X.min(), self.X.max(), 30)

        if color == "":
            c = ("k", "r")

        if isinstance(color, tuple):
            c_data = color[0]
            c_line = color[1]
            data = True
        elif color == "":
            if data:
                c_data = plt.plot([], [], "-")[0].get_color()
                c_line = "k"
            else:
                c_line = "k"
        else:
            c_line = color
            if data:
                c_data = plt.plot([], [], "-")[0].get_color()

        if data:
            p = ax.plot(x, y, color=c_data, ms=ms, mec=mec, marker=marker, ls="")

        ax.plot(xx, self.func(xx), "-", color=c_line, **kwargs)

        if self.log:
            ax.set_xscale("log")
            ax.set_yscale("log")


def plot_covariances(p, cov, names=None, figsize=(12, 12), nsamps=5000, smooth=1):
    p = np.random.multivariate_normal(p, cov, nsamps)
    fig, axs = corner(p, smooth=smooth, names=names, figsize=figsize)
    return fig, axs


def plot_astropy_fit_covariances(fit, fitter):
    p = fit.parameters
    cov = fitter.fit_info["param_cov"]
    ax = plot_covariances(p, cov, names=fit.param_names)
    return ax


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


def wcs_to_grid(header, index=False, verbose=False):
    """wcs_to_grid creates grids (lon,lat) for a given WCS header

    Parameters
    ----------
    header : astropy fits header
        and astropy fits header
    index : bool, optional
        just return indices, by default False
    verbose : bool, optional
        [unused input], by default False

    Returns
    -------
    [tuple]
        (longitude, latitude)
    """

    wcs = WCS(header)
    naxis1 = header["NAXIS1"]  # naxis1
    naxis2 = header["NAXIS2"]  # naxis2
    ij = np.indices(wcs.array_shape)
    if not index:
        coord_grid = wcs.low_level_wcs.array_index_to_world_values(*ij)
    else:
        coord_grid = ij

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


def linear_emcee_fitter(
    x,
    y,
    yerr=None,
    fit_log=False,
    gauss_prior=False,
    nwalkers=10,
    theta_init=None,
    use_lnf=True,
    bounds=([-np.inf, np.inf], [-np.inf, np.inf]),
):
    """
    ## sample call
    sampler,pos = little_emcee_fitter(x,y, theta_init=np.array(mfit.parameters), use_lnf=True)
    samples = sampler.chain[:,1000:,:].reshape((-1,sampler.dim))

    corner.corner(samples,show_titles=True, quantiles=[.16,.84], labels=["$m$", "$b$", r"$\ln\,f$"])
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
            inv_sigma2 = 1.0 / (yerr ** 2 + ymodel ** 2 * np.exp(2 * theta[-1]))
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
def combine_cmap(cmaps, lower, upper, name="custom", N=None, register=True):

    n = len(cmaps)

    for ic, c in enumerate(cmaps):
        if isinstance(c, str):
            cmaps[ic] = mpl.cm.get_cmap(c)

    if N is None:
        N = [256] * n

    values = np.array([])
    colors = np.empty((0, 4))

    for i in range(n):
        step = (upper[i] - lower[i]) / N[i]
        xcols = np.arange(lower[i], upper[i], step)
        values = np.append(values, xcols)
        xcols -= xcols.min()
        xcols /= xcols.max()
        cols = cmaps[i](xcols)
        colors = np.vstack([colors, cols])
    values -= values.min()
    values /= values.max()

    arr = list(zip(values, colors))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, arr)

    if (name != "custom") & register:
        mpl.cm.register_cmap(name=name, cmap=cmap)

    return cmap


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

    for ic, c in enumerate(colormaps):
        if isinstance(c, str):
            colormaps[ic] = mpl.cm.get_cmap(c)

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


def cmap_split(*args, **kwargs):
    """alias for split_cmap"""
    return split_cmap(*args, **kwargs)

def split_cmap(cmapn='viridis',split=0.5,vmin=0, vmaxs=(.5,1),vstep=None,
               vsplit=None,log=False):
    """
    split a colormap at a certain location

    split - where along the colormap will be our split point
            by default this split point is put in the middle
            of the values
    vmin  value for colorbar to start at: should max vim in
            plotting command
    vmaxs  (splitvalue,vmax) - where to start the second segment
            of the color map. cmap(split) will be located
            at valeu=splitvalue
    vplit = instead of giving vmin,vmax,a you can split it at a
            value between 0,1.
    log     doesn't do what anyone would think, don't recommend using



    """
    if vsplit is not None:
        vmin=0
        vmaxs=(vsplit,1)
    vmin1 = vmin
    vmax1 =  vmaxs[0]
    vmin2 = vmax1
    vmax2 =  vmaxs[1]
    if vstep is None:
        vstep=   (vmax2 - vmin1)/1024
    levels1 = np.arange(vmin1, vmax1+vstep, vstep)
    levels2 = np.arange(vmin2, vmax2+vstep, vstep)

    ncols1 = len(levels1)-1
    #ncols1 = int((vmax1-vmin1)//vstep)
    ncols2 = len(levels2)-1
#     ncols1 = int((vmax1-vmin1)//vstep)+1
#     ncols2 = int((vmax2-vmin2)//vstep)+1
    # ncols = ncols1 + ncols2
    split = split
    # Sample the right number of colours
    # from the right bits (between 0 &amp; 1) of the colormaps we want.
    cmap2 = mpl.cm.get_cmap(cmapn)
    if log:
        cmap1 = mpl.cm.get_cmap(cmapn+'_r')
        cols1 = cmap1(np.logspace(np.log10(1-split),0, ncols1))[::-1]
        cols2 = cmap2(np.logspace(np.log10(split), 0, ncols2))
    else:
        cols1 = cmap2(np.linspace(0.0, split, ncols1))
        cols2 = cmap2(np.linspace(split, 1, ncols2))


    #cols2 = cmap2(np.logspace(np.log10(split), 0, ncols2))

    # Combine them and build a new colormap:
    allcols2 = np.vstack( (cols1,cols2) )
    return mpl.colors.LinearSegmentedColormap.from_list('piecewise2', allcols2)

def plot_2dhist(
    X,
    Y,
    xlog=True,
    ylog=True,
    cmap=None,
    norm=mpl.colors.LogNorm(),
    vmin=None,
    vmax=None,
    bins=50,
    statistic=np.nanmean,
    statstd=np.nanstd,
    histbins=None,
    histrange=None,
    cmin=1,
    binbins=None,
    weighted_fit=True,
    ax=None,
    plot_bins=True,
    plot_fit=True,
):
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

    _ = ax.hist2d(
        x,
        y,
        range=histrange,
        bins=histbins,
        cmap=cmap,
        cmin=cmin,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    # bin the data

    if binbins is None:
        binbins = np.linspace(np.nanmin(x), np.nanmax(x), 10)

    st, be, _ = stats.binned_statistic(x, y, statistic=statistic, bins=binbins)
    est, be, _ = stats.binned_statistic(x, y, statistic=statstd, bins=binbins)
    cl = np.isfinite(st) & np.isfinite(est)
    if plot_bins:
        ax.errorbar(
            mavg(be)[cl],
            st[cl],
            yerr=est[cl],
            fmt="s",
            color="r",
            label="binned data",
            lw=1.5,
            zorder=2,
        )

    if weighted_fit:
        p = np.polyfit(mavg(be)[cl][1:], st[cl][1:], 1, w=1 / est[cl][1:] ** 2)
    else:
        p = np.polyfit(mavg(be)[cl][1:], st[cl][1:], 1)
    funcname = "Best fit: {m:0.5G}*x + {b:0.5G}".format(m=p[0], b=p[1])
    if plot_fit:
        ax.plot([0, 64], np.polyval(p, [0, 64]), "dodgerblue", lw=1.5, label=funcname)

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
        order = np.nancumsum(uniqsort)
        return order[invsort].reshape(shape)


def data2norm(H):
    """
    normalize data as percentiles

    """
    bad = ~np.isfinite(H)
    H = np.array(H, copy=True)

    Hflat = H.flatten()
    inds = np.argsort(Hflat)[::-1]
    Hflat = Hflat[inds]
    sm = np.nancumsum(Hflat)
    sm /= sm[-1]
    out = sm[np.argsort(inds)].reshape(H.shape)
    out[bad] = np.nan
    return out


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
    X2 = np.concatenate(
        [X1[0] + before * np.diff(X1[:2]), X1, X1[-1] + after * np.diff(X1[-2:]),]
    )
    Y2 = np.concatenate(
        [Y1[0] + before * np.diff(Y1[:2]), Y1, Y1[-1] + after * np.diff(Y1[-2:]),]
    )

    padn = ((padn, padn), (padn, padn))
    H2 = np.pad(H, padn, mode="constant", constant_values=fill)

    return H2, X2, Y2


def hist2d(
    x,
    y,
    range=None,
    bins=20,
    smooth=False,
    clip=False,
    pad=True,
    normed=True,
    weights=None,
):
    g = np.isfinite(x + y)
    x = np.array(x)[g]
    y = np.array(y)[g]
    if bins is not None:
        if range is None:
            if isinstance(bins, int) or (bins == "auto"):
                xedges = np.histogram_bin_edges(x, bins=bins)
                yedges = np.histogram_bin_edges(y, bins=bins)
            elif check_iterable(bins) & (len(bins) == 2):
                xedges = np.histogram_bin_edges(x, bins=bins[0])
                yedges = np.histogram_bin_edges(y, bins=bins[1])
            bins = [xedges, yedges]
        else:
            if (len(range)==2) & (len(range[0])==2):
                xedges = np.histogram_bin_edges(x, bins=bins, range=range[0])
                yedges = np.histogram_bin_edges(y, bins=bins, range=range[1])
            else:
                xedges = np.histogram_bin_edges(x, bins=bins, range=range)
                yedges = np.histogram_bin_edges(y, bins=bins, range=range)
            bins = [xedges, yedges]
    elif range is None:
        xedges = np.histogram_bin_edges(x, bins=bins)
        yedges = np.histogram_bin_edges(y, bins=bins)
        bins = [xedges, yedges]
        range = None
    else:
        range = list(map(np.sort, range))
    H, X, Y = np.histogram2d(x, y, bins=bins, range=range, weights=weights)

    X1, Y1 = 0.5 * (X[1:] + X[:-1]), 0.5 * (Y[1:] + Y[:-1])

    if pad:
        padn = np.max([2, int(smooth * 2 // 1)])
        H, X1, Y1 = extend_hist(H, X1, Y1, fill=0, padn=padn)

    if smooth:
        if clip:
            oldH = H == 0
        H = nd.gaussian_filter(H, smooth)

    if normed:
        sm = data2norm(H)
    else:
        sm = H

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
        vmin {number} -- min value (default: {0})
        vmax {number} -- max value (default: {max(levels)})
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

    ax.hist(x[np.isfinite(x)], bins="auto", histtype="step", lw=2, **plot_kwargs)
    return ax


def stat_plot2d(
    x,
    y,
    marker="k.",
    bins=20,
    range=None,
    smooth=0,
    xscale=None,
    yscale=None,
    plot_data=False,
    plot_contourf=False,
    plot_contour=False,
    plot_imshow=False,
    plot_binned=True,
    color=None,
    cmap=None,
    levels=None,
    mfc=None,
    mec=None,
    mew=None,
    ms=None,
    vmin=None,
    vmax=None,
    alpha=1,
    rasterized=True,
    linewidths=None,
    data_kwargs=None,
    contourf_kwargs=None,
    contour_kwargs=None,
    data_color=None,
    contour_color=None,
    default_color=None,
    binned_color=None,
    contourf_levels=None,
    contour_levels=None,
    lw=None,
    debug=False,
    zorder=0,
    ax=None,
    plot_datapoints=None,
):
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

    if plot_datapoints is None:
        plot_datapoints = plot_data

    if not (plot_data or plot_contour or plot_contourf):
        # give the user a decent default plot
        plot_data = True
        plot_contour = True
        smooth = 2

    if smooth is None:
        smooth = 0

    g = np.isfinite(x + y)
    x, y = np.asarray(x)[g], np.asarray(y)[g]

    if (x.var() == 0) & (y.var() == 0):
        print(
            "Both variables have Variance=0. So no plot can be generated. Here is a plot to help"
        )
        print("First 10 (or less) elements of x", x[:10])
        print("First 10 (or less) elements of y", y[:10])
        ax.scatter(x, y)
        return 0
    elif x.var() == 0:
        print(
            "Variable X has variance=0. Instead of making an ugly plot, here is a histogram of the remaining variable"
        )
        stat_plot1d(y)
        return 0
    elif y.var() == 0:
        print(
            "Variable Y has variance=0. Instead of making an ugly plot, here is a histogram of the remaining variable"
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
    kwargs_not_set = (contour_kwargs.get("cmap") is None) & (
        contour_kwargs.get("colors") is None
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
            contour_kwargs["linewidths"] = [i for i in np.asarray([lw]).flatten()]

    if contour_kwargs.get("alpha") is None:
        contour_kwargs["alpha"] = alpha

    if contourf_kwargs.get("levels") is None:
        new_levels = np.hstack([[0], levels])

        contourf_kwargs["levels"] = np.unique(new_levels)  # close top contour

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
        p = ax.plot(
            x, y, marker, **data_kwargs, rasterized=rasterized, zorder=zorder + 1
        )
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    else:
        p = None

    # if vmin is None:
    #    vmin = 0
    # if vmax is None:
    #    vmax = levels[-1]

    if plot_contourf:
        cntrf = ax.contourf(
            X1,
            Y1,
            sm_unflat,
            **contourf_kwargs,
            vmin=vmin,
            vmax=vmax,
            zorder=zorder + 2,
        )
    else:
        cntrf = None

    if plot_contour:
        cntr = ax.contour(
            X1, Y1, sm_unflat, **contour_kwargs, vmin=vmin, vmax=vmax, zorder=zorder + 3
        )
    else:
        cntr = None

    if plot_imshow:
        ax.imshow(
            sm_unflat,
            origin="lower",
            extent=[X1.min(), X1.max(), Y1.min(), Y1.max()],
            zorder=zorder + 4,
        )

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


def annotate(
    text,
    x,
    y,
    ax=None,
    horizontalalignment="center",
    verticalalignment="center",
    ha=None,
    va=None,
    transform="axes",
    color="k",
    fontsize=9,
    facecolor="w",
    alpha=0.75,
    bbox=dict(),
    **kwargs,
):

    if ax is None:
        ax = plt.gca()

    horizontalalignment = ha or horizontalalignment
    verticalalignment = va or verticalalignment

    if transform == "axes":
        transform = ax.transAxes
    elif transform == "data":
        transform = ax.transData
    bbox1 = dict(facecolor=facecolor, alpha=alpha)
    bbox1.update(bbox)
    text = ax.text(
        x,
        y,
        text,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        transform=transform,
        color=color,
        fontsize=fontsize,
        bbox=bbox1,
        **kwargs,
    )
    return text


# alias only used becuase of old code


def jhist2d(*args, **kwargs):
    return stat_plot2d(*args, **kwargs)


def corner(pos, names=None, smooth=1, bins=20, figsize=None, **kwargs):
    """produce a corner plot

    Parameters
    ----------
    pos : np.array
        each item should be a row. pos.size = MxN, N items, M
    names : list of strings, optional
        names of variables to be plotted, must have N elements, by default None
    smooth : int, optional
        how much to smooth the contours/histogram, by default 1
    bins : int, optional
        number of bins for histogram, by default 20
    figsize : tuple, optional
        [description], by default 2 * pos.shape[1] + 0.5

    Returns
    -------
    [type]
        [description]
    """
    if figsize is None:
        dim = 2 * pos.shape[1] + 0.5
        figsize = (dim, dim)
    fig, axs = plt.subplots(
        nrows=pos.shape[1],
        ncols=pos.shape[1],
        sharex=False,
        sharey=False,
        figsize=figsize,
    )
    for i in range(pos.shape[-1]):
        for j in range(pos.shape[-1]):
            ax = axs[i, j]
            if i == j:
                stat_plot1d(pos[:, i], ax=axs[i, j])
                ax.set_xlabel(names[j])
            if j < i:
                stat_plot2d(
                    pos[:, j],
                    pos[:, i],
                    ax=ax,
                    bins=bins,
                    smooth=smooth,
                    plot_datapoints=True,
                    plot_contour=True,
                    **kwargs,
                )
                if names is not None:
                    try:
                        if i != j :
                            ax.set_xlabel(names[j])
                            ax.set_ylabel(names[i])


                    except:
                        pass

            if j > i:
                plt.delaxes(axs[i, j])
    fig.tight_layout()
    return fig, axs


def standardize(X, remove_mean=True, remove_std=True):
    if remove_mean:
        mean = np.nanmean(X)
    else:
        mean = 0
    if remove_std:
        std = np.nanstd(X)
    else:
        std = 1

    return (X - mean) / std


def plotoneone(
    color="k",
    lw=2,
    scale=1,
    offset=0,
    p=None,
    invert=False,
    n=50,
    start=None,
    end=None,
    ax=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    if start is None:
        start = np.min([xlim[0], ylim[0]])
    if end is None:
        end = np.max([xlim[1], ylim[1]])
    axscale = ax.get_xscale()
    if axscale == "log":
        xs = np.logspace(np.log10(start), np.log10(end), n)
    else:
        xs = np.linspace(start, end, n)

    if p is not None:
        scale, offset = p
    ys = scale * xs + offset
    if invert:
        ax.plot(ys, xs, color=color, lw=lw, **kwargs)
    else:
        ax.plot(xs, ys, color=color, lw=lw, **kwargs)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def oplot_hist(
    X,
    bins=None,
    ylim=None,
    scale=0.5,
    ax=None,
    show_mean=False,
    show_median=False,
    show_percentiles=None,
):
    if ax is None:
        ax = plt.gca()
    if ylim is None:
        ylim = ax.get_ylim()
    if bins is None:
        bins = "auto"

    H, xedge = np.histogram(
        X, range=np.nanpercentile(X, [0, 100]), bins=bins, density=True
    )
    H = (H / H.max()) * (ylim[1] - ylim[0]) * scale + ylim[0]
    ax.step(mavg(xedge), H, where="mid", color="0.25", alpha=1, zorder=10, lw=1.5)

    if show_mean:
        ax.axvline(np.nanmean(X), 0, 1, color="0.45", ls="--")
    if show_median:
        ax.axvline(np.nanmedian(X), 0, 1, color="0.45", ls="--")
    if not (show_percentiles is None):
        for p in show_percentiles:
            ax.axvline(p, 0, 1, color="0.45", ls="--", alpha=0.5)
    return ax


def find_minima(x, y, yer=-1, err_cut=False, cut=3):
    # srt = np.argsort(x)
    # y = y[srt]
    # x = x[srt]
    # dy = np.gradient(y,axis=0)#/np.gradient(x)
    # dyy = np.gradient(dy,axis=0)#/np.gradient(x)
    dy = np.diff(y, axis=0)
    dy = np.insert(dy, 0, 0, axis=0)
    dyy = np.diff(dy, axis=0)
    dyy = np.insert(dyy, -1, 0, axis=0)

    dysign = np.sign(dy)
    dyysign = np.sign(dyy)

    dysignchange = (np.roll(dysign, 1, axis=0) - dysign) != 0
    dysignchange[0:2] = False
    dyysignchange = (np.roll(dyysign, 1, axis=0) - dyysign) != 0
    dyysignchange[0:2] = False

    if err_cut:
        mins = dysignchange & (dyysign >= 0) & (y > cut * yer)
    else:
        mins = dysignchange & (dyysign >= 0)  # & (np.abs(dy) < yer)
    return np.roll(mins, -1, axis=0)


def find_maxima(x, y, yer=-1, err_cut=False, cut=3):
    # srt = np.argsort(x)
    # y = y[srt,:,:]
    # x = x[srt]
    dy = np.gradient(y, axis=0)  # /np.gradient(x)
    dyy = np.gradient(dy, axis=0)  # /np.gradient(x)

    dysign = np.sign(dy)
    dyysign = np.sign(dyy)

    dysignchange = (np.roll(dysign, 1, axis=0) - dysign) != 0
    dysignchange[0] = False
    dyysignchange = (np.roll(dyysign, 1, axis=0) - dyysign) != 0
    dyysignchange[0] = False

    if err_cut:
        mins = dysignchange & (dyysign <= 0) & (y > cut * yer)
    else:
        mins = dysignchange & (dyysign <= 0)  # & (np.abs(dy) < yer)
    return np.roll(mins, -1, axis=0)


def multi_colored_line_plot(
    x, y, z=None, cmap="viridis", norm=None, vmin=None, vmax=None, ax=None, **kwargs
):
    """
    adapted from matplotlib gallery
    """
    if ax is None:
        ax = plt.gca()
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    if z is None:
        z = y

    # Create a continuous norm to map from data points to colors
    if vmin is None:
        vmin = np.nanmin(z)
    if vmax is None:
        vmax = np.nanmax(z)
    if norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    lc = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm, **kwargs)
    # Set the values used for colormapping

    lc.set_array(z)

    line = ax.add_collection(lc)
    # fig.colorbar(line, ax=axs[0])
    return line


def errorbar_fill(
    x=None,
    y=None,
    yerr=None,
    *args,
    ax=None,
    mid=True,
    color=None,
    alpha=1,
    lw=1,
    ls="-",
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    if mid:
        alpha_fill = alpha * 2
        if alpha_fill >= 1:
            alpha_fill = 1
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)
    if mid:
        ax.plot(x, y, "-", color=color, alpha=alpha, lw=lw, ls=ls)
    return None


def confidence_bands(x, p, cov, func, N=1000, ci=90, ucb=None, lcb=None):
    """
    x, inputs
    p, parameters
    func - function called f(x,*p)

    if linear then p, should be in the order returned
    by np.polyfit. If used PolyRegress then pass p=PolyRegress.b[::-1]
    """
    ps = np.random.multivariate_normal(mean=p, cov=cov, size=N)
    if isinstance(func, str):
        if (func == "line") | (func == "lin"):
            func = lambda x, p: np.polyval(p, x)
    out = np.array([func(x, pi) for pi in ps])
    if ucb is None:
        ucb = 50 + ci / 2
    if lcb is None:
        lcb = 50 - ci / 2
    upper, lower = np.percentile(out, [lcb, ucb], axis=0)
    return lower, upper


def detrend_iter_single_mod(t, f, p=1, low=3, high=3, cutboth=False):
    """
     My iterative detrending algorithm, based on the concept in Vandenberg & Johnson 2015
     borrowed clipping portion from scipy.stats.sigmaclip
     with substantial modifications.
    """
    clip = 1
    c = np.asarray(f).ravel()
    # mask = np.full(c.shape,True,dtype=np.bool)
    mask = np.isfinite(c)
    outmask = np.copy(mask)  # np.full(c.shape,True, dtype=np.bool)
    i = 0
    while clip:
        i += 1
        m = PolyRegress(t[mask], c[mask], P=p, fit=True)
        c_trend = m.y_hat
        c_detrend = c[mask] - c_trend + 1  # get masked detreneded lighcurve
        c_mean = np.median(c_detrend)  # use median for mean
        c_std = c_detrend.std()
        size = c_detrend.size
        critlower = c_mean - c_std * low
        critupper = c_mean + c_std * high
        newmask = (c_detrend >= critlower) & (c_detrend <= critupper)
        outmask[mask] = c_detrend <= critupper
        mask[mask] = newmask
        clip = size - c[mask].size
        print(i, clip, np.sum(mask), len(t))
        # plt.plot(x[mask],c_trend[newmask])
        if i > 50:
            clip = 0
    if cutboth:
        outmask = mask
    return t, c_trend, outmask, m


def plot_to_origin(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])


def jconvolve(h1, h2, x1, x2=None, mode="math", normed=None):
    """convolve two functions with equal sample spacing

    mode: 'math' (default), 'average'
           math: corresonds to simply doing the convolution of two functions
           average: corresponds to averaging random variables
           sum: correspondss to sum of random variables. same as 'math'
    normed: None (default), 'density', 'max', float or int, 'none'
            if a float or in, it will scale the peak to that value
            density makes integral = 1
            max scales to peak = 1
            'none' = forces don't rescale (good for mode = math)

    written for dealing with PDFs,

    """
    if mode == "sum":
        mode = "math"

    if isinstance(normed, int):
        normed = float(normed)

    if x2 is None:
        x2 = x1

    # get sample spacing
    delta = np.diff(x1)[0]

    # get convolution #multiply by sample spacing (integral needs dx)
    # numpy just uses the sum, not the integral
    hconv = np.convolve(h1, h2) * delta

    # math mode should'nt be scaled
    # average mode should return density
    if (mode == "math") & (normed is None):
        normed = None
    elif (mode == "average") & (normed is None):
        normed = "density"

    # the starting point is x1[0] + x2[0]
    Xconv = ((x1[0] + x2[0]) / delta + np.arange(len(hconv))) * delta
    if mode == "math":
        pass
    elif mode == "average":
        # when computing the average, the x-axis is down-weighted
        # by a factor of 2. this is accomplished, by rescaling
        # the x-axis
        Xconv = Xconv / 2  # np.linspace(b[0],b[-1],len(hconv))

    # return properly scaled functions
    if (normed is None) or (normed == "none"):
        return hconv, Xconv

    elif normed == "density":
        hconv /= np.sum(hconv * np.diff(Xconv)[0])
        return hconv, Xconv

    elif (normed == "max") | (normed == 1):
        hconv /= np.max(hconv)
        return hconv, Xconv

    elif isinstance(normed, float):
        hconv /= np.max(hconv)
        hconv *= normed
        return hconv, Xconv


def jconvolve_funcs(
    x1, y1, x2, y2, outx, interp_kind="nearest", fill_value=0, **kwargs
):
    """
    convolve 2 arrays on different x-axes using interpolated functions
    jconvolve_funcs(x1, y1, x2, y2, outx,interp_kind='nearest',fill_value=0)
    fill_value is how the arrays will be extended.

    """
    # take two array and find the c
    # for now assume x1 == x2
    f = interpolate.interp1d(x1, y1, fill_value=fill_value, kind=interp_kind, **kwargs)
    g = interpolate.interp1d(x2, y2, fill_value=fill_value, kind=interp_kind, **kwargs)
    xmin, xmax = minmax(np.append(x1, x2))
    func = lambda tau: f(tau) * g(outx - tau)
    ht = integrate.quad_vec(func, xmin, xmax)[0]
    return ht, outx


def arr_to_rgb(arr, rgb=(0, 0, 0), alpha=1, invert=False, ax=None):
    """
    arr to be made a mask
    rgb:assumed using floats (0..1,0..1,0..1)

    """
    # arr should be scaled to 1
    img = np.asarray(arr, dtype=np.float64)
    img = img - np.nanmin(img)
    img = img / np.nanmax(img)
    im2 = np.zeros(img.shape + (4,))

    if isinstance(rgb, str):
        rgb = mpl.colors.to_rgb(rgb)

    if invert:
        img = 1 - img
    im2[:, :, 3] = img * alpha
    r, g, b = rgb
    im2[:, :, 0] = r
    im2[:, :, 1] = g
    im2[:, :, 2] = b

    #     if ax is None:
    #         ax = plt.gca()
    #     plt.sca(ax)
    #     plt.imshow(im2)

    return im2


def pdf_pareto(t, a, k, xmax=None):
    """PDF of Pareto distribution

    Parameters
    ----------
    t : input
        array
    a : power-law power (a = alpha-1 from real Pareto)
        array
    k : minimum value for power law
        array
    xmax : max value for, optional, by default None

    Returns
    -------
    PDF(t|a,k,xmax)
        numpy array
    """
    if xmax is None:
        out = ((a - 1) / k) * (t / k) ** (-a)
        out[(t < k)] = 0
        return out
    else:
        out = ((a - 1) / (k ** (1 - a) - xmax ** (1 - a))) * t ** (-a)
        out[(t <= k) | (t > xmax)] = 0
        return out


def cdf_pareto(t, a, k, xmax=None):
    """CDF of Pareto distribution

    Parameters
    ----------
    t : input
        array
    a : power-law power (a = alpha-1 from real Pareto)
        array
    k : minimum value for power law
        array
    xmax : max value for, optional, by default None

    Returns
    -------
    CDF(t|a,k,xmax)
        numpy array
    """
    if xmax is None:
        out = 1 - (k / t) ** (a - 1)
        out[t < k] = 0
        return out
    else:
        out = (1 - (t / k) ** (1 - a)) / (1 - (xmax / k) ** (1 - a))
        out[t <= k] = 0
        out[t > xmax] = 1
        return out

from scipy.spatial.distance import cdist
def mahalanobis(X):
    """mahalanobis distance for data
    X = np.array([x,y,z,...])


    Parameters
    ----------
    X : np.array (M x N)
        M x N array, with M varialbes,
            and N observations.
            print(X) should look like
            # [[x1, x2, x3, x4...xn],
            #  [y1, y2, y3, y4...yn].
            #  [z1, z2, z3, z4...zn],
            #   ..]
            # as if X = np.array([x, y, z, ...])

    Returns
    -------
    md: np.array
        the square of maholanobis distance
        it follows a chi2 distribution for normally
        distributed data
    """
    # let scipy do all the lifting
    # but this is a nice way anyways
    # C = np.cov(X.T)
    # P, D, T = eigen_decomp(C)
    # mu = np.mean(X, axis=1)
    # X = (X - mu)
    # wX = wX = X @ np.linalg.inv(T.T) #whitened data
    # #wXT = np.linalg.inv(T) @ X.T
    # #md = wX @ wX.T
    # #md = np.sqrt(md.diagonal())
    # md = np.linalg.norm(wX, axis=1)**2  #norm spannign [xi,yi,zi]
    # # md is distributed as chi2 with d.o.f. = # independent axes
    # md = X @ np.linalg.inv(C) @ X.T

    return cdist(X,np.atleast_2d(X.mean(axis=0)),metric='mahalanobis')[:,0]**2



def eigen_decomp(A, b=[0, 0], return_slope=False):
    """
    eigenvalue decomposition

    A: matrix,
    b: means (if matrix describes the covariance of a distribution)
    returns:
        P, D, T
            P: eigenmatrix P @ D @ P^-1 = A
            D: diagonal(eigenvalues)
            T: transorm matrix, P @ S
            data.(T.T^-1) = whitened data
                projects into the orthogonal eigenspace
                    / the space without covariance
    or if return_slope is True
        slope, intercept of the major axis of 1st two dimensions.
    """


    eVa, eVe = np.linalg.eig(A)
    # relationship in Mathematica is
    # {val, vec} = Eigensystem[m] // N
    # P = Transpose[vec/(Norm /@ vec)] (* normalize eigenvectors *)
    # d = DiagonalMatrix[val]
    # m == P.d.Inverse[P]
    # sometimes P in Mathematica is rotated +- 90 deg w.r.t numpy

    P, D = eVe, np.diag(eVa)
    # cov = P @ D @ np.linalg.inv(P)
    S = D ** 0.5

    T = P @ S  # transform from real to eigenspace
    # Columns of T are scaled eigenvectors

    if return_slope:
        m = P[1] / P[0]
        y_int = -m * b[0] + b[1]
        major = np.argmax(eVa)
        return m[major], y_int[major]
    else:
        return P, D, T

def plot_covariance_ellipse(cov, mu, n=1, ax=None, c='b', lw=1, zorder=100):

    P, D, T = eigen_decomp(cov, mu, return_slope=False)

    m = P[1] / P[0]
    major = np.argmax(D.diagonal())
    angle = np.arctan(m)[major] * 180 / np.pi

    axes = n * np.sqrt(D.diagonal())
    b, a = axes[np.argsort(D.diagonal())]
    # let the width be the length fo the major axis
    pat = mpl.patches.Ellipse(
        angle=angle,
        xy=b,
        width=2*a,
        height=2*b,
        zorder=zorder,
        facecolor="none",
        edgecolor=c,
        lw=lw,
    )

    if ax is None:
        plt.gca().add_artist(pat)
    else:
        ax.add_artist(pat)

    return a, b, angle




def eigenplot(A, b=[0, 0], n=3, plot_data=False, vec_c="r", ell_c="b", ell_lw=2, **kwargs):
    # https://janakiev.com/blog/covariance-matrix/
    eVa, eVe = np.linalg.eig(A)
    b = np.array(b)


    if plot_data:
        data = np.random.multivariate_normal(b, A, 2000)

        plt.plot(*data.T, "k.")

    P, D = eVe, np.diag(eVa)
    S = D ** 0.5

    T = P @ S  # transform from real to eigenspace
    # Columns of T are scaled eigenvectors

    # for eigenvector in T

    for i in T.T:
        i = b + n * i
        plt.plot([b[0], i[0]], [b[1], i[1]], c=vec_c, zorder=100, **kwargs)

    m = P[1] / P[0]
    y_int = -m * b[0] + b[1]
    major = np.argmax(eVa)
    angle = np.arctan(m)[major] * 180 / np.pi
    # print(angle)
    # get the norm of the
    # a1 = 2 * n * np.linalg.norm(T, axis=0)
    a1 = 2 * n * np.sqrt(eVa)
    h, w = a1[np.argsort(eVa)]

    pat = mpl.patches.Ellipse(
        angle=angle,
        xy=b,
        width=w,
        height=h,
        zorder=100,
        facecolor="none",
        edgecolor=ell_c,
        lw=ell_lw,
    )
    plt.gca().add_artist(pat)

    # print(m[major], y_int[major])
    return m[major], y_int[major]


def eigenplot_from_data(x, y, n=3, data=False, vec_c="r", ell_c="b", ell_lw=2):
    g = np.isfinite(x + y)
    cov = np.cov(x[g], y[g])
    b = np.mean(x[g]), np.mean(y[g])
    if data:
        plt.plot(x, y, "k.", zorder=0)
    out = eigenplot(cov, b, data=False, n=n, vec_c=vec_c, ell_c=ell_c, ell_lw=ell_lw)
    return out


def print_bces(bc):
    a, b, erra, errb, covab = bc
    types = ["y/x", "x/y", "bisec", "ortho"]

    for i, t in enumerate(types):
        print(
            f"{t}: \t m:{a[i]:6.3g} +/- {erra[i]:6.3g} \t b:{b[i]:6.3g} +/- {errb[i]:6.3g}"
        )


def figsize(arr, default=[6, 6]):
    arr = np.array(arr)
    norm = np.array(arr.shape) / np.max(arr.shape)
    figsize = (np.array(default) * norm)[::-1]
    return figsize
