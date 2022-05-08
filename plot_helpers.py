import math
import os
import statistics
import sys
from importlib import reload
import warnings

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
from scipy import integrate, interpolate, ndimage, signal, stats
import scipy.special as special
from weighted import quantile
from bces.bces import bces
from astropy.stats import mad_std

from matplotlib.patheffects import withStroke





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
    
    
    
    
def wcsaxis(header, N=6, ax=None, fmt="%0.2f", use_axes=False,label=True):
    oldax = plt.gca()
    if ax is None:
        ax = plt.gca()
    plt.sca(ax)
    xlim = ax.axes.get_xlim()
    ylim = ax.axes.get_ylim()

    if isinstance(header,WCS):
        wcs = header
    else:
        wcs = WCS(header)

    # naxis = header["NAXIS"]  # naxis
    naxis = wcs.wcs.naxis
    # naxis1 = header["NAXIS1"]  # naxis1
    # naxis2 = header["NAXIS2"]  # naxis2
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

    plt.xticks(x, [fmt % i for i in x_tick],rotation=45)
    plt.yticks(y, [fmt % i for i in y_tick])

    if label:
        if wcs.wcs.ctype[0][0].lower() == "g":
            ax.set_xlabel("Galactic Longitude (l)")
            ax.set_ylabel("Galactic Latitude (b)")
        else:
            ax.set_xlabel("Right Ascension (J2000)")
            ax.set_ylabel("Declination (J2000)")

    ax.axes.set_xlim(xlim[0], xlim[1])
    ax.axes.set_ylim(ylim[0], ylim[1])

    plt.sca(oldax)
    return ax
    
    

def color_hue_shift(c, shift=1):
    c = mpl.colors.to_rgb(c)
    h, s, v = mpl.colors.rgb_to_hsv(c)
    h = h + shift % 1
    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))


def adjust_lightness(color, amount=0.5):
    # brighter : amount > 1
    # https://stackoverflow.com/a/49601444
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = rgb_to_hls(*mc.to_rgb(c))
    return hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def plotoneone(scale=1,
    color="k",
    lw=2,
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

def plot_to_origin(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    return None