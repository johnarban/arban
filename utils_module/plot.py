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


def nice_pandas(format="{:3.3g}"):
    pd.set_option("display.float_format", lambda x: format.format(x))


#############################
#############################
####  Plotting commands  ####
#############################
#############################

# Set uniform plot options

# some constants
fwhm = 2 * np.sqrt(2 * np.log(2))

def set_plot_opts(serif_fonts=True):

    if serif_fonts:
        mpl.rcParams["mathtext.fontset"] = "stix"
        mpl.rcParams["font.family"] = "serif"
        mpl.rcParams["font.size"] = 14
    return None


def check_iterable(arr):
    return hasattr(arr, "__iter__")


def color_array(arr, alpha=1):
    """ take an array of colors and convert to
    an RGBA image that can be displayed
    with imshow
    """
    img = np.zeros(arr.shape + (4,))
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            c = mpl.colors.to_rgb(arr[row, col])
            img[row, col, 0:3] = c
            img[row, col, 3] = alpha
    return img



def arr_to_rgb(arr, rgb=(0, 0, 0), alpha=1, invert=False, ax=None):
    """
    arr to be made a mask
    rgb:assumed using floats (0..1,0..1,0..1) or string

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


def wcsaxis(header, N=6, ax=None, fmt="%0.2f", use_axes=False,label=True):
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

    if label:
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


def color_hue_shift(c, shift=1):
    c = mpl.colors.to_rgb(c)
    h, s, v = mpl.colors.rgb_to_hsv(c)
    h = h + shift % 1
    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))

def plot_covariances(p, cov, names=None, figsize=(12, 12), nsamps=5000, smooth=1):
    p = np.random.multivariate_normal(p, cov, nsamps)
    fig, axs = corner(p, smooth=smooth, names=names, figsize=figsize)
    return fig, axs


def plot_astropy_fit_covariances(fit, fitter):
    p = fit.parameters
    cov = fitter.fit_info["param_cov"]
    ax = plot_covariances(p, cov, names=fit.param_names)
    return ax


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
    fmt=None,
    label=None,
    **kwargs,
):
    oldax = plt.gca()
    if ax is None:
        ax = oldax
    plt.sca(ax)

    if mid:
        alpha_fill = alpha * 2
        if alpha_fill >= 1:
            alpha_fill = 1
    plt.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha,label=label,**kwargs)
    if mid:
        plt.plot(x, y, "-", color=color, alpha=alpha, lw=lw, ls=ls,**kwargs)
    plt.sca(oldax)
    return None


def plot_to_origin(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    return None


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


def figsize(arr, default=[6, 6]):
    arr = np.array(arr)
    norm = np.array(arr.shape) / np.max(arr.shape)
    figsize = (np.array(default) * norm)[::-1]
    return figsize

