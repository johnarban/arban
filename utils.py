# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:11:33 2016

@author: johnlewisiii
"""
import os,sys
import numpy as np
from scipy import stats, special, interpolate
import matplotlib.pyplot as plt
from weighted import quantile
from astropy.io import fits
from astropy.table import Table

__location__ = os.path.realpath(
    os.path.join(os.getcwd(), os.path.dirname(__file__)))
    
__filtertable__ = Table.read(os.path.join(__location__,'FilterSpecs.tsv'),format='ascii')
    
# Plot the KDE for a set of x,y values. No weighting
# code modified from 
# http://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
def kdeplot(xp, yp, filled=False, ax=None, grid=None, bw = None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    rvs = np.append(xp.reshape((xp.shape[0], 1)),
                    yp.reshape((yp.shape[0], 1)),
                    axis=1)

    kde = stats.kde.gaussian_kde(rvs.T)
    #kde.covariance_factor = lambda: 0.3
    #kde._compute_covariance()
    kde.set_bandwidth(bw)

    if grid is None:
        # Regular grid to evaluate kde upon
        x_flat = np.r_[rvs[:, 0].min():rvs[:, 0].max():256j]
        y_flat = np.r_[rvs[:, 1].min():rvs[:, 1].max():256j]
    else:
        x_flat = np.r_[0:grid[0]:complex(0,grid[0])]
        y_flat = np.r_[0:grid[1]:complex(0,grid[1])]
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

def sigconf1d(n):
    cdf = (1/2.)*(1+special.erf(n/np.sqrt(2)))
    return (1-cdf)*100,100* cdf,100*special.erf(n/np.sqrt(2))


# In[ Convert tables to arrays]
def table_to_array(table):
    arr = [list(t) for t in table]
    return np.asarray(arr)

def t2a(table):
    return table_to_array(table)

# In[Discrete Colorbar]
def discrete_cmap(colormap, N_colors):
    print 'Not doing anything yet'
    return colormap

# In[WCS axis labels]
def wcsaxis(wcs, N=6, ax=None,fmt='%0.2f'):
    if ax is None:
        ax = plt.gca()
    naxis1 = wcs['NAXIS1']
    naxis2 = wcs['NAXIS2']
    crpix1 = wcs['CRPIX1']
    crpix2 = wcs['CRPIX2']
    crval1 = wcs['CRVAL1']
    crval2 = wcs['CRVAL2']
    try:
        cdelt1 = wcs['CDELT1']
        cdelt2 = wcs['CDELT2']
    except:
        cdelt1 = wcs['CD1_1']
        cdelt2 = wcs['CD2_2']

    offset = (naxis1/N)/5
    x_tick_pix = np.linspace(offset,naxis1-offset,N) #generate 6 values from 0 to naxis1 (150)
    #x_tick_pix = ax.get_xticks()
    x_tick_label = (x_tick_pix - crpix1)*cdelt1 + crval1

    y_tick_pix = np.linspace(offset,naxis2-offset,N) #generate 6 values from 0 to naxis2 (100)
    #y_tick_pix = ax.get_yticks()
    y_tick_label = (x_tick_pix - crpix2)*cdelt2 + crval2
    
    plt.xticks(x_tick_pix, [fmt%i for i in x_tick_label])
    plt.yticks(y_tick_pix, [fmt%i for i in y_tick_label])
    
    if wcs['CTYPE1'][0].lower() == 'g':
        plt.xlabel('Galactic Longitude (l)')
        plt.ylabel('Galactic Latitude (b)')
    else:
        plt.xlabel('Right Ascension (J2000)')
        plt.ylabel('Declination (J2000)')
    
    return ax
    

# In[ writefits]
def writefits(filename, data, wcs = None,clobber=True):
    if wcs is not None:
        wcs = wcs.to_header()
    hdu = fits.PrimaryHDU(data,header=wcs)
    hdu.writeto(filename,clobber=clobber)
    return hdu
    
def grid_data(x,y,z,nxy=(512,512), interp='linear', plot = False,\
             cmap='Greys',levels=None, sigmas = None, filled = False):
    '''
    stick x,y,z data on a grid and return
    XX, YY, ZZ
    '''
    xmin,xmax = x.min(),x.max()
    ymin,ymax = y.min(),y.max()
    nx,ny = nxy
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(ymin, ymax, ny)
    xi, yi = np.meshgrid(xi, yi)

    zi = interpolate.griddata((x,y), z, (xi, yi),method=interp)
    
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
        cont(xi,yi,zi/np.max(zi[np.isfinite(zi)]),cmap=cmap,levels=levels)
    
    return xi, yi, zi


# To make this robust need a disambiguation table
# for filter names, or a decision tree to disamb.
def convert_flux(mag=None,emag=None,filt=None,return_wavelength=False):
    """"Return flux for a given magnitude/filter combo
    
    Input:
    mag -- the input magnitude. either a number or numpy array
    filter -- either filter zeropoint or filer name
    """
    
    if mag is None or filt is None:
        print 'List of filters and filter properties'
        __filtertable__.pprint(max_lines=len(__filtertable__)+3)
        return None
        
    
    if not isinstance(filt,float):
        tab = __filtertable__
        tab['fname'] = [s.lower() for s in tab['fname']]
        if not filt.lower() in tab['fname']:
            print 'Filter %s not found'%filt.lower()
            print 'Please select one of the following'
            print tab['fname'].data
            filt = input('Include quotes in answer (example (\'johnsonK\')): ')
        
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

    for i in xrange(4):
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
    for i in xrange(len(arr) - (n - 1)):
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

def shift_bins(arr,phase=0,nonneg=False):
    # assume original bins are nonneg
    if phase != 0:
        diff = np.diff(arr)
        if np.allclose(diff,diff[::-1]):
            diff = diff[0]
            arr = arr + phase*diff
            #pre = arr[0] + phase*diff
            return arr
        else:
            arr = np.log10(arr)
            diff = np.diff(arr)[0]
            arr = arr + phase * diff
            return np.power(10.,arr)
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
        print 'Error: Defaulting to 10 linears steps'
        n = 10.
        nisNone = False

    # either user specifies log or gives dex and not dx
    log = log or (dxisNone and (not dexisNone))
    if log:
        if xmin == 0:
            print "log(0) is -inf. xmin must be > 0 for log spacing"
        xmin, xmax = np.log10(xmin), np.log10(xmax)
    # print nisNone, dxisNone, dexisNone, log # for debugging logic
    if not nisNone:  # this will make dex or dx if they are not specified
        if log and dexisNone:  # if want log but dex not given
            dex = (xmax - xmin) / n
            # print dex
        elif (not log) and dxisNone:  # else if want lin but dx not given
            dx = (xmax - xmin) / n  # takes floor
            print dx

    if log:
        #return np.power(10, np.linspace(xmin, xmax , (xmax - xmin)/dex + 1))
        return np.power(10, np.arange(xmin, xmax + dex, dex))
    else:
        #return np.linspace(xmin, xmax, (xmax-xmin)/dx + 1)
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
                ra, de = n.split('+')
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
            ra, de = n.split('+')
            sign = '-'
        ra, de = name.split('+')
        ra = ra[0:2] + ':' + ra[2:4] + ':' + ra[4:6] + '.' + ra[6:8]
        de = sign + de[0:2] + ':' + de[2:4] + ':' + de[4:6]
        coord = SkyCoord(ra, de, frame='icrs', unit=('hourangle', 'degree'))
        return np.array(coord.ra.value), np.array(coord.dec.value)






















