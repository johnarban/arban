# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:11:33 2016

@author: johnlewisiii
"""

import numpy as np
from scipy import stats, special, interpolate
import matplotlib.pyplot as plt
from weighted import quantile
import george
from astropy.io import fits
import corner #dfm corner.py needed for quantiles

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
    x_tick_label = (x_tick_pix - crpix1)*cdelt1 + crval1

    y_tick_pix = np.linspace(offset,naxis2-offset,N) #generate 6 values from 0 to naxis2 (100)
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
def convert_flux(mag=None,emag=None,filter=None,return_wavelength=False):
    """"Return flux for a given magnitude/filter combo
    
    Input:
    mag -- the input magnitude. either a number or numpy array
    filter -- either filter zeropoint or filer name
    """
    
    if mag is None or filter is None:
        print 'List of filters and filter properties'
        tab = Table.read('FilterSpecs.tsv',format='ascii')
        tab.pprint()
        return None
        
    
    if not isinstance(filter,float):
        tab = Table.read('FilterSpecs.tsv',format='ascii')
        tab['fname'] = [s.lower() for s in tab['fname']]
        f0 = tab['F0_Jy'][np.where(filter.lower() in tab['fname'])][0]
    else:
        f0 = filter
    
    flux = f0 * 10.**(-mag/2.5)
    
    if emag is not None:
        eflux = 1.08574 * emag * flux
        
    if return_wavelength:
        return flux, eflux, tab['Wavelength'][np.where(filter.lower() in tab['fname'])][0]
    else:
        return flux, eflux
    
    
    
    
    
    






















