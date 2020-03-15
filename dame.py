import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, kernels
from astropy.wcs import WCS

import sys

def header3dto2d(hdr):
    """convert DAME CO Survey header from
    3D cube to 2D image. Be careful
    this changes the header in place

    Parameters
    ----------
    hdr : astropy Header
        3D astropy Header

    Returns
    -------
    0
        Changes the header in place
    """
    for i in [1, 2]:
        hdr[f'CTYPE{i}'] = hdr[f'CTYPE{i+1}']
        hdr[f'CRVAL{i}'] = hdr[f'CRVAL{i+1}']
        hdr[f'CDELT{i}'] = hdr[f'CDELT{i+1}']
        hdr[f'CRPIX{i}'] = hdr[f'CRPIX{i+1}']

    hdr.remove('CTYPE3')
    hdr.remove('CRVAL3')
    hdr.remove('CDELT3')
    hdr.remove('CRPIX3')

    hdr.add_history('MADE 2D HEADER')
    return 0

def getheader2d(filen):
    data = fits.open(filen)

    data[0].data = np.nansum(data[0].data,axis=-1)

    hdr = data[0].header
    header3dto2d(hdr)

    fileout = filen.replace('.', '_wco.')

    data.writeto(fileout,overwrite=True)
    data.close()
    return 0

def get_mass_over_av(akmap, wco=None, df=None,pixel_scale=0.125, dist=450, lim=.1,scale=183, alpha_co = 4.283):

    above_lim = akmap >= lim
    good = np.isfinite(akmap) & above_lim
    # if wco is None:
    #     pass
    # else:
    #     good = good & (wco != 0)
    N = np.sum(good)
    pixel_pc = np.tan(pixel_scale * np.pi / 180) * dist
    if wco is None:
        mass = np.sum(akmap[good] * scale * (pixel_pc ** 2))
    else:
        print('Using Sigma = 4.39 * Wco')
        mass = np.sum(wco[good] * alpha_co * (pixel_pc ** 2))
    radius = np.sqrt(N * (pixel_pc ** 2) / np.pi)
    surfdens = mass/(np.pi * radius**2)
    print(f'Map total pixels: {np.sum(np.isfinite(akmap))}')
    print(f'Map pixels >{lim}: {N}')
    print(f'Mass: {mass:3.3g} Msun')
    print(f'Radius: {radius:3.3g} pc')
    print(f'Σ (Msun/pc^2): {surfdens:4.2f}')
    if df is not None:
        if wco is None:
            df.mass = mass
            df.radius = radius
            df.sigma = surfdens
        else:
            df.mass_co = mass
            df.radius_co = radius
            df.sigma_co = surfdens
    return mass, radius, surfdens



def get_bounds(l=None,b=None,obj=None):
    TA = (l<=180) & (l>=165) & (b<=-10) & (b>=-20)

    CA1 = (l>=155) & (l<=169) & (b>=-10) & (b<=-5)
    CA2 = (l>155) & (l<162) & (b>-15) & (b<-10)
    CA = CA1 | CA2
    # remove L1434
    newCA = CA & ~(b<-13)
    # remove 200 pc foreground
    newCA = newCA & ~((l>=167.5) & (b<-8.75))
    # remove 1kpc background
    newCA = newCA & ~((l<=166) & (b>=-6.5))

    PR = (l>=155) & (l<=165) & (b>=-25) & (b<=-15)
    if obj=='TA':

        return TA
    elif obj=='CA':

        return CA
    elif obj=='newCA':
        return newCA
    elif obj=='PR':
        return PR
    else:
        return None

def getlb(header, mmap,origin=0):
    ys, xs = np.indices(mmap.shape)
    return WCS(header).all_pix2world(xs,ys, origin)

