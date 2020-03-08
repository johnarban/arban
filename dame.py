import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, kernels

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

    fileout = filen.replace('.', '_2d.')

    data.writeto(fileout,overwrite=True)
    data.close()
    return 0

def getpixelscale(filename=None, wcs=None):
    pixel_scale = np.abs(header['CDELT1'])

def smoothmap(filen=None, fitsobj=None, pixel_scale = None, beamin=None, beamout=None, hdu=0, unit='deg'):

    if (filen is None) & (fitsobj is None):
        print('Please give a filename and input and output beam in the same units')
        print('acceptable units are "arcmin" "arcsec" and "deg"')
        print('example')
        print(' smoothmap filename.fits pixel_scale beam_in beam_out arcmin ')
        return -1

    if filen is not None:
        data = fits.getdata(filen, hdu=hdu)
        header = fits.getheader(filen, hdu=hdu)
    elif fitsobj is not None:
        data = fitsobj[hdu].data
        header = fitsobj[hdu].data
    else:
        filen = input('Enter filename: ')
        data = fits.getdata(filen, hdu=hdu)
        header = fits.getheader(filen, hdu=hdu)


    convol_beam = np.sqrt(beamout ** 2 - beamin ** 2)

    convol_beam = convol_beam / pixel_scale

    kernel = kernels.Gaussian2DKernel(convol_beam)
    convol_data = convolve(data, kernel,boundary='wrap')
