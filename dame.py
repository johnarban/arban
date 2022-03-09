import numpy as np
from math import floor,ceil,sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import rv_histogram
from importlib import reload


from astropy.io import fits
from astropy.convolution import convolve, kernels
from astropy.table import Table
from astropy.wcs import WCS
from astropy.constants import G as Ggrav
import astropy.units as u

import pandas as pd
import glob
import os
import utils as ju
import sys

import skimage.morphology as skmorph



from tqdm import tqdm

import john_plot as jplot
reload(jplot)
import sphere as jsphere
reload(jsphere)
import moment_masking as jmm
reload(jmm)

np.seterr(divide="ignore")


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
        hdr[f"CTYPE{i}"] = hdr[f"CTYPE{i+1}"]
        hdr[f"CRVAL{i}"] = hdr[f"CRVAL{i+1}"]
        hdr[f"CDELT{i}"] = hdr[f"CDELT{i+1}"]
        hdr[f"CRPIX{i}"] = hdr[f"CRPIX{i+1}"]

    hdr.remove("CTYPE3")
    hdr.remove("CRVAL3")
    hdr.remove("CDELT3")
    hdr.remove("CRPIX3")

    hdr.add_history("MADE 2D HEADER")
    return 0


def getheader2d(filen):
    data = fits.open(filen)

    data[0].data = np.nansum(data[0].data, axis=-1)

    hdr = data[0].header
    header3dto2d(hdr)

    fileout = filen.replace(".", "_wco.")

    data.writeto(fileout, overwrite=True)
    data.close()
    return 0

def mass(surfd, mask = None, scale = 1, pix_linear_scale = 1, err=None):
    if mask is None:
        mask = np.full_like(surfd,True)

    scale_factor = scale * (pix_linear_scale ** 2)
    surf_d = surfd * scale_factor
    if err is None:
        return np.sum(surf_d[mask])
    else:
        m = np.sum(surf_d[mask])
        e = np.sqrt(np.sum((err * scale_factor)[mask]**2))
        return m, e
    return None


def radius(mask, pix_linear_scale=1):
    """find radius assuming circle (R = sqrt(Area/π))
    """
    N = np.sum(mask * pix_linear_scale**2)
    return np.sqrt(N / np.pi)


# these functions lifted from astrodendro (roslowsky something)
def spatial_mom0(arr,mask=None):
    """The sum of the values"""
    if mask is None:
        mask = np.full(arr.shape,True)
    values = arr[mask]
    return np.nansum(values)

def spatial_mom1(arr, mask=None):
    """The intensity-weighted mean position"""
    """ stolen & modded from astrodendro"""
    if mask is None:
        mask = np.full(arr.shape,True)

    mom_0 = spatial_mom0(arr,mask)
    r,c = np.indices(mask.shape)[:,mask]
    values = arr[mask]
    return np.array([np.nansum(i * values)/mom_0 for i in [r,c]])

def mom2(arr,mask=None):
        """The intensity-weighted covariance matrix"""
        """ stolen abd modded from astrodendro"""
        if mask is None:
            mask = np.full(arr.shape,True)

        mom_1 = spatial_mom1(arr,mask)
        mom_0 = spatial_mom0(arr,mask)

        indices = np.indices(mask.shape)[:,mask]
        values = arr[mask]

        v = values / mom_0

        nd = len(indices)
        zyx = tuple(i - m for i, m in zip(indices, mom_1))

        result = np.zeros((nd, nd),dtype=float)

        for i in range(nd):
            result[i, i] = np.nansum(v * zyx[i] ** 2)
            for j in range(i + 1, nd):
                result[i, j] = result[j, i] = np.nansum(v * zyx[i] * zyx[j])

        return result

def radius2(arr, mask, pix_linear_scale=1):
    """find radius assuming circle (R = sqrt(Area/π))
    """
    Lr, Lc = spatial_mom2(arr,mask)
    return np.sqrt(Lr * Lc) * np.mean(np.atleast_1d(pix_linear_scale))

def virial_mass(sigma, radius):
    sigma_3d_sq = 3 * (sigma * u.km / u.s) ** 2
    r = radius * u.pc
    return (sigma_3d_sq * r / Ggrav).to("Msun")




def mass_radius(surfd, mask, scale, pixscale, err=None):
    """pixscale is the physcial (parsec) scale of a pixel
       scale is the conversion from observed units to Msun/pc^2
    """

    r = radius(mask, pix_linear_scale = pixscale)
    if err is None:
        m = mass(surfd,mask=mask,scale=scale,pix_linear_scale=pixscale, err=err)
        return m, r
    else:
        m, merr = mass(surfd,mask=mask,scale=scale,pix_linear_scale=pixscale, err=err)
        return m, r, merr



def sigma(mass, radius):
    return mass / (np.pi * (radius ** 2))



def get_bounds(l=None, b=None, obj=None):
    """

    Parameters
    ----------
    l : GLON, optional
        array, by default None
    b : GLAT, optional
        array, by default None
    obj : Object name, optional
        , by default None
    Objects:
    Tau: Taurus
    Per: Perseus
    CA: California
    MonOB: MonOB1
    MonR2: MonR2
    W3: W3
    OriA
    OriB
    Herc: Hercules

    """
    with np.errstate(all="ignore"):
        # TA = (l <= 180) & (l >= 165) & (b <= -10) & (b >= -20)
        TA = (l <= 180) & (l >= 165) & (b <= -10.5) & (b >= -19.75)

        # CA1 = (l >= 155) & (l <= 169) & (b >= -10) & (b <= -5)
        CA1 = (l >= 155) & (l <= 169.5) & (b >= -10) & (b <= -5)
        CA2 = (l > 155) & (l < 162) & (b > -15) & (b < -10)
        CA = CA1 | CA2
        # remove L1434
        newCA = CA & ~(b < -13)
        # remove 200 pc foreground
        newCA = newCA & ~((l >= 167.5) & (b < -8.7))
        # remove 1kpc background
        newCA = newCA & ~((l <= 166) & (b >= -6.5))

        PR = (l >= 155) & (l <= 165) & (b >= -25) & (b <= -15)

        monob = (l >= 198) & (l <= 205) & (b >= -0.5) & (b <= 3.25)
        W3 = (l >= 132) & (l <= 135) & (b >= -0.5) & (b <= 2)

        # Oph = (l >= 350) & (l <= 358) & (b >= 12) & (b <= 20)
        Oph = ((l >= 350) | (l <= 11)) & (b >= 12) & (b <= 25)
        Herc = (l >= 41.25) & (l <= 48) & (b >= 7.0) & (b <= 10.75)

        split_a_b = -17 #A and B overlap
        OriA = (l >= 203) & (l <= 217) & (b >= -21) & (b <= split_a_b)
        OriB = (l >= 204) & (l <= 208) & (b >= -18) & (b <= -10)
        if obj == "OriB":
            OriB = (ju.rot_mask(OriB, angle=30)) & (b >= split_a_b)
        MonR2 = (l >= 210) & (l <= 218) & (b >= -14.5) & (b <= -10)

        RCrA = np.full_like(l, True).astype(bool)
        RCrA = ((l > 357.875) | (l < 4.125)) & ((b>-24.125) & (b<-15.875))

        Rose = (l >= 205) & (l <= 209) & (b >= -4) & (b <= 0)

        Pol = (l >= 117) & (l <= 127) & (b >= 20) & (b <= 30)
        notPol = (b < 22) & (l > 123)
        Pol = Pol & ~notPol

    if obj == "Tau":
        return TA.astype(bool)
    elif obj == "CA":
        return newCA.astype(bool)
    elif obj == "Per":
        return PR.astype(bool)
    elif obj == "Herc":
        return Herc.astype(bool)
    elif obj == "W3":
        return W3.astype(bool)
    elif obj == "Oph":
        return Oph.astype(bool)
    elif obj == "MonOB1":
        return monob.astype(bool)
    elif obj == "OriA":
        return OriA.astype(bool)
    elif obj == "OriB":
        return OriB.astype(bool)
    elif obj == "MonR2":
        return MonR2.astype(bool)
    elif obj == "RCrA":
        return RCrA.astype(bool)
    elif obj == "Rose":
        return Rose.astype(bool)
    elif obj == "Pol":
        return Pol.astype(bool)
    else:
        return None

def analysis(
    ak,
    wco,
    boundary,
    co_mask=None,
    tmass=None,
    df=None,
    pixel_scale=0.125,
    dist=1000,
    lim=None,
    ak_scale=183,
    ak_nh2=83.5e20,
    alpha_co=4.389,
    name="Cloud",
):
    """
    boundary: rectangular cloud boundary
    noise_mask: CO noise mask
    CO cloud boundary
    """
    # GET AK DEFINED MASSES
    if lim is None:
        lim = closed_contour(ak,bound=boundary,lim=.95)[1]
    above_lim = ak > lim
    ak_mask = above_lim & boundary
    pixel_pc = pixel_scale**0.5 #np.tan(pixel_scale * np.pi / 180) * dist

    if co_mask is None:
        co_mask = (wco>0) & np.isfinite(ak + wco) & boundary
    else:
        co_mask = co_mask & boundary
    print(f"analysis:: {name}")
    print(f"analysis:: Closed contour {lim:0.2f}")
    print("analysis:: #(Ak): ", np.sum(ak_mask))
    print("analysis:: #(CO): ", np.sum(co_mask))
    print("analysis:: #(CO & Ak): ", np.sum(co_mask & ak_mask))
    print("analysis:: alpha_co: ", alpha_co)
    print("analysis:: ak_scale: ", ak_scale)

    mass_ak_ak = mass(ak, ak_mask, ak_scale, pixel_pc)  # AK MASS, AK BOUNDARIES
    # CO MASS, AK BOUNDARIES
    mass_co_ak = mass(np.nan_to_num(wco), ak_mask, alpha_co, pixel_pc)
    radius_ak = radius(ak_mask, pixel_pc)
    sigma_ak_ak = sigma(mass_ak_ak, radius_ak)
    sigma_co_ak = sigma(mass_co_ak, radius_ak)

    # CO MASS, CO BOUNDARIES
    mass_co_co = mass(np.nan_to_num(wco), co_mask, alpha_co, pixel_pc)
    mass_ak_co = mass(ak, co_mask, ak_scale, pixel_pc)  # CO MASS, CO BOUNDARIES
    radius_co = radius(co_mask, pixel_pc)
    sigma_co_co = sigma(mass_co_co, radius_co)
    sigma_ak_co = sigma(mass_ak_co, radius_co)


    columns = [
        "mass_ak_r_ak",
        "mass_co_r_co",
        "mass_ak_r_co",
        "mass_co_r_ak",
        "radius_co",
        "radius_ak",
        "sigma_m_ak_r_ak",
        "sigma_m_co_r_co",
        "sigma_m_co_r_ak",
        "sigma_m_ak_r_co",
        "Area_ak",
        "Area_co",
        "distance",
        'aco'
    ]
    if tmass is not None:
        columns.insert(7,'mass_2m_r_ak')
        columns.insert(7,'mass_2m_r_co')
        mass_2m_ak = mass(tmass, ak_mask, ak_scale, pixel_pc)
        mass_2m_co = mass(tmass, co_mask, ak_scale, pixel_pc)

    row = [name]
    df = pd.DataFrame(index=row, columns=columns)

    df.mass_ak_r_ak = mass_ak_ak
    df.mass_co_r_ak = mass_co_ak
    df.mass_co_r_co = mass_co_co
    df.mass_ak_r_co = mass_ak_co
    df.radius_co = radius_co
    df.radius_ak = radius_ak
    df.Area_ak = np.pi * radius_ak ** 2
    df.Area_co = np.pi * radius_co ** 2
    df.sigma_m_ak_r_ak = sigma_ak_ak
    df.sigma_m_co_r_ak = sigma_co_ak
    df.sigma_m_co_r_co = sigma_co_co
    df.sigma_m_ak_r_co = sigma_ak_co
    df.distance = dist

    if tmass is not None:
        df.mass_2m_r_ak = mass_2m_ak
        df.mass_2m_r_co = mass_2m_co

    df.Aperpix = np.mean(np.atleast_1d(pixel_pc) ** 2)
    print(f"analysis:: Map total pixels: {np.sum(np.isfinite(boundary))}")

    # measure Xco
    aco = mass_ak_co / mass_co_co #CO & dust mass using CO boundary

    df.aco = aco
    print("\n")
    return df, ak_mask, co_mask

def getlb(header, amap=None, origin=0):
    """get GLAT, GLON coords. Origin=0 indicates
    indices are in numpy format. FITS is origin=1
    """
    wcs = WCS(header)
    if amap is None:
        shape = wcs.array_shape
    else:
        shape = amap.shape
    ys, xs = np.indices(shape)
    return wcs.all_pix2world(xs, ys, origin)


def getv(header3d, naxis=1):
    """get velocity vector
    """
    naxis = int(naxis)
    x = np.arange(header3d[f"NAXIS{naxis}"])
    v = header3d[f"CRVAL{naxis}"] + x * header3d[f"CDELT{naxis}"]
    return v


def dame_bad(arr,header=None, unscaled=False):
    """ Get the blank and 0 filled pixels
    from a moment masked array
    """
    if header is not None:
        blank = header['BLANK']
        zero = dame_itemp(0,header=header)
    else:
        unscale=True
    if unscaled:
        return (arr == blank) , (arr == zero)
    else:
        return np.isnan(arr) #| (arr < 1e-4)

def dame_near_zero(arr, unscaled=False):
    if unscaled:
        return arr == -29248
    else:
        return (arr <1e-5)


def dame_int(a,i,j):
    ai,aj = a.shape[0],a.shape[1]
    im = i-1
    ip = i+1
    jm = j-1
    jp = j+1
    #a_sub = [a[ii,jj] for ii in [im,i,ip] if 0<=ii<ai for jj in [jm,j,jp] if 0<=jj<aj]
    #return sum(a_sub)/len(a_sub)
    cs = [i+1,j],[i,j+1],[i-1,j],[i,j-1]
    a_sub = [a[ii,jj] for ii,jj in cs if (0<=ii<ai) & (0<=jj<aj) ]

    return np.sum(a_sub,axis=0)/len(a_sub)

def dame_nint(i,blank=-32768):
    if hasattr(i, '__iter__'):
        out = np.full_like(i,blank,dtype=int)
        out[i<0] = (i[i<0] - 0.5).astype(int)
        out[i>=0] = (i[i>=0] + 0.5).astype(int)
        return out
    else:
        if i<0:
            return int(i-0.5)
        else:
            return int(i+0.5)

def dame_itemp(t,bscale=1,bzero=0,blank=-32768, header=None):
    """
    /* return FITS integer value corresponding to real value t */

    """
    if header is not None:
        bscale = header['BSCALE']
        bzero = header['BZERO']

    return dame_nint((t - bzero) / bscale)


def dame_temp(i,bscale=1,bzero=0,blank=-32768,header=None):
    """
    /* Return temperature corresponding to FITS integer value ival */
    /* Return 1.e30 (undef) if pixel is blank */
    """

    if header is not None:
        bscale = header['BSCALE']
        bzero = header['BZERO']

    if hasattr(i, '__iter__'):
        t = (i * bscale + bzero).astype(float)
        t[i==blank] = np.nan
    else:
        if i==blank:
            return np.nan
        else:
            t = i * bscale + bzero

    return t



def mask_dame_wco(co, co_raw, noise=None, level=3):
    with np.errstate(all="ignore"):
        bad = np.isnan(co)
        if noise is None:
            noise = np.nanstd(co_raw[bad])
        N = np.nansum(~bad, axis=-1)
        sqrtN = np.sqrt(N)
        wco = np.nansum(co, axis=-1)
        return ((wco / (sqrtN * noise)) > level) & (N > 0), noise, N

def iscontained(bound, labels, label_id, lim=1):
    good = labels == label_id
    if lim == 1:
        return np.all(bound[labels == label_id])
    else:
        min_frac = np.sum(bound) / bound.size
        if lim < min_frac:
            lim = 1
        return np.sum(bound[good]) >= lim * np.sum(good)


def closed_contour(field, bound=None, steps=None, lim=1, min_size=0, nan_value=0,deep=False):
    """lowest value where all contours are closed

    Parameters
    ----------
    field : [type]
        [description]
    bound : [type]
        [description]
    steps : [type]
        [description]
    lim : int, optional
        [description], by default 1
    min_size : int, optional
        [description], by default 0

    Returns
    -------
    [largest_contour, largest_contour_level, labels]
        largest_contour: map of largest object
        largest_contour_level: largest closed contour level
        labels: map of objects at the contour level
    """
    if bound is None:
        bound = np.full(field.shape, True, dtype=bool)
    else:
        bound = bound.copy()  # we don't want to change the bounday

    if steps is None:
        steps = np.linspace(*np.percentile(field[bound],[2,98]),100)

    # we need a external border for the contour
    min_frac = np.sum(bound) / field.size
    if lim < min_frac:
        lim = 1
    if min_frac == 1:
        bound[0, :] = False
        bound[:, 0] = False
        bound[-1, :] = False
        bound[:, -1] = False
        lim = 1

    # get rid of nans (set to zero by default for positivily valued fields)
    field = field.copy()
    field = np.nan_to_num(field, nan=nan_value)

    # want to search from highest to lowest
    # like a dendrogram
    steps = np.sort(steps)[::-1]

    # out_contour :: track largest contour so far
    out_good_labels = None
    out_contour = steps[0]
    for j,i in enumerate(steps):
        shed = field >= i
        label = skmorph.label(shed)
        good = np.unique(label[shed & bound])
        good_labels = np.isin(label, good)
        # find contours contained within boundary
        # good_contained = [iscontained(bound,label,g,lim=1) for g in good]

        if np.sum(good_labels & bound) >= lim * np.sum(good_labels):
            # if np.any(good_contained):
            out_contour = i
            out_good_labels = good_labels.copy()
            # out_good_labels = np.isin(label,good[good_contained])
            out_label = label * 1
            continue
        else:
            if out_good_labels is None:
                out_good_labels = good_labels.copy()
                # out_good_labels = np.isin(label,good[good_contained])
                out_label = label * 1
            break

    if deep:
        step = np.where([steps==out_contour])[0]

        g = (field >= steps[step-1])  & (field <= steps[step+1]) & bound
        new_steps = np.sort(field[g])[::-1]
        out_good_labels, out_contour, out_label = closed_contour(field,bound=bound,steps=new_steps,lim=lim,min_size=min_size,nan_value=nan_value,deep=False)

    return out_good_labels, out_contour, out_label


def largest_closed_contour(field, bound, steps, lim=1, min_size=0, progress=False):
    """find the largest object with a closed contour

    Parameters
    ----------
    field : [type]
        [description]
    bound : [type]
        [description]
    steps : [type]
        [description]
    lim : int, optional
        [description], by default 1
    min_size : int, optional
        [description], by default 0

    Returns
    -------
    [largest_contour, largest_contour_level, labels]
        largest_contour: map of largest object
        largest_contour_level: largest closed contour level
        labels: map of objects at the contour level
    """
    if bound is None:
        bound = np.full(field.shape, True, dtype=bool)

    min_frac = np.sum(bound) / field.size
    h, w = bound.shape
    if lim < min_frac:
        lim = 1
    if min_frac == 1:
        bound[0, :] = False
        bound[:, 0] = False
        bound[-1, :] = False
        bound[:, -1] = False
        lim = 1
    field = np.nan_to_num(field)

    largest_old = min_size

    if progress:
        steps = tqdm(steps)
    for i in steps:
        shed = field >= i
        label = skmorph.label(shed)
        good = np.unique(label[shed & bound])
        # good_labels = np.isin(label,good)
        # find contours contained within boundary
        contained = [iscontained(bound, label, g, lim=lim) for g in good]

        if np.any(contained):
            # print('contained',i)
            contained = good[contained]

            # get contained contour sizes
            sizes = [np.sum(label == l) for l in contained]
            argmax = np.argmax(sizes)
            largest_new = max(sizes)
            largest_label = label == contained[argmax]

            # dont't change if contour size stops increasing
            if largest_new >= largest_old:
                if largest_new > largest_old:
                    largest_contour = largest_label
                    largest_contour_level = i
                    labels = label.copy()

            largest_old = largest_new
    return largest_contour, largest_contour_level, labels

def lon_ptp(lons):
    s = lons.copy()
    w = s >= 180
    s[w] = s[w] - 360
    return np.ptp(s)




def channel_maps(mmap,survey=False,velmin=None,velmax=None,velskip=None,figsize=10,nrows=None,ncols=None,which='interp',line_color='k',ncols_max=np.inf,
                overlay_dust=None,verbose=True,set_bad=0,interpolation='bicubic',colorbar=True,**kwargs):
    print('***********{n}***********'.format(n=mmap.name))
    r,c = np.indices(mmap.shape)

    if not survey:
        sl = slice(*ju.minmax(r[mmap.boundary])),slice(*ju.minmax(c[mmap.boundary]))
    else:
        sl = slice(None,None),slice(None,None)

    zero = (~mmap.bad).astype(int)
    if which[0].lower() == 'r':
        co = mmap.co_raw
        zero = (~dame_bad(mmap.co_raw)).astype(int)
    elif which[0].lower() == 'i':
        co = mmap.co_interp
    else:
        co = mmap.co

    fig, axs = jplot.channel_maps(co[sl[0],sl[1],:],
                            v=mmap.v,dv=mmap.dv,spec_ax=-1,
                            wcs=mmap.wcs[sl[0],sl[1]],
                            velmin=velmin,velmax=velmax,velskip=velskip,nrows=nrows,ncols=ncols,ncols_max=ncols_max,
                            figsize=figsize,verbose=verbose,set_bad=set_bad,interpolation=interpolation,colorbar=colorbar,**kwargs)

    axs = np.array(fig.axes).flat

    if overlay_dust is not None:
        for i in range(len(axs)-1): #the last axis is the colorbar
            ax = axs[i]
            if overlay_dust is not None:
                ax.contour(mmap.planck[sl[0],sl[1]],levels=overlay_dust,colors=[line_color],linewidths=[.5])

    return fig



### LTE
hok = 0.0479924  # K/GHz
c = 2.99792458e10  # cm/s
nu_12CO_10 = 115.2712

def bkgnd(nu=nu_12CO_10, Tbkg = 2.73):
    return  hok * nu / (np.exp(hok * (nu / Tbkg)) - 1.0)

def tex(Tmb_p, nu=nu_12CO_10, Tbkg = 2.73):
        """ calculate excitation temp """

        g = hok * nu  # h*nu/k
        texm = g / np.log(1.0 + g / (Tmb_p + bkgnd(nu,Tbkg)))
        return texm

