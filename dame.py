import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.stats import rv_histogram


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
from fastkde import fastKDE as fk

fastKDE = fk.fastKDE

import plfit
import powerlaw

from tqdm import tqdm

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


def get_mass_over_av(
    akmap, wco, df=None, pixel_scale=0.125, dist=450, lim=0.1, scale=183, alpha_co=4.389
):

    above_lim = akmap >= lim
    good = np.isfinite(akmap) & above_lim

    N = np.sum(good)
    pixel_pc = np.tan(pixel_scale * np.pi / 180) * dist

    mass_ak = np.sum(akmap[good] * scale * (pixel_pc ** 2))
    mass_co = np.sum(wco[good] * alpha_co * (pixel_pc ** 2))

    radius = np.sqrt(N * (pixel_pc ** 2) / np.pi)

    sigma_ak = mass_ak / (np.pi * radius ** 2)
    sigma_co = mass_co / (np.pi * radius ** 2)
    print(f"Map total pixels: {np.sum(np.isfinite(akmap))}")

    # measure Xco
    xco = (akmap * 83.5e20) / wco
    mean_xco = np.nanmean(xco)
    # mean_xco = 10**np.nanmean(np.log10(xco))
    mass_cor = mass_co * mean_xco / 2e20  # scale to new mass
    sigma_cor = mass_cor / (np.pi * radius ** 2)

    print(f"Map pixels >{lim}: {N}")
    print(f"Mass Ak: {mass_ak:3.3g} Msun")
    print(f"Mass CO: {mass_co:3.3g} Msun")
    print(f"Mass CO (corrected): {mass_cor:3.3g} Msun")
    print(f"Xco: {mean_xco:3.3g} (cm^-2)/(K km/s)")
    print(f"Radius: {radius:3.3g} pc")
    print(f"Σ AK (Msun/pc^2): {sigma_ak:4.2f}")
    print(f"Σ CO (Msun/pc^2): {sigma_co:4.2f}")
    print(f"Σ CO cor (Msun/pc^2): {sigma_cor:4.2f}")

    if df is not None:
        df.mass = mass_ak
        df.radius = radius
        df.sigma = sigma_ak

        df.mass_co = mass_co
        df.radius_co = radius
        df.sigma_co = sigma_co
        df.mass_co_cor = mass_cor
        df.xco = mean_xco
        df.sigma_co_cor = sigma_cor
    return 0


def mass(surfd, mask, scale, pixscale, err=None):
    if err is None:
        return np.sum(surfd[mask] * scale * (pixscale ** 2))
    else:
        m = np.sum(surfd[mask] * scale * (pixscale ** 2))
        e = np.sum((err[mask] * scale * pixscale ** 2)) ** 0.5
        return m, e


def radius(mask, pixscale):
    N = np.sum(mask)
    return np.sqrt(N * (pixscale ** 2) / np.pi)


def sigma(mass, radius):
    return mass / (np.pi * (radius ** 2))


def virial_mass(sigma, radius):
    sigma_3d_sq = 3 * (sigma * u.km / u.s) ** 2
    r = radius * u.pc
    return (sigma_3d_sq * r / Ggrav).to("Msun")


def mass_radius(surfd, mask, scale, pixscale, err=None):
    N = np.sum(mask)
    m = np.sum(surfd[mask] * scale * (pixscale ** 2))
    r = np.sqrt(N * (pixscale ** 2) / np.pi)
    if err is None:
        return m, r
    else:
        merr = np.sqrt(np.sum((err[mask] * scale * pixscale) ** 2))
        return m, r, merr


def getmaps(objec, make_global=False, imin=0, imax=0):
    dirs = "/Users/johnlewis/dameco"
    file = glob.glob(f"{dirs}/co_survey/*{objec}*mom.fits")[0]
    print("getmaps::", objec)
    # if make_global:
    #    global planck, header, co, co_raw, wco, peakv, header3d, survey, obj, noise, N

    survey, obj, *_, = os.path.basename(file).split("_")
    print("getmaps::", survey)

    planck = fits.getdata(f"{dirs}/{obj}/{obj}_TAU353.fits")
    header = fits.getheader(f"{dirs}/{obj}/{obj}_TAU353.fits")
    tdust = fits.getdata(f"{dirs}/{obj}/{obj}_TEMP.fits")
    beta = fits.getdata(f"{dirs}/{obj}/{obj}_BETA.fits")
    planck_err = fits.getdata(f"{dirs}/{obj}/{obj}_ERR_TAU.fits")
    planck_fullres = fits.getdata(f"{dirs}/{obj}/{obj}_TAU353_full.fits")
    planck_errfullres = fits.getdata(f"{dirs}/{obj}/{obj}_ERR_TAU_full.fits")
    header_fullres = fits.getheader(f"{dirs}/{obj}/{obj}_TAU353_full.fits")

    momfile = glob.glob(f"{dirs}/co_survey/{survey}*mom.fits")
    rawfile = glob.glob(f"{dirs}/co_survey/{survey}*raw.fits")
    header3d = fits.getheader(momfile[0])
    header3d_raw = fits.getheader(rawfile[0])

    tmass = glob.glob(f"{dirs}/{obj}/nicest_reproj_smooth125.fits")[0]
    tmass = fits.getdata(tmass)
    etmass = glob.glob(f"{dirs}/{obj}/nicest_ivar_reproj_smooth125.fits")[0]
    etmass = fits.getdata(etmass)

    tmass_full = glob.glob(f"{dirs}/{obj}/nicest_reproj_full.fits")[0]
    tmass_full = fits.getdata(tmass_full)
    etmass_full = glob.glob(f"{dirs}/{obj}/nicest_ivar_reproj_full.fits")[0]
    etmass_full = fits.getdata(etmass_full) ** -0.5

    co = fits.getdata(momfile[0])
    co_raw = fits.getdata(rawfile[0])
    if imin == imax:
        wco = np.nansum(co, axis=2) * np.abs(header3d["CDELT1"])
        peakv = np.argmax(np.nan_to_num(co), axis=2)
        frac = 1
        offwco = wco * 0
    else:
        wco = np.nansum(co[:, :, imin:imax], axis=2) * np.abs(header3d["CDELT1"])
        offwco = np.nansum(co[:, :, 0:imin], axis=2) + np.nansum(
            co[:, :, imax:], axis=2
        )
        offwco *= np.abs(header3d["CDELT1"])
        # total = np.nansum(co,axis=2) * np.abs(header3d["CDELT1"])

        peakv = np.argmax(np.nan_to_num(co[:, :, imin:imax]), axis=2)
        with np.errstate(all="ignore"):
            frac = wco / (wco + offwco)  # (np.nansum(co,axis=2)*.65)
        # frac[np.isclose(offwco, 0, atol=0.31)] = 1
        # frac[np.isclose(wco, 0, atol=0.31)] = 0
        # frac[((wco + offwco) < wco) & (wco < 0)] = 0
        # frac[((wco + offwco) < wco) & (wco > 0)] = 1
        # frac[wco < 0] = 0
        # frac[offwco < 0] = 1
        frac[(wco < 0) & (offwco > 0)] = 0
        # frac[(wco < 0) & (offwco<=0) ] = 1
        frac[(wco == 0) & (offwco == 0)] = 1
        frac[offwco <= 0] = 1

    # if co_raw.shape == co.shape:
    #     _, noise, N = mask_dame_wco(co, co_raw)
    noise = np.nan
    bad = dame_bad(co)
    N = np.nansum(~bad, axis=-1)

    return (
        planck * frac,
        header,
        co,
        co_raw,
        wco,
        peakv,
        header3d,
        survey,
        obj,
        noise,
        N,
        planck_fullres,
        header_fullres,
        tdust,
        planck_err,
        frac,
        header3d_raw,
        tmass,
        etmass,
        tmass_full,
        etmass_full,
        planck_errfullres,
        beta,
    )


def analysis(
    ak,
    wco,
    boundary,
    noise_mask,
    co_mask=None,
    df=None,
    pixel_scale=0.125,
    dist=1000,
    lim=0.1,
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
    above_lim = ak >= lim
    akmask = np.isfinite(ak) & above_lim & boundary
    pixel_pc = np.tan(pixel_scale * np.pi / 180) * dist
    if co_mask is None:
        co_mask = noise_mask
    comask = co_mask & noise_mask & np.isfinite(ak + wco) & boundary
    print(f"analysis:: {name}")
    print("analysis:: N(Ak): ", np.sum(akmask))
    print("analysis:: N(CO): ", np.sum(comask))
    print("analysis:: N(CO & Ak): ", np.sum(comask & akmask))
    print("analysis:: alpha_co: ", alpha_co)
    print("analysis:: ak_scale: ", ak_scale)

    mass_ak_ak = mass(ak, akmask, ak_scale, pixel_pc)  # AK MASS, AK BOUNDARIES
    # CO MASS, AK BOUNDARIES
    mass_co_ak = mass(wco, akmask, alpha_co, pixel_pc)
    mass_co_noise_ak = mass(
        wco, noise_mask & akmask, alpha_co, pixel_pc
    )  # CO MASS, AK BOUNDARIES, NOISE CUT
    radius_ak = radius(akmask, pixel_pc)
    sigma_ak_ak = sigma(mass_ak_ak, radius_ak)
    sigma_co_ak = sigma(mass_co_ak, radius_ak)

    # CO MASS, CO BOUNDARIES
    mass_co_co = mass(wco, comask, alpha_co, pixel_pc)
    mass_ak_co = mass(ak, comask, ak_scale, pixel_pc)  # CO MASS, CO BOUNDARIES
    radius_co = radius(comask, pixel_pc)
    sigma_co_co = sigma(mass_co_co, radius_co)
    sigma_ak_co = sigma(mass_ak_co, radius_co)

    columns = [
        "mass_ak_ak",
        "mass_co_ak",
        "mass_co_xco_ak",
        "mass_co_co",
        "mass_ak_co",
        "mass_co_xco",
        "mass_co_noise_ak",
        "radius_co",
        "radius_ak",
        "sigma_ak_ak",
        "sigma_co_ak",
        "sigma_co_co",
        "sigma_ak_co",
        "sigma_co_xco",
        "sigma_co_xco_ak",
        "xco",
        "xco2",
        "xcolog",
        "Aperpix",
        "Area_ak",
        "Area_co",
        "distance",
    ]
    row = [name]
    df = pd.DataFrame(index=row, columns=columns)

    df.mass_ak_ak = mass_ak_ak
    df.mass_co_ak = mass_co_ak
    df.mass_co_co = mass_co_co
    df.mass_ak_co = mass_ak_co
    df.mass_co_noise_ak = mass_co_noise_ak
    df.radius_co = radius_co
    df.radius_ak = radius_ak
    df.Area_ak = np.pi * radius_ak ** 2
    df.Area_co = np.pi * radius_co ** 2
    df.sigma_ak_ak = sigma_ak_ak
    df.sigma_co_ak = sigma_co_ak
    df.sigma_co_co = sigma_co_co
    df.sigma_ak_co = sigma_ak_co
    df.distance = dist

    df.Aperpix = pixel_pc ** 2
    print(f"analysis:: Map total pixels: {np.sum(np.isfinite(boundary))}")

    # measure Xco
    xco = (ak * ak_nh2) / wco
    xco = xco
    g = (noise_mask & boundary).astype(bool)

    g = g & np.isfinite(xco)
    mean_xco = np.nanmean(xco[g])
    mean_xcolog = 10 ** np.nanmean(np.log10(xco[g]))
    # print(np.isfinite(xco[g]).all())
    if not np.isfinite(xco[g]).all():
        print("bad")
        global jox
        jox = xco, g
    xco2 = np.mean(ak[g] * ak_nh2) / np.mean(wco[g])

    mass_co_xco = mass_co_co * mean_xco / 2e20  # scale to new mass
    sigma_co_xco = sigma(mass_co_xco, radius_co)

    mass_co_xco_ak = mass_co_ak * mean_xco / 2e20  # scale to new mass
    sigma_co_xco_ak = sigma(mass_co_xco_ak, radius_ak)

    df.xco = mean_xco
    df.xco2 = xco2
    df.xcolog = mean_xcolog
    df.mass_co_xco = mass_co_xco
    df.sigma_co_xco = sigma_co_xco
    df.mass_co_xco_ak = mass_co_xco_ak
    df.sigma_co_xco_ak = sigma_co_xco_ak
    print("\n")
    return df, akmask, comask


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

        OriA = (l >= 203) & (l <= 217) & (b >= -21) & (b <= -17)
        OriB = (l >= 204) & (l <= 208) & (b >= -18) & (b <= -10)
        if obj == "OriB":
            OriB = (ju.rot_mask(OriB, angle=30)) & (b >= -17)
        MonR2 = (l >= 210) & (l <= 218) & (b >= -14.5) & (b <= -10)

        RCrA = np.full_like(l, True).astype(bool)

        Rose = (l >= 205) & (l <= 209) & (b >= -4) & (b <= 0)

        Pol = (l >= 117) & (l <= 127) & (b >= 20) & (b <= 32)
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


def getlb(header, amap, origin=0):
    ys, xs = np.indices(amap.shape)
    return WCS(header).all_pix2world(xs, ys, origin)


def getv(header3d, naxis=1):
    naxis = int(naxis)
    x = np.arange(header3d[f"NAXIS{naxis}"])
    v = header3d[f"CRVAL{naxis}"] + x * header3d[f"CDELT{naxis}"]
    return v


def dame_bad(arr, bad_val=20):
    with np.errstate(all="ignore"):
        c1 = -np.log2(np.abs(arr)) > bad_val - 5
        c2 = np.isnan(arr)
        c3 = arr == -32768
        c4 = arr == 0
    # with np.errstate(invalid='ignore'):
    #    c4 = np.log2(arr) < -10
    return c1 | c2 | c3 | c4


def mask_dame_wco(co, co_raw, noise=None, level=3):
    with np.errstate(all="ignore"):
        bad = dame_bad(co)
        if noise is None:
            noise = np.nanstd(co_raw[bad])
        N = np.nansum(~bad, axis=-1)
        sqrtN = np.sqrt(N)
        wco = np.nansum(co, axis=-1)
        return ((wco / (sqrtN * noise)) > level) & (N > 0), noise, N


def channel_maps(
    chmap,
    v,
    start=0,
    stop=-1,
    step=1,
    vmin=-0.2,
    vmax=10,
    xlim=None,
    ylim=None,
    **kwargs,
):
    if stop == -1:
        stop = chmap.shape[-1]
    chan = np.arange(start, stop + step, step)
    chan[-1] = len(v) - 1
    N = len(chan)
    sN = np.sqrt(N)
    nrow = int(np.ceil(sN))
    ncol = int(np.ceil(N / nrow))
    nrow, ncol
    ny, nx, _ = chmap.shape
    if ny > nx:
        ac = 1  # ny/nx
        ar = 1
    else:
        ac = 1
        ar = 1  # nx/ny
    fig, axs = plt.subplots(
        nrows=nrow,
        ncols=ncol,
        figsize=(4.0 * nrow * ar, 4.0 * ncol * ac),
        sharex=True,
        sharey=True,
    )
    for i, ax in enumerate(axs.flatten()):
        if i < N - 1:
            summed = np.nansum(chmap[:, :, chan[i] : chan[i + 1]], axis=-1)
            ax.imshow(summed, vmin=vmin, vmax=vmax, **kwargs)
            ju.annotate(
                f"{v[chan[i]]:0.1f}:{v[chan[i+1]]:0.1f}", 0.8, 0.9, ax=ax, fontsize=25
            )
            if xlim is not None:
                ax.set_xlim(*xlim)
            if ylim is not None:
                ax.set_ylim(*ylim)
            # plt.ylim(15, 45)
        else:
            plt.delaxes(ax)

    return fig, axs


def iscontained(bound, labels, label_id, lim=1):
    good = labels == label_id
    if lim == 1:
        return np.all(bound[labels == label_id])
    else:
        min_frac = np.sum(bound) / bound.size
        if lim < min_frac:
            lim = 1
        return np.sum(bound[good]) >= lim * np.sum(good)


def closed_contour(field, bound, steps, lim=1, min_size=0, nan_value=0):
    """[summary]

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
    field = np.nan_to_num(field, nan=nan_value)

    # want to search from highest to lowest
    # like a dendrogram
    steps = np.sort(steps)[::-1]

    # out_contour :: track largest contour so far
    out_good_labels = None
    out_contour = steps[0]
    for i in steps:
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

    return out_good_labels, out_contour, out_label


def largest_closed_contour(field, bound, steps, lim=1, min_size=0, progress=False):
    """[summary]

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

