import numpy as np
from astropy.io import fits
from astropy.convolution import convolve, kernels
from astropy.table import Table
from astropy.wcs import WCS
import pandas as pd
import glob
import os
import utils as ju
import sys
import matplotlib.pyplot as plt


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
    akmap, wco, df=None, pixel_scale=0.125, dist=450, lim=0.1, scale=183, alpha_co=4.383
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


def mass(surfd, mask, scale, pixscale):
    return np.sum(surfd[mask] * scale * (pixscale ** 2))


def radius(mask, pixscale):
    N = np.sum(mask)
    return np.sqrt(N * (pixscale ** 2) / np.pi)


def sigma(mass, radius):
    return mass / (np.pi * (radius ** 2))


def getmaps(objec, make_global=False, vmin=0, vmax=0):
    dirs = "/Users/johnlewis/dameco"
    file = glob.glob(f"{dirs}/co_survey/*{objec}*mom.fits")[0]
    print("getmaps::", file)
    if make_global:
        global planck, header, co, co_raw, wco, peakv, header3d, survey, obj, noise, N

    survey, obj, *mid, = os.path.basename(file).split("_")

    planck = 3233 * fits.getdata(f"{dirs}/{obj}/{obj}_TAU353.fits")
    header = fits.getheader(f"{dirs}/{obj}/{obj}_TAU353.fits")

    momfile = glob.glob(f"{dirs}/co_survey/{survey}*mom.fits")
    rawfile = glob.glob(f"{dirs}/co_survey/{survey}*raw.fits")
    header3d = fits.getheader(momfile[0])

    co = fits.getdata(momfile[0])
    co_raw = fits.getdata(rawfile[0])
    if vmin == vmax:
        wco = np.nansum(co, axis=2) * np.abs(header3d["CDELT1"])
        peakv = np.argmax(np.nan_to_num(co), axis=2)
        frac = 1
        offwco = wco * 0
    else:
        wco = np.nansum(co[:, :, vmin:vmax], axis=2) * np.abs(header3d["CDELT1"])
        offwco = np.nansum(co[:, :, 0:vmin], axis=2) + np.nansum(
            co[:, :, vmax:], axis=2
        )
        peakv = np.argmax(np.nan_to_num(co[:, :, vmin:vmax]), axis=2)
        frac = (wco) / (wco + offwco)  # (np.nansum(co,axis=2)*.65)
        frac[np.isclose(offwco, 0, atol=0.31)] = 1
        frac[np.isclose(wco, 0, atol=0.31)] = 0
        frac[wco < 0] = 0
        frac[offwco < 0] = 1

    if co_raw.shape == co.shape:
        _, noise, N = mask_dame_wco(co, co_raw)
        print("getmaps::", obj, noise)
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
    alpha_co=4.383,
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

    print("analysis:: N(Ak)", np.sum(akmask))
    print("analysis:: N(CO)", np.sum(comask))
    print("analysis:: N(CO & Ak)", np.sum(comask & akmask))
    print("analysis:: alpha_co", alpha_co)
    print("analysis:: ak_scale", ak_scale)

    mass_ak_ak = mass(ak, akmask, ak_scale, pixel_pc)  # AK MASS, AK BOUNDARIES
    mass_co_ak = mass(wco, akmask, alpha_co, pixel_pc)  # CO MASS, AK BOUNDARIES
    mass_co_noise_ak = mass(
        wco, noise_mask & akmask, alpha_co, pixel_pc
    )  # CO MASS, AK BOUNDARIES, NOISE CUT
    radius_ak = radius(akmask, pixel_pc)
    sigma_ak_ak = sigma(mass_ak_ak, radius_ak)
    sigma_co_ak = sigma(mass_co_ak, radius_ak)

    mass_co_co = mass(wco, comask, alpha_co, pixel_pc)  # CO MASS, CO BOUNDARIES
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
        "Apix",
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
    df.sigma_ak_ak = sigma_ak_ak
    df.sigma_co_ak = sigma_co_ak
    df.sigma_co_co = sigma_co_co
    df.sigma_ak_co = sigma_ak_co

    df.Apix = pixel_pc ** 2
    print(f"Map total pixels: {np.sum(np.isfinite(boundary))}")

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

    return df, akmask, comask


class DameMap(object):
    def __init__(self, name, distance=1000):
        cat = Table.read("mycat.csv").to_pandas()
        cat.set_index("Name", inplace=True)
        self.name = cat.loc[name].name
        self.survey = cat.loc[name].survey
        self.vmin = cat.loc[name].vmin
        self.vmax = cat.loc[name].vmax
        self.noise = cat.loc[name].noise
        allvar = getmaps(self.survey, make_global=False, vmin=self.vmin, vmax=self.vmax)
        # planck, header, co, co_raw, wco, peakv, header3d, survey, obj, noise, N
        self.planck = allvar[0]
        self.header = allvar[1]
        self.co = allvar[2]
        self.co_raw = allvar[3]
        self.wco = allvar[4]
        self.peakv = allvar[5]
        self.header3d = allvar[6]
        self.survey2 = allvar[7]
        self.obj = allvar[8]
        # self.noise = allvar[9]
        self.N = allvar[10]
        self.distance = cat.loc[name].distance
        self.survey_noise = cat.loc[name].noise
        self.shape = self.planck

        l, b = getlb(self.header, self.wco)
        self.l = l
        self.b = b
        self.v = getv(self.header3d)
        self.dv = self.header3d["CDELT1"]  # km/s

        self.boundary = get_bounds(l, b, self.name)
        co_mask, _, N = mask_dame_wco(co, co_raw, level=3, noise=self.noise)
        self.co_mask = co_mask
        self.N = N

        # co_mask = noise_mask
        self.pixelscale = np.abs(self.header["CDELT2"])
        self.pixel_pc = np.tan(self.pixelscale * np.pi / 180) * self.distance

        self.df = None

    def analysis(
        self,
        ak_limit=0.1,
        co_limit=None,
        snr=3,
        wco_limit=None,
        save=True,
        ak_scale=183,
        alpha_co=4.383,
    ):
        if wco_limit is None:
            co_mask = mask_dame_wco(co, co_raw, level=snr, noise=self.noise)[0]
        else:
            co_mask = wco > wco_limit
        df, akmask, comask = analysis(
            self.planck,
            self.wco,
            self.boundary,
            co_mask,
            pixel_scale=self.pixelscale,
            dist=self.distance,
            lim=ak_limit,
            ak_scale=ak_scale,
            alpha_co=alpha_co,
            name=self.name,
        )
        if save:
            self.df = df

        return df, akmask, comask

    def mass_ak(self, ak_lim=0.1, ak_scale=183):
        if ju.check_iterable(ak_lim):
            return list(zip(*[self.mass_ak(i, ak_scale=ak_scale) for i in ak_lim]))
        mask = self.boundary
        mask = mask & (self.planck > ak_lim)
        m = mass(self.planck, mask, ak_scale, self.pixel_pc)
        r = radius(mask, self.pixel_pc)
        return m, r

    def mass_co(
        self, snr_lim=None, wco_lim=None, alpha_co=4.383,
    ):

        mask = self.boundary
        if snr_lim is not None:
            if ju.check_iterable(snr_lim):
                f = lambda x: self.mass_co(snr_lim=x, alpha_co=alpha_co)
                return list(zip(*[f(i) for i in snr_lim]))

            noise = np.sqrt(self.N) * self.noise * self.dv
            mask = mask & ((wco > snr_lim * noise) & (self.N > 0))

        elif wco_lim is not None:
            if ju.check_iterable(wco_lim):
                f = lambda x: self.mass_co(wco_lim=x, alpha_co=alpha_co)
                return list(zip(*[f(i) for i in wco_lim]))

            mask = mask & (wco > wco_lim)
        else:
            snr_lim = 3
            noise = np.sqrt(self.N) * self.noise * self.dv
            mask = mask & ((wco > snr_lim * noise) & (self.N > 0))

        m = mass(self.wco, mask, alpha_co, self.pixel_pc)
        r = radius(mask, self.pixel_pc)
        return m, r

    def mass(
        self,
        ak=None,
        co=None,
        ak_limit=None,
        snr_limit=None,
        wco_limit=None,
        ak_scale=183,
        alpha_co=4.383,
    ):
        co_mask = self.boundary
        if ak_limit is not None:
            pass
            # mask = mask & (self.planck > ak_limit)
        else:
            ak_limit = 0.1
        if snr_limit is not None:
            noise = np.sqrt(self.N) * self.noise * self.dv
            mask = mask & ((wco > snr_limit * noise) & (self.N > 0))
        if wco_limit is not None:
            mask = mask & (wco > wco_limit)
        df, *_ = analysis(
            self.planck,
            self.wco,
            self.boundary,
            co_mask,
            pixel_scale=self.pixelscale,
            dist=self.distance,
            lim=ak_limit,
            ak_scale=ak_scale,
            alpha_co=alpha_co,
            name=self.name,
        )
        return df



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
    TA = (l <= 180) & (l >= 165) & (b <= -10) & (b >= -20)

    CA1 = (l >= 155) & (l <= 169) & (b >= -10) & (b <= -5)
    CA2 = (l > 155) & (l < 162) & (b > -15) & (b < -10)
    CA = CA1 | CA2
    # remove L1434
    newCA = CA & ~(b < -13)
    # remove 200 pc foreground
    newCA = newCA & ~((l >= 167.5) & (b < -8.75))
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
    elif obj == "MonOB":
        return monob.astype(bool)
    elif obj == "OriA":
        return OriA.astype(bool)
    elif obj == "OriB":
        return OriB.astype(bool)
    elif obj == "MonR2":
        return MonR2.astype(bool)
    elif obj == "RCrA":
        return RCrA.astype(bool)
    else:
        return None


def getlb(header, amap, origin=0):
    ys, xs = np.indices(amap.shape)
    return WCS(header).all_pix2world(xs, ys, origin)


def getv(header3d):
    x = np.arange(header3d["NAXIS1"])
    v = header3d["CRVAL1"] + x * header3d["CDELT1"]
    return v


def dame_bad(arr, bad_val=2 ** (-20)):
    return (arr == bad_val) | np.isnan(arr) | (arr == -32768) | (np.log2(arr) < -10)


def mask_dame_wco(co, co_raw, noise=None, level=3):
    bad = dame_bad(co)
    if noise is None:
        noise = np.nanstd(co_raw[bad])
    N = np.nansum(~bad, axis=-1)
    sqrtN = np.sqrt(N)
    wco = np.nansum(co, axis=-1)
    return ((wco / (sqrtN * noise)) > level) & (N > 0), noise, N


def channel_maps(chmap, v, start=0, stop=-1, step=1):
    if stop == -1:
        stop = chmap.shape[-1]
    chan = np.arange(start, stop + step, step)
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
            ax.imshow(summed, vmin=-0.2, vmax=2)
            ju.annotate(
                f"{v[chan[i]]:0.1f}:{v[chan[i+1]]:0.1f}", 0.8, 0.9, ax=ax, fontsize=25
            )
            plt.xlim(580, 630)
            plt.ylim(15, 45)
        else:
            plt.delaxes(ax)

    return fig, axs

