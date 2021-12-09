from scipy.ndimage import distance_transform_edt, gaussian_filter,binary_opening
from scipy.stats import sigmaclip
import numpy as np


# import numpy as np


def find_emission_free_region_rms(T, low=4, high=4):
    emission_free_T, lower, upper = sigmaclip(T[np.isfinite(T)], low=low, high=high)
    return emission_free_T.std()


def nan_gaussian_filter(T, fwhm, mode="constant", cval=0, preserve_nan=True, **kwargs):
    """default parameters mimic
    convolve(x,kernels.Gaussian2DKernel(fwhm),preserve_nan=True,boundary='fill',fill_value=np.nan)
    fill_value = np.nan basically continues the interpolation beyond the boundary
    """
    V = T.copy()
    V[np.isnan(T)] = 0
    VV = gaussian_filter(V, fwhm, mode=mode, cval=cval, **kwargs)

    W = np.ones_like(T)
    W[np.isnan(T)] = 0
    WW = gaussian_filter(W, fwhm, mode=mode, cval=cval, **kwargs)

    Z = VV / WW

    if preserve_nan:
        Z[np.isnan(T)] = np.nan

    return Z


def dame_moment_masking(
    T,
    ds=1 / 8,
    dv=0.65,
    noise=0.26,
    fwhm_s=1 / 4,
    fwhm_v=2.5,
    clip_n=5,
    truncate=2.5,
    sigma_clip_high=None,
    nneigh=None,
    verbose=True,
):
    """
    dame_moment_masking python implementation of Dame (2011) moment masking method

    Parameters
    ----------
    T : numpy array
        this is the array to be masked
    ds : pixel scale, optional
        pixel scale in degrees, by default 1/8
    dv : float, optional
        velocity scale in km/s, by default 0.65
    noise : float, optional
        noise, by default 0.26
    fwhm_s : float, optional
        spatial FWHM of gaussian, by default 1/4
    fwhm_v : float, optional
        velocity FWHM of gaussian, by default 2.5
    clip_n : int, optional
        clipping level for smoothed spectra, by default 5
    truncate : float, optional
        tuncation of gaussian, by default 2.5
    sigma_clip_high : float, optional
        obsolete, by default None
    nneigh : int or tuple of ints, optional
        number if pixels to add, optional, by default None
    verbose : bool, optional
        print out useful output, by default True

    Returns
    -------
    Tm, Ts, Mx, Tc
        Tm is the masked array
        Ts is the smoothed array
        Mx is mask (dialated)
        Tc is the clipping level
    """
    # define fwhm vector
    fwhm = (fwhm_s / ds, fwhm_s / ds, fwhm_v / dv)
    if verbose:
        print(
            f"FWHM vector: dy: {fwhm[0]:5.2f}  dx: {fwhm[1]:5.2f}  dv: {fwhm[2]:5.2f}"
        )

    # define sigma for gaussian filter
    sigma = np.divide(fwhm, 2.3548)
    Ts = nan_gaussian_filter(T, sigma, preserve_nan=True, truncate=truncate)

    # find robust noise
    new_rms = np.nanstd(Ts)
    old_rms = 0
    i = 0
    while (old_rms != new_rms) & (i < 20):
        old_rms = new_rms
        new_rms = np.sqrt(np.mean((Ts[Ts < (clip_n * new_rms)] ** 2)))
        # i+=1
    # if i==20:
    #     new_rms = np.nanstd(stats.sigmaclip(Ts)[0])

    Tc = clip_n * new_rms

    # Make Mask
    M = np.full_like(T, False, dtype=bool)
    M[Ts > Tc] = True

    if verbose:
        print(f"Clipping RMS: {new_rms:0.5f}")

    # get size of clipping area (automatically)
    if nneigh is None:
        size = lambda y: tuple(map(lambda x: 2 * int(x / 2 + 0.5) + 1, y))
        nneigh = size(fwhm)
    else:
        if not hasattr(nneigh, "__iter__"):
            nneigh = (nneigh, nneigh, nneigh)

    if verbose:
        print("Neighbors", nneigh)

    # expand

    # ~nd.binary_opening(M)
    edt, inds = distance_transform_edt(~binary_opening(M), return_indices=True)
    # iy, ix, iv = np.indices(T.shape)
    # add one cuz distance needs to be plus 1
    dy, dx, dv = (np.abs(np.indices(T.shape) - inds) + 1).astype(int)
    dspace = np.sqrt(dy ** 2 + dx ** 2)
    # dv = np.abs(iv - inds[2])
    expand = (dspace < np.sqrt(nneigh[0] ** 2 + nneigh[1] ** 2)) & (dv < nneigh[2])
    # expand = (dy < nneigh[0]) & (dx < nneigh[1]) & (dv <nneigh[2])
    # expand = edt <= np.mean(np.array(nneigh)**2)**.5 #

    M[expand] = True

    Tm = T * M
    Tm[np.logical_not(M)] = np.nan

    return Tm, Ts, M, Tc
