from scipy.ndimage import distance_transform_edt, gaussian_filter,binary_opening,binary_dilation
from scipy.stats import sigmaclip, norm
from astropy.stats import sigma_clipped_stats, mad_std
import numpy as np
import matplotlib.pyplot as plt

# import numpy as np
def interp_cube(arr):
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

    T = arr.copy()

    bad = np.isnan(T).all(axis=2) # bad pixels
    contiguous_bad  = binary_opening(bad) # contiguous bad
    contiguous_good = binary_opening(~bad) # contiguous good

    g = bad & (~contiguous_bad) & (~contiguous_good) # bad, but not contigous bad, and not contiguous good
    i,j = np.where(g)

    # good but not contiguous
    T[(~bad) & (~contiguous_good)] = np.nan

    for i,j in zip(i,j):
        T[i,j,:] = dame_int(arr,i,j)

    return T


def find_emission_free_region_rms(T, low=4, high=4, axis=None, cenfunc='mean',stdfunc='std'):
    if axis is None:
        #emission_free_T, lower, upper = sigmaclip(T[np.isfinite(T)], low=low, high=high)
        # sigma_clipped_stats returns mean, median, stddev
        _,_,emission_free_T = sigma_clipped_stats(T[np.isfinite(T)],sigma_lower = low, sigma_upper=high,cenfunc=cenfunc,stdfunc=stdfunc)
        return emission_free_T
    else:
        shape = list(T.shape)
        shape.pop(axis)
        emission_free_T = np.zeros(shape)
        func = lambda x: find_emission_free_region_rms(x,low=low, high=high, cenfunc=cenfunc, stdfunc=stdfunc)
        # for i in range(shape[0]):
        #     for j in range(shape[1]):
                # use recursion to fill emission free region
        emission_free_T = np.array([[func(T[i,j]) for j in range(shape[1])] for i in range(shape[0])])
        return emission_free_T


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

def mask_with_nan(arr,mask):
    """ mask array with nan values"""
    masked_arr = np.copy(arr)
    masked_arr[mask] = np.nan
    return masked_arr


def chauvenet(data,max_step_percent = .05, nbad_divisor=5,verbose=False, debug=False, return_mask=True,return_median=False,return_steps=False,return_mean=False):
    ## assume noise is centered on zero
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.nan,np.nan,np.nan
    srt = np.argsort(data[finite])
    f_data = data[finite][srt]



    N = len(f_data)
    if debug:
        print(f'Originally we have {N} pixels')

    bad, i = [True], 0
    ii = 0
    steps = 0
    nsingles = 0
    ii_steps = []
    while (i<(N-1)) & np.any(bad):
        steps += 1
        i+=ii
        if debug:
            print(i,N-i,f_data.shape,type(N-i))
        median = np.median(f_data[:N-i])
        stddev = mad_std(f_data[:N-i]-median)

        if debug:
            print(f'std:{stddev:0.3g}  med: {median:0.3g}  i: {i}  ii: {ii}')


        P = 1/(2*(N-i))

        bad = norm.sf(f_data[:N-i]-median,scale=stddev) < P

        ii = int(np.sum(bad))
        # ii = int(max(1,np.sum(bad)/nbad_divisor) + 0.5)
        if  ii / (N-i) > max_step_percent:
            # print('warning: small steps')
            ii = int(max_step_percent * (N-i) / nbad_divisor + 0.5)
            nsingles += 1
        ii_steps.append(ii)


    # stddev = mad_std(f_data[:N-i]-median)
    if verbose:
        print(f'chauvenet rejected {i} out of {len(f_data)} after {steps} steps ({nsingles} single steps)')
        print(f'Initial stddev: {np.std(f_data):10.3g}   Final stddev: {stddev:10.3g}')


    out = (stddev,)

    if return_mask:
        mask = np.zeros(data.shape)
        iy,ix,iz = np.indices(data.shape)
        iy = iy[finite][srt[:N-i]]
        ix = ix[finite][srt[:N-i]]
        iz = iz[finite][srt[:N-i]]
        mask[iy,ix,iz] = 1

        out = out + (mask.astype(bool),)

    if return_median:
        out = out + (median,)

    if return_steps:
        out = out + (ii_steps,)

    if return_mean:
        out = np.mean(f_data[:N-i])

    return out




def find_robust_rms(Ts, axis=None, sigma_clip = None, clip_n = 5, max_iter = 50, robust_clip = None ,max_step_percent=0.05, nbad_divisor=5, debug=False):
    """
    find robust rms of a array of Ts

    Parameters
    ----------
    Ts : array_like

    clip_n : int
        number of sigma to clip

    axis : int
        axis to compute rms

    max_step_percent : float
        maximum percentage of data to reject

    nbad_divisor : int
        number of times to divide the number of bad pixels

    debug : bool
        print debug info

    Returns
    -------
    rms : float
        robust rms of Ts



    """
    use_chauvenet = False

    if sigma_clip is not None:
        if not sigma_clip:
            use_chauvenet = True
    elif robust_clip is not None:
        if robust_clip:
            use_chauvenet = True

    if debug:
        print('*** find_robust_rms ***')
        print('use chauv.',use_chauvenet)

    if use_chauvenet:
        new_rms, mask = chauvenet(Ts,max_step_percent=max_step_percent,nbad_divisor=nbad_divisor,verbose=debug,debug=debug,return_mask=True)
        if debug:
            print('RMS(c):',new_rms)
        return new_rms, mask
    else:
        i = 0
        new_rms = np.nanstd(Ts)
        old_rms = 0
        while (old_rms != new_rms) & (i < max_iter):
            old_rms = new_rms
            new_rms = np.sqrt(np.nanmean((Ts[Ts < (clip_n * new_rms)] ** 2)))
            if debug:
                print('RMS:',new_rms)
        return new_rms, Ts < (clip_n * new_rms)

def dilated_mask(mask, ni,nj,nk, test=False):

    mask = mask.astype(bool)

    dilated = np.zeros(mask.shape)

    # s = np.ones((2*ni+1,1,1))
    # dilated = np.logical_or(dilated, binary_dilation(mask,s))

    # s = np.ones((1,2*nj+1,1))
    # dilated = np.logical_or(dilated, binary_dilation(mask,s))

    # s = np.ones((1,1,2*nk+1))
    # dilated = np.logical_or(dilated, binary_dilation(mask,s))

    s = np.ones((ni, nj, nk))
    dilated = np.logical_or(dilated, binary_dilation(mask,s))

    if test:
        dilated = dilated.astype(int)
        dilated[mask] = 2

    return dilated







def dame_moment_masking(T, ds=1 / 8, dv=0.65, fwhm_s=1 / 4, fwhm_v=2.5, \
                        nneigh=None, clip_n=5, truncate=4, follow_dame = True, rms_map = True, \
                        specax = 2, verbose=True,debug=False,clean_mask=False, robust_rms = True,\
                        mode = "constant", expand_mode = 'dilate'):
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
    debug : bool, optional
        print out debug info, by default False
    clean_mask : bool, optional
        clean mask, by default False
    follow_dame : bool, optional
        Follow the prescription of Dame (2011), by default True
        Note this overrides all other parameters and is the default
        but is not the best. The best options (imo) are the default
        inputs when follow_dame=False is set.
    rms_map : bool, optional
        return a rms map, by default True
    specax : int, optional
        spectral axis, by default 2
    mode : str, optional
        mode for interpolation, by default "constant"
    expand_mode : str, optional
        mode for expanding the mask, by default 'dilate'
        'dilate' : expand the mask by dilating the mask
        'ellipse' : expand mask using the distance transform

    Returns
    -------
    Tm, Ts, Mx, Tc, rms
        Tm is the masked array
        Ts is the smoothed array
        Mx is mask (dialated)
        Tc is the clipping level
        rms is the rms or rms map of the input array
    """

    if follow_dame:
        # do what dame 2011 does
        if verbose:
            print('following dame 2011 prescription')
            print('\tSetting default values [overrides user input]')
            print('\tfwhm_s = 2 * ds')
            print('\tfwhm_v = 2.5 km/s')
            print('\tclip_n = 5')
            print('\tnneigh = (3,3,3)')
        nneigh = (3,3,3)
        rms_map = False
        truncate = 2.5
        fwhm_s = 2 * ds
        fwhm_v = 2.5
        clean_mask = False
        clip_n = 5
        robust_rms = False
        expand_mode='dilate'


    if np.nanmedian(T)==0:
        T = mask_with_nan(T, T==0)

    # define fwhm vector
    fwhm = (fwhm_s / ds, fwhm_s / ds, fwhm_v / dv)
    if verbose:
        print(f"FWHM vector: dy: {fwhm[0]:5.2f}  dx: {fwhm[1]:5.2f}  dv: {fwhm[2]:5.2f}")



    # Generate smoothed array
    Ts = nan_gaussian_filter(T, np.divide(fwhm, 2.3548), preserve_nan=True, truncate=truncate)

    if debug:
        print('T len(nan) ',np.sum(np.isnan(T)))
        print('Ts len(nan)',np.sum(np.isnan(Ts)))

    # find robust noise
    smooth_rms, mask = find_robust_rms(Ts, clip_n=clip_n, robust_clip = robust_rms, debug=debug)
    if verbose:
            print('RMS(Ts):',smooth_rms)
    if follow_dame:
        if verbose:
            print('with follow_dame=True, we dont generate a rms map')
        smooth_rms = np.sqrt(np.nanmean(mask_with_nan(Ts, ~mask)**2))
    else:
        if rms_map:
            smooth_rms = mad_std(mask_with_nan(Ts, ~mask),ignore_nan=True,axis=specax)
        else:
            smooth_rms = mad_std(mask_with_nan(Ts, ~mask),ignore_nan=True)


    #  Tc can always be 3D
    Tc = np.atleast_3d(clip_n * smooth_rms)

    # Make Mask
    M = Ts > Tc

    if verbose:
        print(f"Clipping (smooth) RMS: {np.nanmean(np.atleast_1d(smooth_rms)):0.5f}")

    # get size of clipping area (automatically, if not set)
    if nneigh is None:
        size = lambda y: np.asarray(tuple(map(lambda x: 2 * int(x / 2 + 0.5) + 1, y)))
        nneigh = size(fwhm)
    else:
        if not hasattr(nneigh, "__iter__"):
            nneigh = np.asarray((nneigh, nneigh, nneigh))

    if verbose:
        print("Neighbors", nneigh)

    # expand
    if clean_mask:
        # clean mask using a binary erosion and dilation
        if verbose:
            print('cleaning mask')
        Mc = binary_opening(M, iterations=1)
    else:
        Mc = M.copy()

    if expand_mode[0].lower() == 'd':
        M = dilated_mask(Mc, *nneigh)
        M[np.isnan(T)] = False

    else: # expand_mode[0].lower() == 'e':

        ### get distances to nearest True element and element location
        ### sampling keyword is set so that nneigh pixels = 1
        npix = np.array(fwhm)/2
        if verbose:
            print('npix',npix)
        edt, inds = distance_transform_edt(~Mc, sampling=1/npix, return_indices=True)
        expand = edt <= 1
        # # add one cuz distance needs to be plus 1
        # dy, dx, dv = (np.abs(np.indices(T.shape) - inds) + 1+0.5).astype(int)
        # expand = (dy**2 / nneigh[0]**2 + dx**2 / nneigh[1]**2 + dv**2 / nneigh[2]**2) < 1

        # dspace = np.sqrt(dy ** 2 + dx ** 2)
        # # expand = (dy < nneigh[0]) & (dx < nneigh[1]) & (dv <nneigh[2])
        # expand = (dspace < np.sqrt(nneigh[0] ** 2 + nneigh[1] ** 2)) & (dv < nneigh[2])

        M[expand & (~np.isnan(T))] = True




    if rms_map:
        new_rms = mad_std(mask_with_nan(T, M),ignore_nan=True,axis=specax)
    else:
        new_rms = mad_std(mask_with_nan(T, M),ignore_nan=True)

    if verbose:
        print(f"Clipping (T) RMS: {np.nanmean(np.atleast_1d(new_rms)):0.5f}")

    if np.isnan(np.nanmean(np.atleast_1d(new_rms))):
        print('WARNING: nan in rms map')
        import pdb; pdb.set_trace()
    # return masked array
    Tm = mask_with_nan(T, ~M)

    return Tm, Ts, M, Tc, new_rms

    Tm = T * M
    # Tm[np.logical_not(M)] = np.nan



    return Tm, Ts, M, Tc, new_rms



class MaskedCube(object):
    """
    Class to perform moment masking on a cube.
    """
    def __init__(self, data, ds=1 / 8, dv=0.65, specax = None,):
        """
        Initialize the masked cube.

        Parameters
        ----------
        data : array_like
            The data cube to be masked.
        """
        self._data = data
        self._mask = None
        self._rms = None
        self._ds = ds
        self._dv = dv
        self._specax = specax

        self.set_defaults()

    def set_defaults(self, **kwargs):
        """
        #truncate = 5, fwhm_s = 2*ds, fwhm_v = 2.5,
                    #  clip_n = 5, robust_rms = True, rms_map = True,
                    #  clean_mask = False, follow_dame = False,
                    #  nneigh = None, verbose = False, debug = False)
        """

        self._truncate = kwargs.get('truncate', 5)
        self._fwhm_s = kwargs.get('fwhm_s', 2. * self._ds)
        self._fwhm_v = kwargs.get('fwhm_v', 2.5)
        self._clip_n = kwargs.get('clip_n', 5)
        self._robust_rms = kwargs.get('robust_rms', True)
        self._rms_map = kwargs.get('rms_map', False)
        self._clean_mask = kwargs.get('clean_mask', False)
        self._expand_mode = kwargs.get('expand_mode', 'd')
        self._follow_dame = kwargs.get('follow_dame', False)
        self._nneigh = kwargs.get('nneigh', None) # determine from fwhm_s/ds and fwhm_v/dv
        self._verbose = kwargs.get('verbose', False)
        self._debug = kwargs.get('debug', False)

        if self._follow_dame:
            self.set_dame(self._follow_dame)

    def set_dame(self, dame=True, **kwargs):
        """
        Set the DAME flag.

        Parameters
        ----------
        dame : bool
            If True, use DAME masking.
        """
        self._follow_dame = dame
        if dame:
            self._truncate = kwargs.get('truncate', 2.5)
            self._clip_n = kwargs.get('clip_n', 5)
            self._robust_rms = kwargs.get('robust_rms', False)
            self._rms_map = kwargs.get('rms_map', False)
            self._expand_mode = kwargs.get('expand_mode', 'd')
            self._nneigh = kwargs.get('nneigh', (3,3,3))
            self._fwhm_s = kwargs.get('fwhm_s', 2 * self._ds)
            self._fwhm_v = kwargs.get('fwhm_v', 2.5)

    def set_mask_params(self,
                        fwhm_s = None,
                        fwhm_v = None,
                        n_ds = None,
                        n_dv = None,):

        if fwhm_s is not None:
            self._fwhm_s = fwhm_s
        elif n_ds is not None:
            self._fwhm_s = n_ds * self._ds

        if fwhm_v is not None:
            self._fwhm_v = fwhm_v
        elif n_dv is not None:
            self._fwhm_v = n_dv * self._dv

