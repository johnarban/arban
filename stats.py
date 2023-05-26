import numpy as np
from scipy import stats
from scipy import special


def minmax(arr, axis=None):
    return np.nanmin(arr, axis=axis), np.nanmax(arr, axis=axis)



def weighted_generic_moment(x, k, w=None):
    x = np.asarray(x, dtype=np.float64)
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
    else:
        w = np.ones_like(x)

    return np.sum(x ** k * w) / np.sum(w)


def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)


def weighted_std(x, w):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    SS = np.sum(w * (x - weighted_mean(x, w)) ** 2) / np.sum(w)
    # quantile(x, w, 0.5)
    return np.sqrt(SS)


def weighted_percentile(x, w, percentile, p=0):
    k = np.isfinite(x + w)
    clean_x = np.asarray(x[k], dtype=np.float64)
    clean_w = np.asarray(w[k], dtype=np.float64)
    srt = np.argsort(clean_x)
    sorted_w = clean_w[srt]
    sorted_x = clean_x[srt]
    Sn = np.cumsum(sorted_w)
    Pn = (Sn - 0.5 * sorted_w) / Sn[-1]
    return np.interp(np.asarray(percentile) / 100.0, Pn, sorted_x)


def weighted_median(x, w):
    return weighted_percentile(x, w, 50)


def weighted_mad(x, w, stddev=True):
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    if stddev:
        return 1.4826 * weighted_median(np.abs(x - weighted_median(x, w)), w)
    else:
        return weighted_median(np.abs(x - weighted_median(x, w)), w)



def rolling_window(arr, window):
    """[summary]
    Arguments:
        arr {[numpy.ndarray]} -- N-d numpy array
        window {[int]} -- length of window
    Returns:
        out -- array s.t. np.mean(arr,axis=-1) gives the running mean along rows (or -1 axis of a)
            out.shape = arr.shape[:-1] + (arr.shape[-1] - window + 1, window)
    """
    shape = arr.shape[:-1] + (
        arr.shape[-1] - window + 1,
        window,
    )  # the new shape (a.shape)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)



def mavg(arr, n=2, axis=-1):
    """
    returns the moving average of an array.
    returned array is shorter by (n-1)
    applied along last axis by default
    """
    return np.mean(rolling_window(arr, n), axis=axis)



def mgeo(arr, n=2, axis=-1):
    """rolling geometric mean

    Arguments:
        arr {no.array} -- array

    Keyword Arguments:
        n {int} -- window size (default: {2})
        axis {int} -- axis to roll over (default: {-1})

    Returns:
        [type] -- [description]
    """
    return stats.gmean(rolling_window(arr, n), axis=axis)



def pdf(values, bins=None, range=None):
    """
    ** Normalized differential area function. **
    (statistical) probability denisty function
    normalized so that the integral is 1
    and. The integral over a range is the
    probability of the value is within
    that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    """
    if isinstance(bins, "__getitem__") and (range is None):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, x = np.histogram(values, bins=bins, range=range, density=False)
    # From the definition of Pr(x) = dF(x)/dx this
    # is the correct form. It returns the correct
    # probabilities when tested
    pdf = h / (np.sum(h, dtype=float) * np.diff(x))
    return pdf, bin_center(x)


def pdf2(values, bins=None, range=None):
    """
    N * PDF(x)
    The ~ PDF normalized so that
    the integral is equal to the
    total amount of a quantity.
    The integral over a range is the
    total amount within that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    """
    if hasattr(bins, "__getitem__") and (range is None):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    pdf, x = np.histogram(values, bins=bins, range=range, density=False)
    pdf = pdf.astype(float) / np.diff(x)
    return pdf, bin_center(x)


def edf(data, pdf=False):
    y = np.arange(len(data), dtype=float)
    x = np.sort(data).astype(float)
    return y, x


def cdf(values, bins):
    """
    CDF(x)
    (statistical) cumulative distribution function
    Integral on [-inf, b] is the fraction below b.
    CDF is invariant to binning.
    This assumes you are using the entire range in the binning.
    Returns array of size len(bins)
    Plot versus bins[:-1]
    """
    if hasattr(bins, "__getitem__"):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range, density=False)  # returns int

    # cumulative fraction below bin_k
    c = np.cumsum(h / np.sum(h, dtype=float))
    # append 0 to beginning because P(X < min(x)) = 0
    return np.append(0, c), bins


def cdf2(values, bins):
    """
    # # Exclusively for area_function which needs to be unnormalized
    (statistical) cumulative distribution function
    Value at b is total amount below b.
    CDF is invariante to binning

    Plot versus bins[:-1]
    Not normalized to 1
    """
    if hasattr(bins, "__getitem__"):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    h, bins = np.histogram(values, bins=bins, range=range, density=False)
    c = np.cumsum(h).astype(float)
    return np.append(0.0, c), bins


def area_function(extmap, bins, scale=1):
    """
    Complimentary CDF for cdf2 (not normalized to 1)
    Value at b is total amount above b.
    """
    c, bins = cdf2(extmap, bins)
    return scale * (c.max() - c), bins


def diff_area_function(extmap, bins, scale=1):
    """
    See pdf2
    """
    s, bins = area_function(extmap, bins)
    dsdx = -np.diff(s) / np.diff(bins)
    return dsdx * scale, bin_center(bins)


def log_diff_area_function(extmap, bins):
    """
    See pdf2
    """
    s, bins = diff_area_function(extmap, bins)
    g = s > 0
    dlnsdlnx = np.diff(np.log(s[g])) / np.diff(np.log(bins[g]))
    return dlnsdlnx, bin_center(bins[g])


def mass_function(values, bins, scale=1, aktomassd=183):
    """
    M(>Ak), mass weighted complimentary cdf
    """
    if hasattr(bins, "__getitem__"):
        range = (np.nanmin(bins), np.nanmax(bins))
    else:
        range = None

    if scale != 1:
        aktomassd = scale

    h, bins = np.histogram(
        values,
        bins=bins,
        range=range,
        density=False,
        weights=values * aktomassd * scale,
    )
    c = np.cumsum(h).astype(float)
    return c.max() - c, bins



def ortho_dist(x, y, m, b):
    """
    get the orthogonal distance
    from a point to a line
    """
    ortho_dist = (y - m * x - b) / np.sqrt(1 + m ** 2)
    return ortho_dist


def mad(X, stddev=True, axis=None):
    if stddev:
        return 1.4826 * np.nanmedian(np.abs(X - np.nanmedian(X, axis=axis)), axis=axis)
    else:
        return np.nanmedian(np.abs(X - np.nanmedian(X, axis=axis)), axis=axis)


def mean_mad(X, stddev=True, axis=None):
    if stddev:
        return 1.4826 * np.nanmedian(np.abs(X - np.nanmeam(X, axis=axis)), axis=axis)
    else:
        return np.nanmedian(np.abs(X - np.nanmean(X, axis=axis)), axis=axis)


def rms(X, axis=None):
    return np.sqrt(np.nanmean(X ** 2, axis=axis))


def standardize(X, remove_mean=True, remove_std=True):
    if remove_mean:
        mean = np.nanmean(X)
    else:
        mean = 0
    if remove_std:
        std = np.nanstd(X)
    else:
        std = 1

    return (X - mean) / std



def pdf_pareto(t, a, k, xmax=None):
    """PDF of Pareto distribution

    Parameters
    ----------
    t : input
        array
    a : power-law power (a = alpha-1 from real Pareto)
        array
    k : minimum value for power law
        array
    xmax : max value for, optional, by default None

    Returns
    -------
    PDF(t|a,k,xmax)
        numpy array
    """
    if xmax is None:
        out = ((a - 1) / k) * (t / k) ** (-a)
        out[(t < k)] = 0
        return out
    else:
        out = ((a - 1) / (k ** (1 - a) - xmax ** (1 - a))) * t ** (-a)
        out[(t <= k) | (t > xmax)] = 0
        return out


def cdf_pareto(t, a, k, xmax=None):
    """CDF of Pareto distribution

    Parameters
    ----------
    t : input
        array
    a : power-law power (a = alpha-1 from real Pareto)
        array
    k : minimum value for power law
        array
    xmax : max value for, optional, by default None

    Returns
    -------
    CDF(t|a,k,xmax)
        numpy array
    """
    if xmax is None:
        out = 1 - (k / t) ** (a - 1)
        out[t < k] = 0
        return out
    else:
        out = (1 - (t / k) ** (1 - a)) / (1 - (xmax / k) ** (1 - a))
        out[t <= k] = 0
        out[t > xmax] = 1
        return out

from scipy.spatial.distance import cdist
def mahalanobis(X,X2=None):
    """mahalanobis distance for data
    X = np.array([x1,x2,x3,...])


    Parameters
    ----------
    X : np.array (M x N)
        M x N array, with M varialbes,
            and N observations.
            print(X) should look like
            # [[x1, x2, x3, x4...xn],
            #  [y1, y2, y3, y4...yn].
            #  [z1, z2, z3, z4...zn],
            #   ..]
            # as if X = np.array([x, y, z, ...])

    Returns
    -------
    md: np.array
        the square of maholanobis distance
        it follows a chi2 distribution for normally
        distributed data
    """
    # let scipy do all the lifting
    # but this is a nice way anyways
    # C = np.cov(X.T)
    # P, D, T = eigen_decomp(C)
    # mu = np.mean(X, axis=1)
    # X = (X - mu)
    # wX = X @ np.linalg.inv(T.T) #whitened data
    # md = np.linalg.norm(wX, axis=1)**2  #norm spannign [xi,yi,zi]
    # #wXT = np.linalg.inv(T) @ X.T
    # #md = wX @ wX.T
    # #md = np.sqrt(md.diagonal())
    # #md is distributed as chi2 with d.o.f. = # independent axes
    if X2 is None:
        return cdist(X,np.atleast_2d(X.mean(axis=0)),metric='mahalanobis')[:,0]**2
    else:
        C = np.cov(X.T)
        P, D, T = eigen_decomp(C)
        mu = np.mean(X2, axis=1)
        wX = (X2-mu) @ np.linalg.inv(T.T)
        md = np.linalg.norm(wX, axis=1)** 2
        return md




def sigconf1d(n):
    """
    calculate the percentile corresponding to n*sigma
    for a 1D gaussian
    """
    cdf = (1 / 2.0) * (1 + special.erf(n / np.sqrt(2)))
    return (1 - cdf) * 100, 100 * cdf  # , 100 * special.erf(n / np.sqrt(2))

def nsigma(dim=1, n=1,return_interval=False):
    """Generalized n-sigma relation

    Parameters
    ----------
    dim : float, optional
        dimensionality, by default 1
    n : float, optional
        N-sigma, by default 1

    Returns
    -------
    float
        the percential/100 corresponding the given sigma

    References:
        https://math.stackexchange.com/a/3668447
        https://mathworld.wolfram.com/RegularizedGammaFunction.html

    The generalized N-sigma relation for M dimensions is given
    by the Regularized Lower Incomplete Gamma Function -
        P(a,z) = γ(a,z)/Γ(a), where γ(a,z) is the lower incomplete gamma function
    The Incomplete Gamma Function is defined
        $\Gamma(a,z0,z1) = \int_z0^z1 t^{a-1} e^{-t} dt$

    For 1D: $Erf(n/sqrt(2)) = \Gamma(1,0,n^2/2)/\Gamma(1)$ gives the Percentile for n-sigma
    For 2D: 1 - exp(-m^2 /2) gives the Percentile for n-sigma
    P(m/2,n^2 / 2) generalizes this to m dimensions
    We need the regularized lower incomplete gamma, which is Gamma(a,z0,z1)/Gamma(a,0,inf)
    this is the incomp. reg. gamma func P(a,z) in

    If we want to think about this in terms of Mahalanobis distance
    Then, well, the Mahalanobis distance is distributed like
    a chi2-distribution with k = m degrees of freedom (assuming the
    eigenvectors are of the covariance matrix are all independent)
    So this covariance is also written as the
    SurvivalFunction(χ^2(k=m),x=n**2) where n = mahalanobis distance
    this would be written stats.chi2(m).cdf(n**2), but this is half
    the speed of using special.gammainc
    @astrojthe3
    """
    if return_interval:
        p = special.gammainc(dim/2,n**2 /2)/2
        return (1 - p)/2, (1 + p)/2
    return special.gammainc(dim/2,n**2 /2)
