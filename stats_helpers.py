import numpy as np
import scipy.statistics as stats


def mavg(arr, n=2, axis=-1):
    """
    returns the moving average of an array.
    returned array is shorter by (n-1)
    applied along last axis by default
    """
    return np.mean(rolling_window(arr, n), axis=axis)


def weighted_generic_moment(x, k, w=None):
    x = np.asarray(x, dtype=np.float64)
    if w is not None:
        w = np.asarray(w, dtype=np.float64)
    else:
        w = np.ones_like(x)

    return np.sum(x ** k * w) / np.sum(w)


def weighted_mean(x, w=1.):
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

    If we want to think about this in terms of Mahalanobis (or n-sigma) distance
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


# 