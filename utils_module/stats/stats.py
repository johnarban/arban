

def sigconf1d(n):
    """
    calculate the percentile corresponding to n*sigma
    for a 1D gaussian
    """
    cdf = (1 / 2.0) * (1 + special.erf(n / np.sqrt(2)))
    return (1 - cdf) * 100, 100 * cdf #, 100 * special.erf(n / np.sqrt(2))



def minmax(arr,axis=None):
    return np.nanmin(arr, axis=axis),np.nanmax(arr,axis=axis)


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


# weighted statistics
def weighted_std(x, w):

    SS = np.sum(w * (x - x.mean()) ** 2) / np.sum(w)
    quantile(x, w, 0.5)
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
    return np.interp(percentile/100., Pn, sorted_x)

def weighted_median(x,w):
    return weighted_percentile(x,w,50)

def weighted_mad(x, w, stddev=True):
    def median(arr, wei): return weighted_median(arr, wei)
    if stddev:
        return 1.4826 * median(np.abs(x - median(x, w)), w)
    else:
        return median(np.abs(x - median(x, w)), w)

def weighted_mean(x, w):
    return np.sum(x * w) / np.sum(w)



# error propogated vectorized functions
def err_div(x, y, ex, ey):
    """do division with error propogation

    Parameters
    ----------
    x : float or numpy array
        divisor (x/y)
    y : float or numpy array
        divident (x/y)
    ex : float or numpy array
        error in x
    ey : float or numpy array
        error in y

    Returns
    -------
    returns 2 items
        quotient, error in quotient
    """
    Q = x / y
    dQ = np.abs(Q) * np.sqrt((ex / x) ** 2 + (ey / y) ** 2)
    return Q, dQ


