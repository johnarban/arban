from utils import *


def _mavg(arr, n=2, mode="valid"):
    """
    returns the moving average of an array.
    returned array is shorter by (n-1)
    """
    # weigths = np.full((n,), 1 / n, dtype=float)
    if len(arr) > 400:
        return signal.fftconvolve(arr, [1.0 / float(n)] * n, mode=mode)
    else:
        return signal.convolve(arr, [1.0 / float(n)] * n, mode=mode)

def mgeo(arr, n=2, axis=-1):
    """
    Returns array of lenth len(arr) - (n-1)

    # # written by me
    # # slower for short loops
    # # faster for n ~ len(arr) and large arr
    a = []
    for i in xrange(len(arr)-(n-1)):
        a.append(stats.gmean(arr[i:n+i]))

    # # Original method# #
    # # written by me ... ~10x faster for short arrays
    b = np.array([np.roll(np.pad(arr,(0,n),mode='constant',constant_values=1),i)
              for i in xrange(n)])
    return np.product(b,axis=0)[n-1:-n]**(1./float(n))
    """
    # a = []
    # for i in range(len(arr) - (n - 1)):
    #    a.append(stats.gmean(arr[i:n + i]))

    return stats.gmean(rolling_window(arr, n), axis=axis)



