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



##### from sklearn.linear_model import Lasso,LassoLarsIC
# generate data
from sklearn.linear_model import Lasso,LassoLarsIC

thresholds = np.percentile(x, np.linspace(0, 1, k+2)[1:-1]*100)
m,n = np.log10([11,30])
thresholds = np.linspace(m,n,k)

def get_turnovers(x, y, thresholds, k = 100, use_aic = False,
                  plot_line=False, plot_features=False,plot_data=False,
                  ax=None,unlogx=False,unlogy=False,unlog=False,zorder=100):

    g = np.isfinite(1/x + 1/y)
    x = x[g]
    y = y[g]

    basis = np.hstack([x[:, np.newaxis],  np.maximum(0,  np.column_stack([x]*k)-thresholds)])
    if use_aic:
        model = LassoLarsIC().fit(basis, y)
    else:
        model = Lasso(0.0003).fit(basis, y)

    if ax is None:
        ax = plt.gca()

    fx = lambda z: z
    fy = lambda z: z

    if unlogx:
        fx = lambda z: 10**z
        fy = lambda z: z
    if unlogy:
        fx = lambda z: z
        fy = lambda z: 10**z
    if unlog:
        fx = lambda z: 10**z
        fy = lambda z: 10**z




    if plot_data:
        ax.plot(fx(x),fx(y),'k.',zorder=zorder)
    if plot_line:
        ax.plot(fx(x), fy(model.predict(basis)),'.', color='b',zorder=zorder)

    if plot_features:
        for th in thresholds[~np.isclose(model.coef_[1:],0)]:
            ax.axvline(fx(th),0,1,color='r',alpha=0.5,zorder=zorder)

    return model, thresholds



def pwl_w2brks(x, x1, x2, k1, k2, k3, b1):
    c1 = x < x1
    c2 = (x>= x1) & (x<x2)
    c3 = x>=x2

    f1 = lambda x: x * k1 + b1
    b2 = (k1-k2) * x1 + b1
    f2 = lambda x: x * k2 + b2
    b3 = (k2 - k3) * x2 + b2
    f3 = lambda x: x * k3 + b3
    return np.piecewise(x, [c1,c2,c3], [f1,f2,f3])


def radec2vec(ra,dec):
    """Convert RA/Dec angle (in radians) to a vector"""
    v1 = np.cos(dec)*np.cos(ra)
    v2 = np.cos(dec)*np.sin(ra)
    v3 = np.sin(dec)
    return np.array([v1, v2, v3])


def vec2radec(v):
    """Convert a vector to Ra/Dec (in radians)"""
    el=np.arcsin(v[2]/np.sqrt(v[0]**2+v[1]**2+v[2]**2))
    ccc=v[0]**2+v[1]**2
    if (v[1] > 0):
        s=1
    else:
        s=-1
    if (ccc != 0):
        az=(abs(np.arccos(v[0]/np.sqrt(v[0]**2+v[1]**2)))*s+(4*np.pi)) % (2*np.pi)
    else:
        az=0
    return np.array([az,el])