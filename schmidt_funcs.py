import numpy as np
from scipy import interpolate, ndimage, stats, signal, integrate, misc
from astropy.io import ascii, fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c
from IPython.display import display, Math, Markdown
import triangle
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from astropy.modeling.fitting import LevMarLSQFitter, SimplexLSQFitter
import emcee
import triangle
import pdb

######################
# make iPython print immediately
import sys
oldsysstdout = sys.stdout


class flushfile():

    def __init__(self, f):
        self.f = f

    def __getattr__(self, name):
        return object.__getattribute__(self.f, name)

    def write(self, x):
        self.f.write(x)
        self.f.flush()

    def flush(self):
        self.f.flush()
sys.stdout = flushfile(sys.stdout)


def comp(arr):
    '''
    returns the compressed version
    of the input array if it is a
    numpy MaskedArray
    '''
    try:
        return arr.compressed()
    except:
        return arr


def mavg(arr, n=2, mode='valid'):
    '''
    returns the moving average of an array.
    returned array is shorter by (n-1)
    '''
    if len(arr) > 400:
        return signal.fftconvolve(arr, [1. / float(n)] * n, mode=mode)
    else:
        return signal.convolve(arr, [1. / float(n)] * n, mode=mode)


def mgeo(arr, n=2):
    '''
    Returns array of lenth len(arr) - (n-1)

    ## written by me
    ## slower for short loops
    ## faster for n ~ len(arr) and large arr
    a = []
    for i in xrange(len(arr)-(n-1)):
        a.append(stats.gmean(arr[i:n+i]))

    ## Original method##
    ## written by me ... ~10x faster for short arrays
    b = np.array([np.roll(np.pad(arr,(0,n),mode='constant',constant_values=1),i)
              for i in xrange(n)])
    return np.product(b,axis=0)[n-1:-n]**(1./float(n))
    '''
    a = []
    for i in xrange(len(arr) - (n - 1)):
        a.append(stats.gmean(arr[i:n + i]))

    return np.array(a)

def avg(arr, n=2):
    '''
    NOT a general averaging function
    return bin centers (lin and log)
    '''

    if np.sum(np.diff(np.float16(np.diff(arr)))) is 0:
        return mavg(arr, n=n)
    else:
        return mgeo(arr, n=n)

def llspace(mx, mn, n, log=False, dex = None):
    '''
    get values evenly spaced in linear or log spaced
    mx, mn = max and min values
    n = number of values
    '''
    if log or (dex is not None):
        if dex is not None:
            n = int((np.log10(mx) - np.log10(mn))/dex)
        return np.logspace(np.log10(mx), np.log10(mn), n)
    else:
        return np.linspace(mx, mn, n)


def nametoradec(name):
    '''
    Get names formatted as
    hhmmss.ss+ddmmss to Decimal Degree
    '''
    if 'string' not in str(type(name)):
        rightascen = []
        declinatio = []
        for n in name:
            ra, de = n.split('+')
            ra = ra[0:2] + ':' + ra[2:4] + ':' + ra[4:6] + '.' + ra[6:8]
            de = de[0:2] + ':' + de[2:4] + ':' + de[4:6]
            coord = SkyCoord(ra, de, frame='icrs',
                             unit=('hourangle', 'degree'))
            rightascen.append(coord.ra.value)
            declinatio.append(coord.dec.value)
        return np.array(rightascen), np.array(declinatio)
    else:
        ra, de = name.split('+')
        ra = ra[0:2] + ':' + ra[2:4] + ':' + ra[4:6] + '.' + ra[6:8]
        de = de[0:2] + ':' + de[2:4] + ':' + de[4:6]
        coord = SkyCoord(ra, de, frame='icrs', unit=('hourangle', 'degree'))
        return np.array(coord.ra.value), np.array(coord.dec.value)


def get_ext(extmap, errmap, extwcs, ra, de):
    '''
    Get the extinction (errors) for a particular position or
    list of positions
    '''
    try:
        xp, yp = extwcs.all_world2pix(
            np.array([ra]).flatten(), np.array([de]).flatten(), 0)
    except:
        xp, yp = WCS(extwcs).all_world2pix(
            np.array([ra]).flatten(), np.array([de]).flatten(), 0)
    ext = []
    err = []
    for i in range(len(np.array(xp))):
        try:
            ext.append(extmap[yp[i], xp[i]])
            err.append(errmap[yp[i], xp[i]])
        except IndexError:
            ext.append(np.nan)
            err.append(np.nan)
    return np.array(ext), np.array(err)


def cdf(values, bins):
    '''
    (statistical) cumulative distribution function
    Integral on [-inf, b] is the fraction below b.
    CDF is invariant to binning.
    This assumes you are using the entire range in the binning.
    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    '''
    c = np.cumsum(pdf(values, bins) * np.diff(bins))

    return np.append(0,c), bins


def cdf(values, bins):
    '''
    (statistical) cumulative distribution function
    Value at b is total amount below b.
    CDF is invariante to binning

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    '''
    c = np.cumsum(np.histogram(values, bins=bins, density=False)[0])
    return np.append(0, c), bins


def area_function(extmap, bins):
    '''
    Complimentary CDF for cdf2
    Value at b is total amount above b.
    '''
    c = cdf2(extmap, bins)
    return c.max() - c, bins


def pdf(values, bins):
    '''
    (statistical) probability denisty function
    normalized so that the integral is 1
    and. The integral over a range is the
    probability of the value is within
    that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    '''
    pdf, x = np.histogram(values, bins=bins, density=False)
    pdf = pdf / (np.sum(pdf) * np.diff(bins))
    return pdf, avg(bins)


def pdf2(values, bins):
    '''
    The ~ PDF normalized so that
    the integral is equal to the
    total amount of a quantity.
    The integral over a range is the
    total amount within that range.

    Returns array of size len(bins)-1
    Plot versus bins[:-1]
    '''
    pdf, x = np.histogram(values, bins=bins, density=False)
    pdf = pdf / np.diff(bins)
    return pdf, avg(bins)


def diff_area_function(extmap, bins):
    '''
    See pdf2
    '''
    s = area_function(extmap, bins)
    dsdx = -np.diff(s) / np.diff(bins)

    return dsdx , avg(bins)


def hist(values, bins):
    '''
    really just a wrapper for numpy.histogram
    '''
    hist, x = np.histogram(values, bins=bins, density=False)

    return hist.astype(np.float), avg(bins)


def bootstrap(ext, err):
    return ext + err * np.random.randn()


def num_above(values, level):
    return np.sum((values >= level) & np.isfinite(values), dtype=np.float)


def num_below(values, level):
    return np.sum((values < level) & np.isfinite(values), dtype=np.float)


def surfd(object_vals, valuemap, bins,
          object_val_err=None, valuerr=None, scale=1.):
    '''
    call: surfd(object_vals, valuemap, bins,
                    object_val_err = None, valuerr = None, scale = 1.)
    '''
    valmap = comp(valmap)
    valuerr = comp(valuerr)
    if yso_err is not None:
        n = hist(bootstrap(object_vals, object_val_err), bins)
    else:
        n = hist(object_vals, bins)

    if errmap is not None:
        s = hist(bootstrap(valuemap, valuerr), bins) * scale
    else:
        s = hist(valuemap, bins) * scale

    return n / s


def alpha(y, x):
    a = np.array(list(set(np.nonzero(y)[0]) & set(np.nonzero(x)[0])))
    al = np.diff(np.log(y[a])) / np.diff(np.log(x[a]))
    return np.mean(al[np.isfinite(al)])


def Heaviside(x):
    return 0.5 * (np.sign(x) + 1.)


def schmidt_law(Ak, theta):
    beta, kappa = theta
    return kappa * (Ak ** beta)


def fit_lmfit_schmidt(x, y, yerr, init=None):
    @custom_model
    def model(x, Ak0=init[0], beta=init[1], kappa=init[2]):
        return np.log10(schmidt_law(x, (Ak0, beta, kappa)))

    m_init = model()
    fit = LevMarLSQFitter()
    m = fit(m_init, x, np.log10(y), weights=1. / (yerr / y)**2, maxiter=1000)

    return m


def emcee_schmidt(x, y, yerr, pos=[2., 1.], pose=[
                  2., 1.], nwalkers=None, nsteps=None):
    def model(x, theta):
        '''
        theta = (Ak0, beta, kappa)
        '''
        return np.log(schmidt_law(x, theta))

    def lnlike(theta, x, y, yerr):
        mod = model(x, theta)
        #inv_sigma2 = 1/yerr**2
        mu = yerr**2
        x = np.abs(y - mod)
        logL = np.sum(x * np.log(mu)) - np.sum(mu) - np.sum(misc.factorial(x))
        return logL
        # return -0.5*(np.sum((y-mod)**2 * inv_sigma2))

    def lnprior(theta):
        beta, kappa = theta
        c1 = 0 <= beta
        c2 = 0 < kappa
        #c3 = 0. <= Ak0 <= 1.
        if c1 and c2:  # and  c3:
            return 0.0
        return -np.inf

    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y, yerr)

    ndim, nwalkers = len(pos), nwalkers

    pos = [np.array(pos) + np.array(pose) * 2 *
           (0.5 - np.random.rand(ndim)) for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob, args=(x, np.log(y), yerr / y))

    sampler.run_mcmc(pos, nsteps)

    # Get input values
    #x, y, yerr = sampler.args
    samples = sampler.chain[:, burnin:, :].reshape((-1, sampler.dim))

    ## Print out final values ##
    theta_mcmc = np.percentile(samples, [16, 50, 84], axis=0).T

    print sampler.acor

    for i, item in enumerate(theta_mcmc):
        #j=[r'A_{K,0}',  r'\beta', r'\kappa']
        j = [r'\beta', r'\kappa']
        display(Math(j[i] + ' = ' +
                     r'{:.2f}'.format(item[1]) +
                     r'~~(+{:.2f}'.format(item[2] - item[1]) +
                     r', ' +
                     r'-{:.2f})'.format(item[1] - item[0])))

    return sampler, np.median(samples, axis=0), np.std(samples, axis=0)


def schmidt_results_plots(sampler, model, x, y, yerr, burnin=200,
                          akmap=None, bins=None, scale=None, triangle_plot=True):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    try:
        mpl.style.use('john')
    except:
        None
    # Get input values
    #x, y, yerr = sampler.args
    samples = sampler.chain[:, burnin:, :].reshape((-1, sampler.dim))

    ## Print out final values ##
    theta_mcmc = np.percentile(samples, [16, 50, 84], axis=0).T

    for i, item in enumerate(theta_mcmc):
        #j=[r'A_{K,0}',  r'\beta', r'\kappa']
        j = [r'\beta', r'\kappa']
        display(Math(j[i] + ' = ' +
                     r'{:.2f}'.format(item[1]) +
                     r'~~(+{:.2f}'.format(item[2] - item[1]) +
                     r', ' +
                     r'-{:.2f})'.format(item[1] - item[0])))

    # Plot corner plot
    if triangle_plot is True:
        fig = triangle.corner(samples, labels=['beta', 'kappa'],
                              truths=theta_mcmc[:, 1], quantiles=[.16, .84],
                              verbose=False)

    randsamp = samples[np.random.randint(len(samples), size=100)]

    # Plot fits
    fig = plt.figure()

    if (akmap is not None) & (bins is not None):
        plt.plot(bins[:-1], 1. / (diff_area_function(akmap, bins)
                                  * scale), 'o', mec='none', alpha=0.75)

    plt.errorbar(x, y, yerr, fmt='rs', alpha=0.7, mec='none')
    plt.legend(['Inv. Diff, Area function', 'Data'],
               loc='upper left', fontsize=12)

    for samp in randsamp:
        plt.plot(bins[:-1], schmidt_law(bins[:-1], samp), 'k', alpha=0.1)

    if True:
        plt.semilogx()
        # ax2.semilogx()
    if True:
        plt.semilogy()

    return plt.gca()


def fit(bins, samp, samperr, maps, mapserr, scale=1., pos=None, pose=None, title=None,
        sampler=None, no_plot=False, triangle_plot=True, nwalkers=100, nsteps=1e4):

    print 'Fitting your sources'
    x = bins[:-1]

    yp = np.array([yso_surfd(samp, maps, bins,
                             yso_err=samperr,
                             errmap=mapserr,
                             scale=scale) for i in xrange(100)])

    nyerr = []
    for i in np.ma.MaskedArray(yp, mask=(
            ~np.isfinite(yp) & ~np.isfinite(1 / yp))).T:
        nyerr.append(np.std(np.array(i)[np.nonzero(i)]))
    nyerr = np.array(nyerr)
    y = yso_surfd(samp, maps, bins, scale=scale)

    pyerr = np.sqrt(hist(samp, bins))

    yerr = y / pyerr
    # yerr = np.sqrt((y * np.sqrt(pyerr**2.))**2)# + nyerr**2)

    uni = np.isfinite(y) & np.isfinite(
        1. / y) & np.isfinite(yerr) & np.isfinite(1. / yerr)

    x = x[uni]
    y = np.array(y)[uni]
    yerr = np.array(yerr)[uni]

    if pos is None:
        pos = [2, 1]
    if pose is None:
        pose = [1., 1.]

    if sampler is None:
        print 'Sampler autocorrelation times . . .'
        sampler, t, et = emcee_schmidt(x, y, yerr,
                                       pos=pos, pose=pose,
                                       nwalkers=nwalkers, nsteps=nsteps)
    else:
        print 'Next time don\'t give me a ' + str(type(sampler)) + '.'

    '''
    ## Don't plot at the same time
    if not no_plot:
        ax = schmidt_results_plots(sampler, schmidt_law,x,y,yerr,
                                 burnin = 2000, akmap = maps,
                                 bins = bins, scale = scale, triangle_plot = triangle_plot)
    else:
        print 'No plots for you!'


    if title is None:
        None
    else:
        ax.set_title(title)
    '''

    # pdb.set_trace()
    try:
        return sampler, x, y, yerr, ax, t
    except:
        return sampler, x, y, yerr


def tester():
    print 'hi yall'
