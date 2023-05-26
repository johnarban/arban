import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt
import emcee





def linear_emcee_fitter(
    x,
    y,
    yerr=None,
    fit_log=False,
    gauss_prior=False,
    nwalkers=10,
    theta_init=None,
    use_lnf=True,
    bounds=([-np.inf, np.inf], [-np.inf, np.inf]),
):
    """
    ## sample call
    sampler,pos = little_emcee_fitter(x,y, theta_init=np.array(mfit.parameters), use_lnf=True)
    samples = sampler.chain[:,1000:,:].reshape((-1,sampler.dim))

    corner.corner(samples,show_titles=True, quantiles=[.16,.84], labels=["$m$", "$b$", r"$\ln\,f$"])
    ---------------------------------------------

    Arguments:
        x {np.array} -- x values as numpy array
        y {np.array} -- y values as numpy array

    Keyword Arguments:
        model {function that is called model(x,theta)} -- (default: {linear model})
        yerr {yerr as numpy array} -- options (default: {.001 * range(y)})
        loglike {custom loglikelihood} -- (default: {chi^2})
        lnprior {custom logprior} --  (default: {no prior})
        nwalkers {number of walkers} -- (default: {10})
        theta_init {initial location of walkers} -- [required for operation] (default: {Noalne})
        use_lnf {use jitter term} -- (default: {True})

    Returns:
        sampler, pos -- returns sampler and intial walker positions
    """

    if fit_log:
        x, y = np.log(x), np.log(y)

    if yerr is None:
        yerr = np.full_like(y, 0.001 * (np.nanmax(y) - np.nanmin(y)))

    g = np.isfinite(x + y + 1 / yerr ** 2)
    x, y, yerr = x[g], y[g], yerr[g]

    bounds = np.sort(bounds, axis=1)

    def model(x, theta):
        return theta[0] * x + theta[1]

    if theta_init is None:
        theta_init, cov = np.polyfit(x, y, 1, cov=True)
    if use_lnf:
        theta_init = np.append(theta_init, -1)
        newcov = np.zeros((3, 3))
        newcov[0:2, 0:2] = cov
        newcov[2, 2] = 0.0001
        cov = newcov
    ndim = len(theta_init)

    pos = np.random.multivariate_normal(theta_init, cov, size=nwalkers)

    def lnlike(theta, x, y, yerr, use_lnf=use_lnf):
        ymodel = model(x, theta)
        if use_lnf:
            inv_sigma2 = 1.0 / (yerr ** 2 + ymodel ** 2 * np.exp(2 * theta[-1]))
        else:
            inv_sigma2 = 1.0 / yerr ** 2
        return -0.5 * (np.sum((y - ymodel) ** 2 * inv_sigma2 - np.log(inv_sigma2)))

    def lnprior(theta):
        if gauss_prior:
            return stats.multivariate_normal(theta[:-1], theta_init[:-1], cov)
        else:
            c1 = (theta[0] > bounds[0].min()) & (theta[0] < bounds[0].max())
            c2 = (theta[1] > bounds[1].min()) & (theta[0] < bounds[1].max())
            if c1 & c2 & np.all(np.isfinite(theta)):
                return 0.0
            return -np.inf

    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, x, y, yerr, use_lnf=use_lnf)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
    return sampler, pos


def plot_walkers(sampler, limits=None, bad=None):
    """
    sampler :  emcee Sampler class
    """

    if hasattr(sampler, "__getitem__"):
        chain = sampler
        ndim = chain.shape[-1]
    else:
        chain = sampler.chain
        ndim = sampler.ndim

    fig = plt.figure(figsize=(8 * ndim, 4 * ndim))

    for w, walk in enumerate(chain[:, limits:, :]):
        if bad is None:
            color = "k"
        elif bad[w]:
            color = "r"
        else:
            color = "k"
        for p, param in enumerate(walk.T):
            ax = plt.subplot(ndim, 1, p + 1)
            ax.plot(param, color, alpha=0.75, lw=0.75)
            # ax.set_ylim(param.min()*0.5,param.max()*1.5)
            # ax.semilogy()
    plt.tight_layout()
    return fig
