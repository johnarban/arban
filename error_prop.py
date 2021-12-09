import numpy as np

def err_div(x, y, ex, ey):
    Q = x / y
    dQ = np.abs(Q) * np.sqrt((ex / x) ** 2 + (ey / y) ** 2)
    return Q, dQ


def err_multiply(x, y, ex, ey):
    Q = x * y
    dQ = np.abs(Q) * np.sqrt( (ex/x)**2 + (ey/y)**2 )
    return Q, dQ


def err_add(x, y, ex, ey):
    Q = x + y
    dQ = np.sqrt(ex**2 + ey**2)
    return Q, dQ

def err_power_1(x, p, ex):
    """error propogation for
    x^p where p is a constant
    """
    Q = x ** p
    dQ = np.abs(p * x**(p-1) * ex)
    return Q, dQ

def err_power_2(x, p, ex, ep):
    """error propogation for
    x^p where x & p have errors
    """
    Q = x ** p
    dQ_sq = Q**2 * ( (p * ex / x)**2 + (np.log(x) * ep)**2 )
    return Q, dQ_sq**0.5


def log_err(x, dx):
    return dx / (np.log(10) * x)

def log_errorbar(x, dx):
    logxerr = log_err(x,dx)
    logxupp = np.log10(x) + logxerr
    logxlow = np.log10(x) - logxerr
    xupp = 10**logxupp - x
    xlow = x - 10**logxlow
    return xupp, xlow


# 1st moment error is v * sqrt(2) sigma_w / w

# 2nd moment error
