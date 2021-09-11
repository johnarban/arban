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