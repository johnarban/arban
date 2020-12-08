
import numpy as np



def covariance_ellipse(cov,mu=(0,0)):
    # https://cookierobotics.com/007/
    a, b, c = cov[0, 0], cov[0, 1], cov[1, 1]

    # the eigenvalues of the covariance matrix
    lambda1 = 0.5 * (a + np.sqrt(4 * b ** 2 + (a - c)**2) + c)  # major axis^2
    lambda2 = 0.5 * (a - np.sqrt(4 * b ** 2 + (a - c)**2) + c)  # minor axis^2

    if (b == 0) & (a >= c):
        theta = 0
    elif (b == 0) & (a < c):
        theta = np.pi / 2
    else:
        theta = np.arctan2(lambda1 - a, b)

    x = lambda t: np.sqrt(lambda1) * np.cos(theta) * np.cos(t) + np.sqrt(lambda2) * np.sin(theta) * np.sin(t)
    y = lambda t: np.sqrt(lambda1) * np.sin(theta) * np.cos(t) - np.sqrt(lambda2) * np.cos(theta) * np.sin(t)

    t = np.linspace(0,2*np.pi, 100)
    line = (x(t)+mu[0],y(t)+mu[1])

    return lambda1, lambda2, theta, line



def whiten_data(Y):
    """whiten data by projecting to eigenbases

    Parameters
    ----------
    Y : np.array([X1,X2,...])
        array where each column is a variable
        this is the same as np.cov requires

    Returns
    -------
    np.array
        mean subtracted, whitened data

    help:https://janakiev.com/blog/covariance-matrix/
    """

    cov = np.cov(Y)
    mu = np.mean(Y,axis=1)
    eVa, eVe = np.linalg.eig(cov)
    ## eigendecomposition finds
    ## cov P = P D (where D is diag(lambdas))
    ## Finds the P that diagonalizes covariance matrix

    P, D = eVe, np.diag(eVa)
    # cov = P @ D @ inv(P)
    S = D ** 0.5

    T = P @ S  # transform from real to eigenspace
    # inv(P) = P.T:: P is orthogonal
    #  S.T = S, S is symmetric
    # T.T = S P.T
    # Columns of T are scaled eigenvectors

    return (Y.T - mu).dot(np.linalg.inv(T.T))