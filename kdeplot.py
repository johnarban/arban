import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def kdeplot(xp,yp,filled=False,ax=None,*args,**kwargs):
    if ax is None:
        ax = plt.gca()
    rvs = np.append(xp.reshape((xp.shape[0],1)),
                    yp.reshape((yp.shape[0],1)),
                    axis=1)

    kde = stats.kde.gaussian_kde(rvs.T)
    #kde.covariance_factor = lambda : .45
    #kde._compute_covariance()

    # Regular grid to evaluate kde upon
    x_flat = np.r_[rvs[:,0].min():rvs[:,0].max():256j]
    y_flat = np.r_[rvs[:,1].min():rvs[:,1].max():256j]
    x,y = np.meshgrid(x_flat,y_flat)
    grid_coords = np.append(x.reshape(-1,1),y.reshape(-1,1),axis=1)

    z = kde(grid_coords.T)
    z = z.reshape(256,256)
    if filled:
        cont = ax.contourf
    else:
        cont = ax.contour
    cs = cont(x_flat,y_flat,z,*args,**kwargs)
    return cs
