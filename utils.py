# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 18:11:33 2016

@author: johnlewisiii
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# In[Plot the KDE for a set of x,y values. No weighting]

def kdeplot(xp, yp, filled=False, ax=None, grid=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    rvs = np.append(xp.reshape((xp.shape[0], 1)),
                    yp.reshape((yp.shape[0], 1)),
                    axis=1)

    kde = stats.kde.gaussian_kde(rvs.T)
    kde.covariance_factor = lambda: .3
    kde._compute_covariance()

    if grid is None:
        # Regular grid to evaluate kde upon
        x_flat = np.r_[rvs[:, 0].min():rvs[:, 0].max():256j]
        y_flat = np.r_[rvs[:, 1].min():rvs[:, 1].max():256j]
    else:
        x_flat = np.r_[0:grid[0]:256j]
        y_flat = np.r_[0:grid[1]:256j]
    x, y = np.meshgrid(x_flat, y_flat)
    grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

    z = kde(grid_coords.T)
    z = z.reshape(256, 256)
    if filled:
        cont = ax.contourf
    else:
        cont = ax.contour
    cs = cont(x_flat, y_flat, z, *args, **kwargs)
    return cs

# In[ Median SEDs ]

class MedianSED():

    # Initilize class
    def __init__(self, sed, wavelength):
        if isinstance(sed, type([]) ):
            sed = np.asarray(sed)
        if isinstance(wavelength, type([]) ):
            wavelength = np.asarray(wavelength)

        self.sort = np.argsort(wavelength) # data wrangling
        self.x = np.sort(wavelength)
        self.sed = sed[:,self.sort]
        self.valid = self.validate()  # data validation
        self.median = self.get_median(self.sed, self.valid) # calculate median

    def __update__(self):
        # update valid array and median
        self.valid = self.validate(self.sed)
        self.median = self.get_median(self.sed, self.valid)

    def get_median(self):
        # DEF: Calculate median SED
        validseds = self.seds
        validseds[~self.valid] = np.nan   # set invalid elements to NaN
        return np.nanmedian(validseds, axis=1)

    def validate(self):
        nonfinite = np.isfinite(self.seds)  # Identify nans and infinities
        zeros = ~np.iszero(self.seds)   # Identify zero fluxes
        return nonfinite & zeros     # Return valid array


    def __append__(self,sed):
        # Make sure we can use numpy indexing
        if isinstance(sed, type([]) ):
            sed = np.array(sed)
        self.sed = np.append(self.sed, sed[self.sort])

    def update_seds(self, seds):
        # append seds and update
        if ~isinstance(seds[0], list):
            # make sure a list is used
            seds = [seds]
        for sed in seds:
            self.__append__(sed)

        self.__update__()

    def xy(self):
        return self.x[self.valid], self.median[self.valid]



# In[More stuff]































