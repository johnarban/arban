
from __future__ import print_function

import copy
import gc
import glob
import os
import sys
import warnings
from importlib import reload

import aplpy
import astropy.visualization as vis
import colorcet
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import astropy.constants as const
from astropy import units as u
from astropy.convolution import (Gaussian2DKernel, convolve,
                                 interpolate_replace_nans)

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import fitting, models
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate, interpolate
from scipy import ndimage as ndimage
from scipy import optimize, signal, special, stats
from scipy.ndimage import median_filter, morphology
from scipy.optimize import curve_fit, minimize
from scipy.signal import argrelextrema

import utils as jutils

# define short aliases
ju = jutils
nd = ndimage

warnings.simplefilter('ignore', category=AstropyWarning)
np.seterr(divide='ignore', invalid='ignore');

#ipython
%matplotlib inline
%config InlineBackend.print_figure_kwargs={'bbox_inches':None}

#from scipy.integrate import simps
#from scipy.interpolate import UnivariateSpline, griddata
#from scipy.ndimage.morphology import binary_dilation