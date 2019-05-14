import os
import sys
import glob
from importlib import reload

# import math stuff
print('loading numpy and scipy')
import numpy as np
from scipy import integrate, interpolate
from scipy import ndimage as ndimage
from scipy.ndimage import morphology
from scipy import optimize, signal, special, stats
from scipy.optimize import curve_fit, minimize

print('loading colorcet, matplotlib, and seaborn')
import colorcet
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

print('loading astropy into namespace')
# import astropy stuff
from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import fitting, models
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.constants as const
from astropy import units as u
from astropy.convolution import (Gaussian2DKernel, convolve,
                                 interpolate_replace_nans)                           

print('loading personal utils into namespace as: jutils & alias ju')
# personal import and aliases
import utils as jutils
# define short aliases
ju = jutils
nd = ndimage

print('Muting divide by zero error and AstropyWarnings')
# do some warning handling
import warnings
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
np.seterr(divide='ignore', invalid='ignore')
