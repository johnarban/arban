print('Import common astro modules')
import glob
import os
import sys
# do some warning handling
import warnings
from importlib import reload

#get astrophysical constants
import astropy.constants as const
import astropy.convolution as convolution
import colorcet # better colors (cet_*)
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import astropy stuff
from astropy import units as u
from astropy.convolution import (convolve, convolve_fft,
                                 interpolate_replace_nans, kernels)
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import fitting, models
from astropy.table import Table
from astropy.utils.exceptions import AstropyWarning
from astropy.wcs import WCS
# matplotlib 3d stuff
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
#import scipy stuff
from scipy import integrate, interpolate
from scipy import ndimage as ndimage
from scipy import optimize, signal, special, stats
from scipy.ndimage import morphology
from scipy.optimize import curve_fit, minimize

# personal import and aliases
import utils as jutils

# define short aliases
ju = jutils
nd = ndimage

print('Muting divide by zero error and AstropyWarnings')
warnings.simplefilter('ignore', category=AstropyWarning)
np.seterr(divide='ignore', invalid='ignore')
