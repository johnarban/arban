import time
istart = time.time()
print(f'Standard imports')
import glob
import os
import sys
from importlib import reload
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# scipy imports
from scipy import optimize
from scipy import signal
from scipy import special
from scipy import stats
from scipy import integrate
from scipy import interpolate
from scipy import ndimage as nd


print(f'Import common astropy modules')
#get astrophysical constants
from astropy import constants as const
from astropy import units as u
from astropy import convolution as convolution
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.modeling import fitting, models, custom_model



import colorcet  # better colors (cet_*)
import cmasher # more colors (cmr.*)
#import seaborn as sns
from tqdm import tqdm


# import astropy stuff
from astropy.convolution import (convolve, convolve_fft,
                                 interpolate_replace_nans, kernels)
from astropy.utils.exceptions import AstropyWarning
# matplotlib 3d stuff
#from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import morphology

print('load custom utils (ju)')
import utils as ju

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')


print(f'load time: {time.time()-istart+0.3:3.1f}s')