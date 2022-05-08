
def color_hue_shift(c, shift=1):
    c = mpl.colors.to_rgb(c)
    h, s, v = mpl.colors.rgb_to_hsv(c)
    h = h + shift % 1
    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))


def adjust_lightness(color, amount=0.5):
    # brighter : amount > 1
    # https://stackoverflow.com/a/49601444
    try:
        c = mc.cnames[color]
    except Exception:
        c = color
    c = rgb_to_hls(*mc.to_rgb(c))
    return hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    
    
import matplotlib.colors as mc
from colorsys import rgb_to_hls,hls_to_rgb #builtin
def adjust_lightness(color, amount=0.5):
    # brighter : amount > 1
    #https://stackoverflow.com/a/49601444

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = rgb_to_hls(*mc.to_rgb(c))
    return hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def invert_color(ml, mode = 'rgb'):
	rgb = mpl.colors.to_rgb(ml)
	if mode == 'hsv':
    	hsv = mpl.colors.rgb_to_hsv(rgb)
	    h, s, v = hsv
	    h = 1 - h
	    s = 1 - s
	    v = 1 - v
	    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))
	 else:
    	return mpl.colors.to_hex([1 - i for i in rgb])
    


def icol(*args, **kwargs):
    return invert_color(*args, **kwargs)


def color_array(arr, alpha=1):
    """ take an array of colors and convert to
    an RGBA image that can be displayed
    with imshow
    """
    img = np.zeros(arr.shape + (4,))
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            c = mpl.colors.to_rgb(arr[row, col])
            img[row, col, 0:3] = c
            img[row, col, 3] = alpha
    return img



def arr_to_rgb(arr, rgb=(0, 0, 0), alpha=1, invert=False, ax=None):
    """
    arr to be made a mask
    rgb:assumed using floats (0..1,0..1,0..1) or string

    """
    #check if boolean or single value
    is_bool = ((arr==0) | (arr==1)).all() or (arr.dtype is np.dtype(bool))

    # arr should be scaled to 1
    img = np.asarray(arr, dtype=np.float64)

    if not is_bool:
        img = img - np.nanmin(img)
        img = img / np.nanmax(img)

    im2 = np.zeros(img.shape + (4,))

    if isinstance(rgb, str):
        rgb = mpl.colors.to_rgb(rgb)

    if invert:
        img = 1 - img
    im2[:, :, 3] = img * alpha
    
    im2[:, :, 0] = rgb[0]
    im2[:, :, 1] = rgb[1]
    im2[:, :, 2] = rgb[2]

    return im2


# Make it scale properly
# How does matplotlib
# scaling work
def combine_cmap(cmaps, lower, upper, log = False,name="custom", N=None, register_cmap=True, norm=False):
    """
    colormaps : a list of N matplotlib colormap classes
    lower : the lower limits for each colormap: array or tuple
    upper : the upper limits for each colormap: array or tuple
    		# ranges should not overlap
    log   : Do you want to plot logscale. This will create
            a color map that is usable with LogNorm()
    """
    if log:
        upper = [np.log10(i / lower[0]) for i in upper]
        lower = [np.log10(i / lower[0]) for i in lower]
    else:
        lower = lower
        upper = upper

    n = len(cmaps)

    for ic, c in enumerate(cmaps):
        if isinstance(c, str):
            cmaps[ic] = mpl.cm.get_cmap(c)

    if N is None:
        N = [256] * n

    values = np.array([])
    colors = np.empty((0, 4))

    for i,cmap in enumerate(cmaps):
    	# get steps for i-th color map
        step = (upper[i] - lower[i]) / N[i]
        xcols = np.arange(lower[i], upper[i], step)
        # store xcols
        values = np.append(values, xcols)
        # scale range to 0-1 to get full colormap
        xcols -= xcols.min()
        xcols /= xcols.max()
        cols = cmap(xcols)
        colors = np.vstack([colors, cols])
    values -= values.min()
    values /= values.max()

    arr = list(zip(values, colors))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, arr)

    if (name != "custom") & register_cmap:
        mpl.cm.register_cmap(name=name, cmap=cmap)

    if norm:
        if log:
            norm = mpl.colors.LogNorm(vmin=min(lower), vmax=max(upper))
        else:
            norm = mpl.colors.Normalize(vmin=min(lower), vmax=max(upper))
        return cmap, norm

    return cmap



# def custom_cmap(colormaps, lower, upper, log=(0, 0)):
#     """
#     colormaps : a list of N matplotlib segmented colormap classes
#     lower : the lower limits for each colormap: array or tuple
#     upper : the upper limits for each colormap: array or tuple
#     log   : Do you want to plot logscale. This will create
#             a color map that is usable with LogNorm()
#     """
#     if isinstance(log, tuple):
#         for lg in log:
#             if lg:
#                 upper = [np.log10(i / lower[0]) for i in upper]
#                 lower = [np.log10(i / lower[0]) for i in lower]
#                 norm = upper[-1:][0]
#             else:
#                 lower = lower
#                 upper = upper
#                 norm = upper[-1:][0]
#     elif log:
#         upper = [np.log10(i / lower[0]) for i in upper]
#         lower = [np.log10(i / lower[0]) for i in lower]
#         norm = upper[-1:][0]
#     else:
#         lower = lower
#         upper = upper
#         norm = upper[-1:][0]
# 
#     for ic, c in enumerate(colormaps):
#         if isinstance(c, str):
#             colormaps[ic] = mpl.cm.get_cmap(c)
# 
#     cdict = {"red": [], "green": [], "blue": []}
# 
#     for color in ["red", "green", "blue"]:
#         for j, col in enumerate(colormaps):
#             # print j,col.name,color
#             x = [i[0] for i in col._segmentdata[color]]
#             y1 = [i[1] for i in col._segmentdata[color]]
#             y0 = [i[2] for i in col._segmentdata[color]]
#             x = [(i - min(x)) / (max(x) - min(x)) for i in x]
#             x = [((i * (upper[j] - lower[j])) + lower[j]) / norm for i in x]
#             if (j == 0) & (x[0] != 0):
#                 x[:0], y1[:0], y0[:0] = [0], [y1[0]], [y0[0]]
#             for i in range(len(x)):  # first x needs to be zero
#                 cdict[color].append((x[i], y1[i], y0[i]))
# 
#     return colors.LinearSegmentedColormap("my_cmap", cdict)



def concat_cmap(cmaps=['viridis','turbo'],split_fracs=0.5):
    """
    split a colormap at a certain location

    split - where along the colormap will be our split point
            by default this split point is put in the middle
            of the values
    vmin  value for colorbar to start at: should max vim in
            plotting command
    vmaxs  (splitvalue,vmax) - where to start the second segment
            of the color map. cmap(split) will be located
            at valeu=splitvalue
    vplit = instead of giving vmin,vmax,a you can split it at a
            value between 0,1.
    log     doesn't do what anyone would think, don't recommend using

    """


    ncolors  = 1024

    cols1 = mpl.cm.get_cmap(cmaps[0])(np.linspace(0, 1, int(ncolors * split_fracs)))
    cols2 = mpl.cm.get_cmap(cmaps[1])(np.linspace(0, 1, int(ncolors * (1 - split_fracs))))

    # Combine them and build a new colormap:
    return mpl.colors.ListedColormap(np.vstack( (cols1,cols2) ))



def cmap_split(*args, **kwargs):
    """alias for split_cmap"""
    return split_cmap(*args, **kwargs)

def split_cmap(cmapn='viridis',split=0.5,vmin=0, vmaxs=(.5,1),vstep=None,
               vsplit=None,log=False,return_norm=False):
    """
    split a colormap at a certain location

    split - where along the colormap will be our split point
            by default this split point is put in the middle
            of the values
    vmin  value for colorbar to start at: should max vim in
            plotting command
    vmaxs  (splitvalue,vmax) - where to start the second segment
            of the color map. cmap(split) will be located
            at valeu=splitvalue
    vplit = instead of giving vmin,vmax,a you can split it at a
            value between 0,1.
    log     doesn't do what anyone would think, don't recommend using



    """
    if vsplit is not None:
        vmin=0
        vmaxs=(vsplit,1)
    vmin1 = vmin
    vmax1 =  vmaxs[0]
    vmin2 = vmax1
    vmax2 =  vmaxs[1]
    if vstep is None:
        vstep=   (vmax2 - vmin1)/1024
    levels1 = np.arange(vmin1, vmax1+vstep, vstep)
    levels2 = np.arange(vmin2, vmax2+vstep, vstep)

    ncols1 = len(levels1)-1
    #ncols1 = int((vmax1-vmin1)//vstep)
    ncols2 = len(levels2)-1
#     ncols1 = int((vmax1-vmin1)//vstep)+1
#     ncols2 = int((vmax2-vmin2)//vstep)+1
    # ncols = ncols1 + ncols2
    split = split
    # Sample the right number of colours
    # from the right bits (between 0 &amp; 1) of the colormaps we want.
    cmap2 = mpl.cm.get_cmap(cmapn)
    if log:
        cmap1 = mpl.cm.get_cmap(cmapn+'_r')
        cols1 = cmap1(np.logspace(np.log10(1-split),0, ncols1))[::-1]
        cols2 = cmap2(np.logspace(np.log10(split), 0, ncols2))
    else:
        cols1 = cmap2(np.linspace(0.0, split, ncols1))
        cols2 = cmap2(np.linspace(split, 1, ncols2))


    #cols2 = cmap2(np.logspace(np.log10(split), 0, ncols2))

    # Combine them and build a new colormap:
    allcols2 = np.vstack( (cols1,cols2) )
    cmap = mpl.colors.LinearSegmentedColormap.from_list('piecewise2', allcols2)
    if return_norm:
        return cmap,mpl.colors.Normalize(vmin=vmin,vmax=vmaxs[1])

    return cmap



def clean_color(color, reverse=False):
    if isinstance(color, str):
        if color[-2:] == "_r":
            return color[:-2], True
        elif reverse is True:
            return color, True
        else:
            return color, False
    else:
        return color, reverse


def color_cmap(c, alpha=1, to_white=True, reverse=False):
    if to_white:
        end = (1, 1, 1, alpha)
    else:
        end = (0, 0, 0, alpha)

    color, reverse = clean_color(c, reverse=reverse)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("density_cmap", [color, end])
    if reverse:
        return cmap.reversed()
    else:
        return cmap


def contour_level_colors(cmap, levels, vmin=None, vmax=None, center=True):
    """get colors corresponding to those produced by contourf

    Arguments:
        cmap {string or cmap} -- colormap
        levels {list or array} -- desired levels

    Keyword Arguments:
        vmin {number} -- min value (default: {0})
        vmax {number} -- max value (default: {max(levels)})
        center {True} -- contourf uses center=True values.
                         False will produce a border effect (default: {True})

    Returns:
        [ndarray] -- [list of colors]
    """
    vmin = vmin or 0
    vmax = vmax or max(levels)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    # offset = np.diff(levels)[0] * .5
    # colors = mpl.cm.get_cmap(cmap)(norm(levels-offset))
    levels = np.r_[0, levels]
    center_levels = 0.5 * (levels[1:] + levels[:-1])
    return mpl.cm.get_cmap(cmap)(norm(center_levels))