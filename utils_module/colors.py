


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



def color_hue_shift(c, shift=1):
    c = mpl.colors.to_rgb(c)
    h, s, v = mpl.colors.rgb_to_hsv(c)
    h = h + shift % 1
    return mpl.colors.to_hex(mpl.colors.hsv_to_rgb((h, s, v)))


# Make it scale properly
# How does matplotlib
# scaling work
def combine_cmap(cmaps, lower, upper, name="custom", N=None, register=True):

    n = len(cmaps)

    for ic, c in enumerate(cmaps):
        if isinstance(c, str):
            cmaps[ic] = mpl.cm.get_cmap(c)

    if N is None:
        N = [256] * n

    values = np.array([])
    colors = np.empty((0, 4))

    for i in range(n):
        step = (upper[i] - lower[i]) / N[i]
        xcols = np.arange(lower[i], upper[i], step)
        values = np.append(values, xcols)
        xcols -= xcols.min()
        xcols /= xcols.max()
        cols = cmaps[i](xcols)
        colors = np.vstack([colors, cols])
    values -= values.min()
    values /= values.max()

    arr = list(zip(values, colors))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, arr)

    if (name != "custom") & register:
        mpl.cm.register_cmap(name=name, cmap=cmap)

    return cmap


def custom_cmap(colormaps, lower, upper, log=(0, 0)):
    """
    colormaps : a list of N matplotlib colormap classes
    lower : the lower limits for each colormap: array or tuple
    upper : the upper limits for each colormap: array or tuple
    log   : Do you want to plot logscale. This will create
            a color map that is usable with LogNorm()
    """
    if isinstance(log, tuple):
        for lg in log:
            if lg:
                upper = [np.log10(i / lower[0]) for i in upper]
                lower = [np.log10(i / lower[0]) for i in lower]
                norm = upper[-1:][0]
            else:
                lower = lower
                upper = upper
                norm = upper[-1:][0]
    elif log:
        upper = [np.log10(i / lower[0]) for i in upper]
        lower = [np.log10(i / lower[0]) for i in lower]
        norm = upper[-1:][0]
    else:
        norm = upper[-1:][0]

    for ic, c in enumerate(colormaps):
        if isinstance(c, str):
            colormaps[ic] = mpl.cm.get_cmap(c)

    cdict = {"red": [], "green": [], "blue": []}

    for color in ["red", "green", "blue"]:
        for j, col in enumerate(colormaps):
            # print j,col.name,color
            x = [i[0] for i in col._segmentdata[color]]
            y1 = [i[1] for i in col._segmentdata[color]]
            y0 = [i[2] for i in col._segmentdata[color]]
            x = [(i - min(x)) / (max(x) - min(x)) for i in x]
            x = [((i * (upper[j] - lower[j])) + lower[j]) / norm for i in x]
            if (j == 0) & (x[0] != 0):
                x[:0], y1[:0], y0[:0] = [0], [y1[0]], [y0[0]]
            for i in range(len(x)):  # first x needs to be zero
                cdict[color].append((x[i], y1[i], y0[i]))

    return colors.LinearSegmentedColormap("my_cmap", cdict)


def cmap_split(*args, **kwargs):
    """alias for split_cmap"""
    return split_cmap(*args, **kwargs)

def split_cmap(cmapn='viridis',split=0.5,vmin=0, vmaxs=(.5,1),vstep=None,
               vsplit=None,log=False):
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
    # split = split
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
    return mpl.colors.LinearSegmentedColormap.from_list('piecewise2', allcols2)


def color_cmap(c, alpha=1, to_white=True, reverse=False,name='density_cmap'):
    if to_white:
        end = (1, 1, 1, alpha)
    else:
        end = (0, 0, 0, alpha)

    color, reverse = clean_color(c, reverse=reverse)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(name, [color, end])
    if reverse:
        return cmap.reversed()
    else:
        return cmap
