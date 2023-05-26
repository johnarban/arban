import matplotlib.colors as colors
import numpy as np
from matplotlib import cm as mcx
from matplotlib import colors as mcolor

# TODO

#Make it scale properly
#How does matplotlib
#scaling work

def custom_cmap(colormaps, lower, upper, log = 0):
    '''
    colormaps : a list of N matplotlib colormap classes
    lower : the lower limits for each colormap: array or tuple
    upper : the upper limits for each colormap: array or tuple
    log   : Do you want to plot logscale. This will create 
            a color map that is usable with LogNorm()
    '''
    
    if log == 1:
        upper = [np.log10(i/lower[0]) for i in upper]
        lower = [np.log10(i/lower[0]) for i in lower]
        norm = upper[-1:][0]
    else:
        lower = lower
        upper = upper
        norm = upper[-1:][0]
        
    cdict = { 'red':[], 'green':[],'blue':[] }
    
    for color in ['red','green','blue']:
        for j,col in enumerate(colormaps):
            #print j,col.name,color
            x = [i[0] for i in col._segmentdata[color]]
            y1 = [i[1] for i in col._segmentdata[color]]
            y0 = [i[2] for i in col._segmentdata[color]]
            x = [(i-min(x))/(max(x)-min(x)) for i in x]
            x = [((i * (upper[j] - lower[j]))+lower[j])/norm for i in x]
            if (j == 0) & (x[0] != 0):
                x[:0],y1[:0],y0[:0] = [0],[y1[0]],[y0[0]]
            for i in range(len(x)): #first x needs to be zero
                cdict[color].append((x[i],y1[i],y0[i]))
                
    return colors.LinearSegmentedColormap('my_cmap',cdict)



# TODO

# Make it scale properly
# How does matplotlib
# scaling work
def combine_cmap_val(cmaps,values=None, lower=None, upper=None, log = False,name="custom", N=None, register=True, norm=False):
    """
    colormaps : a list of N matplotlib colormap classes
    lower : the lower limits for each colormap: array or tuple
    upper : the upper limits for each colormap: array or tuple
    log   : Do you want to plot logscale. This will create
            a color map that is usable with LogNorm()
    """

    if (values is None) & (lower is None) & (upper is None):
        values = np.linspace(0,1,len(cmaps))

    if (upper is None) & (lower is None):
        upper = values[1:]
        lower = values[:-1]


    if log:
        upper = [np.log10(i / lower[0]) for i in upper]
        lower = [np.log10(i / lower[0]) for i in lower]
        norm = upper[-1:][0]
    else:
        lower = lower
        upper = upper
        norm = upper[-1:][0]

    n = len(cmaps)

    for ic, c in enumerate(cmaps):
        if isinstance(c, str):
            cmaps[ic] = mcx.get_cmap(c)

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
    cmap = mcolor.LinearSegmentedColormap.from_list(name, arr)

    if (name != "custom") & register:
        mcx.register_cmap(name=name, cmap=cmap)

    if norm:
        if log:
            norm = mcolor.LogNorm(vmin=min(lower), vmax=max(upper))
        else:
            norm = mcolor.Normalize(vmin=min(lower), vmax=max(upper))
        return cmap, norm

    return cmap



def combine_cmap(cmaps=None,split_fracs=None):
    """
    split a colormap at a certain location
    cmaps - a list of N matplotlib colormap classes (defaults to viridis)
    split_fracs - list of fractions to split at (length = len(cmaps)-1)
                  defaults to evenly distributing the colormaps


    """

    if cmaps is None:
        cmaps = [mcx.viridis]

    nmaps = len(cmaps)


    if split_fracs is None:
        # distribute evenly between 0 and 1
        split_fracs = np.diff(np.linspace(0, 1, nmaps + 1))
    else:
        split_fracs = np.atleast_1d(split_fracs)
        if len(split_fracs) == nmaps - 1:
            split_fracs = np.diff(np.r_[0, split_fracs, 1])
        elif len(split_fracs) > nmaps - 1:
            split_fracs = np.diff(np.r_[0, split_fracs[:nmaps-1], 1])
        else:
            split_fracs = np.diff(np.linspace(0, 1, nmaps + 1))

    ncolors  = 256 * nmaps

    # create a list of colors
    cols = []
    for i,c in enumerate(cmaps):
        cmap = mcx.get_cmap(c)
        ncolor = int(ncolors * split_fracs[i])
        cols.append(cmap(np.linspace(0, 1, ncolor)))

    # cols1 = mcx.get_cmap(cmaps[0])(np.linspace(0, 1, int(ncolors * split_fracs)))
    # cols2 = mcx.get_cmap(cmaps[1])(np.linspace(0, 1, int(ncolors * (1 - split_fracs))))

    # Combine them and build a new colormap:
    return mcolor.ListedColormap(np.vstack(cols))



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
    cmap2 = mcx.get_cmap(cmapn)
    if log:
        cmap1 = mcx.get_cmap(cmapn+'_r')
        cols1 = cmap1(np.logspace(np.log10(1-split),0, ncols1))[::-1]
        cols2 = cmap2(np.logspace(np.log10(split), 0, ncols2))
    else:
        cols1 = cmap2(np.linspace(0.0, split, ncols1))
        cols2 = cmap2(np.linspace(split, 1, ncols2))


    #cols2 = cmap2(np.logspace(np.log10(split), 0, ncols2))

    # Combine them and build a new colormap:
    allcols2 = np.vstack( (cols1,cols2) )
    cmap = mcolor.LinearSegmentedColormap.from_list('piecewise2', allcols2)
    if return_norm:
        return cmap,mcolor.Normalize(vmin=vmin,vmax=vmaxs[1])

    return cmap


def combine_colormaps(colormaps, limits, log=False):
    """
    colormaps : a list of N matplotlib colormap classes
    lower : the lower limits for each colormap: array or tuple
    upper : the upper limits for each colormap: array or tuple
    log   : Do you want to plot logscale. This will create
            a color map that is usable with LogNorm()
    """

    limits

    if isinstance(log, tuple):
        if len(log) == len(limits)-1:
            if np.any(log):
                for lg in log:
                    if lg:
                        upper = [np.log10(i / lower[0]) for i in upper]
                        lower = [np.log10(i / lower[0]) for i in lower]
                        norm = upper[-1:][0]
                    else:
                        lower = lower
                        upper = upper
                        norm = upper[-1:][0]
        else:
            raise ValueError('log must be a tuple of length len(limits)-1')
    elif log:
        upper = [np.log10(i / lower[0]) for i in upper]
        lower = [np.log10(i / lower[0]) for i in lower]
        norm = upper[-1:][0]
    else:
        lower = lower
        upper = upper
        norm = upper[-1:][0]

    for ic, c in enumerate(colormaps):
        if isinstance(c, str):
            colormaps[ic] = mcx.get_cmap(c)

    cdict = {"red": [], "green": [], "blue": []}


    for j, col in enumerate(colormaps):
        if isinstance(col, mcolor.LinearSegmentedColormap):
            segmentdata = col._segmentdata
            for color in ['red','green','blue']:
                c  = segmentdata[color]
                x  = [i[0] for i in c]
                y1 = [i[1] for i in c]
                y0 = [i[2] for i in c]

                x = ((x * (upper[j] - lower(j))) + lower[j]) / norm
                cdict[color].append((x, y0, y1))
        else:
            x = np.linspace(0, 1, col.N)
            xcol = ((x * (upper[j] - lower(j))) + lower[j]) / norm
            r,g,b = col.colors.T # returns r,g,b,a

            cdict["red"].append(  (xcol, r, r))
            cdict["green"].append((xcol, g, g))
            cdict["blue"].append( (xcol, b, b))

    return colors.LinearSegmentedColormap("my_cmap", cdict)



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
