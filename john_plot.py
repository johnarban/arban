
from utils import *
import utils as ju

from  math import floor,sqrt,ceil
from mpl_toolkits.axes_grid1.inset_locator import inset_axes





def annotate(text, x, y, ax=None,
    horizontalalignment="center",
    verticalalignment="center",
    ha=None, va=None, transform="axes",
    fontsize=9,
    color="k",
    facecolor="w",
    edgecolor='0.75',
    alpha=0.75,
    text_alpha=1,
    bbox=dict(),
    stroke = None,
    **kwargs,
):
    """wrapper for Axes.text

    Parameters
    ----------
    text : str
        text
    x : number
        x coordinate
    y : number
        x coordinate
    ax : axes, optional
        [description], by default None
    horizontalalignment : str, optional
        by default "center"
    verticalalignment : str, optional
         by default "center"
    ha : alias for horizontalalignment
    va : alias for verticalalignment
    transform : str, optional
        use 'axes' (ax.transAxes) or 'data' (ax.transData) to interpret x,y
    fontsize : int, optional
         by default 9
    color : str, optional
        text color, by default "k"
    facecolor : str, optional
        color of frame area, by default "w"
    edgecolor : str, optional
        color of frame edge, by default '0.75'
    alpha : float, optional
        transparency of frame area, by default 0.75
    text_alpha : int, optional
        transparency of text, by default 1
    bbox : [type], optional
        dictionary defining the bounding box or frame, by default dict()
    stroke : (list, mpl.patheffects,dict), optional
        most often should be dict with {'foregroud':"w", linewidth:3}
        if using stroke, use should set bbox=None

    Returns
    -------
    text
       the annotation
    """
    if ax is None:
        ax = plt.gca()

    horizontalalignment = ha or horizontalalignment
    verticalalignment = va or verticalalignment

    if transform == "axes":
        transform = ax.transAxes
    elif transform == "data":
        transform = ax.transData
    if bbox is None:
        bbox1 = dict(facecolor='none', alpha=0,edgecolor='none')

    else:
        bbox1 = dict(facecolor=facecolor, alpha=alpha,edgecolor=edgecolor)
        bbox1.update(bbox)
    text = ax.text(
        x,
        y,
        text,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        transform=transform,
        color=color,
        fontsize=fontsize,
        bbox=bbox1,
        **kwargs,
    )

    if stroke is not None:
        if type(stroke) == dict:
            text.set_path_effects([withStroke(**stroke)])
        elif isinstance(stroke,(list,tuple)):
            text.set_path_effects([*stroke])
        elif isinstnace(stroke,mpl.patheffects.AbstractPathEffect):
            text.set_path_effects([stroke])
    return text

def get_square(n,aspect=1.):
    """
    return rows, cols for grid of n items
    n <= 3 returns single row

    aspect = width/height (cols/rows)
    """
    def empty(n,rows,cols):
        empty_spaces = (rows * cols) - n
        empty_rows = floor(empty_spaces / cols)
        return floor(empty_spaces / cols)
    if n < 4:
        return 1, n
    if aspect < 2:
        aspect = 1.
    if (sqrt(n) % 1)  > 0:
        ceil = int(np.ceil(sqrt(n)))
        rows = cols = ceil
        if aspect == 1:
            rows -= empty(n,rows,cols)
        else:
            rows, cols = round(rows * sqrt(aspect)), round(cols / sqrt(aspect))
            rows -= empty(n,rows,cols)
    else:
        rows, cols = sqrt(n),sqrt(n)
        if aspect != 1:
            print(rows,cols)
            rows, cols = round(rows * sqrt(aspect)), round(cols / sqrt(aspect))
            print(rows,cols,)
            rows -= empty(n,rows,cols)
    return int(rows), int(cols)


def channel_maps(cube,v=None,dv=None,spec_ax=-1,wcs=None,
                velmin=None,velmax=None,velskip=None,
                figsize=10,verbose=True,nrows=None,ncols=None, set_bad = np.nan,colorbar=True, **kwargs):
    """Create grid of channel maps

    Uses matplotlib sublplots to create nicely sized grids.

    cube: 3d cube laid out as p - p - v
    v = velocity vector
    dv = width of channels
    spec_ax : don't use this it may not work right
    wcs = none, optional wcs to get wcs coordinates on the axes
    velmin, velmax, velskip : velocity integration min, max, & channels
        if no v is given, then these are interpreted as indices
    figsize=width of figure
    nrows, ncols: can pass in a number of nrows, ncols : can try, but
            this program finds good values for those by itself.
            [[[[ curently these are not implemented ]]]]
    **kwargs get passed to imshow
    """


    ## CHANNEL SELECTION STUFF
    # if a vector of velocities is not provided, use the length
    # of the spectral axis to define channel index as "velocity"
    if v is None:
        v = np.arange(cube.shape[spec_ax])
    if dv is None:
        dv = 1

    # gett the shape of the cube. rearrange from VEL-LAT-LON -> LAT-LON-VEL
    # if needed
    if spec_ax != -1:
            # swap VEL<>LON        swap LON<>LAT
        cube = cube.swapaxes(0,-1).swapaxes(0,1)
    shape = (cube.shape[0],cube.shape[1])

    # pick default velocity integration range (velskip)
    if velskip is None:
        if (velmax is None) & (velmin is None):
            velskip = dv * 5
        else:
            velskip = dv
    if velmin is None:
        velmin = v[0]
    if velmax is None:
        velmax = v[-1]

    # I don't intentionally implement -1 integration, so don't allow it
    if velskip < 0:
        print(" don't be getting fancy trying to go backwards.\n setting velskip = abs(velskip)\n I see you")
        velskip = abs(velskip)

    # we can't integrate over less than 1 channel
    if velskip < dv:
        velskip = dv

    # get 1st index
    imin = int((velmin - v[0])/dv)
    if (imin < 0) | (imin>len(v)-1):
        imin = 0
        velmin = v[imin]
    if verbose:
        print(f'Nearest channel to vel_min={velmin:0.4g}: channel {imin}, {v[imin]:0.4g} km/s')

    # get last index
    imax = int(((velmax - v[0])/dv) + 0.5)
    if (imax < 0) | (imax>len(v)-1):
        imax = len(v)-1
        velmax = v[imax]
    if verbose:
        print(f'Nearest channel to vel_max={velmax:0.4g}: channel {imax}, {v[imax]:0.4g} km/s')




    if verbose:
        print(f'channel width: {dv:0.2f}')
    # round to best velskip
    if velskip >= velmax-velmin:
        print('velskip cannot be > velmax - velmin. setting them equal')
        iskip = imax - imin

    else:
        iskip = int(velskip / dv + 0.5)
    if verbose:
        print(f'Summing every {iskip} ({iskip * dv:0.4g} km/s) channels')

    ## Need to be careful of how numpy indices work
    ## We will integrate from chans[i], chans[i+1]
    ## There will be n_images = len(chans)-1 images ===  each images goes from [0,1),[1,2), [2,3)...
    ## i runs from 0,....,n_images-1  # numpy is 0-based indexing
    ## this means if last element in chans, chans[-1] <= imax
    ## then imax will not show up in the channel map output
    ## this is not what a user expects. So the last value in
    ## chan_end should be >= imax.


    if iskip > 1:
        chan_start = np.arange(imin,imax+iskip,iskip) # we might have to start integration at imax
        chan_end = np.arange(imin+iskip,imax + iskip,iskip)-1 # but we can end on imax.

        if chan_end[-1] > len(v):
            chan_end[-1] = len(v)-1
        chans = list(zip(chan_start,chan_end))  ## ordered pairs [(chan_start[0], chan_end[0]),...]
        if (chan_end[-1] < imax):
            cstart = chan_end[-1] + 1
            cend = imax + (imax-cstart)
            chans.append([cstart,cend])
    else:
        # if we're not integrating, we will just be taking each channel
        chans = np.arange(imin,imax+iskip,iskip)
        if chans[-1] >= len(v):
            chans[-1]=len(v)-1
    #print(chans)
    nchans = len(chans)
    nimage = nchans
    if verbose:
        print(f'Number of images: {nimage}')


    ### Set up the axes and figure shape
    a = ju.get_aspect(cube[:,:,0])

    if nrows is None:
        nrows = 0
    if ncols is None:
        ncols = 0

    if nrows * ncols < nimage:
        nr,nc = get_square(nimage,aspect=a)
    else:
        nr, nc = nrows,ncols

    if verbose:
        print(f'Rows: {nr}  Columns: {nc} Aspect: {a:0.2g}')

    # ds is the width of the figure
    ds = figsize #* np.sqrt(nc)
    figsize = ds
    dpi = 72
    figsize = figsize
    figsize *= np.array([nc*a,nr])

    figsize = figsize * ds / figsize[0]
    if np.sqrt(nc*nr) % 1 == 0:
        figsize[0] = figsize[0]* 1.2


    fig,axs = plt.subplots(nrows=nr,ncols=nc,figsize=figsize,sharex=True,sharey=True,
                          gridspec_kw={'hspace':0.,'wspace':0.,
                                       'left':0.05,'right':0.92,'top':0.95,'bottom':0.05},
                          subplot_kw={'projection':wcs},facecolor='tan') # wcs can be None
    fig.set_tight_layout(False) # don't use tight layout


    axs = np.asarray(np.atleast_1d(axs).flat)
    #set_limits(ax=axs[0],expand=3,square=False,)
    for i in range(nimage):
        ax = axs[i]


        if abs(iskip) == 1:
            c_start = chans[i]
            c_end = c_start + 1
            im = ax.imshow(cube[:,:,c_start]*dv,**kwargs)
        elif abs(iskip) > 1:
            c_start = chans[i][0]
            c_end = chans[i][1]+1 #make it inclusive
            sub = cube[:,:,c_start:c_end]
            img = np.nansum(sub,axis=-1) * dv
            img[np.isnan(sub).all(axis=-1)] = np.nan #nansum got rid of badvals->0, I want them back
            #img[img==set_bad] = np.nan
            im = ax.imshow(img,**kwargs)

        vsub = v[c_start:c_end]
        v_min, v_max = vsub.min(),vsub.max()
        if v_min == v_max:
            t = ju.annotate(fr'${v_min:0.1f}\ km/s$' + '\n' + fr'$\rm chan: {c_start}$',0.05,0.95,ha='left',va='top',
                            alpha=1,ax=ax,bbox=None,stroke={'foreground':'w','linewidth':3})
        else:

            #v_min -= dv/2
            #v_max += dv/2
            t = ju.annotate(fr'$({v_min:0.2f},{v_max:0.2f})\ km/s$'+'\n'+fr'$\rm chan: {c_start}-{c_end}$',0.05,0.95,ha='left',va='top',
                            alpha=1,ax=ax,bbox=None,stroke={'foreground':'w','linewidth':3})

    if colorbar:
        #colorbar on last axis
        cax = inset_axes(ax,
                        width="8%", # width = 10% of parent_bbox width
                        height="90%", # height : 50%
                        loc=3,
                        bbox_to_anchor=(1.01, 0, 1, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0.
                        )
        cbr = plt.colorbar(im, cax=cax)
        if iskip == 1:
            cbr.set_label(r"$\rm T_{mb}\,\Delta v\ [K km/s]$")
        else:
            cbr.set_label(r"$\rm T_{mb}\,\Delta v\ [K km/s]$")
        cbr.set_ticks(np.linspace(*im.get_clim(),5))

    col_0_axs = axs[0::nc]
    last_row_axs = axs[-nc:]

    if wcs is None:
        # sharex and sharey automatically hide ticks
        for ax in axs.reshape(nr,nc)[:-1,1:].flat:
            pass
            #ax.set_axis_off()
            # ax.set_xticklabels([])
            # ax.set_yticklabels([])
            # ax.set_xlabel('')
            # ax.set_ylabel('')

        for ax in axs:
            pass

    else:
        for ax in axs:#reshape(nr,nc)[:-1,1:].flat:
            #ax.set_axis_off()
            ax.coords[0].set_ticklabel_visible(False)
            ax.coords[1].set_ticklabel_visible(False)

        for ax in axs:
            ax.coords[1].set_axislabel(' ')
            ax.coords[0].set_axislabel(' ')

        for ax in axs.reshape(nr,nc)[:,0].flat:
            ax.coords[1].set_ticklabel_visible(True)
            ax.coords[1].set_axislabel('GLAT')

        for ax in axs.reshape(nr,nc)[-1,:]:
            ax.coords[0].set_axislabel('GLON')
            ax.coords[0].set_ticklabel_visible(True)
            ax.coords[0].set_ticklabel(exclude_overlapping=True)


    for i in range(nimage,len(axs)):
        plt.delaxes(axs[i])

    return fig,axs.reshape(nr,nc)
    #axs = [g for g in gr]


