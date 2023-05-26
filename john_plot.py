
from utils import *
import utils as ju

from  math import floor,sqrt,ceil
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.cm import get_cmap
import numpy as np




def channel_maps(cube,v=None,dv=None,spec_ax=-1,wcs=None,
                velmin=None,velmax=None,velskip=None,ncols_max=np.inf,debug=False,
                figsize=10,verbose=True,nrows=None,ncols=None, set_bad = None,colorbar=True, fig=None,ax=None, **kwargs):
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
        # imagine imin = 0, imax = 10, iskip = 3
        # then we want to get chan_end = [3,6,9]
        # and chan_start = [0,3,6]
        chan_start = np.arange(imin,imax+iskip,iskip) # we might have to start integration at imax
        chan_end = np.arange(imin+iskip,imax + iskip,iskip)-1 # but we can end on imax.

        if chan_end[-1] > len(v):
            chan_end[-1] = len(v)-1 # make sure we don't go past the end of the spectrum
        chans = list(zip(chan_start,chan_end))  ## ordered pairs [(chan_start[0], chan_end[0]),...]
        if (chan_end[-1] < imax):
            cstart = chan_end[-1] + 1
            cend = imax + (imax-cstart)
            chans.append([cstart,cend])
    else:
        # if we're not integrating, we will just be taking each channel
        if imin!=imax:
            chans = np.arange(imin,imax+iskip,iskip)
        else:
            chans = [imin]
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

    if (nrows * ncols >= nimage):
        nr, nc = nrows,ncols
    elif ncols > 0:
        nc = ncols
        nr = int(nimage / ncols + 1)

    else:
        nr,nc = get_square(nimage,aspect=a)

    if nc > ncols_max:
        nc = ncols_max
        nr = int(nimage / ncols_max + 1)

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

    gridspec_kw={'hspace':0.,'wspace':0.,'left':0.1,'right':0.92,'top':0.95,'bottom':0.1}
    subplot_kw={'projection':wcs}
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(nrows=nr,ncols=nc,figure=fig,**gridspec_kw)
    axs = gs.subplots(subplot_kw=subplot_kw,sharex=True,sharey=True,)
    axs = np.atleast_1d(axs).ravel()
    fig.set_constrained_layout(False)

    # fig,axs = plt.subplots(nrows=nr,ncols=nc,figsize=figsize,sharex=True,sharey=True,constrained_layout=False,
    #                         gridspec_kw=gridspec_kw,subplot_kw=subplot_kw,facecolor='tan') # wcs can be None
    #fig.set_tight_layout(False) # don't use tight layout


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
            img = np.nansum(sub,axis=-1) * dv * 1.
            img[np.isnan(sub).all(axis=-1)] = np.nan #nansum got rid of badvals->0, I want them back
            if set_bad is not None:
                img[(sub==set_bad).all(axis=-1)] = np.nan
            im = ax.imshow(img,**kwargs)
        elif abs(iskip) == 0:
            c_start = chans[i]
            c_end = c_start + 1
            im = ax.imshow(cube[:,:,c_start]*dv,**kwargs)


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

    #fig.supxlabel('Galactic Longitude',y=0.0)
    #fig.supylabel('Galactic Latitude',x=0.0)


    return fig,axs.reshape(nr,nc)
    #axs = [g for g in gr]




def renzogram(cube,v=None,dv=None,wcs=None,
                velmin=None,velmax=None,velskip=None,
                figsize=10,verbose=True,cmap='RdBu_r',smooth=0,levels=[1],lw=1, ax=None,filled=False,alpha=1,**kwargs):
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
        v = np.arange(cube.shape[-1])
    if dv is None:
        dv = 1

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

    nr = 1
    nc = 1
    ds = figsize #* np.sqrt(nc)
    figsize = figsize
    figsize *= np.array([nc*a,nr])
    figsize = figsize * ds / figsize[0]

    if ax is None:
        subplot_kw={'projection':wcs}
        fig, ax = plt.subplots(nr, nc, figsize=figsize,subplot_kw=subplot_kw)
        fig.set_constrained_layout(False)
    else:
        fig = ax.figure


    if filled:
        levels = np.append(levels,np.inf)

    for i in range(nimage):
        color = get_cmap(cmap,nimage)(i/nimage)
        if abs(iskip) == 1:
            c_start = chans[i]
            c_end = c_start + 1
            layer = cube[:,:,c_start]*dv

        elif abs(iskip) > 1:
            c_start = chans[i][0]
            c_end = chans[i][1]+1 #make it inclusive
            sub = cube[:,:,c_start:c_end]
            img = np.nansum(sub,axis=-1) * dv
            #img[np.isnan(sub).all(axis=-1)] = np.nan #nansum got rid of badvals->0, I want them back
            #img[img==set_bad] = np.nan
            layer = img

        if smooth>0:
            layer = ju.nan_gaussian_filter(layer,smooth)
        if filled:
            cntr = ax.contourf(layer,levels=levels,linewidths=lw,colors=[color], alpha=alpha,**kwargs)
        else:
            cntr = ax.contour(layer,levels=levels,linewidths=lw,colors=[color], alpha=alpha,**kwargs)


    ax.set_aspect('equal')
    if wcs is None:
        pass
    else:
        ax.coords[1].set_axislabel('GLAT')
        ax.coords[0].set_axislabel('GLON')

    imin,imax = np.mean(np.atleast_1d(chans[0])),np.mean(np.atleast_1d(chans[-1]))
    vmin,vmax = np.interp([imin,imax],np.arange(len(v)),v)


    return fig,mpl.cm.get_cmap(cmap,nimage), mpl.colors.Normalize(vmin=vmin,vmax=vmax)





# function to plot spectra over their position on a collapsed cube
def overlay_spectra_plot(array, nrow=5,ncol=5,**kwargs):
    """
    Overlay spectra on a collapsed cube.

        Parameters
        ----------
        array : 3D numpy array

        nrow : int
            Number of rows in the figure.
        ncol : int
            Number of columns in the figure.
        **kwargs : dict
            Keyword arguments passed to `ax.plot` for the spectra

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        """


    cube = np.nan_to_num(array)

    fig,ax = plt.subplots(subplot_kw={'projection':mmap.wcs},figsize=(10,10))
    fig.set_constrained_layout(False)

    collapsed_cube = np.nanmean(cube,axis=2)

    vmin,vmax = np.percentile(collapsed_cube[collapsed_cube>0], [0.1,99.9])
    ax.imshow(collapsed_cube,cmap='Greys',norm=mpl.colors.LogNorm(vmin=vmin,vmax=vmin))


    w = 1/ncol # in figure coords
    h = 1/nrow # in figure coords


    dr,dc = collapsed_cube.shape

    # create grid of inset_axes on figure
    for i in range(nrow):
        for j in range(ncol):
            b,l = i*h, j*w
            #print(f'left:{l:0.1f} col: {j} bottom:{b:0.1f} row:{i}')
            bl = [b,l]
            ax2 = ax.inset_axes([l,b,w,h])
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_facecolor('none')
            #ax.add_patch(mpl.patches.Rectangle([l,b],w,h,transform=ax.transAxes,color='r',alpha=0.5))
            #ju.annotate(f'row:{i} col:{j}',l,b,ha='left',va='bottom',ax=ax,transform='axes')


            #print(f'{int(b*dr)}:{int((b+h)*dr)},{int(l*dc)}:{int((l+w)*dc)}')
            line = np.nanmean(mmap.co[sl][int(b*dr):int((b+h)*dr),int(l*dc):int((l+w)*dc),vsl],axis=(0,1))

            ax2.plot(mmap.v[vsl],ju.scale_ptp(line),'r',lw=1,**kwargs)



            ax2.set_ylim(ax2.get_ylim()[0],max(ax2.get_ylim()[1],.3))
            #ax2.set_axis_off()

            #ax.add_patch(mpl.patches.Rectangle([bl[0],bl[1]],w*dc,h*dr,transform=ax.transData,alpha=0.25))
    return fig


# Plot the KDE for a set of x,y values. No weighting code modified from
# http://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
def kdeplot(xp, yp, filled=False, ax=None, grid=None, bw=None, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    rvs = np.append(xp.reshape((xp.shape[0], 1)), yp.reshape((yp.shape[0], 1)), axis=1)

    kde = stats.kde.gaussian_kde(rvs.T)
    # kde.covariance_factor = lambda: 0.3
    # kde._compute_covariance()
    kde.set_bandwidth(bw)

    # Regular grid to evaluate kde upon
    if grid is None:
        x_flat = np.r_[rvs[:, 0].min() : rvs[:, 0].max() : 256j]
        y_flat = np.r_[rvs[:, 1].min() : rvs[:, 1].max() : 256j]
    else:
        x_flat = np.r_[0 : grid[0] : complex(0, grid[0])]
        y_flat = np.r_[0 : grid[1] : complex(0, grid[1])]
    x, y = np.meshgrid(x_flat, y_flat)
    grid_coords = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

    z = kde(grid_coords.T)
    z = z.reshape(x.shape[0], x.shape[1])
    if filled:
        cont = ax.contourf
    else:
        cont = ax.contour
    cs = cont(x_flat, y_flat, z, *args, **kwargs)
    return cs




def errorbar_fill(
    x=None,
    y=None,
    yerr=None,
    *args,
    ax=None,
    mid=True,
    color=None,
    fill_color=None,
    alpha=1,
    lw=1,
    ls="-",
    fmt=None,
    label=None,
    **kwargs,
):
    oldax = plt.gca()
    if ax is None:
        ax = oldax
    #plt.sca(ax)

    if mid:
        alpha_fill = alpha * 2
        if alpha_fill >= 1:
            alpha_fill = 1
    if color is None:
        color = ax.plot([],[])[0].get_color()
    if fill_color is None:
        fill_color = adjust_lightness(color,1.5)
    ax.fill_between(x, y - yerr, y + yerr, color=fill_color, alpha=alpha,label=label,**kwargs)
    if mid:
        ax.plot(x, y, color=color, alpha=alpha, lw=lw, ls=ls,**kwargs)
    #plt.sca(oldax)
    return None