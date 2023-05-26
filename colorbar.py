from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt

def _normalize_location_orientation(location):
    loc_settings = {
        "left": {
            "location": "left",
            "orientation": "vertical",
            "anchor": (1.0, 0.5),
            "panchor": (0.0, 0.5),
            "pad": 0.10,
        },
        "right": {
            "location": "right",
            "orientation": "vertical",
            "anchor": (0.0, 0.5),
            "panchor": (1.0, 0.5),
            "pad": 0.05,
        },
        "top": {
            "location": "top",
            "orientation": "horizontal",
            "anchor": (0.5, 0.0),
            "panchor": (0.5, 1.0),
            "pad": 0.05,
        },
        "bottom": {
            "location": "bottom",
            "orientation": "horizontal",
            "anchor": (0.5, 1.0),
            "panchor": (0.5, 0.0),
            "pad": 0.15,
        },
    }
    
    orientation = loc_settings[location.lower()]["orientation"]

    return loc_settings[location.lower()], orientation


def get_cax(ax=None, position=None, frac=0.03, pad=0.02):
    """get a colorbar axes of the same height as current axes
    position: "left" "right" ( vertical | )
              "top"  "bottom"  (horizontal --- )

    """
    if ax is None:
        ax = plt.gca()

    size = f"{frac*100}%"
    divider = make_axes_locatable(ax)

    if position is None:
        position = 'right'

    if position is 'bottom':
        pad += 0.15

    if position is 'right':
        left = 1 + pad
        width = frac
        bottom = 0.0
        height = 1.0
    elif position is 'bottom':
        left = 0.0
        width = 1.0
        bottom = 0 - pad
        height = frac*2
    else:
        raise ValueError(f"position {position} not supported")

    p = [left, bottom, width, height]
    cax = ax.inset_axes(p, transform=ax.transAxes)

    plt.sca(ax)
    return cax


def colorbar(mappable=None, cax=None, ax=None, size=0.03, pad=0.05, position=None, orientation='vertical', **kw):
    """wrapper for pyplot.colorbar.

    """
    if ax is None:
        ax = plt.gca()
    if orientation[0].lower()=='h':
        pos = 'bottom'
    elif orientation[0].lower()=='v':
        pos = 'right'

    if cax is None:
        cax = get_cax(ax=ax, frac=size, pad=pad, position=position)
    elif (cax == 'inset') & (orientation[0].lower()=='h'):
        cax = ax.inset_axes([0.2,.1,0.6,0.05])
    elif (cax == 'inset') & (orientation[0].lower()=='v'):
        cax = ax.inset_axes([0.85,.1,0.05,0.8])

    ret = plt.colorbar(mappable, cax=cax, ax=ax, **kw)
    return ret
