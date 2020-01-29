import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors


def clean_color(color, reverse=False):
    if isinstance(color, str):
        if color[-2:] == '_r':
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

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "density_cmap", [color, end])
    if reverse:
        return cmap.reversed()
    else:
        return cmap


def contour_level_colors(cmap, levels, vmin=None, vmax=None):
    vmin = vmin or 0
    vmax = vmax or max(levels)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    #offset = np.diff(levels)[0] * .5
    #colors = mpl.cm.get_cmap(cmap)(norm(levels-offset))
    levels = np.r_[0, levels]
    center_levels = 0.5 * (levels[1:] + levels[:-1])
    return mpl.cm.get_cmap(cmap)(norm(center_levels))
