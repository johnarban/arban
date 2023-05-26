import matplotlib.colors as mcolor
from colorsys import rgb_to_hls,hls_to_rgb #builtin
import numpy as np


def color_array(arr, alpha=1):
    """ take an array of colors and convert to
    an RGBA image that can be displayed
    with imshow
    """
    img = np.zeros(arr.shape + (4,))
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            c = mcolor.to_rgb(arr[row, col])
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
        rgb = mcolor.to_rgb(rgb)

    if invert:
        img = 1 - img
    im2[:, :, 3] = img * alpha
    r, g, b = rgb
    im2[:, :, 0] = r
    im2[:, :, 1] = g
    im2[:, :, 2] = b

    #     if ax is None:
    #         ax = plt.gca()
    #     plt.sca(ax)
    #     plt.imshow(im2)

    return im2




def invert_color(ml, *args, **kwargs):
    rgb = mcolor.to_rgb(ml)
    hsv = mcolor.rgb_to_hsv(rgb)
    h, s, v = hsv
    h = 1 - h
    s = 1 - s
    v = 1 - v
    return mcolor.to_hex(mcolor.hsv_to_rgb((h, s, v)))


def icol(*args, **kwargs):
    return invert_color(*args, **kwargs)



def color_hue_shift(c, shift=1):
    c = mcolor.to_rgb(c)
    h, s, v = mcolor.rgb_to_hsv(c)
    h = h + shift % 1
    return mcolor.to_hex(mcolor.hsv_to_rgb((h, s, v)))




def adjust_lightness(color, amount=0.5):
    # brighter : amount > 1
    # https://stackoverflow.com/a/49601444
    try:
        c = mcolor.cnames[color]
    except Exception:
        c = color
    c = rgb_to_hls(*mcolor.to_rgb(c))
    return hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
