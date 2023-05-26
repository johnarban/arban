from math import sqrt, floor, ceil

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
        ceily = int(ceil(sqrt(n)))
        rows = cols = ceily
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



def get_xylim(ax=None):
    if ax is None:
        ax = plt.gca()
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    return xlim, ylim


def set_xylim(xlim=None, ylim=None, ax=None, origin=None):
    """set xylims with tuples
    xlim: tuple of x axis limits
    ylim: tuple of y axis limits
    origin: sometimes you just want to change the origin
            so you can keep the axis limits the same
            but just change origin


    """
    if ax is None:
        ax = plt.gca()

    if xlim is None:
        xlim = ax.get_xlim()
    if ylim is None:
        ylim = ax.get_ylim()

    if isinstance(xlim, tuple):
        xlim = list(xlim)
    if isinstance(ylim, tuple):
        ylim = list(ylim)
    if origin is not None:
        if origin is True:
            if ax.get_xaxis().get_scale()[:3] != "log":
                xlim[0] = 0
            if ax.get_yaxis().get_scale()[:3] != "log":
                ylim[0] = 0
        else:
            xlim[0] = origin[0]
            ylim[0] = origin[1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return tuple(xlim), tuple(ylim)



def plot_to_origin(ax=None):
    if ax is None:
        ax = plt.gca()

    ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])

    return None