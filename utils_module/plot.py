def annotate(text, x, y,ax=None,
            horizontalalignment='center',
            verticalalignment='center',
            ha = None,
            va = None,
            transform='axes',
            color='k',
            fontsize=9,
            facecolor='w',
            alpha=0.75,
            bbox=dict(), **kwargs):

    if ax is None:
        ax = plt.gca()

    horizontalalignment = ha or horizontalalignment
    verticalalignment = va or verticalalignment

    if transform=='axes':
         transform = ax.transAxes
    elif transform == 'data':
        transform = ax.transData
    bbox1 = dict(facecolor=facecolor, alpha=alpha)
    bbox.update(bbox1)
    text = ax.text(x,y,text,horizontalalignment=horizontalalignment,
                                verticalalignment=verticalalignment,
                                transform=transform,
                                color=color,
                                fontsize=fontsize,
                                bbox=bbox, **kwargs)
    return text

