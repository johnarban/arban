import matplotlib.colors as colors
import numpy as np

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
