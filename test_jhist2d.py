import numpy
color = ['match', None, 'purple','purple_r']
cmap = ['Reds', None]
data_color = ['match',None,'b']
contour_color = [None,'g',['g','r']]
fkw = [None,{'cmap':'turbo'}]
nkw = [None,{'cmap':'magma'},{'colors':['pink','w']}]
dkw = [None,{'color':'orange'}]
x, y = np.random.multivariate_normal((0,0),cov=[[-1,0],[3,-2]],size=1000).T
i=0
fig, axs = plt.subplots(ncols=6,nrows=12,figsize=(6*6,4*12))
for c in color:
    for cm in cmap:
        for dc in data_color:
            for cc in contour_color:
                #print(f'{c}:{cm}:{dc}:{cc}')
                try:
                    ax, *plots = ju.jhist2d(x,y,'o',
                                         plot_datapoints=True,
                                         plot_contour=True,
                                         plot_contourf=True,
                                         levels=[.1,.25,.5,.75,.90],
                                         bins=100, smooth=5,zorder=10,
                                         color=c,
                                         cmap = cm,
                                         data_color = dc,
                                         contour_color = cc,
                                         ax=axs.ravel()[i],debug=False)
                except:
                    ax, *plots = ju.jhist2d(x,y,'o',
                                         plot_datapoints=True,
                                         plot_contour=True,
                                         plot_contourf=True,
                                         levels=[.1,.25,.5,.75,.90],
                                         bins=100, smooth=5,zorder=10,
                                         color=c,
                                         cmap = cm,
                                         data_color = dc,
                                         contour_color = cc,
                                         ax=axs.ravel()[i],debug=True)
                i+=1
                ax.set_title(f'c-{c} cm-{cm} dc-{dc} cc-{cc}')