import numpy as np
from scipy import stats, special, interpolate
import matplotlib.pyplot as plt
from weighted import quantile
import george
from astropy.io import fits
import corner #dfm corner.py needed for quantiles

from utils import *

class MedianSED():

    # Initilize class
    def __init__(self, sed, wavelength, error = [None], valid = [None], which = [None], nl = [None]):
        
        
        if isinstance(sed, type([]) ):
            #print 'Fixed sed type'
            sed = np.asarray(sed)
        if isinstance(wavelength, type([]) ):
            #print 'Fixed wavelength type'
            wavelength = np.asarray(wavelength)
        if isinstance(error, type([]) ):
            #print 'Fixed error type'
            error = np.asarray(error)
        if isinstance(valid, type([]) ):
            #print 'Fixed valid type'
            valid = np.asarray(valid)
        if len(nl) == 1:
            #print 'Fixed noorm_limits'
            self.nl = np.asarray([20,25])
            
        elif isinstance(nl,type([]) ):
            self.nl = np.sort(np.asarray(nl))
        
        
        self.sort = np.argsort(wavelength) # data wrangling #rid of W1, W2  
        #self.sort = np.argsort(wavelength) 
        self.wavelength = wavelength[self.sort]
        self.wav = self.wavelength # alias
        self.frequency = 2.99792458e-14/self.wavelength
        self.nu = self.frequency
    
        if error[0] is not None:
            self.error = error[:,self.sort] # sort error
        else:
            self.error = None
        if valid[0] is not None:
            self.valid = valid[:,self.sort] == 1 # sort error
        else:
            self.valid = True
        
        self.sed = sed[:,self.sort] #sorted SEDS
        
        
        self.valid = self.validate() & self.valid  # set valid data mask
        normalization_factor = self.normalization_factor()
        self.normalized_sed = self.sed / normalization_factor # normalize the SEDs by <22-70 micron flux>
        if self.error is not None:
            self.normalized_error = self.error / normalization_factor # normalize the SEDs by <22-70 micron flux>
            self.weights = 1./(self.normalized_error/self.normalized_sed)**2 # log space weights
        
        self.median = self.get_quantile(0.5) # calculate median
        return None
   
   
    def normalization_factor(self):
        x = self.wavelength
        norm_region = np.where((x>self.nl[0]) & (x<self.nl[1]))[0]
        #print x[norm_region]
        norm = np.nanmean(self.sed[:,norm_region],axis=1) #[7-13) are 22 22 24 24 70 70
        #norm = np.nanmean(self.sed[:,3:4]/self.wav[3:4],axis=1) 
        return norm[:,None]

    def get_quantile(self, q = 0.5):
        # DEF: Calculate median SED
        vs = self.normalized_sed.copy()
        vs[~self.valid] = np.nan   # set invalid elements to NaN
        if self.error is None:
            return np.nanpercentile(vs,q*100, axis=0)
        else:
            keep = np.isfinite(vs)
            quantile_sed = []
            for i in xrange(len(self.wav)):
                try:
                    quantile_sed.append(quantile(vs[keep[:,i],i],self.weights[keep[:,i],i], q))
                except:
                    print 'Error:  computing un-weighted quantile'
                    quantile_sed.append(np.nanpercentile(vs[keep[:,i],i],q*100))
            #quantile_sed = np.asarray([quantile(vs[keep[:,i],i],self.weights[keep[:,i],i], q) for i in xrange(len(self.wav))])
            return np.asarray(quantile_sed)
        

    def validate(self):
        #nonfinite = np.isfinite(self.sed)  # Identify nans and infinities
        zeros = np.isfinite(np.log(self.sed))   # Identify zero fluxes
        if self.error is None:
            zero_error = True
        else:
            zero_error = np.isfinite(np.log(self.error))
        return zeros & zero_error    # Return valid array

    def xy(self):
        return self.wavelength, self.median
        
    def plot_medsed(self, sig = 1.,ax = None,colors = None, shapes = None):
        if ax is None:
            ax = plt.gca()
        if colors is None:
            colors=['k']*len(self.wav)
        if shapes is None:
            shapes=['o']*len(self.wav)
        drop = np.isfinite(np.log10(self.median))
        p = special.erf(sig)
        ax.fill_between(self.wav[drop], \
            (self.get_quantile(p)/self.wav)[drop], \
            (self.get_quantile(1.-p)/self.wav)[drop],color='0.7')
        for i in xrange(len(self.wav)):
            ax.plot([self.wav[i]],[(self.median/self.wav)[i]],colors[self.sort[i]]+shapes[self.sort[i]])
        ax.loglog()
        
        return ax
        
    
    def GP_median_sed(self,kernel = george.kernels.ExpSquaredKernel(3)):
        x = self.wav
        y = self.median
        #yerr = np.nanmean(np.abs(self.normalized_sed - self.median),axis=0)
        #yerr = np.sqrt(np.nansum(self.normalized_error**2,axis=0))
        yerr = np.nanmedian(np.abs(self.normalized_error),axis=0)
        t = np.logspace(np.log10(.1),np.log10(200),30)
        gp = george.GP(kernel)
        gp.compute(np.log10(x),yerr/y)
        mu, cov = gp.predict(np.log(y),np.log10(t))
        std = np.sqrt(np.diag(cov))
        return t, np.exp(mu), np.exp(std)
    
    
    def plot_gp(self, ax = None):
        if ax is None:
            ax = plt.gca()
        
        t, mu, std = self.GP_median_sed()
        
        ax.plot(t, mu)
        
        return ax
        
    def alpha(self,x,y,e=None, range = (0,1000)):
        rng = np.sort(range)
        rng = np.where((x>rng[0]) & (x<rng[1]))[0]
        logx = np.log(x)[rng]
        logy = np.log(y)[rng]
        
        logx = logx[np.isfinite(logy)]
        logy = logy[np.isfinite(logy)]
        
        if e is None:
            m, b = np.polyfit(logx,logy,1)
            return m, b
        else:
            e = e/y
            m, b, cov = np.polyfit(logx, logy, 1, w = 1/e**2, cov = True)
            return m, b, cov