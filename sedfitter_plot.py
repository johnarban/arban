from astropy.io import ascii,fits
from sedfitter import sed, extinction,fit_info
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.style.use('sns_paper')
import numpy as np
from scipy.integrate import cumtrapz,simps
import astropy.units as u
from astropy.analytic_functions import blackbody_lambda as bl
from astropy.analytic_functions import blackbody_nu as bn
%matplotlib inline


mids = ['333','740','1246','1283','1512','2393','2561','2707','2748']
#Get FitInfo for all fits
fin = fit_info.FitInfoFile('/Users/johnlewisiii/Charlie/sedfitter_cf/OrionProtostars/low.fitinfo','r')
#iterable fin to list
info = np.asarray([f for f in fin])

def dosomething(source_id,):
    #select SED of BEST FIT for M#
    source_index = np.where([source_id==inf.source.name for inf in g])[0]
    
    filen = info[source_index].model_name[0]+'_sed.fits.gz'
    av = info[thisone].av[0]
    #OPEN as HDU for Stellar Flux
    hdulist = fits.open('/Users/johnlewisiii/PythonStuff/sedfitter/models_r06/seds/%s/%s'%(filen[:5],filen))
    #use SED class for everything else
    s=sed.SED.read('/Users/johnlewisiii/PythonStuff/sedfitter/models_r06/seds/%s/%s'%(filen[:5],filen))
    #load all individual headers
    wav = hdulist[1].data.field('WAVELENGTH') * sed.sed.UNIT_MAPPING[hdulist[1].columns[0].unit]
    nu = hdulist[1].data.field('FREQUENCY') * sed.sed.UNIT_MAPPING[hdulist[1].columns[1].unit]
    star = hdulist[1].data.field('STELLAR_FLUX') * sed.sed.UNIT_MAPPING[hdulist[1].columns[2].unit]
    ap = hdulist[2].data.field('APERTURE') * u.AU #sed.sed.UNIT_MAPPING[hdulist[2].columns[0].unit]
    flux = hdulist[3].data.field('TOTAL_FLUX') * sed.sed.UNIT_MAPPING[hdulist[3].columns[0].unit]
    error = hdulist[3].data.field('TOTAL_FLUX_ERR') * sed.sed.UNIT_MAPPING[hdulist[3].columns[1].unit]
    #extinction_law
    ext_law = g[0].meta.extinction_law.get_av(wav)
    #--------- Start Plotting -----------#
    plt.subplot(1,9,i+1)
    #plot SED and stellar flux
    plt.plot(wav,flux[0],'b')
    plt.plot(wav,flux[0]*(10**(ext_law * av)),'k')
    plt.plot(wav,star,'r')
    rat = (2.95*u.Rsun.to(u.cm)/u.kpc.to(u.cm))**2
    plt.plot(wav,np.pi*nu*bn(nu,3000)*rat,'y')
    plt.ylim(1e-16,1e-10)
    plt.loglog()
    plt.title(source_id)
    #calculate flux by temperature of star
    #(simps(np.pi*bn(nu,3032)*rat,nu) * flux.unit * (4*np.pi) * s.distance**2).to(u.Lsun)
    #print (simps(star/nu,nu) * flux.unit * (4*np.pi) * s.distance**2).to(u.Lsun)
    #print (simps(flux[49]/nu,nu) * flux.unit * (4*np.pi) * s.distance**2).to(u.Lsun)
#plt.savefig('/Users/johnlewisiii/Desktop/TTS.png',dpi=400)