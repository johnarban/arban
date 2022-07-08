import astropy.units as u
import astropy.constants as const
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits

square_beam = 1/np.sqrt(np.pi / (4 * np.log(2)))

def beam_area(BMAJ, BMIN=None, square = False):
    """
    Calculate the beam area in square arcsec
    assuming a gausian beam, unless square is set to True
    """
    if not isinstance(BMAJ,u.Quantity):
        BMAJ = BMAJ * u.arcsec
    if (BMIN is not None) and (not isinstance(BMIN,u.Quantity)):
        BMIN = BMIN * u.arcsec

    BMAJ = BMAJ.to(u.arcsec)
    BMIN = BMIN.to(u.arcsec) if BMIN is not None else BMAJ


    θsq = BMAJ * BMIN
    if square:
        return θsq

    Ω = np.pi * θsq / (4 * np.log(2))

    return Ω

def RJ_temp(S,freq):
    """
    S : source flux in Jy
    freq : frequency in GHz
    """

    if not isinstance(S,u.Quantity):
        S = S * u.Jy

    if not isinstance(freq,u.Quantity):
        freq = freq * u.GHz

    λ = (const.c / freq).si
    coeff = 2 * const.k_B / λ**2
    return (S / coeff).to(u.K)

def RJ(T,freq):
    """
    T : source brightness in K
    freq : frequency in GHz
    """

    if not isinstance(T,u.Quantity):
        T = T * u.K

    if not isinstance(freq,u.Quantity):
        freq = freq * u.GHz

    λ = (const.c / freq).si
    coeff = 2 * const.k_B / λ**2
    return (T * coeff).to(u.Jy)



def convert_K_to_Jy(Tb, res_arcsec, freq_GHz,unit='mJy',manual=False, square=False):
    """
    Convert brightness temperature (K) to flux density (Jy/beam or Jy/pixel)
    """

    Ω = beam_area(res_arcsec,square=square)
    ν = freq_GHz * u.GHz if not isinstance(freq_GHz,u.Quantity) else freq_GHz
    if not isinstance(Tb,u.Quantity):
        Tb = Tb * u.K

    if manual:
        λ = (const.c / ν).si
        Snu = 2 * const.k_B * Ω * Tb / λ**2
        return Snu.si.to(unit,equivalencies=u.dimensionless_angles())
    else:
        equiv = u.brightness_temperature(ν,Ω)
        return Tb.to(unit,equivalencies=equiv)

def convert_Jy_to_K(Snu, res_arcsec, freq_GHz,unit='K',manual=False,square=False):
    """
    Convert flux density (Jy/beam or Jy/pixel) to brightness temperature (K)
    """
    Ω = beam_area(res_arcsec,square=square)
    ν = freq_GHz * u.GHz if not isinstance(freq_GHz,u.Quantity) else freq_GHz


    if not isinstance(Snu,u.Quantity):
        Snu = Snu * u.Jy

    if manual:
        λ = (const.c / ν).si
        Tb = Snu * λ**2 / (2 * const.k_B * Ω)
        return Tb.si.to(unit,equivalencies=u.dimensionless_angles())
    else:
        equiv = u.brightness_temperature(ν,Ω)
        return (Snu).to(unit,equivalencies=equiv)


def convert_JyBeam_to_JyPixel(S_Jy_per_beam, BMAJ, BMIN, arcsec_per_pix):
    """
    S_Jy_per_beam : source flux in Jy/beam
    BMAJ, BMIN = beam major/minor axis in arcsec
    arcsec_per_pixel = pixel scale in arcsec
    """
    # gaussian beam
    beam_arcsec2 = BMAJ * BMIN * np.pi / (4*np.log(2)) # in arcsec**2

    pixels_per_beam = beam_arcsec2 / (arcsec_per_pix)**2

    S_Jy_per_pix = S_Jy_per_beam / pixels_per_beam

    return S_Jy_per_pix


def convert_JyPixel_to_JyBeam(S_Jy_per_pixel, BMAJ, BMIN, arcsec_per_pix):
    """
    S_Jy_per_pixel : source flux in Jy/pixel
    BMAJ, BMIN = beam major/minor axis in arcsec
    arcsec_per_pixel = pixel scale in arcsec
    """
    # gaussian beam
    beam_arcsec2 = BMAJ * BMIN * np.pi / (4*np.log(2)) # in arcsec**2

    pixels_per_beam = beam_arcsec2 / (arcsec_per_pix)**2

    S_Jy_per_beam = S_Jy_per_pixel  * pixels_per_beam

    return S_Jy_per_beam


def convert_deltav_deltaf(v_kms = 5, freq_ghz=230,unit='MHz'):
    f = freq_ghz * u.GHz
    Δv = v_kms * u.km/u.s
    Δf = (f * Δv/const.c).to(unit)
    return Δf

def convert_deltaf_deltav(df_mhz = 3.8, freq_ghz=230,unit='km/s'):
    f = freq_ghz * u.GHz
    Δf = df_mhz * u.MHz
    Δv = (const.c * Δf / f).to(unit)
    return Δv

def TP_sens(delta_nu,tint,nant=3):
    ηap = 0.69
    Aeff = ηap * np.pi * (5.6336616 * u.m)**2
    ηq = 0.96
    ηc = 0.88
    Δν = delta_nu * u.MHz
    t = tint * u.s
    n_p = 2
    wr = 1
    Tsys = 114 * u.K
    a = 2 * const.k_B * Tsys
    b = ηq * ηc * Aeff * np.sqrt(nant * n_p * Δν.si * t)
    return (a/b)

def siggas_to_ico(sgas,aco=4.3):
    aco= aco * u.Msun * (u.K * u.km/u.s)**-1 / u.pc**2
    sig = sgas * (u.Msun/u.pc**2)
    return (sig/aco).to(u.K * u.km/u.s)

def get_spec_ax(header3d,N=None):
    if not isinstance(header3d, WCS):
        wcs = WCS(header3d)
    else:
        wcs = header3d
    if wcs.has_spectral:
        # wcs.spectral :: spectral axies
        # np.indices(wcs.spectral.array_shape)[0] get indices for spectral axis
        # wcs.spectral.array_shape is the shape of the spectral axis
        # wcs.spectral.array_index_to_world with units.
        if wcs.spectral.array_shape is None:
            array_shape = (N,)
        else:
            array_shape = wcs.spectral.array_shape
        return wcs.spectral.array_index_to_world(np.indices(array_shape)[0])



def get_beam(filen):
    hdul = fits.open(filen)
    header = hdul[0].header
    BMAJ = header['BMAJ']
    BMIN = header['BMIN']
    BPA = header['BPA']
    hdul.close()
    return BMAJ, BMIN, BPA





def mad(X, astropy=True, axis=None):
    if  astropy:
        return mad_std(X,axis=axis,ignore_nan=True)
    else:
        return 1.482602218505602 * np.nanmedian(np.abs(X - np.nanmedian(X, axis=axis)), axis=axis)

