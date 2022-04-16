# Functions to help with ALMA data
from astropy.wcs import WCS
from tqdm import tqdm_notebook
from numpy import ndarray
from astroquery.simbad import Simbad
from astroquery.ipac.ned import Ned
import re
import numpy as np
import astropy.units as u

# Generate ALMA archive download scripts


def generate_download_script(
    list_of_urls,
    download_dir,
    download_name,
    download_template="downloadRequest_template.sh",
    download_script="downloadRequest.sh",
):
    """
    edit the download template to
    download


    """

    with open(download_template, "r") as f:
        contents = f.readlines()

    index = contents.index("listoffiles\n")
    contents.pop(index)
    for value in list_of_urls[::-1]:
        contents.insert(index, value + "\n")

    with open(download_script, "w") as f:
        contents = "".join(contents)
        f.write(contents)


class BoolArray(ndarray):
    def __new__(*args, **kwargs):

        obj = super().__new__(*args, **kwargs)
        return obj

    def __bool__(self):
        return self.size > 0

    def __round__(self, *args, **kwargs):
        return self.round(*args, **kwargs)


def get_NED_name(name):
    """
    Get the NED name of an object
    """
    # loop over name if name is iterable
    if hasattr(name, "__iter__") and not isinstance(name, str):
        return np.asarray([get_NED_name(n) for n in name])

    try:
        q = Ned.query_object(name)
        t = q["Object Name"].data[0]
    except:
        t = ""

    return t


def get_SIMBAD_name(name):
    """
    Get the SIMBAD name of an object
    """
    # loop over name if name is iterable
    if hasattr(name, "__iter__") and not isinstance(name, str):
        return np.asarray([get_SIMBAD_name(n) for n in name])

    try:
        q = Simbad.query_object(name)
        t = q["MAIN_ID"].data[0]
    except:
        t = ""

    return t


def get_ned_simbad_name(name):
    """
    Get the NED and SIMBAD names of an object
    """
    # loop over name if name is iterable
    if hasattr(name, "__iter__") and not isinstance(name, str):
        return np.asarray([get_ned_simbad_name(n) for n in name])

    if name.strip() == "":
        return ["", ""]

    t = get_NED_name(name)
    if t == "":
        t = "failed NED"

    s = get_SIMBAD_name(name)
    if s == "":
        s = "failed SIMBAD"

    return [t, s]


def get_name(name, return_redshift=False, resolve=True, debug=False):
    oldname = name
    name = " ".join(re.split("_", name))  # _'s are space place holders
    # if the first value is a digit, that digit is a modifier. remove it

    # split runs of letters and digits
    s = list(filter(None, re.split("(\d+)|(-)", name)))

    # remove "#-" from beginning of names
    if s[0].isdigit() and not name.isdigit():
        if s[1] == "-":
            s = s[2:]
    name = "".join(s)

    # Handle NGC designations

    # chang N### to NGC ####
    if len(name) > 1:
        if (name[0] is "N") & (name[1].isdigit()):
            name = "LMC"  # 'NGC' + name[1:]
    # BAD CHOICE - N### might be LMC/SMC

    if name == "30 doradus":
        name = "LMC"

    if "NGC" in name:
        name = "".join(name.split())

        name = name.split("Field")[0]
        name = name.split("HubbleV")[0]

        s = list(filter(None, re.split("(\d+)", name)))
        # if len(s)>1:
        #     if not s[-1].isdigit():
        #         s = ''.join(s[:-1])
        #     else:
        #         s = ''.join(s)
        # else:
        s = "".join(s)
        s = list(filter(None, re.split("(\d+)", s)))
        name = " ".join(s)
        # print(oldname,' -->  ',name)
        # s = name.split('-')
        # if len(s)>1:
        #     print(s)
        #     name = ' '.joino(s[:2])

    # group LMC, SMC sub regions into one
    if "SMC" == name[:3]:
        name = "SMC"
    if "LMC" == name[:3]:
        name = "LMC"
    if "WLM" == name[:3]:
        name = "DDO 221"
    if "ESO" == name[:3]:
        name = "".join(name.split())
        name = name.replace("ESO", "ESO ")

    if "Target" in name:
        name = name[: name.index("Target")]

    if debug:
        print(oldname, " -->  ", name)
        # pass

    if not resolve:
        if return_redshift:
            return name, np.nan
        else:
            return name

    if (name == "LMC") or (name == "SMC"):
        if return_redshift:
            return name, np.nan
        else:
            return name
    try:
        # print('try ned')
        q = Ned.query_object(name)
        t = q["Object Name"].data[0]  # + ' (NED)'
        z = float(q["Redshift"].data[0])
    except:
        # print('try simbad')
        q = Simbad.query_object(name)
        if q is not None:
            t = q["MAIN_ID"].data[0]  # + ' (Simbad)'
            z = np.nan
        else:
            t = ""
            z = np.nan

    if return_redshift:
        return t, z
    else:
        return t


def get_names(names, return_redshift=True, resolve=True, debug=False):
    """
    Get the names of objects in the list
    """
    out = []
    for name in tqdm_notebook(names):
        out.append(get_name(name, return_redshift=return_redshift, resolve=resolve, debug=debug))
    if return_redshift:
        names, z = list(zip(*out))
        return np.asarray(names), np.asarray(z)
    else:
        return np.asarray(out)


def sesameURL(name, ned=True, simbad=True):
    name = name.replace(" ", "+")
    url = f"http://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oxp/~"
    if ned:
        url += "N"
    if simbad:
        url += "S"
    if ned and simbad:
        url += "A"  # get both results

    url += f"?{name}"

    return url


# pd.read_xml(requests.get(sesameURL('Arp 299')).text,xpath='.//Resolver',parser='etree')

# adjust astropy fits pixel scale to adjust from one distance to another


def adjust_pixel_scale(header, old_distance, new_distance):
    """
    Adjust the pixel scale of a fits image to adjust from one distance to another
    """

    # get scaling factor
    scale_factor = old_distance / new_distance
    header2 = header.copy()
    # set the new pixel scale
    header2["CDELT1"] = header["CDELT1"] * scale_factor
    header2["CDELT2"] = header["CDELT2"] * scale_factor

    return header2


def method_new_distance_new_physical_scale_or_resolution(
    header,
    pixel_scale,
    resolution,
    distance,
    new_distance=None,
    observing_resolution=None,
    new_resolution=None,
    new_physical_resolution=None,
):
    """
    method_new_distance_new_physical_scale_or_resolution _summary_

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        _description_
    pixel_scale : float
        pixel scale from header
    resolution : resolution in arcsec

    distance : distance in Mpc

    new_distance : new distance in Mpc, optional

    Result options (must give one)
    observing_resolution : [arcsec], optional
       use if you want to observe target at given resolution, by default None
    new_resolution : [arcsec], optional
        use if you want to examine your system at a given angular resolution, by default None
    new_physical_resolution : [pc], optional
        use if you want to examine your system at a given physical resolution, by default None

    Returns
    -------
    fwhm : float
        fwhm of gaussian required for convolution to desired state
    header : astropy.io.fits.header.Header
        header (modified if necessary)

    """


    if new_distance is not None:
        scale = new_distance / distance
        header2 = adjust_pixel_scale(header.copy(), distance, new_distance)
    else:
        scale = 1
        new_distance = distance
        header2 = header.copy()

    physical_resolution = np.deg2rad(resolution / 3600) * distance * 1e6

    print("=======================")
    print("Original System")
    print(f"native pixel scale {pixel_scale:0.2f} arcsec")
    print(f"native resolution {resolution:0.2f} arcsec")
    print(f"native physical resolution {physical_resolution:0.2f} pc")
    print(f"distance {distance:0.2f} Mpc")
    print("\n")
    pixel_scale = pixel_scale / scale  # make pixels smaller if new_distance is larger
    resolution = resolution / scale
    physical_resolution = np.deg2rad(resolution / 3600) * new_distance * 1e6  # this should not change

    if scale != 1:
        print("Moved System")
        print(f"moved pixel scale {pixel_scale:0.2f} arcsec")
        print(f"moved resolution {resolution:0.2f} arcsec")
        print(f"moved physical resolution {physical_resolution:0.2f} pc (this should not change)")
        print(f"distance {new_distance:0.2f} Mpc")
    else:
        print("System is not moved")

    print("\n")
    print("New System")
    print(f"scale factor: {scale:0.3f}")
    if observing_resolution is not None:
        print("new resolution is given by observing resolution")
        new_resolution = observing_resolution
        convolution_fwhm_pixel = observing_resolution / pixel_scale
    elif new_resolution is not None:
        print("new resolution is {:0.2f} arcsec as specified by user".format(new_resolution))
        convolution_fwhm_pixel = (np.sqrt(new_resolution ** 2 - resolution ** 2) / pixel_scale)
    elif new_physical_resolution is not None:
        new_resolution = new_physical_resolution / (new_distance * 1e6)
        new_resolution = np.rad2deg(new_resolution) * 3600
        if new_resolution < resolution:
            print('new resolution is smaller than original resolution')
            print('returning fwhm=0 and header (modified if necessary)')
            return 0, header2
        convolution_fwhm_pixel = np.sqrt(new_resolution ** 2 - resolution ** 2) / pixel_scale
        print("new resolution is {:0.2f} arcsec".format(new_resolution))
    else:
        raise ValueError(
            "need to specify new_distance, new_resolution or new_physical_resolution"
        )

    print(f"new physical resolution (calculated) {np.tan(np.deg2rad(new_resolution/3600)) * new_distance * 1e6:0.2f} pc")
    print(f"convolution_fwhm_pixel {convolution_fwhm_pixel:0.2f} pixels")

    return convolution_fwhm_pixel, header2


# short hand for the above function
def move_target(
    header,
    pixel_scale,
    resolution,
    distance,
    new_distance=None,
    observing_resolution=None,
    new_resolution=None,
    new_physical_resolution=None,
):
    """
    move_target(
    header,
    pixel_scale,
    resolution,
    distance,
    new_distance=None,
    observing_resolution=None,
    new_resolution=None,
    new_physical_resolution=None
    )

    Parameters
    ----------
    header : astropy.io.fits.header.Header
        _description_
    pixel_scale : float
        pixel scale from header
    resolution : resolution in arcsec

    distance : distance in Mpc

    new_distance : new distance in Mpc, optional

    Result options (must give one)
    observing_resolution : [arcsec], optional
       use if you want to observe target at given resolution, by default None
    new_resolution : [arcsec], optional
        use if you want to examine your system at a given angular resolution, by default None
    new_physical_resolution : [pc], optional
        use if you want to examine your system at a given physical resolution, by default None

    Returns
    -------
    fwhm : float
        fwhm of gaussian required for convolution to desired state
    header : astropy.io.fits.header.Header
        header (modified if necessary)

    """
    return method_new_distance_new_physical_scale_or_resolution(
        header,
        pixel_scale,
        resolution,
        distance,
        new_distance=new_distance,
        observing_resolution=observing_resolution,
        new_resolution=new_resolution,
        new_physical_resolution=new_physical_resolution,
    )


def count(tb):
    """ print stats on pointings from MS"""
    direc = tb.getcol('DIRECTION')
    anten = tb.getcol('ANTENNA_ID')
    p=tb.getcol('TARGET')
    nuniq = np.unique(direc[:,0,:].T,axis=0).shape[0]
    # nuniq = np.unique(p[:,0,:].T,axis=0).shape
    nanten = len(np.unique(anten))
    npoint = len(anten)
    print(f'There are {npoint} pointings')
    print(f'There are {nanten} antennas')
    print(f'There are {npoint/nanten} pointings per antenna')
    print(f'There are {nuniq} unique positions')


