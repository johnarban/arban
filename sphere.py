## Functions relating to spherical geometry.
# The notation in this is consistent with
# Wolfram Mathworld (hopefully, sometimes, maybe)
# also will use lat, lon sometimes

# Area of a square patch on a sphere

import numpy as np


def Square_Patch(lat, lon, delta_lat, delta_lon, r=1.,indeg=False):
    """Find the area of a square patch on a sphere

    Parameters
    ----------
    lat : latitude (center of patch)
        angle of latitude, spans -pi/2 to pi/2
    lon : longitude (center of patch)
        angle of longitude, spance 0, 2 pi
    delta_lat : change in latitude

    delta_lon : change in longitude

    r : float, optional
        distance, by default 1

    Returns
    -------
    area : in same type as entered
        mathematically correct area of a spherical patch
    """
    if indeg:
        lat = np.deg2rad(lat)
        lon = np.deg2rad(lon)
        delta_lat = np.deg2rad(delta_lat)
        delta_lon = np.deg2rad(delta_lon)

    return 2 * r**2 * delta_lon * np.sin(delta_lat/2) * np.cos(lat)

