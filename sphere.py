## Functions relating to spherical geometry.
# The notation in this is consistent with
# Wolfram Mathworld (hopefully, sometimes, maybe)
# also will use lat, lon sometimes

# Area of a square patch on a sphere
def Square_Patch(lat, lon, delta_lat, delta_lon, r=1.):
    """Find the area of a square patch on a sphere

    Parameters
    ----------
    lat : latitude
        angle of latitude, spans -pi/2 to pi/2
    lon : longitude
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
    return 4 * r**2 * delta_lon * np.sin(delta_lat) * np.cos(lat)