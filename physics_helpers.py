
def thermal_v(T, mu=None, mol=None):
    """thermal_v(T,mu)
    get thermal velocity for a temperature & molecular mass mu

    Arguments:
        T {[float]} -- [temperature in kelvin]

    Keyword Arguments:
        mu {float} -- [mean molecular mass] (default: {1})
                mol ISM w/ He: 2.33
                ISM w/ He : 1.37
                12CO, 13CO,C18O = 18,19,20

    Returns:
        [type] -- [description]
    """

    return np.sqrt(constants.k_B * T * u.Kelvin / (mu * constants.m_p)).to("km/s").value


def virial(sig, mass, r):
    s = 1.33 * (sig * (u.km / u.s)) ** 2
    r = r * u.pc
    m = constants.G * (mass * u.Msun)
    return (s * r / m).si.value


def numdens(mass, radius):
    """number density from mass/radius
    assuming spherical symmetry

    Parameters
    ----------
    mass : float
        in solar masses
    radius : float
        in parsecs

    Returns
    -------
    float
        in cm^-3
    """
    mass = mass * u.solMass
    radius = radius * u.pc
    volume = (4 / 3) * np.pi * (radius ** 3)
    dens = (mass / volume) / (2.33 * constants.m_p)
    return dens.to(u.cm ** -3).value


def jeansmass(temp, numdens, mu=2.33):  # 12.03388 msun T=n=1
    """
    temp in K
    numdens in cm^-3
    mu is mean molecular weight [default: 2.33, ISM w/ Helium corr]
    returns Mjeans in solar masses
    .5 * (5 * kb / G)^3 * (3/4π) * (1/2.33 mp)^4 * T^3 / n
    """
    mj = (5 * constants.k_B / (constants.G)) ** 3
    mj *= 3 / (4 * np.pi)
    mj *= (1 / (mu * constants.m_p)) ** 4
    mj = mj * (temp * u.K) ** 3
    mj = mj * (u.cm ** 3 / numdens)
    mj = mj ** 0.5
    return mj.to(u.solMass).value

