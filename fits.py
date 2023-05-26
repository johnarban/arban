"""
routines for reading in 
header values from fits 
files without astropy
"""

from astropy.io.fits import open as fits_open

# Get header value from a FITS file
def get_header_card_value(line):
    if b'/' in line:
        # only stuff after the last / is comment
        *card_value, comment = line.split(b'/')
        card_value = b'/'.join(card_value)
    else:
        card_value = line.split(b'/')[0].strip()
        comment = b''

    # history items may have equal signs in them
    if (b'=' in line) & (line[:7]!=b'HISTORY') & (line[:7]!=b'COMMENT'):
        card,value = card_value.split(b'=')
    else:
        card,value = card_value[:7],card_value[7:]

    card = card.strip().decode()
    value = value = value.strip().decode().replace("'", "")
    comment = comment.strip().decode()
    return card,value,comment

def getheader_val_fast(fname, card_name):
    """
    Pure python function to get header values
    from a fits file

    """
    with open(fname, 'rb') as f:
        history = ''
        line = ''
        while line.strip() != b'END':
            line = f.read(80)
            card, value, comment = get_header_card_value(line)
            if (card=='HISTORY') | (card=='COMMENT'):
                if card == card_name:
                    history += '\n'+value
            elif card == card_name:
                try:
                    return float(value)
                except ValueError:
                    return value

        return history


# get header value from a FITS file
def getheader_val(filename,card):
    """
    Pure python function to get header values
    from a fits file

    """
    with fits_open(filename) as hdul:
        header = hdul[0].header
        return header[card]
    
    
def nside2resol(nside,):
    """nside2resol: get healpix resolution from Nsides

    Parameters
    ----------
    nside : int
        Nsides

    Returns
    -------
    Astropy Quantity [arcmin]
        resolution in arcmin
    """
    resol = 60 * (180 / np.pi) * np.sqrt(np.pi / 3) / nside
    return resol * u.arcmin