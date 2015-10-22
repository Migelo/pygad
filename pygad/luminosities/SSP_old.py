'''
Using Padova Simple stellar populations (SSPs) from Girardi
http://stev.oapd.inaf.it/cgi-bin/cmd
Marigo+ (2008), Girardi+ (2010)

Examples:
    >>> from ..snapshot import Snap
    >>> from ..environment import module_dir
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_320', physical=False)
    >>> s.load_double_prec = True
    >>> s.pos -= [34783.1, 35589.7, 33614.8]
    load block pos... done.
    convert to double precision... done.
    >>> s.to_physical_units()
    convert block pos to physical units... done.
    convert boxsize to physical units... done.
    >>> gal = s[s.r < '12.269 kpc']
    derive block r... done.
    >>> mbol = calc_mags(gal.stars)
    skipping Z_0_0150.dat
    skipping Z_0_0160.dat
    skipping Z_0_0170.dat
    skipping Z_0_0190.dat
    skipping Z_0_0200.dat
    skipping Z_0_0210.dat
    load block elements... done.
    convert to double precision... done.
    convert block elements to physical units... done.
    derive block H... done.
    derive block He... done.
    derive block metals... done.
    derive block Z... done.
    load block age... done.
    convert to double precision... done.
    convert block age to physical units... done.
    load block inim... done.
    convert to double precision... done.
    convert block inim to physical units... done.
    >>> mbol
    UnitArr([ -9.83588477,  -9.60060157,  -9.0921185 , ..., -10.70487702,
             -10.29691583,  -9.13260201], units="mag")
    >>> -2.5129*np.log10(sum(10**(-0.4*mbol)))
    -22.572481221547424
    >>> mag_b = calc_mags(gal.stars, 'b')
    skipping Z_0_0150.dat
    skipping Z_0_0160.dat
    skipping Z_0_0170.dat
    skipping Z_0_0190.dat
    skipping Z_0_0200.dat
    skipping Z_0_0210.dat
    >>> mag_b
    UnitArr([ -8.4467461 ,  -8.5647344 ,  -7.91316907, ..., -10.14487127,
              -8.95035609,  -7.83551164], units="mag")
'''
__all__ = ['read_table', 'combine_tables', 'calc_mags']

import numpy as np
from scipy.interpolate import interpn
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from .. import environment
from ..units import *

# effective wavelength midpoint
_band_widths = {
        'u':  364.2,      # ultroviolet
        'b':  446.0,      # blue
        'v':  550.2,      # visual
        'r':  655.7,      # red
        'i':  803.7,      # infrared
        'j': 1231. ,      # J-band
        'h': 1637. ,      # H-band
        'k': 2194. ,      # K-band
    }
# standard weights (A_lambda / A_V)
_band_weights = {
        'u':  1.55,
        'b':  1.29,
        'v':  1.01,
        'r':  0.82,
        'i':  0.60,
        'j':  0.29,
        'h':  0.18,
        'k':  0.11,
    }
# the full width half maximum bandwidth (in wavelength)
_band_widths = {
        'u':   64.,
        'b':   92.,
        'v':   92.,
        'r':  164.,
        'i':  126.,
        'j':  200.,
        'h':  233.,
        'k':  320.,
    }

def read_table(filename):
    '''
    Read a single table from a table file.

    Args:
        filename (str):     The path to the table.

    Returns:
        head (str):         The header of the table (first 10 lines).
        names (list):       A list of the table column names.
        table (np.ndarray): The table itself.
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    head = ''.join(lines[1:11])
    names = lines[11][1:].split()
    table = np.loadtxt(filename)
    return head, names, table

def combine_tables(band, required_shape=(353,11)):
    '''
    Load the tables of single stellar population (SSP) magnitudes for a given
    band.

    Using the tables stored in ./SSP-model, which are taken from
    'http://stev.oapd.inaf.it/cgi-bin/cmd'. For more information see
    ./SSP-model/README.txt and the tables.

    Args:
        band (str):     The requested band Supported bands are: U, B, V, R, I, J,
                        H, and K. Furthermore one can request bolometric
                        magnitudes ('Mbol').
    Returns:
        Zs (np.ndarray):    The age metallicities for the individual table points.
        ages (np.ndarray):  The age values (in years) for the individual table
                            points.
        tab (np.ndarray):   The table of magnitudes.
    '''
    from ..environment import module_dir
    directory = module_dir+'luminosities/SSP-model/'
    band = band.upper()
    if band == 'MBOL':
        band = 'mbol'

    tab = []
    Zs = []
    ages = None

    for f in sorted(os.listdir(directory)):
        if not f.startswith('Z_'):
            continue
        Z = float( f[2:8].replace('_','.') )
        h, n, t = read_table(directory + f)
        if t.shape != required_shape:
            if environment.verbose: print 'skipping', f
            continue
        Zs.append(Z)
        if ages is None:
            ages = t[:,1].copy()
        i = n.index(band)
        tab.append( t[:,i] )
    tab = np.array(tab)
    Zs = np.array(Zs)

    return Zs, ages, tab

def calc_mags(s, band='mbol'):
    '''
    Calculate the magnitudes of the star particles in the given band.

    Uses tables from 'combine_tables'.

    Args:
        s (Snap):       The (sub-)snapshot of stars only to calculate the
                        magnitudes for.
        band (str):     The band to calculate the magnitude for. Supported bands
                        are: U, B, V, R, I, J, H, and K. Furthermore one can
                        request bolometric magnitudes (Mbol).

    Returns:
        mag (np.ndarray):   The magnitude of the individual star particles.
    '''
    from ..analysis import mass_weighted_mean
    Zs, ages, tab = combine_tables(band)

    # get array that can be modified without changing the snapshot itself
    Z = s.Z.view(np.ndarray).copy()
    age = s.age.in_units_of('yr',copy=True).view(np.ndarray)

    # handle values beyond boundaries
    Z[Z < np.min(Zs)] = np.min(Zs)
    Z[Z > np.max(Zs)] = np.max(Zs)
    age[age < np.min(ages)] = np.min(ages)
    age[age > np.max(ages)] = np.max(ages)

    M = interpn((Zs, np.log10(ages)), tab, np.array(zip(Z, np.log10(age))))
    # multiply luminosity(!) by initial(!) mass:
    # M = -2.5129 * log10( 10.0**(-0.4*M) * s.inim.view(np.ndarray) )
    #   = M - 2.5129 * log10( s.inim.view(np.ndarray) )
    M -= 2.5129 * np.log10( s.inim.in_units_of('Msol').view(np.ndarray) )

    # promote to UnitArr
    M = M.view(UnitArr)
    M.units = 'mag'
    return M

