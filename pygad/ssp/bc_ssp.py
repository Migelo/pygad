'''
This module reads and interpolates Bruzual & Charlot single stellar population
(SSP) model tables.

Examples:
    >>> ages, u_b = load_table('U-B', 'm42')
    >>> u_b[::10]
    array([-1.0793, -1.0793, -1.0793, -1.0793, -1.0082, -0.6092, -0.6533,
           -0.7324, -0.7617, -0.5991, -0.5185, -0.3466, -0.1312,  0.0532,
            0.1679,  0.227 ,  0.2277,  0.2713,  0.2879,  0.2926,  0.2845,
            0.275 ])
    >>> for Z in [0.001, 0.01, 0.03]:
    ...     print(inter_bc_qty('1.23 Myr', Z, qty='Mbol'),
    ...           inter_bc_qty('15.2 Myr', Z, qty='Mbol'),
    ...           inter_bc_qty('0.8 Gyr', Z, qty='Mbol'),
    ...           inter_bc_qty('2.5 Gyr', Z, qty='Mbol'))
    -2.617717e+00 -5.640066e-01 3.811986460678126 4.700866682636425
    -2.657609e+00 -4.655381e-01 3.9231958298509078 4.904316681049632
    -2.784935e+00 -2.826923e-01 4.045632843503789 5.05050001420477

    >>> from ..environment import module_dir
    >>> from ..snapshot import Snapshot
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_320', physical=True)
    >>> inter_bc_qty(s.stars['age'], s.stars['metallicity'], qty='log Nly')
    load block form_time... done.
    derive block age... done.
    load block Z... done.
    derive block elements... done.
    derive block H... done.
    derive block He... done.
    derive block metals... done.
    derive block metallicity... done.
    UnitArr([40.85281952, 40.80583952, 41.17077814, ..., 41.2113266 ,
             41.21994522, 41.1978302 ])

    >>> inter_bc_qty(s.stars['age'], s.stars['metallicity'], qty='Vmag', IMF='Chabrier')
    UnitArr([5.5082267 , 5.47273851, 6.07288295, ..., 6.03545475, 5.89715848,
             6.07693768])
    >>> inter_bc_qty(s.stars['age'], s.stars['metallicity'], qty='Vmag', IMF='Salpeter')
    UnitArr([5.86339955, 5.81575021, 6.39816042, ..., 6.29992533, 6.1988034 ,
             6.33598423])

    magnitudes in the SSP tables are normalized to 1 solar mass -
    `derive_rules.get_luminosities` takes care for scaling correctly:
    >>> from ..snapshot import get_luminosities
    >>> L = get_luminosities(s.stars, band='V')
    load block mass... done.
    >>> L
    UnitArr([227724.53566959, 217788.83689963, 126963.87412006, ...,
             126999.67565879, 142102.98979917, 122512.21536806], units="Lsol")

    # To check
    #UnitArr([ 255272.44859605,  236937.66146024,  141531.12688612, ...,
    #          101504.63406881,  143090.23019079,   97253.25881182], units="Lsol")
    >>> lum_to_mag(L)
    UnitArr([-8.64352456, -8.59508904, -8.00920041, ..., -8.00950653,
             -8.13150804, -7.97044848], units="mag")
'''
__all__ = ['load_table', 'inter_bc_qty', 'lum_to_mag']

import numpy as np
from .. import gadget
from ..units import UnitQty, UnitScalar
from ..physics import solar
from .. import environment
import sys

metallicity_name = {
        1e-4: 'm22',
        4e-4: 'm32',
        4e-3: 'm42',
        8e-3: 'm52',
        2e-2: 'm62',
        5e-2: 'm72',
        }

table_names = {
        '1ABmag': ['log-age-yr', 'Mbol', 'g_AB', '(u-g)AB', '(g-r)AB',
                   '(g-i)AB', '(g-z)AB', '(FUV-NUV)AB', '(FUV-r)AB', '(FUV-R)AB',
                   'F(1500A)'],
        '1color': ['log-age-yr', 'Mbol', 'Umag', 'Bmag', 'Vmag', 'Kmag', '14-V',
                   '17-V', '22-V', '27-V', 'U-J', 'J-F', 'F-N', 'U-B', 'B-V'],
        '2color': ['log-age-yr', 'Rmag', 'J2Mmag', 'Kmag', 'V-R', 'V-I', 'V-J',
                   'V-K', 'R-I', 'J-H', 'H-K', 'V-K\'', 'V-Ks', '(J-H)2M ',
                   '(J-Ks)2M'],
        '3color': ['log-age-yr', 'B(4000)', 'B4_VN', 'B4_SDSS', 'B(912)', 'NLy',
                   'NHeI', 'NHeII', 'Mbol', 'Bol_Flux', 'SNR/yr/Lo', 'N(BH)',
                   'N(NS)', 'PNBR/yr/Lo', 'N(WD)', 'M(Remnants)'],
        '4color': ['log-age-yr', 'Mbol', 'Bmag', 'Vmag', 'Kmag', 'M*_liv',
                   'M_remnants', 'M_ret_gas', 'M_galaxy', 'SFR/yr',
                   'M*_liv+M_rem', 'M*_tot/Lb', 'M*_tot/Lv', 'M*_tot/Lk',
                   'M*_liv/Lb', 'M*_liv/Lv', 'M*_liv/Lk'],
        '5color': ['log-age-yr', 'Mbol', 'b(t)*\'s/yr', 'B(t)/yr/Lo',
                   'Turnoff_mass', 'BPMS/BMS'],
        '6lsindx_ffn': ['log-age', 'CN_1', 'CN_2', 'Ca4227', 'G4300', 'Fe4383',
                        'Ca4455', 'Fe4531', 'Fe4668', 'H\\beta', 'Fe5015', 'Mg_1',
                        'Mg_2', 'Mg-b'],
        '6lsindx_sed': ['log-age', 'CN_1', 'CN_2', 'Ca4227', 'G4300', 'Fe4383',
                        'Ca4455', 'Fe4531', 'Fe4668', 'H\\beta', 'Fe5015', 'Mg_1',
                        'Mg_2', 'Mg-b'],
        '6lsindx_sed_lick_system': ['log-age', 'CN_1', 'CN_2', 'Ca4227', 'G4300',
                                    'Fe4383', 'Ca4455', 'Fe4531', 'Fe4668',
                                    'H\\beta', 'Fe5015', 'Mg_1', 'Mg_2', 'Mg-b'],
        '7lsindx_ffn': ['log-age', 'Fe5270', 'Fe5335', 'Fe5406', 'Fe5709',
                        'Fe5782', 'Na-D', 'TiO_1', 'TiO_2', 'H\\delta_A',
                        'H\\gamma_A', 'H\\delta_F', 'H\\gamma_F', 'D(4000)'],
        '7lsindx_sed': ['log-age', 'Fe5270', 'Fe5335', 'Fe5406', 'Fe5709',
                        'Fe5782', 'Na-D', 'TiO_1', 'TiO_2', 'H\\delta_A',
                        'H\\gamma_A', 'H\\delta_F', 'H\\gamma_F', 'D(4000)',
                        'B4_VN', 'CaII8498', 'CaII8542', 'CaII8662', 'MgI8807',
                        'H8_3889', 'H9_3835', 'H10_3798', 'BH-HK'],
        '7lsindx_sed_lick_system': ['log-age', 'Fe5270', 'Fe5335', 'Fe5406',
                                    'Fe5709', 'Fe5782', 'Na-D', 'TiO_1', 'TiO_2',
                                    'H\\delta_A', 'H\\gamma_A', 'H\\delta_F',
                                    'H\\gamma_F', 'D(4000)', 'B4_VN', 'CaII8498',
                                    'CaII8542', 'CaII8662', 'MgI8807', 'H8_3889',
                                    'H9_3835', 'H10_3798', 'BH-HK'],
        '8lsindx_sed_fluxes': ['log-age', 'Nx', 'Im', 'Flux_Blue', 'Flux_Red',
                               'Flux_Ctrl', 'Flux_Line', 'Index'],
        '9color': ['log-age-yr', 'Kmag', 'K-I3.5', 'I3.5-I4.5', 'I4.5-I5.7',
                   'I5.7-I7.9', 'I7.9-I12', 'I12-I25', 'I25-I60', 'I60-I100',
                   'M24-M70', 'M70-M160', 'I100-M160'],
        'acs_wfc_color': ['log-age-yr', 'Vmag', 'Kmag', 'V-F220w', 'V-F250w',
                          'V-F330w', 'V-F410w', 'V-F435w', 'V-F475w', 'V-F555w',
                          'V-F606w', 'V-F625w', 'V-F775w', 'V-F814w', 'log Nly',
                          'Mt/Lb', 'Mt/Lv', 'Mt/Lk'],
        #'ised' -> binary file,
        'wfc3_color': ['log-age-yr', 'Vmag', 'Kmag', 'V-F110W', 'V-F125W',
                       'V-F160W', 'V-F225W', 'V-F336W', 'V-FR388N', 'V-F438W',
                       'V-F555W', 'V-F814W', 'V-ACS220W', 'V-ACS625W', 'log Nly',
                       'Mt/Lb', 'Mt/Lv', 'Mt/Lk'],
        'wfc3_uvis1_color': ['log-age-yr', 'Vmag', 'Kmag', 'V-F225w', 'V-F336w',
                             'V-F438w', 'V-F547m', 'V-F555w', 'V-F606w',
                             'V-F625w', 'V-F656n', 'V-F657n', 'V-F658n',
                             'V-F814w', 'log Nly', 'Mt/Lb', 'Mt/Lv', 'Mt/Lk'],
        'wfpc2_johnson_color': ['log-age-yr', 'Vmag', 'V-F300w', 'V-F300rl',
                                'V-F336w', 'V-F439w', 'V-F450w', 'V-F555w',
                                'V-F606w', 'V-F675w', 'V-F814w', 'V-U', 'V-B',
                                'V-R', 'V-I', 'V-J', 'V-K', 'V-L'],
    }
available_qty = set(np.concatenate(list(table_names.values()))) - set(['log-age-yr', 'log-age'])

table_base_name = {
        'Kroupa': 'bc2003_hr_stelib_%s_kroup_ssp',
        'Chabrier': 'bc2003_hr_stelib_%s_chab_ssp',
        'Salpeter': 'bc2003_hr_stelib_%s_salp_ssp',
        }

def load_table(qty, metal_code, folder=None, base=None, ending=None, IMF=None):
    '''
    Read the column of a quantity from the SSP tables.

    The table path is constructed as in:
        folder + '/' + (base%metal_code) + '.' + ending

    Args:
        qty (str):          The name of the quantity as in the tables.
        metal_code (str):   The part of the file name that encodes the metallicity
                            (values are stored in `metallicity_name`).
        folder (str):       The folder where the SSP model tables are located.
                            Default: as given in the `gadget.cfg` plus a subfolder
                            called as the IMF from `gadget.cfg` in lower cases.
        base (str):         The base name of the tables. Shall include exactly one
                            "%s" which is going to be substituted by the metal
                            code. Default: the string in `table_base_name` of the
                            IMF defined in the `gadget.cfg`.
        ending (str):       The file name ending. If None, it iterates over the
                            keys in `table_names` until the quantity `qty` is
                            found in the table.
        IMF (str):          The name of the IMF to use, if folder and base is not
                            given (undefined behaviour otherwise). By default the
                            one given in `gadget.cfg` is used. Values that are not
                            in `table_base_name.keys()` are invalid.

    Returns:
        age_bins (np.ndarray):  The first column, which are the age bins for the
                                different bins.
        qty (np.ndarray):       The column of the quantity asked for.

    Raises:
        IOError:            If any of the tables does not exist.
        RuntimeError:       If the quantity does not exists in one of the tables.
    '''
    if IMF is None:
        IMF=gadget.general['IMF']
    if folder is None:
        folder = gadget.general['SSP_dir'] + '/' + IMF.lower()
    if base is None:
        base = table_base_name[IMF]

    found = False
    if ending is None:
        for ending, columns in table_names.items():
            if qty in columns:
                tbl = np.loadtxt(folder + '/' + (base%metal_code) + '.' + ending)
                found = True
                break
    else:
        columns = table_names[ending]
        if qty in columns:
            tbl = np.loadtxt(folder + '/' + (base%metal_code) + '.' + ending)
            found = True

    if not found:
        raise RuntimeError('Could not find the quantity "%s" in the ' % qty + \
                           'SSP tables.')
    else:
        if environment.verbose >= environment.VERBOSE_TALKY:
            print("Using SSP table: " + folder + '/' + (base%metal_code) + '.' + ending)


    return tbl[:,0], tbl[:,columns.index(qty)]

def inter_bc_qty(age, Z, qty, units=None, IMF=None):
    '''
    Interpolate a quantity for given age and metallicity from the SSP model
    tables.

    Args:
        age (UnitQty):  The ages. If no units are given, they are assumed to be in
                        years.
        Z (UnitQty):    The metallicities (in absolute units, not solar units).
        qty (str):      The quantity to interpolate (as in the tables).
        IMF (str):      The name of the IMF to use, if folder and base is not
                        given (undefined behaviour otherwise). By default the one
                        given in `gadget.cfg` is used. Values that are not in
                        `table_base_name.keys()` are invalid.

    Returns:
        Q (UnitArr):    The interpolated quantity.

    Raises:
        IOError:            If any of the tables does not exist.
        RuntimeError:       If the quantity does not exists in one of the tables.
        ValueError:         If the length of the ages and the metallicity Z do not
                            fit eachother.
    '''
    if environment.verbose >= environment.VERBOSE_TALKY:
        print('interpolate SSP tables for qty "%s"...' % qty)

    age = np.log10(UnitQty(age, 'yr')).view(np.ndarray)
    Z = UnitQty(Z).view(np.ndarray)

    # allow for single values
    single_value = False
    if not age.shape or not Z.shape:
        single_value = True
        if not age.shape:
            age = np.array([float(age)])
        if not Z.shape:
            Z = np.array([float(Z)])
    if not len(age)==len(Z):
        raise ValueError('The array of ages and metallicities need to have ' + \
                         'the length!')

    if environment.verbose >= environment.VERBOSE_TALKY:
        print('read SSP tables...')
        sys.stdout.flush()
    # load all tables (just a few MB)
    tbls = {}
    for mtl in list(metallicity_name.keys()):
        age_bins, tbls[mtl] = load_table(qty, metallicity_name[mtl], IMF=IMF)

    if environment.verbose >= environment.VERBOSE_TALKY:
        print('table limits:')
        print('  age [yr]:    %.2e - %.2e' % (10**age_bins.min(),
                                              10**age_bins.max()))
        print('  metallicity: %.2e - %.2e' % (min(metallicity_name.keys()),
                                              max(metallicity_name.keys())))

    if environment.verbose >= environment.VERBOSE_TALKY:
        print('interpolate in age...')
        sys.stdout.flush()
    # interpolate in age in all metallicities simultaneously (creating blocks for
    # each tabled metallicity for each particle, i.e. len(metallicity_name) entire
    # blocks -- there are much less metallcity bins than age bins!):
    Q_mtl = {mtl:np.empty(len(age)) for mtl in metallicity_name.keys()}
    for k in range(len(age_bins)+1):
        if k == 0:
            age_mask = age<age_bins[0]
            n = [0, 0]
            a_age = np.zeros(np.sum(age_mask))
        elif k == len(age_bins):
            age_mask = age_bins[-1]<=age
            n = [-1, -1]
            a_age = np.zeros(np.sum(age_mask))
        else:
            age_mask = (age_bins[k-1]<=age) & (age<age_bins[k])
            n = [k-1, k]
            a_age = (age[age_mask] - age_bins[n[0]]) / (age_bins[n[1]] - age_bins[n[0]])

        for mtl in metallicity_name.keys():
            tbl = tbls[mtl]
            Q_mtl[mtl][age_mask] = a_age*tbl[n[1]] + (1.0-a_age)*tbl[n[0]]

    if environment.verbose >= environment.VERBOSE_TALKY:
        print('interpolate in metallicity...')
        sys.stdout.flush()
    # now interpolate in metallicity:
    Q = np.empty(len(age))
    for i in range(len(metallicity_name)+1):
        if i == 0:
            z = list(metallicity_name.keys())[0]
            Z_mask = Z<z
            Q[Z_mask] = Q_mtl[z][Z_mask]
        elif i == len(metallicity_name):
            z = list(metallicity_name.keys())[-1]
            Z_mask = z<=Z
            Q[Z_mask] = Q_mtl[z][Z_mask]
        else:
            z = [list(metallicity_name.keys())[i-1], list(metallicity_name.keys())[i]]
            Z_mask = (z[0]<=Z) & (Z<z[1])
            a_Z = (Z[Z_mask] - z[0]) / (z[1] - z[0])
            Q[Z_mask] = a_Z * Q_mtl[z[1]][Z_mask] + \
                    (1.0-a_Z) * Q_mtl[z[0]][Z_mask]

    if single_value:
        return UnitScalar(float(Q), units)
    return UnitQty(Q, units)

def lum_to_mag(L, subs=None):
    '''
    Convert luminosities to magnitudes.

    Args:
        L (UnitQty):    The luminosities to convert (if not units are given, they
                        are assumed to be in 'Lsol').

    Returns:
        mag (UnitArr):  The magnitudes.
    '''
    L = UnitQty(L, 'Lsol', subs=subs, dtype=float)
    return UnitQty( solar.abs_mag - 2.5 * np.log10(L), 'mag')

