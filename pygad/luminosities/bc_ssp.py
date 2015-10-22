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
    ...     print calc_bc_qty('1.23 Myr', Z, 'Mbol'),
    ...     print calc_bc_qty('15.2 Myr', Z, 'Mbol'),
    ...     print calc_bc_qty('0.8 Gyr', Z, 'Mbol'),
    ...     print calc_bc_qty('2.5 Gyr', Z, 'Mbol')
    -2.63760769231 -2.63760769231 -2.63760769231 -2.63760769231
    -2.73153108405 -2.81110600157 -2.93654167692 -2.97260382741
    -3.09563469756 -3.44045934015 -3.98401393332 -4.14028325209
'''
__all__ = ['load_table', 'calc_bc_qty']

import numpy as np
from .. import gadget
from ..units import UnitScalar

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
available_qty = set(np.sum(table_names.values())) - set(['log-age-yr', 'log-age'])

table_base_name = {
        'Kroupa': 'bc2003_hr_stelib_%s_kroup_ssp',
        'Chabrier': 'bc2003_hr_stelib_%s_chab_ssp',
        'Salpeter': 'bc2003_hr_stelib_%s_salp_ssp',
        }

def load_table(qty, metal_code, folder=None, base=None, ending=None):
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

    Returns:
        ages (np.ndarray):  The first column, which are the ages for the different
                            bins.
        qty (nnp.ndarray):  The column of the quantity asked for.
    '''
    if folder is None:
        folder = gadget.general['SSP_dir'] + '/' + gadget.general['IMF'].lower()
    if base is None:
        base = table_base_name[gadget.general['IMF']]

    found = False
    if ending is None:
        for ending, columns in table_names.iteritems():
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

    return tbl[:,0], tbl[:,columns.index(qty)]

def calc_bc_qty(age, Z, qty):
    '''
    Interpolate a quantity for given age and metallicity from the SSP model
    tables.

    Args:
        age (...):
        Z (...):
        qty (str):      The quantity to interpolate (as in the tables).

    Returns:
        Q (...):
    '''
    age = float( np.log10(UnitScalar(age, 'yr')) )

    z = [None] * 2
    for mtl, mtl_code in metallicity_name.iteritems():
        if mtl < Z:
            z[0] = mtl
        if mtl > Z:
            z[1] = mtl
            break
    if z[0] is None: z[0] = min(metallicity_name.keys())
    if z[1] is None: z[1] = max(metallicity_name.keys())

    tbl = {}
    for mtl in z:
        ages, tbl[mtl] = load_table(qty, metallicity_name[mtl])

    i = np.argmin(ages - age)
    if ages[i] > age:
        i = [i-1, i]
    else:
        i = [i, i+1]
    if i[0] < 0:
        for mtl, table in tbl.iteritems():
            Q[mtl] = table[0]
    elif i[1] >= len(tbl):
        for mtl, table in tbl.iteritems():
            Q[mtl] = table[-1]
    else:
        a = (age - ages[i[0]]) / (ages[i[1]] - ages[i[0]])
        Q = {}
        for mtl, table in tbl.iteritems():
            Q[mtl] = a*table[i[1]] + (1.0-a)*table[i[0]]

    if z[0] == z[1]:
        Q = Q[z[0]]
    else:
        a = (Z - z[0]) / (z[1] - z[0])
        Q = a*Q[z[1]] + (1.0-a)*Q[z[0]]

    return Q

