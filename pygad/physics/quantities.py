'''
Some general quantities and constants.

Examples:
    >>> solar.Z()
    0.012946810000000086
    >>> assert abs(sum(solar.Z_massfrac) - 1.0) < 1e-3
    >>> c
    UnitArr(2.997925e+08, units="m s**-1")
    >>> if abs(G - '6.67408e-11 m**3/kg/s**2') > '1e-14 m**3/kg/s**2':
    ...     print(G)
    >>> if abs(kB - '1.380649e-23 J/K') > '2e-29 J/K':
    ...     print(kB)
    >>> if abs(N_A - '6.022141e23 mol**-1') > '2e17 mol**-1':
    ...     print(N_A)
    >>> if abs(m_u - '1.660539e-27 kg') > '2e-33 kg':
    ...     print(m_u)
    >>> for m in [m_p, m_n, 1822.9*m_e]:
    ...     assert abs(m/m_u - 1.0) < 0.02

    TODO: Why is this actually not closer to one?!
'''
__all__ = ['alpha_elements',
           'constants',
           'G', 'c', 'h', 'hbar', 'kB', 'N_A', 'epsilon0', 'mu0', 'q_e', 'm_p',
           'm_n', 'm_u', 'm_e', 'm_H', 'm_He', 'R',
           'solar',
           'SMH_Moster_2013', 'SMH_Behroozi_2013', 'SMH_Kravtsov_2014',
           'Reff_van_der_Wel_2014', 'SFR_Elbaz_2007',
           'Jeans_length', 'Jeans_mass']

import numpy as np
import warnings
import scipy.constants
from ..units import *
import sys

# alpha elements, produced in (or actually before) SNII:
alpha_elements = ['O', 'C', 'Ne', 'Si', 'Mg', 'S', 'Ca']
# sometimes Carbon & Nitrogen are also considered alpha elements; sometimes Oxygen
# is not considered an alpha element; Ar and Ti are not followed...

# All scipy constants:
constants = {}
for name in scipy.constants.find():
    try:
        value = scipy.constants.value(name)
        units = scipy.constants.unit(name)
        constants[name] = UnitArr(value,
                                  units.replace('^', '**').replace('ohm', 'Ohm')
                                  if units else None)
    except UnitError:
        pass
    except SyntaxError:
        pass
# Some useful constants:
try:
    G = constants['Newtonian constant of gravitation']
    c = constants['speed of light in vacuum']
    h = constants['Planck constant']
    hbar = constants['reduced Planck constant']
    kB = constants['Boltzmann constant']
    N_A = constants['Avogadro constant']
    epsilon0 = constants['vacuum electric permittivity']
    mu0 = constants['vacuum mag. permeability']
    q_e = constants['elementary charge']
    m_p = constants['proton mass']
    m_n = constants['neutron mass']
    m_u = constants['atomic mass constant']
    m_e = constants['electron mass']
    m_H = UnitArr(1.00794, 'u')
    m_He = UnitArr(4.002602, 'u')
    R = N_A * kB  # ideal gas constant
except KeyError:
    print("Constant missing from scipy.constants, update scipy to the latest"\
          " version!")
    raise

class solar(object):
    '''
    Class serving as a structure for holding solar values.

    These are taken from the tables from Wiersma+(2008) and are the CLOUDY
    default values.
    webpage: http://www.strw.leidenuniv.nl/WSS08/
    The link above is dead, alternative source on our bitbucket: https://bitbucket.org/broett/pygad/downloads/
    '''

    Z_massfrac = [0.70649785,
                  0.28055534,
                  0.0020665436,
                  8.3562563e-4,
                  0.0054926244,
                  0.0014144605,
                  5.907064E-4,
                  6.825874E-4,
                  4.0898522E-4,
                  6.4355E-5,
                  0.0011032152]

    abs_mag = 4.75

    @staticmethod
    def H():
        '''The Hydrogen mass fraction.'''
        return solar.Z_massfrac[0]

    @staticmethod
    def He():
        '''The Helium mass fraction.'''
        return solar.Z_massfrac[1]

    @staticmethod
    def Z():
        '''The solar metallicty.'''
        return 1.0 - (solar.Z_massfrac[0] + solar.Z_massfrac[1])

    @staticmethod
    def C():
        '''The Carbon mass fraction.'''
        return solar.Z_massfrac[2]

    @staticmethod
    def N():
        '''The Nitrogen mass fraction.'''
        return solar.Z_massfrac[3]

    @staticmethod
    def O():
        '''The Oxygen mass fraction.'''
        return solar.Z_massfrac[4]

    @staticmethod
    def Ne():
        '''The Neon mass fraction.'''
        return solar.Z_massfrac[5]

    @staticmethod
    def Mg():
        '''The Magnesium mass fraction.'''
        return solar.Z_massfrac[6]

    @staticmethod
    def Si():
        '''The Silicon mass fraction.'''
        return solar.Z_massfrac[7]

    @staticmethod
    def S():
        '''The Sulfur mass fraction.'''
        return solar.Z_massfrac[8]

    @staticmethod
    def Ca():
        '''The Calcium mass fraction.'''
        return solar.Z_massfrac[9]

    @staticmethod
    def Fe():
        '''The Iron mass fraction.'''
        return solar.Z_massfrac[10]

    @staticmethod
    def alpha():
        '''The mass fraction of the alpha elements.'''
        return sum(getattr(solar, e)() for e in alpha_elements)

    @staticmethod
    def log10_Fe_H():
        '''log_10(Fe / H)'''
        return np.log10(solar.Fe() / solar.H())

    @staticmethod
    def log10_O_Fe():
        '''log_10(O/ Fe)'''
        return np.log10(solar.O() / solar.Fe())

    @staticmethod
    def log10_alpha_Fe():
        '''log_10(alpha-elements / Fe)'''
        return np.log10(solar.alpha() / solar.Fe())


def SMH_Moster_2013(M_halo, z=0.0, return_scatter=False):
    '''
    Calculate the stellar mass to halo mass ratio as in Moster et al. (2013).

    The formula was given by Benjamin Moster in a private communication. It simply
    comes from a simple propagation of uncertainties of the formula of the mean,
    plus a constant 0.15 dex for a "physical scatter" (as explained in the paper).

    Args:
        M_halo (UnitQty):       The halo mass (in solar masses, if as float).
        z (float):              The redshift to calculate for.
        return_scatter (bool):  Whether to also return the scatter.

    Returns:
        SMH (float, np.dnarray):    The stellar mass to halo mass ratio.
       [lower (float, np.dnarray):  The lower bound of the 1-sigma scatter.]
       [upper (float, np.dnarray):  The upper bound of the 1-sigma scatter.]

    Example:
        >>> SMH_Moster_2013([1e11, 1e12, 1e13])
        array([0.01012983, 0.03429659, 0.00974028])
        >>> for a in SMH_Moster_2013([1e11, 1e12, 1e13], return_scatter=True):
        ...     print(a)
        [0.01012983 0.03429659 0.00974028]
        [0.00341559 0.0186693  0.00423019]
        [0.03004267 0.06300485 0.0224276 ]
        >>> for z in [0.5, 1.0, 2.0, 4.0]:
        ...     print(SMH_Moster_2013('1e11 Msol',z,return_scatter=True))
        ...     print(SMH_Moster_2013('1e13 Msol',z,return_scatter=True))
        (0.004314181996911216, 0.0013140763985384505, 0.0141636866191142)
        (0.009955587054568093, 0.004043156124677078, 0.02451394666561377)
        (0.003235536108623059, 0.0008746580727273953, 0.011968898746409244)
        (0.010323427157890868, 0.003921558687759114, 0.027176221693924904)
        (0.0026531006668289547, 0.0006021546514267902, 0.011689593581399173)
        (0.010560160483693697, 0.0037325186267171905, 0.02987714211072727)
        (0.002390900353994381, 0.00044896686807466534, 0.012732352672802963)
        (0.0103622356309934, 0.0034261919431899795, 0.03133972907900081)
    '''
    param = {
        'M1': [11.590470, 1.194913],
        'N': [0.035113, -0.024729],
        'beta': [1.376177, -0.825820],
        'gamma': [0.608170, 0.329275],
    }
    one_minus_a = float(z / (z + 1.0))

    M_halo = UnitQty(M_halo, 'Msol', dtype=float)
    from collections import Iterable
    if getattr(M_halo, 'shape', None) and M_halo.shape[0] > 1:
        ret = []
        for Mh in M_halo:
            ret.append(SMH_Moster_2013(Mh, z=z, return_scatter=return_scatter))
        ret = np.array(ret)
        if return_scatter:
            return ret[:, 0], ret[:, 1], ret[:, 2]
        else:
            return ret
    M_halo = float(M_halo)

    inter = {}
    for key, vals in param.items():
        inter[key] = vals[0] + vals[1] * one_minus_a
    inter['M1'] = 10 ** inter['M1']

    SMH = 2.0 * inter['N'] / ((M_halo / inter['M1']) ** -inter['beta'] +
                              (M_halo / inter['M1']) ** inter['gamma'])
    if return_scatter:
        param.update({
            'M1e': [0.236067, 0.353477],
            'Ne': [0.00577173, 0.00693815],
            'betae': [0.19344, 0.285018],
            'gammae': [0.0993274, 0.212919],
        })

        eta = M_halo / inter['M1']
        alpha = eta ** -inter['beta'] + eta ** inter['gamma']
        dmd = {}
        dmd['M1'] = [(inter['gamma'] * eta ** inter['gamma'] \
                      - inter['beta'] * eta ** (-inter['beta'])) / alpha,
                     (inter['gamma'] * eta ** inter['gamma'] \
                      - inter['beta'] * eta ** (-inter['beta'])) / alpha \
                     * one_minus_a]
        dmd['N'] = [np.log10(np.e) / inter['N'],
                    np.log10(np.e) / inter['N'] * one_minus_a]
        dmd['beta'] = [np.log10(np.e) / alpha * np.log(eta) \
                       * eta ** (-inter['beta']),
                       np.log10(np.e) / alpha * np.log(eta) \
                       * eta ** (-inter['beta']) * one_minus_a]
        dmd['gamma'] = [-np.log10(np.e) / alpha * np.log(eta) \
                        * eta ** inter['gamma'],
                        -np.log10(np.e) / alpha * np.log(eta) \
                        * eta ** inter['gamma'] * one_minus_a]
        sigma = 0.0
        for key in dmd.keys():
            sigma += (dmd[key][0] * param[key + 'e'][0]) ** 2 \
                     + (dmd[key][1] * param[key + 'e'][1]) ** 2
        sigma = np.sqrt(sigma) + 0.15

        log10_SM = np.log10(SMH * M_halo)
        lower, upper = 10 ** (log10_SM - sigma) / M_halo, 10 ** (log10_SM + sigma) / M_halo

    if return_scatter:
        return SMH, lower, upper
    else:
        return SMH


def _Behroozi_function(log10_M, log10_M1, log10_eps, alpha, delta, gamma):
    '''
    The function form from Behroozi et al. (2013) for calculating the logarithm
    of stellar mass as a function of the logarithm of halo mass.
    '''

    def f(x, alpha, delta, gamma):
        return -np.log10(10 ** (alpha * x) + 1.) + \
               delta * np.log10(1 + np.exp(x)) ** gamma / (1 + np.exp(10 ** -x))

    return log10_eps + log10_M1 + f(log10_M - log10_M1, alpha, delta, gamma) \
           - f(0, alpha, delta, gamma)


def SMH_Behroozi_2013(M_halo, z=0.0, return_scatter=False):
    '''
    Calculate the stellar mass to halo mass ratio as in Behroozi et al. (2013).

    Args:
        M_halo (UnitQty):       The halo mass (in solar masses, if as float).
        z (float):              The redshift to calculate for.
        return_scatter (bool):  Whether to also return the scatter.

    Returns:
        SMH (float, np.dnarray):    The stellar mass to halo mass ratio.
       [lower (float, np.dnarray):  The lower bound of the 1-sigma scatter.]
       [upper (float, np.dnarray):  The upper bound of the 1-sigma scatter.]

    Example:
        >>> SMH_Behroozi_2013([1e11, 1e12, 1e13])
        array([0.0046755 , 0.02679825, 0.00898848])
        >>> for e in SMH_Behroozi_2013([1e11, 1e12, 1e13], return_scatter=True):
        ...     print(e)
        [0.0046755  0.02679825 0.00898848]
        [0.00283027 0.01622207 0.00544109]
        [0.00772375 0.04426968 0.01484862]
        >>> for z in [0.5, 1.0, 2.0, 4.0]:
        ...     print(SMH_Behroozi_2013(1e11,z))
        ...     print(SMH_Behroozi_2013(1e13,z))
        0.004210670265291123
        0.009949059926042463
        0.0031404619764022197
        0.010782128125330971
        0.002244488003457852
        0.011437102413458175
        0.003460606635214448
        0.007272677770327794
    '''
    M_halo = UnitQty(M_halo, 'Msol', dtype=float).view(np.ndarray)

    z = float(z)
    a = 1 / (1. + z)
    nu = np.exp(-4. * a ** 2)
    log10_eps = -1.777 + (-0.006 * (a - 1.) - 0.000 * z) * nu - 0.119 * (a - 1.)
    log10_M1 = 11.514 + (-1.793 * (a - 1.) - 0.251 * z) * nu
    alpha = -1.412 + (0.731 * (a - 1.)) * nu
    delta = 3.508 + (2.608 * (a - 1.) - 0.043 * z) * nu
    gamma = 0.316 + (1.319 * (a - 1.) + 0.279 * z) * nu

    mu = -0.020 + 0.081 * (a - 1.)
    kappa = 0.045 - 0.155 * (a - 1.)

    xi = 0.218 - 0.023 * (a - 1.)

    log10_SM = _Behroozi_function(np.log10(M_halo), log10_M1, log10_eps, alpha, delta, gamma)
    # TODO: do properly
    # log10_SM = log10_SM - mu
    if return_scatter:
        return 10 ** log10_SM / M_halo, \
               10 ** (log10_SM - xi) / M_halo, 10 ** (log10_SM + xi) / M_halo
    else:
        return 10 ** log10_SM / M_halo


def SMH_Kravtsov_2014(M_halo, type='200c', return_scatter=False):
    '''
    Calculate the stellar mass to halo mass ratio as in Kravtsov et al. (2014).

    Note:
        The fits themselves are in Appendix A.

    Args:
        M_halo (UnitScalar):    The halo mass (in solar masses, if as float).
        return_scatter (bool):  Whether to also return the scatter.

    Returns:
        SMH (float):                The stellar mass to halo mass ratio.
       [scatter ([float, float]):   The scatter.]

    Example:
        >>> SMH_Kravtsov_2014([1e11, 1e12, 1e13])
        array([0.00807214, 0.04218667, 0.0205807 ])
        >>> for a in SMH_Kravtsov_2014([1e11, 1e12, 1e13], return_scatter=True):
        ...     print(a)
        [0.00807214 0.04218667 0.0205807 ]
        [0.00509317 0.02661799 0.01298554]
        [0.01279348 0.06686137 0.03261821]
        >>> for M in [1e11, 1e12, 1e13, 1e14, 1e15]:
        ...     print(SMH_Kravtsov_2014(M, return_scatter=True))
        (0.008072138072586098, 0.005093174802556399, 0.01279347667984786)
        (0.042186669544032114, 0.02661798898796599, 0.06686136537294297)
        (0.02058069982657301, 0.012985543710118501, 0.03261821105122113)
        (0.007212575378877716, 0.004550827407919915, 0.01143116161809591)
        (0.0022639956645691835, 0.0014284846924312448, 0.003588191316537168)
    '''
    M_halo = UnitQty(M_halo, 'Msol', dtype=float).view(np.ndarray)

    log10_M1 = {'200c': 11.35, '200m': 11.41, 'vir': 11.39}[type]
    log10_eps = {'200c': -1.642, '200m': -1.720, 'vir': -1.685}[type]
    alpha = {'200c': -1.779, '200m': -1.727, 'vir': -1.740}[type]
    delta = {'200c': 4.345, '200m': 4.305, 'vir': 4.335}[type]
    gamma = {'200c': 0.619, '200m': 0.544, 'vir': 0.531}[type]

    log10_SM = _Behroozi_function(np.log10(M_halo), log10_M1, log10_eps, alpha, delta, gamma)
    if return_scatter:
        warnings.warn('Scatter of 0.2 dex in Kravtsov et al. (2014) not sure!')
        return 10 ** log10_SM / M_halo, \
               10 ** (log10_SM - 0.2) / M_halo, 10 ** (log10_SM + 0.2) / M_halo
    else:
        return 10 ** log10_SM / M_halo


def Reff_van_der_Wel_2014(M_stars, z, type, return_scatter=False):
    '''
    A fit to the measure effective radii from van der Wel et al. (2014).

    Args:
        M_stars (UnitQty):      The stellar mass (in solar masses if float).
        z (float):              The redshift. Has to be less than 3.
        type ('ETG', 'LTG'):    The type of the galaxy.
        return_scatter (bool):  Whether to also return the scatter.

    Returns:
        Reff (UnitQty):     The half mass radius in kpc.
       [sigma (float):      The logarithmic scatter in Reff:
                            sigma(log10(Reff/kpc)). It is also for arrays of Reff
                            just a scalar!]

    Examples:
        >>> Reff_van_der_Wel_2014(1e10, z=1.2, type='LTG')
        UnitArr(3.5828332753571375, units="kpc")
        >>> Reff_van_der_Wel_2014([1e9,1e10,1e11], z=0.5, type='LTG')
        UnitArr([2.63476599, 4.52628659, 7.7757457 ], units="kpc")
        >>> Reff_van_der_Wel_2014([1e10,1e11,1e12], z=0.5, type='ETG',
        ...                       return_scatter=True)
        (UnitArr([ 0.99942888,  5.36725089, 28.82384388], units="kpc"), 0.10500000000000001)
    '''
    z_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    z = float(z)
    if z > z_edges[-1]:
        raise ValueError('The redshift has to be less than %.3f.' % z_edges[-1])

    A_bins = {'ETG': [0.60, 0.42, 0.22, 0.09, -0.05, -0.06],
              'LTG': [0.86, 0.78, 0.70, 0.65, 0.55, 0.51]}
    alpha_bins = {'ETG': [0.75, 0.71, 0.76, 0.76, 0.76, 0.79],
                  'LTG': [0.25, 0.22, 0.22, 0.23, 0.22, 0.18]}
    sigma_bins = {'ETG': [0.10, 0.11, 0.12, 0.14, 0.14, 0.14],
                  'LTG': [0.16, 0.16, 0.17, 0.18, 0.19, 0.19]}

    z_bins = (z_edges[1:] + z_edges[:-1]) / 2.0
    for i in range(1, len(z_bins)):
        if z < z_bins[i] or i == len(z_bins) - 1:
            xi_z = (z - z_bins[i]) / (z_bins[i - 1] - z_bins[i])
            A = A_bins[type][i - 1] * xi_z + A_bins[type][i] * (1. - xi_z)
            alpha = alpha_bins[type][i - 1] * xi_z + alpha_bins[type][i] * (1. - xi_z)
            sigma = sigma_bins[type][i - 1] * xi_z + sigma_bins[type][i] * (1. - xi_z)
            break

    M_stars = UnitQty(M_stars, 'Msol', dtype=float) / UnitArr(5e10, 'Msol')
    M_stars = M_stars.view(np.ndarray)  # needed for broken power of alpha
    Reff = UnitArr(10 ** A * M_stars ** alpha, 'kpc')
    if return_scatter:
        return Reff, sigma
    else:
        return Reff


def SFR_Elbaz_2007(M_star, z=0.0, return_scatter=False):
    '''
    The SFR as a function of stellar mass from Elbaz et al. (2007).

    The paper provides fits to observational data at z=0 and z=1.
    Here, it is interpolated between those two values, if z is
    inbetween or >1.

    Args:
        M_star (UnitQty):       The stellar mass of the galaxy.
        z (float):              The redshift. Has to be less than 1.5.
        return_scatter (bool):  Whether to also return the scatter.

    Returns:
        SFR (UnitQty):          The star-formation rate.
       [lower (UnitQty):        The lower bound of the 1-sigma scatter.]
       [upper (UnitQty):        The upper bound of the 1-sigma scatter.]

    Examples:
        >>> SFR_Elbaz_2007('1e9 Msol')
        UnitArr(0.2509107407720147, units="Msol yr**-1")
        >>> SFR_Elbaz_2007([1e10, 1e11])
        UnitArr([1.47747198, 8.7       ], units="Msol yr**-1")
        >>> SFR_Elbaz_2007([1,2e10,5e11], z=1)
        UnitArr([7.20000000e-09, 1.34356751e+01, 2.43447602e+02], units="Msol yr**-1")
        >>> for e in SFR_Elbaz_2007(2e10, z=0.1, return_scatter=True):
        ...     print(e)
        3.611106995357125 [Msol yr**-1]
        1.974967367437606 [Msol yr**-1]
        6.883386251196164 [Msol yr**-1]
        >>> for e in SFR_Elbaz_2007([1,2e10,5e11], z=0.7, return_scatter=True):
        ...     print(e)
        [1.38838325e-08 1.01608191e+01 1.79425865e+02] [Msol yr**-1]
        [7.60266234e-09 5.13688082e+00 9.03862837e+01] [Msol yr**-1]
        [2.64461727e-08 2.02086955e+01 3.57505029e+02] [Msol yr**-1]
    '''
    M_star = UnitQty(M_star, 'Msol', dtype=float).view(np.ndarray)

    SFR_0 = 8.7 * (M_star / 1e11) ** 0.77
    SFR_1 = 7.2 * (M_star / 1e10) ** 0.9

    sigma_0 = [5.0 * (M_star / 1e11) ** 0.77, 16.1 * (M_star / 1e11) ** 0.77]
    sigma_1 = [3.6 * (M_star / 1e10) ** 0.9, 14.4 * (M_star / 1e10) ** 0.9]

    xi_z = (float(z) - 1.0) / (0.0 - 1.0)
    SFR = xi_z * SFR_0 + (1. - xi_z) * SFR_1
    sigma = [xi_z * s[0] + (1. - xi_z) * s[1] for s in zip(sigma_0, sigma_1)]

    SFR = UnitArr(SFR, 'Msol/yr')
    sigma = [UnitArr(s, 'Msol/yr') for s in sigma]

    if return_scatter:
        return SFR, sigma[0], sigma[1]
    else:
        return SFR


def Jeans_length(T, rho, mu='1 u', units='kpc'):
    '''
    The Jeans length.

    ... as in:

        /    15 kB T    \ 1/2
       | --------------- |
        \ 4 pi G mu rho /

    Note:
        You can also calculate Jeans lengthes for arrays of parameters as long as
        they have all the same length (the Jeans length is then calculated
        elementwise for all entries in the array) or only on parameter is given
        as an array while all others are still scalars.

    Args:
        T (UnitArr):        The temperature of the gas (floats are interpreted in
                            units of Kelvin).
        rho (UnitArr):      The gas density (floats -> g/cm**3).
        mu (UnitArr):       The (effective) particle mass (floats -> g).
        units (str, Unit):  The units to return the Jeans length in.

    Return:
        L (UnitQty):        The Jeans length.

    Raises:
        ValueError:         If multiple parameters have nonidentical shapes.

    Examples:
        >>> if abs(Jeans_length('10 K', '1e6 u/cm**3', units='pc') -
        ...        '0.03067 pc') > '1e-4 pc':
        ...     print(Jeans_length('10 K', '1e6 u/cm**3', units='pc'))
        >>> if abs(Jeans_length(1e4, '1 u/cm**3', units='pc') -
        ...        '969.8 pc') > '0.5 pc':
        ...     print(Jeans_length(1e4, '1 u/cm**3', units='pc'))
    '''
    k = []  # array for lengths of given arrays
    T = UnitQty(T, units='K', dtype=np.float64)
    if sum(T.shape) > 0:
        k.append(sum(T.shape))

    rho = UnitQty(rho, units='g/cm**3', dtype=np.float64)
    if sum(rho.shape) > 0:
        k.append(sum(rho.shape))

    mu = UnitQty(mu, units='g', dtype=np.float64)
    if sum(mu.shape) > 0:
        k.append(sum(mu.shape))

    # check if all given arrays are of the same shape, if there are more than one
    if len(k) > 1:
        if k.count(k[0]) != len(k):
            raise ValueError('If more than one parameter is passed as an ' + \
                             'array, they need to have the same shape!')

    L2 = 15. * kB * T / (4. * np.pi * G * mu * rho)
    L2.convert_to(Unit(units) ** 2)
    return np.sqrt(L2)


def Jeans_mass(T, rho, mu='1 u', units='Msol'):
    '''
    The Jeans mass.

    ... as in:

        4      3          /     3    \ 1/2  / 5 kB T \ 3/2
       --- pi L  rho  =  | ---------- |    | -------- |
        3                 \ 4 pi rho /      \  G mu  /

    Note:
        You can also calculate Jeans lengthes for arrays of parameters as long as
        they have all the same length (the Jeans length is then calculated
        elementwise for all entries in the array) or only on parameter is given
        as an array while all others are still scalars.

    Args:
        T (UnitArr):        The temperature of the gas (floats are interpreted in
                            units of Kelvin).
        rho (UnitArr):      The gas density (floats -> g/cm**3).
        mu (UnitArr):       The (effective) particle mass (floats -> g).
        units (str, Unit):  The units to return the Jeans mass in.

    Return:
        M (UnitQty):        The Jeans mass.

    Raises:
        ValueError:         If multiple parameters have nonidentical shapes.

    Examples:
        >>> M = Jeans_mass('10 K', '1e6 u/cm**3')
        >>> if abs(M - '2.964 Msol') > '1e-3 Msol':
        ...     print(M)
        >>> M = Jeans_mass(UnitArr([1e2,1e3,1e4,1e5],'K'), '1.0 u/cm**3')
        >>> if np.any(np.abs(M -
        ...         UnitArr([9.372e+04,2.964e+06,9.372e+07,2.964e+09],'Msol'))
        ...         / M > 1e-3):
        ...     print(M)
        ...     print(np.abs(M -
        ...         UnitArr([9.372e+04,2.964e+06,9.372e+07,2.964e+09],'Msol')) / M)
        >>> M = Jeans_mass(UnitArr([1e4,1e4,1e2],'K'), UnitArr([1e-2,1e0,1e2],'u/cm**3'))
        >>> if np.any(np.abs(M -
        ...         UnitArr([9.372e+08, 9.372e+07, 9.372e+03],'Msol')) / M > 1e-3):
        ...     print(M)
    '''
    rho = UnitQty(rho, units='g/cm**3', dtype=np.float64)

    M = 4. * np.pi / 3. * Jeans_length(T, rho, mu) ** 3 * rho
    M.convert_to(units)
    return M
