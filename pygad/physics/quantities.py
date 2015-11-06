'''
Some general quantities and constants.

Example:
    >>> solar.Z()   # TODO: Why so low, isn't it more like 0.02?
    0.012946810000000086
    >>> G
    UnitArr(6.673840e-11, units="m**3 kg**-1 s**-2")
    >>> m_p
    UnitArr(1.672622e-27, units="kg")

    TODO: Why is this actually not closer to one?!
    >>> sum(solar.Z_massfrac)
    0.99971229335

    >>> Jeans_mass('10 K', '1e6 u/cm**3')
    UnitArr(2.9637322241, units="Msol")
    >>> Jeans_length('10 K', '1e6 u/cm**3', units='pc')
    UnitArr(0.0306686617722, units="pc")
    >>> Jeans_mass(UnitArr([1e2,1e3,1e4,1e5],'K'), '1.0 u/cm**3')
    UnitArr([  9.37214420e+04,   2.96373222e+06,   9.37214420e+07,
               2.96373222e+09], units="Msol")
    >>> Jeans_mass('1e4 K', UnitArr([1e-2,1e0,1e2],'u/cm**3'))
    UnitArr([  9.37214420e+08,   9.37214420e+07,   9.37214420e+06], units="Msol")
'''
__all__ = ['alpha_elements', 'G', 'c', 'kB', 'N_A', 'R', 'm_p', 'm_n', 'm_u',
           'm_e', 'solar', 'SMH_Moster_2013', 'SMH_Behroozi_2013',
           'SMH_Kravtsov_2014', 'Reff_van_der_Wel_2014', 'SFR_Elbaz_2007',
           'Jeans_length', 'Jeans_mass']

import numpy as np
import sys
import warnings
import scipy.constants
from ..units import *

# alpha elements, produced in (or actually before) SNII:
alpha_elements = ['O', 'C', 'Ne', 'Si', 'Mg', 'S', 'Ca']
# sometimes Carbon & Nitrogen are also considered alpha elements; sometimes Oxygen
# is not considered an alpha element; Ar and Ti are not followed...

# Some usefule constants:
G = UnitArr(scipy.constants.value('Newtonian constant of gravitation'),
            scipy.constants.unit('Newtonian constant of gravitation')
                                .replace('^','**'))
c = UnitArr(scipy.constants.value('speed of light in vacuum'),
            scipy.constants.unit('speed of light in vacuum').replace('^','**'))
kB = UnitArr(scipy.constants.value('Boltzmann constant'),
             scipy.constants.unit('Boltzmann constant').replace('^','**'))
N_A = UnitArr(scipy.constants.value('Avogadro constant'),
              scipy.constants.unit('Avogadro constant').replace('^','**'))
R = N_A * kB    # ideal gas constant
m_p = UnitArr(scipy.constants.value('proton mass'),
              scipy.constants.unit('proton mass').replace('^','**'))
m_n = UnitArr(scipy.constants.value('neutron mass'),
              scipy.constants.unit('neutron mass').replace('^','**'))
m_u = UnitArr(scipy.constants.value('atomic mass constant'),
              scipy.constants.unit('atomic mass constant').replace('^','**'))
m_e = UnitArr(scipy.constants.value('electron mass'),
              scipy.constants.unit('electron mass').replace('^','**'))

class solar(object):
    '''
    Class serving as a structure for holding solar values.
    
    These are taken from the tables from Wiersma+(2008) and are the CLOUDY
    default values.
    webpage: http://www.strw.leidenuniv.nl/WSS08/
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
        return 1.0 - (solar.Z_massfrac[0]+solar.Z_massfrac[1])

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
        M_halo (UnitScalar):    The halo mass (in solar masses, if as float).
        z (float):              The redshift to calculate for.
        return_scatter (bool):  Whether to also return the scatter.

    Returns:
        SMH (float):                The stellar mass to halo mass ratio.
       [scatter ([float, float]):   The scatter.]

    Example:
        >>> for M in [1e11, 1e12, 1e13]:
        ...     print SMH_Moster_2013(M)
        0.010129827533
        0.0342965939809
        0.00974028215475
        >>> for z in [0.5, 1.0, 2.0, 4.0]:
        ...     print SMH_Moster_2013(1e11,z,return_scatter=True)
        ...     print SMH_Moster_2013(1e13,z,return_scatter=True)
        (0.004314181996911216, [0.0013140763985384505, 0.014163686619114201])
        (0.009955587054568093, [0.0040431561246770777, 0.024513946665613771])
        (0.003235536108623059, [0.00087465807272739534, 0.011968898746409244])
        (0.010323427157890868, [0.0039215586877591139, 0.027176221693924904])
        (0.0026531006668289547, [0.00060215465142679015, 0.011689593581399173])
        (0.010560160483693697, [0.0037325186267171905, 0.02987714211072727])
        (0.002390900353994381, [0.00044896686807466534, 0.012732352672802963])
        (0.0103622356309934, [0.0034261919431899795, 0.03133972907900081])
    '''
    param = {
            'M1':    [11.590470, 1.194913],
            'N':     [ 0.035113,-0.024729],
            'beta':  [ 1.376177,-0.825820],
            'gamma': [ 0.608170, 0.329275],
            }
    one_minus_a = float(z / (z + 1.0))

    M_halo = float( UnitScalar(M_halo, 'Msol', dtype=float) )

    inter = {}
    for key, vals in param.iteritems():
        inter[key] = vals[0] + vals[1] * one_minus_a
    inter['M1'] = 10**inter['M1']

    SHM = 2.0 * inter['N'] / ( (M_halo/inter['M1'])**-inter['beta'] +
                               (M_halo/inter['M1'])**inter['gamma'] )
    if return_scatter:
        param.update( {
                'M1e':    [ 0.236067  , 0.353477  ],
                'Ne':     [ 0.00577173, 0.00693815],
                'betae':  [ 0.19344   , 0.285018  ],
                'gammae': [ 0.0993274 , 0.212919  ],
                } )

        eta = M_halo / inter['M1']
        alpha = eta**-inter['beta'] + eta**inter['gamma']
        dmd = {}
        dmd['M1'] = [ (inter['gamma']*eta**inter['gamma'] \
                        - inter['beta']*eta**(-inter['beta'])) / alpha,
                      (inter['gamma']*eta**inter['gamma'] \
                        - inter['beta']*eta**(-inter['beta'])) / alpha \
                        * one_minus_a ]
        dmd['N'] = [ np.log10(np.e) / inter['N'],
                     np.log10(np.e) / inter['N'] * one_minus_a ]
        dmd['beta'] = [ np.log10(np.e) / alpha * np.log(eta) \
                            * eta**(-inter['beta']),
                        np.log10(np.e) / alpha * np.log(eta) \
                            * eta**(-inter['beta']) * one_minus_a ]
        dmd['gamma'] = [ -np.log10(np.e) / alpha * np.log(eta) \
                            * eta**inter['gamma'],
                         -np.log10(np.e) / alpha * np.log(eta) \
                            * eta**inter['gamma'] * one_minus_a ]
        sigma = 0.0
        for key in dmd.iterkeys():
            sigma += ( dmd[key][0] * param[key+'e'][0] )**2 \
                    + ( dmd[key][1] * param[key+'e'][1] )**2
        sigma = np.sqrt( sigma ) + 0.15

        log10_SM = np.log10(SHM*M_halo)
        lower, upper = 10**(log10_SM-sigma)/M_halo, 10**(log10_SM+sigma)/M_halo

    if return_scatter:
        return SHM, [lower, upper]
    else:
        return SHM

def _Behroozi_function(log10_M, log10_M1, log10_eps, alpha, delta, gamma):
    '''
    The function form from Behroozi et al. (2013) for calculating the logarithm 
    of stellar mass as a function of the logarithm of halo mass.
    '''
    def f(x, alpha, delta, gamma):
        return -np.log10(10**(alpha*x)+1.) + \
                delta * np.log10(1+np.exp(x))**gamma / (1+np.exp(10**-x))

    return log10_eps + log10_M1 + f(log10_M-log10_M1,alpha,delta,gamma) \
            - f(0,alpha,delta,gamma)

def SMH_Behroozi_2013(M_halo, z=0.0, return_scatter=False):
    '''
    Calculate the stellar mass to halo mass ratio as in Behroozi et al. (2013).

    Args:
        M_halo (UnitScalar):    The halo mass (in solar masses, if as float).
        z (float):              The redshift to calculate for.
        return_scatter (bool):  Whether to also return the scatter.

    Returns:
        SMH (float):                The stellar mass to halo mass ratio.
       [scatter ([float, float]):   The scatter.]

    Example:
        >>> for M in [1e11, 1e12, 1e13]:
        ...     print SMH_Behroozi_2013(M)
        0.00467550099072
        0.0267982464569
        0.00898847634918
        >>> for z in [0.5, 1.0, 2.0, 4.0]:
        ...     print SMH_Behroozi_2013(1e11,z)
        ...     print SMH_Behroozi_2013(1e13,z)
        0.00421067026529
        0.00994905992604
        0.0031404619764
        0.0107821281253
        0.00224448800346
        0.0114371024135
        0.00346060663521
        0.00727267777033
        >>> SMH_Behroozi_2013(1e11, z=0.1, return_scatter=True)
        (0.0047309550870295175, [0.0028500856797309547, 0.0078530748021593891])
    '''
    M_halo = float( UnitScalar(M_halo, 'Msol', dtype=float) )

    z = float(z)
    a           = 1 / (1.+z)
    nu          = np.exp(-4.*a**2)
    log10_eps   = -1.777 + (-0.006*(a-1.) - 0.000*z)*nu - 0.119*(a-1.)
    log10_M1    = 11.514 + (-1.793*(a-1.) - 0.251*z)*nu
    alpha       = -1.412 + ( 0.731*(a-1.)          )*nu
    delta       =  3.508 + ( 2.608*(a-1.) - 0.043*z)*nu
    gamma       =  0.316 + ( 1.319*(a-1.) + 0.279*z)*nu

    mu          = -0.020 + 0.081*(a-1.)
    kappa       =  0.045 - 0.155*(a-1.)

    xi          =  0.218 - 0.023*(a-1.)

    log10_SM = _Behroozi_function(np.log10(M_halo), log10_M1, log10_eps, alpha, delta, gamma)
    #TODO: do properly
    #log10_SM = log10_SM - mu
    if return_scatter:
        return 10**log10_SM/M_halo, [10**(log10_SM-xi)/M_halo, 10**(log10_SM+xi)/M_halo]
    else:
        return 10**log10_SM/M_halo

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
        >>> for M in [1e11, 1e12, 1e13, 1e14, 1e15]:
        ...     print SMH_Kravtsov_2014(M, return_scatter=True)
        (0.0080721380725860981, [0.0050931748025563987, 0.01279347667984786])
        (0.042186669544032114, [0.026617988987965989, 0.066861365372942974])
        (0.020580699826573009, [0.012985543710118501, 0.032618211051221133])
        (0.0072125753788777162, [0.0045508274079199152, 0.01143116161809591])
        (0.0022639956645691835, [0.0014284846924312448, 0.0035881913165371681])
    '''
    M_halo = float( UnitScalar(M_halo, 'Msol', dtype=float) )

    log10_M1 =  {'200c': 11.35, '200m': 11.41, 'vir': 11.39}[type]
    log10_eps = {'200c':-1.642, '200m':-1.720, 'vir':-1.685}[type]
    alpha =     {'200c':-1.779, '200m':-1.727, 'vir':-1.740}[type]
    delta =     {'200c': 4.345, '200m': 4.305, 'vir': 4.335}[type]
    gamma =     {'200c': 0.619, '200m': 0.544, 'vir': 0.531}[type]

    log10_SM = _Behroozi_function(np.log10(M_halo), log10_M1, log10_eps, alpha, delta, gamma)
    if return_scatter:
        warnings.warn('Scatter of 0.2 dex in Kravtsov et al. (2014) not shure!')
        #print >> sys.stderr, 'WARNING: scatter of 0.2 dex in Kravtsov et al. (2014) not shure!'
        return 10**log10_SM/M_halo, [10**(log10_SM-0.2)/M_halo, 10**(log10_SM+0.2)/M_halo]
    else:
        return 10**log10_SM/M_halo

def Reff_van_der_Wel_2014(M_stars, z, type, return_scatter=False):
    '''
    A fit to the measure effective radii from van der Wel et al. (2014).

    Args:
        M_stars (UnitScalar):   The stellar mass (in solar masses if float).
        z (float):              The redshift. Has to be less than 3.
        type ('ETG', 'LTG'):    The type of the galaxy.
        return_scatter (bool):  Whether to also return the scatter.

    Returns:
        Reff (float):    The half mass radius in kpc.
        or
        sigma (float):  The logarithmic scatter in Reff: sigma(log10(Reff)).
    '''
    z_edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    z = float(z)
    if z > z_edges[-1]:
        raise ValueError('The redshift has to be less than %.3f.' % z_edges[-1])

    A_bins = {'ETG': [ 0.60,  0.42,  0.22,  0.09, -0.05, -0.06],
              'LTG': [ 0.86,  0.78,  0.70,  0.65,  0.55,  0.51]}
    alpha_bins = {'ETG': [0.75, 0.71, 0.76, 0.76, 0.76, 0.79],
                  'LTG': [0.25, 0.22, 0.22, 0.23, 0.22, 0.18]}
    sigma_bins = {'ETG': [0.10, 0.11, 0.12, 0.14, 0.14, 0.14],
                  'LTG': [0.16, 0.16, 0.17, 0.18, 0.19, 0.19]}

    z_bins = (z_edges[1:]+z_edges[:-1])/2.0
    for i in xrange(1,len(z_bins)):
        if z < z_bins[i] or i==len(z_bins)-1:
            xi_z = (z - z_bins[i]) / (z_bins[i-1] - z_bins[i])
            A     = A_bins[type][i-1]     * xi_z + A_bins[type][i]     * (1.-xi_z)
            alpha = alpha_bins[type][i-1] * xi_z + alpha_bins[type][i] * (1.-xi_z)
            sigma = sigma_bins[type][i-1] * xi_z + sigma_bins[type][i] * (1.-xi_z)
            break

    M_stars = float( UnitScalar(M_stars, 'Msol', dtype=float) )
    Reff = 10**A * (M_stars/5e10)**alpha
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
        M_star (UnitScalar):    The stellar mass of the galaxy.
        z (float):              The redshift. Has to be less than 1.5.
        return_scatter (bool):  Whether to also return the scatter.

    Returns:
        SFR (float):    The star-formation rate in solar masses per year.

    Examples:
        >>> SFR_Elbaz_2007(1e9)
        0.2509107407720147
        >>> SFR_Elbaz_2007(1e10)
        1.4774719776417176
        >>> SFR_Elbaz_2007(1e11)
        8.7
        >>> SFR_Elbaz_2007(2e10, z=1)
        13.435675078130027
        >>> SFR_Elbaz_2007(2e10, z=0)
        2.5194883194934694
        >>> SFR_Elbaz_2007(2e10, z=0.5)
        7.977581698811748
        >>> SFR_Elbaz_2007(2e10, z=0.1, return_scatter=True)
        (3.611106995357125, [1.974967367437606, 6.883386251196164])
    '''
    M_star = float( UnitScalar(M_star, 'Msol', dtype=float) )

    SFR_0 = 8.7*(M_star/1e11)**0.77
    SFR_1 = 7.2*(M_star/1e10)**0.9

    sigma_0 = [5.0*(M_star/1e11)**0.77, 16.1*(M_star/1e11)**0.77]
    sigma_1 = [3.6*(M_star/1e10)**0.9,  14.4*(M_star/1e10)**0.9]

    xi_z = (float(z) - 1.0) / (0.0 - 1.0)
    SFR = xi_z*SFR_0 + (1.-xi_z)*SFR_1
    sigma = [xi_z*s[0] + (1.-xi_z)*s[1] for s in zip(sigma_0, sigma_1)]

    if return_scatter:
        return SFR, sigma
    else:
        return SFR

def Jeans_length(T, rho, mu=m_u, units='kpc'):
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
    '''
    k = [] # array for lengths of given arrays
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

    L2 = 15.*kB*T/(4.*np.pi*G*mu*rho)
    L2.convert_to(Unit(units)**2)
    return np.sqrt(L2)
    
def Jeans_mass(T, rho, mu=m_u, units='Msol'):
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
    ''' 
    rho = UnitQty(rho, units='g/cm**3', dtype=np.float64)

    M = 4.*np.pi/3. * Jeans_length(T,rho,mu)**3 * rho
    M.convert_to(units)
    return M

