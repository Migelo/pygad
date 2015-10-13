'''
This module handles basic Lambda-CDM Friedman-Lemaitre-Robertson-Walter
cosmologies.

Example:
    >>> from ..environment import module_dir
    >>> cosmo = WMAP7()
    >>> cosmo.universe_age()
    UnitArr(13.7689702445, units="Gyr")
    >>> cosmo.cosmic_time(1.5)
    UnitArr(4.3752151566, units="Gyr")
    >>> z = cosmo.lookback_time_2z('5 Gyr') # reshift before 5 Gyr
    >>> z
    0.4915040090605358
    >>> z2a(z)                              # and corresponding scale factor
    0.6704641716852489
'''
__all__ = ['a2z', 'z2a', 'Planck2013', 'WMAP7', 'FLRWCosmo']

import numpy as np
from ..units import *
from quantities import *

def a2z(a):
    """
    Convert a scalefactor to redshift.

    Example:
        >>> a2z(1)
        0.0
        >>> a2z(0.1234)
        7.103727714748785
        >>> round(a2z(z2a(1.234)), 10)
        1.234
    """
    return (1.0 - a) / a

def z2a(z):
    """
    Convert a scalefactor to redshift.

    Example:
        >>> z2a(0)
        1.0
        >>> z2a(12.345)
        0.07493443237167478
        >>> round(z2a(a2z(0.363)), 10)
        0.363
    """
    return 1.0 / (1.0 + z)

def _rho_crit(H, G):
    '''
    Critical density of the universe as inferred from the Hubble 'constant' H and
    the gravitational constant G.

    It simply executes:

                     2
                  3 H
    rho      =  --------
       crit      8 pi G

    (No care is taken of units!)

    Example:
        >>> _rho_crit(H=2.2815e-18, G=6.67384e-8)
        9.309932895584995e-30
    '''
    return 3.0 * H**2 / (8.0 * np.pi * G)

def Planck2013():
    '''
    Create a FLRW cosmology with the Planck 2013 results.

    Returns:
        FLRWCosmo(h_0=0.6711, Omega_Lambda=0.6825, Omega_m=0.3175,
                  Omega_b=0.0491, sigma_8=0.8344, n_s=0.9624)

    Example:
        >>> planck = Planck2013()
        >>> planck.universe_age()
        UnitArr(13.8344731089, units="Gyr")
        >>> planck.f_c
        0.15464566929133858
    '''
    return FLRWCosmo(h_0=0.6711, Omega_Lambda=0.6825, Omega_m=0.3175,
                     Omega_b=0.0491, sigma_8=0.8344, n_s=0.9624)

def WMAP7():
    '''
    Create a FLRW cosmology with the parameters of the 7-year Wilkinson Microwave
    Anisotropy Probe.

    Returns:
        FLRWCosmo(h_0=0.704, Omega_Lambda=0.728, Omega_m=0.272, Omega_b=0.0455,
                  sigma_8=810, n_s=0.967)

    Example:
        >>> wmap = WMAP7()
        >>> wmap.universe_age()
        UnitArr(13.7689702445, units="Gyr")
        >>> wmap.f_c
        0.16727941176470587
    '''
    return FLRWCosmo(h_0=0.704, Omega_Lambda=0.728, Omega_m=0.272, Omega_b=0.0455,
                     sigma_8=0.810, n_s=0.967)

class FLRWCosmo(object):
    '''
    A Friedman-Lemaitre-Robertson-Walter cosmology that can calculate various
    cosmological values.

    Args:
        h_0 (float):            The Hubble parameter h (as in
                                H = 100 h km s**-1 Mpc**-1) at z=0.
        Omega_Lambda (float):   The density parameter of dark energy at z=0.
        Omega_m (float):        The density parameter of all (dark + baryonic)
                                matter at z=0.
        Omega_b (float):        The density parameter of baryonic matter at z=0.
                                Defaults to 0.16*Omega_m.
        sigma_8 (float):        Amplitude of the (linear) power spectrum at a
                                scale of 8 h**-1 Mph. (Not used yet.)
        n_s (float):            Scalar spectral index. (Not used yet.)

    Note:
        Radiation densities are neglected throughout.

    Example:
        >>> cosmo = FLRWCosmo(h_0=0.7, Omega_Lambda=0.7, Omega_m=0.3,
        ...                   Omega_b=0.05, sigma_8=0.8, n_s=0.96)
        >>> cosmo.Omega_tot
        1.0
        >>> cosmo.is_flat()
        True
        >>> cosmo.h(1)
        1.2324771803161305
        >>> cosmo.f_c
        0.16666666666666669
        >>> cosmo.universe_age()
        UnitArr(13.4762079087, units="Gyr")
        >>> cosmo.cosmic_time(z=2.0)
        UnitArr(3.2288370786, units="Gyr")
        >>> cosmo.rho_crit()
        UnitArr(135.961969367, units="Msol kpc**-3")
        >>> cosmo.rho_crit(z=10)
        UnitArr(5.438479e+04, units="Msol kpc**-3")
        >>> cosmo.lookback_time_2z('5 Gyr')
        0.4938323725410206
        >>> cosmo.H()
        UnitArr(70.0, units="km s**-1 Mpc**-1")
        >>> cosmo.H(z=1.0)
        UnitArr(123.247718032, units="km s**-1 Mpc**-1")
    '''

    def __init__(self, h_0, Omega_Lambda, Omega_m, Omega_b=None, sigma_8=None,
                 n_s=None):
        if sigma_8:
            assert sigma_8 > 0
        self._h_0          = h_0
        self._Omega_Lambda = Omega_Lambda
        self._Omega_m      = Omega_m
        self._Omega_b      = 0.16*Omega_m if Omega_b is None else Omega_b
        self._sigma_8      = sigma_8
        self._n_s          = n_s    # not used by now

    def __repr__(self):
        return 'FLRWCosmo(h_0=%.4g, ' % self._h_0 + \
               'O_Lambda=%.4g, ' % self._Omega_Lambda + \
               'O_m=%.4g, ' % self._Omega_m + \
               'O_b=%.4g, ' % self._Omega_b + \
               'sigma_8=%s, ' % ('None' if self._sigma_8 is None \
                                    else '%.4g' % self._sigma_8) + \
               'n_s=%s)' % ('None' if self._n_s is None else '%.4g' % self._n_s)

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        return (self._h_0 == other._h_0 and
                self._Omega_Lambda == other._Omega_Lambda and
                self._Omega_m == self._Omega_m and
                self._Omega_b == self._Omega_b and
                self._sigma_8 == self._sigma_8 and
                self._n_s == self._n_s)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __cmp__(self, other):
        return None

    def copy(self):
        '''Create a copy of the class instance.'''
        copy = object.__new__(FLRWCosmo)
        copy.__dict__ = self.__dict__.copy()
        return copy

    @property
    def h_0(self):
        '''Dimensionless Hubble parameter at z=0.'''
        return self._h_0

    @property
    def Omega_Lambda(self):
        '''Dark energy density parameter (at z=0).'''
        return self._Omega_Lambda

    @property
    def Omega_m(self):
        '''Matter (total) density parameter (at z=0).'''
        return self._Omega_m

    @property
    def Omega_b(self):
        '''Baryon density parameter (at z=0).'''
        return self._Omega_b

    @property
    def Omega_k(self):
        '''Curvature density parameter (at z=0).'''
        return 1.0 - self._Omega_m - self._Omega_Lambda

    @property
    def Omega_tot(self):
        '''Density (total) parameter (at z=0).'''
        return self._Omega_m + self._Omega_Lambda

    @property
    def f_c(self):
        '''Cosmic baryon fraction (at z=0).'''
        return self._Omega_b / self._Omega_m

    @property
    def sigma_8(self):
        '''Density (total) parameter (at z=0).'''
        return self._sigma_8

    @property
    def n_s(self):
        '''Density (total) parameter (at z=0).'''
        return self._n_s

    @h_0.setter
    def h_0(self, value):
        self._h_0 = value

    @Omega_Lambda.setter
    def Omega_Lambda(self, value):
        self._Omega_Lambda = value

    @Omega_m.setter
    def Omega_m(self, value):
        assert value >= 0
        self._Omega_m = value

    @Omega_b.setter
    def Omega_b(self, value):
        assert (value >= 0 and value <= self._Omega_m)
        self._Omega_b = value

    @sigma_8.setter
    def sigma_8(self, value):
        assert sigma_8 >= 0
        self._sigma_8 = value

    @n_s.setter
    def n_s(self, value):
        self._n_s = value

    def is_flat(self, tol=0.0):
        '''Test whether the cosmology is a flat one within the tolerance 'tol'
        (abs(self.Omega_k) <= tol).'''
        return abs(self.Omega_k) <= tol

    def _E(self, z=0.0):
        '''Helper function for e.g. FLRWCosmo.h() and
        FLRWCosmo.lookback_time().'''
        if z < -1:
            raise ValueError('Redshift cannot be smaller -1.')
        return np.sqrt(  self._Omega_m * (1.0 + z)**3
                       + self.Omega_k * (1.0 + z)**2
                       + self._Omega_Lambda)

    def h(self, z=0.0):
        '''Calculate the dimensionless Hubble parameter at redshift z.'''
        return self._h_0 * self._E(z)

    def H(self, z=0.0):
        '''Calculate the Hubble parameter at redshift z.'''
        return UnitArr(100.0*self.h(z=z), 'km/s / Mpc')

    def rho_crit(self, z=0.0):
        '''
        Calculate the critical density at redshift z.

        Note:
            The function uses G, from the quantities module.

        Args:
            z (float):  The redshift to calculate rho_crit for.

        Returns:
            rho_crit (UnitArr): The critical density (by default in units of
                                'Msol / kpc**3').
        '''
        from ..snapshot.snapshot import _Snap
        rho_crit = 3.0*self.H(z)**2 / (8.0*np.pi*G)
        rho_crit.convert_to('Msol / kpc**3')
        return rho_crit

    def lookback_time_in_Gyr(self, z):
        '''
        Lookback time (as float) for an object with redshift z in Gyr.

        Note:
            This function is a few times faster than FLRWCosmo.lookback_time(z)
            and uses 1 pc = 3.08567758149e+16 m and 1 yr = 365 d = 31536000 s.
        '''
        # this import is suspended until here, because it takes some time
        from scipy.integrate import quad

        if z < -1:
            raise ValueError('Redshift cannot be smaller -1.')
        t, err = quad(lambda zz: 1.0/((1.0+zz)*self._E(zz)), 0.0, z)
        # divide by H to get t in units of 1/(km s**-1 Mpc**-1)
        t /= 100.0 * self._h_0
        # 978.461942380137 Gyr = 1 / (km*s**-1 * Mpc**-1)
        return t * 978.461942380137

    def lookback_time(self, z):
        '''
        Lookback time for an object with redshift z.

        Note:
            Internally this function uses self.lookback_time_in_Gyr.
        
        Args:
            z (float):  The redshift to calculate the lookback time for.
        
        Returns:
            lookback_time (UnitArr):    Lookback time.
        '''
        return UnitArr(self.lookback_time_in_Gyr(z), 'Gyr')

    def universe_age(self):
        '''
        The present day age of the universe.
        
        Note:
            The function calls FLRWCosmo.lookback_time(z=oo) in the back.
        '''
        return self.lookback_time(z=np.inf)

    def cosmic_time(self, z):
        '''
        The cosmic time at a given redshift.

        Args:
            z (float):  The redshift to calculate the cosmic time for. 
        '''
        return self.universe_age() - self.lookback_time(z=z)

    def lookback_time_2z(self, t):
        '''
        Convert a lookback time in given units to a redshift.

        Note:
            The algorithm only converges, if the result for the redshift is
            smaller than ~1e5 (about 550 years after the big bang in our universe)
            and t is positive.

        Args:
            t (UnitScalar): The cosmic time. (If a float is passed, it is
                            interpreted as Gyr.)
        '''
        # this import is suspended until here, because it takes some time
        from scipy.optimize import brentq

        t = UnitScalar(t, 'Gyr', dtype=float)

        if self.universe_age() < t:
            raise ValueError('Lookback time (%s) is larger than the age ' % t + \
                              'of the universe (%s). ' % self.universe_age() + \
                              'Cannot convert to redshift.')
        if t < 0:
            raise ValueError('Does not support negative lookback times.')
        t = float(t.in_units_of('Gyr'))
        return brentq(lambda z: self.lookback_time_in_Gyr(z) - t, -1e-10, 1e5,
                      maxiter=1000, disp=True)

