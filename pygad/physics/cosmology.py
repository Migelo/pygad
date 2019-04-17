'''
This module handles basic Lambda-CDM Friedman-Lemaitre-Robertson-Walter
cosmologies.

Example:
    >>> from ..environment import module_dir
    >>> cosmo = WMAP7()
    >>> if abs(cosmo.universe_age() - '13.77 Gyr') > '1e-2 Gyr':
    ...     print(cosmo.universe_age())
    >>> if abs(cosmo.cosmic_time(1.5) - '4.375 Gyr') > '1e-3 Gyr':
    ...     print(cosmo.cosmic_time(1.5))
    >>> z = cosmo.lookback_time_2z('5 Gyr') # reshift before 5 Gyr
    >>> assert abs(z - 0.4915) < 1e-4
    >>> assert abs(z2a(z) - 0.6705) < 1e-4  # and corresponding scale factor
'''
__all__ = ['a2z', 'z2a', 'Planck2013', 'WMAP7', 'FLRWCosmo']

import numpy as np
from ..units import *
from .quantities import *


def a2z(a):
    """
    Convert a scalefactor to redshift.

    Limited to z=1e15 (i.e. a2z(0)=1e15) in order to make the numerics more
    stable.

    Example:
        >>> a2z(1)
        0.0
        >>> round(a2z(0.1234), 4)
        7.1037
        >>> round(a2z(z2a(1.234)), 10)
        1.234
        >>> round( np.log10( a2z(0) ), 3 )
        15.0
    """
    return (1.0 - a) / (a + 1e-15)


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
    return 3.0 * H ** 2 / (8.0 * np.pi * G)


def Planck2013():
    '''
    Create a FLRW cosmology with the Planck 2013 results.

    Parameters are taken from Table 9 of the Planck Collaboration results of 2013.

    Returns:
        FLRWCosmo(h_0=0.6777, Omega_Lambda=0.683, Omega_m=0.307,
                  Omega_b=0.04825, sigma_8=0.8288, n_s=0.9611)

    Example:
        >>> planck = Planck2013()
        >>> planck.universe_age()
        UnitArr(13.786917087815011, units="Gyr")
        >>> planck.f_c
        0.15716612377850164
    '''
    return FLRWCosmo(h_0=0.6777, Omega_Lambda=0.683, Omega_m=0.307,
                     Omega_b=0.04825, sigma_8=0.8288, n_s=0.9611)


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
        UnitArr(13.768970244523558, units="Gyr")
        >>> wmap.f_c
        0.16727941176470587
    '''
    return FLRWCosmo(h_0=0.704, Omega_Lambda=0.728, Omega_m=0.272, Omega_b=0.0455,
                     sigma_8=0.810, n_s=0.967)


class FLRWCosmo(object):
    '''
    A Friedman-Lemaitre-Robertson-Walker cosmology that can calculate various
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
        >>> assert abs(cosmo.h(1) - 1.2325) < 1e-4
        >>> assert abs(cosmo.f_c - 0.166667) < 1e-4
        >>> assert abs(cosmo.universe_age() - '13.48 Gyr') < '0.02 Gyr'
        >>> assert abs(cosmo.cosmic_time(2.0) - '3.229 Gyr') < '0.01 Gyr'
        >>> assert abs(cosmo.rho_crit()-'136.0 Msol/kpc**3') < '1. Msol/kpc**3'
        >>> assert abs(cosmo.rho_crit(10)-'5.44e4 Msol/kpc**3') < '1e2 Msol/kpc**3'
        >>> assert abs(cosmo.lookback_time_2z('5 Gyr') - 0.49383) < 1e-4
        >>> cosmo.H()
        UnitArr(70.0, units="Mpc**-1 km s**-1")
        >>> assert abs(cosmo.H(z=1.0) - '123.25 km/s/Mpc') < '0.1 km/s/Mpc'
        >>> z = 1.234
        >>> assert abs(cosmo.comoving_distance(z) - '3.839 Gpc') < '1e-3 Gpc'
        >>> assert abs(cosmo.trans_comoving_distance(z) - '3.839 Gpc') < '1e-3 Gpc'
        >>> assert abs(cosmo.angular_diameter_distance(z) - '1.719 Gpc') < '1e-3 Gpc'
        >>> assert abs(cosmo.luminosity_distance(z) - '8.577 Gpc') < '1e-3 Gpc'
        >>> assert (abs(cosmo.apparent_angle('1 kpc', z) - '0.1200 arcsec')
        ...         < '1e-4 arcsec')
        >>> assert (abs(cosmo.angle_to_length('1 arcsec', z) - '8.332 kpc')
        ...         < '1e-3 kpc')
        >>> assert abs(cosmo.angle_to_length(1e-6, z) - '1.719 kpc') < '1e-3 kpc'
        >>> assert (abs(cosmo.angle_to_length(cosmo.apparent_angle('1 kpc',z), z) -
        ...         '1 kpc') < '1e-5 kpc')
    '''

    def __init__(self, h_0, Omega_Lambda, Omega_m, Omega_b=None, sigma_8=None,
                 n_s=None):
        if sigma_8:
            assert sigma_8 > 0
        self._h_0 = h_0
        self._Omega_Lambda = Omega_Lambda
        self._Omega_m = Omega_m
        self._Omega_b = 0.16 * Omega_m if Omega_b is None else Omega_b
        self._sigma_8 = sigma_8
        self._n_s = n_s  # not used by now

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
        return np.sqrt(self._Omega_m * (1.0 + z) ** 3
                       + self.Omega_k * (1.0 + z) ** 2
                       + self._Omega_Lambda)

    def h(self, z=0.0):
        '''Calculate the dimensionless Hubble parameter at redshift z.'''
        return self._h_0 * self._E(z)

    def H(self, z=0.0):
        '''Calculate the Hubble parameter at redshift z.'''
        return UnitArr(100.0 * self.h(z=z), 'km/s / Mpc')

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
        from ..snapshot.snapshot import Snapshot
        rho_crit = 3.0 * self.H(z) ** 2 / (8.0 * np.pi * G)
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

        if z <= -1:
            raise ValueError('Redshift cannot be smaller -1.')
        t, err = quad(lambda zz: 1.0 / ((1.0 + zz) * self._E(zz)), 0.0, z)
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
        return self.lookback_time(z=np.inf) - self.lookback_time(z=z)

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

    def comoving_distance(self, z, units='Gpc'):
        '''
        Comoving distance for an object with redshift z.

        Args:
            z (float):          The redshift to calculate the distance for.
            units (str, Unit):  The units to return the distance in.

        Returns:
            comoving_distance (UnitArr):    Comoving distance.
        '''
        from scipy.integrate import quad

        if z <= -1:
            raise ValueError('Redshift cannot be smaller -1.')
        d_c, err = quad(lambda zz: 1.0 / self._E(zz), 0.0, z)
        # divide by H to get d_c (com. dist. over speed of light) in units of
        # 1/(km s**-1 Mpc**-1)
        d_c /= 100.0 * self._h_0
        # 978.461942380137 Gyr = 1 / (km*s**-1 * Mpc**-1)
        d_c *= 978.461942380137
        return (c * UnitArr(d_c, 'Gyr')).in_units_of(units, subs={'a': 1.0})

    def trans_comoving_distance(self, z, units='Gpc'):
        '''
        The transverse comoving distance for an object with redshift z.

        Args and Returns as in comoving_distance.
        '''
        dc = self.comoving_distance(z=z, units=units)
        if self.Omega_k == 0.0:
            return dc

        dh = (c / self.H()).in_units_of(units)
        if self.Omega_k > 0.0:
            sqrt_Ok = np.sqrt(self.Omega_k)
            return dh / sqrt_Ok * np.sinh(sqrt_Ok * dc / dh)
        else:  # self.Omega_k < 0.0
            sqrt_Ok = np.sqrt(-self.Omega_k)
            return dh / sqrt_Ok * np.sin(sqrt_Ok * dc / dh)

    def angular_diameter_distance(self, z, units='Gpc'):
        '''
        The angular diameter distance for an object with redshift z.

        Args and Returns as in comoving_distance.
        '''
        return self.trans_comoving_distance(z, units=units) / (1.0 + z)

    def luminosity_distance(self, z, units='Gpc'):
        '''
        The luminosity distance for an object with redshift z.

        Args and Returns as in comoving_distance.
        '''
        return (1.0 + z) * self.trans_comoving_distance(z, units=units)

    def apparent_angle(self, l, z, units='arcsec'):
        '''
        Convert a physical extent of some object at redshift z to its apparent
        angle.

        Args:
            l (UnitScalar):     The physical length of the object.
            z (float):          The redshift of the object.
            units (str, Unit):  The units to return the angle in (has to be an
                                angular unit, cannot be unitless!).

        Returns:
            theta (UnitArr):    The object's apparent angle.
        '''
        l = UnitScalar(l)
        dA = self.angular_diameter_distance(z, units=l.units)
        theta = UnitArr(float(l / dA), 'rad')
        return theta.in_units_of(units)

    def angle_to_length(self, theta, z, units='kpc'):
        '''
        Convert an apparent angle of some object at redshift z into its physical
        extent.

        Args:
            theta (UnitScalar, float):
                                The apparent angle of the object. If a float is
                                passed, it is interpreted as radiants.
            z (float):          The redshift of the object.
            units (str, Unit):  The units to return the extent in.

        Returns:
            ext (UnitArr):      The object's physical extent.
        '''
        theta = UnitScalar(theta, 'rad')
        dA = self.angular_diameter_distance(z, units=units)
        return float(theta) * dA

