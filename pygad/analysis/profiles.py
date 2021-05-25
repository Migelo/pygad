'''
A collection of analysis functions to create profiles.

Example:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snapshot
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_320', physical=True)
    >>> center = UnitArr([33816.9, 34601.1, 32681.0], s['pos'].units)
    load block pos... done.
    >>> radially_binned(s.stars, 'Fe/mass', av='mass', center=center,
    ...                 r_edges=np.linspace(0,15,30))
    load block Z... done.
    derive block elements... done.
    derive block Fe... done.
    load block mass... done.
    UnitArr([0.00080558, 0.00065764, 0.00053465, 0.00049411, 0.00046332,
             0.00043347, 0.00043268, 0.00042417, 0.00040976, 0.00039034,
             0.00039317, 0.00037786, 0.00036477, 0.00035503, 0.00035541,
             0.00034816, 0.00030562, 0.00031812, 0.00030737, 0.00032145,
             0.00029885, 0.00028331, 0.00028867, 0.00029243, 0.00029605,
             0.00028672, 0.00029282, 0.00027797, 0.00026345], dtype=float32)
    >>> profile_dens(s, 'mass', center=center,
    ...     r_edges=[0]+list(np.logspace(-1,2,30))) # doctest:+ELLIPSIS
    UnitArr([4.94145445e+09, 2.23835198e+09, 1.00754934e+09, 1.45437910e+09,
             1.62901107e+09, 2.11259075e+09, 1.67041258e+09, 1.19817515e+09,
             8.61809349e+08, 6.60993375e+08, 4.51407479e+08, 3.60605928e+08,
             2.54610455e+08, 1.74468352e+08, 1.23857127e+08, 8.19490250e+07,
             5.28403138e+07, 3.39792774e+07, 2.04022431e+07, 1.18351359e+07,
             6.89966816e+06, 4.16610461e+06, 2.27211531e+06, 1.30861066e+06,
             7.09065373e+05, 3.94554812e+05, 2.16913872e+05, 1.50912836e+05,
             9.56563740e+04, 5.45304130e+04], units="Msol kpc**-3")
    >>> rho0, Rs, err = NFW_fit(s, center, Rmax='300 kpc')  # R200 ~ 178 kpc
    >>> if (abs(rho0 - '4.46e7 Msol/kpc**3') > '0.03e7 Msol/kpc**3'
    ...             or abs(Rs - '9.68 kpc') > '0.02 kpc'):
    ...     print(rho0, Rs)
    >>> if abs(err - 0.13) > 0.01:
    ...     print(err)
    >>> smassprof1 = radially_binned(s.stars, 'mass', center=center)
    >>> from ..transformation import Translation
    >>> Translation(-center).apply(s)
    apply Translation to "pos" of "snap_M1196_4x_320"... done.
    >>> smassprof2 = radially_binned(s.stars, 'mass')
    derive block r... done.
    >>> assert (np.abs(smassprof1-smassprof2)/smassprof1).max() < 1e-3
'''
__all__ = ['radially_binned', 'profile_dens', 'NFW', 'NFW_fit']

import numpy as np
from ..units import *
from ..utils import dist

def radially_binned(s, qty, av=None, r_edges=None, proj=None, center=None):
    '''
    Bin a quantity radially.

    Args:
        s (Snap):               The snapshot to use.
        qty (str, Unitqty):     The quantity to do the binning of.
        av (str, Unitqty):      The quantity to average over. Otherwise as 'qty'.
        r_edges (array-like):   The edges of the radial bins.
                                Default: 0 - 20 kpc in 50 bins
        proj (int):             If None bin spherically, otherwise bin projected
                                along this coordinate: (0:'x', 1:'y', 2:'z').
        center (array-like):    The center with respect to which the radii are
                                calculated.

    Returns:
        Q (UnitArr):            The binned quantity.
    '''
    if isinstance(qty, str):
        qty = s.get(qty)
    else:
        qty = UnitQty(qty)
    if av is not None:
        if isinstance(av, str):
            av = s.get(av)
        else:
            av = UnitQty(av)

    if center is None:
        center = [0,0,0] if proj is None else [0,0]
    center = UnitQty(center, s['pos'].units, subs=s)

    if proj is None:
        r = s['r'] if np.all(center==0) else dist(s['pos'],center)
    else:
        if proj not in list(range(3)):
            raise ValueError('Have to project along 0, 1, or 2!')
        if np.all(center==0) and proj==2:
            r = s['rcyl']
        else:
            axes = tuple( set([0,1,2]) - set([proj]) )
            r = dist(s['pos'][:,axes],center)
    r_ind = r.argsort()

    if r_edges is None:
        r_edges = UnitArr(np.linspace(0,20,51), 'kpc')
    r_edges = UnitQty(r_edges, s['pos'].units, subs=s)

    # r_edges as indices in r_ind
    ind_edges = np.array( [np.abs(r[r_ind]-rr).argmin() for rr in r_edges] )

    sorted_qty = qty[r_ind]
    if av is not None:
        sorted_av = av[r_ind]
        sorted_qty = sorted_qty * sorted_av
    Q = []
    for i in range(1, len(r_edges)):
        val = sorted_qty[ind_edges[i-1]:ind_edges[i]].sum()
        if av is not None:
            val /= sorted_av[ind_edges[i-1]:ind_edges[i]].sum()
        Q.append( val )

    Q = UnitArr(Q,qty.units)

    if av is not None:
        Q[np.isnan(Q)] = 0.0

    return Q

def profile_dens(s, qty, av=None, r_edges=None, proj=None, center=None):
    '''
    Create a radial profile for the density of a quantity.

    It calls radially_binned and then devides the bins by their volume / area.

    Args:
        See radially_binned.

    Returns:
        profile (UnitArr):      The profile in units of the quantity per
                                volume/area.
    '''
    if r_edges is None:
        r_edges = UnitArr(np.linspace(0,20,51), 'kpc')
    r_edges = UnitQty(r_edges, s['pos'].units, subs=s)

    Q = radially_binned(s, qty, av=av, r_edges=r_edges, proj=proj, center=center)
    if proj is None:
        V = (4.0/3.0*np.pi)*r_edges**3
    else:
        V = np.pi*r_edges**2
    V = V[1:] - V[:-1]
    return Q/V

def NFW(r, rho0, Rs):
    '''
    The function for a Navarro-Frenk-White (NFW) profile.

    Note:
        If both r and Rs are integers, an integer division is performed there and
        the result is probably wrong!

    Returns:
        rho = rho0 / ( r/Rs * (1.0 + r/Rs)**2 )

    Examples:
        >>> NFW(0.1, 42.0, 1.234)
        443.4892634342499
        >>> NFW(12.34, 2.1, 12.34)
        0.525
        >>> NFW(123.4, 1.0, 43.21)
        0.0235524185442844
    '''
    r_norm = r/Rs
    return rho0 / ( r_norm * (1.0 + r_norm)**2 )

def _NFW_log10(lr, lrho0, lRs):
    '''
    The logarithmic version of the NFW profile for fitting.

    Doctests:
        >>> assert abs(_NFW_log10(-1,1.23,0.12)
        ...             - np.log10(NFW(0.1,10**1.23,10**0.12))) < 1e-5
        >>> assert abs(_NFW_log10(2.1,12.3,1.2)
        ...             - np.log10(NFW(10.0**2.1,10**12.3,10**1.2))) < 1e-5
    '''
    r_norm = 10.0**lr / 10.0**lRs
    return lrho0 - np.log10( r_norm * (1.0 + r_norm)**2 )
def NFW_fit(s, center=None, Rmin=None, Rmax=None, Nfit=100, r_edges=None):
    '''
    Do a Navarro-Frenk-White (NFW) profile fit.

    First a logarithmic density profile is created with logarithmic radial bins.
    Then a least-square fit of a NFW profile (cf. NFW function) is done to this
    profile.

    Args:
        s (Snap):               The (sub-)snapshot to use (all particles are
                                used).
        center (array-like):    The center to take the profiles from.
        Rmin (float, UnitArr):  The minimal radius for the profile. Default 1 kpc.
        Rmax (float, UnitArr):
                                The radius to fit up to. (Should be close to
                                virial radius). Default: 100 kpc.
        Nfit (int):             The number of points of the profile to fit.
        r_edges (array-like):   The edges of the radial bins. If given Rmin, Rmax,
                                and Nfit are ignored.

    Returns:
        rho0 (UnitArr): The logarithm of the density parameter of the fit.
        Rs (UnitArr):   The scale radius.
        err (array):    The standard deviation error in the *logarithmic* fit.
    '''
    from scipy.optimize import curve_fit

    if r_edges is None:
        if Rmin is None: Rmin = '1 kpc'
        if Rmax is None: Rmax = '100 kpc'
        Rmin = float(UnitScalar(Rmin,s['pos'].units,subs=s))
        Rmax = float(UnitScalar(Rmax,s['pos'].units,subs=s))
        r_edges = np.array([0]+list(np.logspace(np.log10(Rmin),np.log10(Rmax),Nfit)))
    else:
        r_edges = UnitQty(r_edges, s['pos'].units, subs=s)
    # get something like the midpoints
    r = np.array( [r_edges[1]] + list((r_edges[2:]+r_edges[1:-1])/2.) )
    rho = profile_dens(s, 'mass', center=center, r_edges=r_edges, proj=None)

    # logarithmic fitting
    log_r   = np.log10(r).view(np.ndarray)
    log_rho = np.log10(rho).view(np.ndarray)
    log_Rs      = log_r[-1] - 1.0   # divide (a bit less than) Rmax by 10
    log_rho0    = log_rho[np.abs(log_r-log_Rs).argmin()]
    (log_rho0, log_Rs), cov = curve_fit(_NFW_log10, log_r, log_rho,
                                        p0=(log_rho0,log_Rs))

    # fit quality
    err = (_NFW_log10(log_r, log_rho0, log_Rs) - log_rho).std()

    return (UnitArr(10.0**log_rho0, rho.units),
            UnitArr(10.0**log_Rs, s['pos'].units),
            err)

