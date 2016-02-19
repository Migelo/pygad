'''
A collection of analysis functions that are somewhat connected to halo properties.

Examples:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_320', physical=True)
    >>> center = shrinking_sphere(s.stars, center=[s.boxsize/2]*3,
    ...                           R=s.boxsize*np.sqrt(3)) # doctest: +ELLIPSIS
    load block pos... done.
    convert block pos to physical units... done.
    do a shrinking sphere...
      starting values:
        center = ...
        R      = ...
    load block mass... done.
    convert block mass to physical units... done.
    done.
    >>> if np.linalg.norm( center - UnitArr([33816.9, 34601.1, 32681.0], 'kpc') ) > 1.0:
    ...     print center

    >>> R200, M200 = virial_info(s, center)
    >>> if abs(R200 - '177 kpc') > 3 or abs(M200 - '1e12 Msol') / '1e12 Msol' > 0.1:
    ...     print R200, M200
    >>> Translation(-center).apply(s)
    apply Translation to "pos" of "snap_M1196_4x_320"... done.

    >>> FoF, N_FoF = find_FoF_groups(s.highres.dm, '2.5 kpc')    # doctest: +ELLIPSIS
    perfrom a FoF search (l = 2.5 [kpc], N >= 100)...
    found 106 groups
    the 3 most massive ones are:
      group 0:   3.57e+11 [Msol]  @  [1..., -1..., 1...] [kpc]
      group 1:   8.05e+10 [Msol]  @  [4..., 9..., 4...] [kpc]
      group 2:   2.52e+10 [Msol]  @  [8..., 1...e+03, 7...] [kpc]
    >>> halo0 = s.highres.dm[FoF==0]
    >>> if abs(halo0['mass'].sum() - '3.57e11 Msol') > '0.02e11 Msol':
    ...     print halo0['mass'].sum()

    # find galaxies (exclude those with almost only gas)
    >>> galaxies = generate_FoF_catalogue(s.baryons, l='3 kpc', min_N=3e2,
    ...             exclude=lambda g,s: g.Mgas/g.mass>0.9)  # doctest: +ELLIPSIS
    perfrom a FoF search (l = 3 [kpc], N >= 300)...
    found 6 groups
    the 3 most massive ones are:
      group 0:   4.09e+10 [Msol]  @  [-0..., 1..., -0...] [kpc]
      group 1:   7.17e+09 [Msol]  @  [4..., 9..., 4...] [kpc]
      group 2:   5.29e+09 [Msol]  @  [8..., 1...e+03, 7...] [kpc]
    initialize halos from FoF group IDs...
    load block ID... done.
    initialized 3 halos.
    >>> galaxies[0] # doctest: +ELLIPSIS
    <Halo N = 71,... /w M = 4.1e+10 [Msol] @ com = [-0..., 1..., -0...] [kpc]>
    >>> gal = s[galaxies[0]]
    >>> assert len(gal) == len(galaxies[0])
    >>> assert set(gal['ID']) == set(galaxies[0].IDs)
    >>> assert np.all(gal.parts == np.array(galaxies[0].parts))
    >>> assert gal['mass'].sum() == galaxies[0].props['mass']
    >>> assert gal.stars['mass'].sum() == galaxies[0].Mstars
    >>> galaxies[0]._calc_prop('ssc', s)
    >>> if np.linalg.norm(galaxies[0].ssc - galaxies[0].com) > '1.0 kpc':
    ...     print galaxies[0].ssc
    ...     print galaxies[0].com
'''
__all__ = ['shrinking_sphere', 'virial_info', 'find_FoF_groups', 'Halo',
           'generate_FoF_catalogue', 'find_most_massive_progenitor']

import numpy as np
from .. import utils
from ..units import *
from ..utils import *
import sys
from ..transformation import *
from properties import *
from ..snapshot import *
from .. import environment
from .. import C

def shrinking_sphere(s, center, R, periodic=True, shrink_factor=0.93,
                     stop_N=10, verbose=environment.verbose):
    '''
    Find the densest point by shrinking sphere technique.

    Cf. TODO: cite!!!

    Args:
        s (Snap):               The (sub-)snapshot to find the densest point for.
        center (array-like):    The center to start with.
        R (float, UnitArr, str):The initial radius.
        periodic (bool):        Whether to assume a periodic box (with the
                                sidelength of snap.props['boxsize']) or not.
        shrink_factor (float):  The factor to shrink the sphere in each step.
        stop_N (int):           If so many or less particles are left in the
                                sphere, stop.

    Returns:
        center (UnitArr):       The center.
    '''
    center0 = UnitQty(center,s['pos'].units,subs=s,dtype=np.float64)
    R = UnitScalar(R,s['pos'].units,subs=s)

    if verbose:
        print 'do a shrinking sphere...'
        print '  starting values:'
        print '    center = %s' % center0
        print '    R      = %s' % R
        sys.stdout.flush()

    if not 0 < shrink_factor < 1:
        raise ValueError('"shrink_factor" must be in the interval (0,1)!')
    if not 0 < stop_N:
        raise ValueError('"stop_N" must be positive!')

    pos  = s['pos'].astype(np.float64)
    mass = s['mass'].astype(np.float64)
    assert len(pos) == len(mass)
    boxsize = float( s.boxsize.in_units_of(s['pos'].units) )

    # needed since C does not know about stridings
    if pos.base is not None:
        pos  = pos.copy()
        mass = mass.copy()

    center = np.empty((3,), dtype=np.float64)
    if periodic:
        C.cpygad.shrinking_sphere_periodic(
                                  C.c_void_p(center.ctypes.data),
                                  C.c_size_t(len(pos)),
                                  C.c_void_p(pos.ctypes.data),
                                  C.c_void_p(mass.ctypes.data),
                                  C.c_void_p(center0.ctypes.data), C.c_double(R),
                                  C.c_double(shrink_factor), C.c_size_t(stop_N),
                                  C.c_double(boxsize))
    else:
        C.cpygad.shrinking_sphere_nonperiodic(
                                  C.c_void_p(center.ctypes.data),
                                  C.c_size_t(len(pos)),
                                  C.c_void_p(pos.ctypes.data),
                                  C.c_void_p(mass.ctypes.data),
                                  C.c_void_p(center0.ctypes.data), C.c_double(R),
                                  C.c_double(shrink_factor), C.c_size_t(stop_N))
    center = center.view(UnitArr)
    center.units = s['pos'].units

    if verbose:
        print 'done.'
        sys.stdout.flush()

    return center

def virial_info(s, center=None, odens=200.0, N_min=10):
    '''
    Return the virial radius (R_odens) and the virial mass (M_odens) of the
    structure at 'center'.
    
    The virial radius is the radius, where the average density within that radius
    is <odens> times the critical density of the universe. If the inner <N_min>
    particles do not exceed a density of <odens> times the critical density,
    R_<odens> and M_<odens> are both returned as zero.

    Args:
        s (Snap):               The (sub-)snapshot to use.
        center (array-like):    The center of the structure to calculate the
                                properties for. (default: (0,0,0))
        odens (float):          Overdensity parameter (rho/rho_crit) used as
                                threshold for calculated radius and enclosed
                                mass to be returned.
        N_min (int):            The minimum number of particles required to form
                                the halo (have a density higher than
                                odens * rho_crit). If not reached, the return
                                values are NaN.

    Returns:
        R_odens, M_odens (both UnitArr):
                                The virial radius & mass (or corresponding
                                R_odens, M_odens)
    '''
    if center is None:
        center = [0,0,0]
    center = UnitQty(center,s['pos'].units,subs=s).view(np.ndarray)

    rho_crit = s.cosmology.rho_crit(z=s.redshift)
    rho_crit = rho_crit.in_units_of(s['mass'].units/s['pos'].units**3, subs=s)

    # make use of the potentially precalculated derived block r
    if np.all(center==0):
        r = s['r']
    else:
        r = dist(s['pos'], center)

    mass = s['mass'].astype(np.float64)
    r    = r.astype(np.float64)
    if mass.base is not None:
        mass = mass.copy()
        r    = r.copy()
    info = np.empty((2,), dtype=np.float64)
    C.cpygad.virial_info(C.c_size_t(len(r)),
                         C.c_void_p(mass.ctypes.data),
                         C.c_void_p(r.ctypes.data),
                         C.c_double(odens*rho_crit),
                         C.c_size_t(N_min),
                         C.c_void_p(info.ctypes.data),
    )
    if info[0] == 0.0:
        info[:] = np.nan
    return UnitArr(info[0],s['pos'].units), \
           UnitArr(info[1],s['mass'].units)

def find_FoF_groups(s, l, min_N=100, sort=True, verbose=environment.verbose):
    '''
    Perform a friends-of-friends search on a (sub-)snapshot.

    Args:
        s (Snap):           The (sub-)snapshot to perform the FoF finder on.
        l (UnitScalar):     The linking length to use for the FoF finder.
        min_N (int):        The minimum number of particles in a FoF group to
                            actually define it as such.
        sort (bool):        Whether to sort the groups by mass. If True, the group
                            with ID 0 will be the most massive one and the in
                            descending order.

    Returns:
        FoF (np.ndarray):   A block of FoF group IDs for the particles of `s`.
                            (IDs are ordered in mass, if `sort=True`).
        N_FoF (int):        The number of FoF groups found.
    '''
    l = UnitScalar(l, s['pos'].units, subs=s)
    sort = bool(sort)
    min_N = int(min_N)

    if verbose:
        print 'perfrom a FoF search (l = %.2g %s, N >= %g)...' % (
                l, l.units, min_N)
        sys.stdout.flush()

    pos = s['pos'].astype(np.float64)
    mass = s['mass'].astype(np.float64)
    if pos.base is not None:
        pos = pos.copy()
    if mass.base is not None:
        mass = mass.copy()
    FoF = np.empty(len(s), dtype=np.uintp)
    boxsize = float(s.boxsize.in_units_of(s['pos'].units))

    C.cpygad.find_fof_groups(C.c_size_t(len(s)),
                             C.c_void_p(pos.ctypes.data),
                             C.c_void_p(mass.ctypes.data),
                             C.c_double(l),
                             C.c_size_t(min_N),
                             C.c_int(int(sort)),
                             C.c_void_p(FoF.ctypes.data),
                             C.c_double(boxsize),
                             None,  # build new tree
    )

    # do not count the particles with no halo!
    N_FoF = len(set(FoF)) - 1

    if verbose:
        print 'found %d groups' % N_FoF
        N_list = min(N_FoF, 3)
        if N_list:
            if N_list==1:
                print 'the most massive one is:'
            else:
                print 'the %d most massive ones are:' % N_list
            for i in xrange(N_list):
                FoF_group = s[FoF==i]
                com = center_of_mass(FoF_group)
                M = FoF_group['mass'].sum()
                print '  group %d:   %8.3g %s  @  [%.3g, %.3g, %.3g] %s' % (
                        i, M, M.units, com[0], com[1], com[2], com.units)
        sys.stdout.flush()

    return FoF, N_FoF

class Halo(object):
    '''
    A class representing an halo or some group of an snapshot represented by IDs.

    Internally the defining IDs are stored and on creation some properties (such
    as total mass and R200) are calculated.

    Args:
        halo (Snap):                The sub-snapshot that defines the halo.

        alternatively:

        IDs (set, array-like):      The IDs that define the halo.
        snap (Snap):                Some (sub-)snapshot that entirely contains the
                                    halo defining IDs.

        calc (set, list, tuple):    A list of the properties to calculate at
                                    isinstantiation. 'mass', 'com', and 'parts'
                                    are always calculated.
    '''
    __long_name_prop__ = {
            'mass':     'total mass',
            'Mstars':   'total stellar mass',
            'Mgas':     'total gas mass',
            'Mdm':      'total dark matter mass',
            'parts':    'number of particles per species',
            'com':      'center of mass',
            'ssc':      'shrinking sphere center',
            'Rmax':     'maximum distance of a particle from com',
            'Rvir':     'Rvir (Mo+ 2002): (3*M / (4*pi * 18*pi**2 * rho_c))**(1/3.)',
            'R200':     'R200 (Mo+ 2002): (3*M / (4*pi * 200*rho_c))**(1/3.)',
            'R500':     'R500 (Mo+ 2002): (3*M / (4*pi * 500*rho_c))**(1/3.)',
            'lowres_part':  'number of low-resolution particles',
            'lowres_mass':  'mass of low-resolution particles',
    }

    @staticmethod
    def all_prop_list():
        return Halo.__long_name_prop__.keys()

    def __init__(self, halo=None, IDs=None, snap=None, calc=None):
        if halo is None:
            halo = snap[IDMask(IDs)]
            if len(halo) != len(IDs):
                raise ValueError('The given snapshot does not contain all ' + \
                                 'specified IDs!')
        else:
            if IDs is not None or snap is not None:
                raise ValueError('If the sub-snapshot`halo` is given, the ' + \
                                 'Halo will be defined by this and `IDs` ' + \
                                 'and `snap` must be None to avoid confusion!')

        # the defining property:
        self._IDs = halo['ID'].copy()

        # derive some properties
        self._props = {}
        if calc is None:
            # remove time consuming calculations
            calc = set(Halo.all_prop_list()) - set(['ssc'])
        if calc == 'all':
            calc = Halo.all_prop_list()
        calc = set(calc) - set(['mass', 'com', 'parts'])
        self._calc_prop('mass', halo=halo)
        self._calc_prop('com', halo=halo)
        self._calc_prop('parts', halo=halo)
        for prop in calc:
            self._calc_prop(prop, halo=halo)

    def _calc_prop(self, prop, snap=None, halo=None):
        if halo is None:
            halo = snap[self.mask]

        if prop == 'mass':
            val = halo['mass'].sum()
        elif prop == 'com':
            val = center_of_mass(halo)
        elif prop == 'parts':
            val = tuple(halo.parts)   # shall not change!
        elif prop in ['Mstars', 'Mgas', 'Mdm']:
            sub = getattr(halo, prop[1:], None)
            if sub is None:
                val = UnitScalar(0.0, halo['mass'].units)
            else:
                val = sub['mass'].sum()
        elif prop == 'ssc':
            R_max = np.percentile(periodic_distance_to(halo['pos'],
                                                       self.com,
                                                       halo.boxsize),
                                  70)
            val = shrinking_sphere(halo, center=self.com,
                                   R=R_max, verbose=False)
        elif prop == 'Rmax':
            val = periodic_distance_to(halo['pos'], self.com,
                                       halo.boxsize).max()
        elif prop in ['Rvir', 'R200', 'R500']:
            if prop == 'Rvir':
                odens = 18. * np.pi**2
            else:
                odens = int(prop[1:])
            rho_crit = halo.cosmology.rho_crit( z=halo.redshift )
            rho_crit.convert_to(halo['mass'].units/halo['pos'].units**3,
                                subs=halo)
            r3 = 3.0 * self.mass / (4.0*np.pi * odens*rho_crit)
            val = r3 ** Fraction(1,3)
        elif prop == 'lowres_part':
            low = getattr(halo, 'lowres', None)
            val = 0 if low is None else len(low)
        elif prop == 'lowres_mass':
            low = getattr(halo, 'lowres', None)
            if low is None:
                val = UnitScalar(0.0, halo['mass'].units)
            else:
                val = low['mass'].sum()
        else:
            raise ValueError('Unknown property "%s"!' % prop)

        self._props[prop] = val

    @property
    def IDs(self):
        return self._IDs.copy()

    @property
    def mask(self):
        '''The ID mask of the halo to mask a snapshot.'''
        # avoid copying the IDs
        mask = IDMask.__new__(IDMask)
        SnapMask.__init__(mask)
        mask._IDs = self._IDs
        return mask

    @property
    def props(self):
        '''All properties.'''
        return self._props.copy()

    def prop_descr(self, name):
        '''Long description of a property.'''
        return Halo.__long_name_prop__.get(name, 'unknown')

    def __getattr__(self, name):
        attr = self._props.get(name, None)
        if attr is not None:
            return attr
        raise AttributeError('%s has no attribute "%s"!' % (self, name))

    def __repr__(self):
        r = '<Halo N = %s' % nice_big_num_str(len(self))
        r += ' /w M = %.2g %s' % (self.mass, self.mass.units)
        r += ' @ com = [%.2g, %.2g, %.2g] %s' % (
                tuple(self.com) + (self.com.units,))
        r += '>'
        return r

    def __len__(self):
        return sum(self._props['parts'])

def generate_FoF_catalogue(s, l=None, calc=None, FoF=None, exclude=None,
                           verbose=environment.verbose, **kwargs):
    '''
    Generate a list of Halos defined by FoF groups.

    Args:
        s (Snap):           The snapshot to generate the FoF groups for.
        l (UnitScalar):     The linking length for the FoF groups.
        calc (set, list, tuple):
                            A list of the properties to calculate at instantiation
                            of the Halo instances.
        FoF (np.array):     An array with group IDs for each particle in the
                            snapshot `s`. If given, `l` is ignored and the FoF
                            groups are not calculated within this function.
        exclude (function): Exclude all halos h for which `exclude(h,s)` returns
                            a true value (note: s[h] gives the halo as a
                            sub-snapshot).
        verbose (bool):     Verbosity.
        **kwargs:           passed to `find_FoF_groups`.

    Returns:
        halos (list):       A list of all the groups as Halo instances. (Sorted in
                            mass, it not `sort=False` in the `kwargs`.)
    '''
    calc = kwargs.pop('calc', None)

    if FoF is None:
        FoF, N_FoF = find_FoF_groups(s, l=l, **kwargs)
    else:
        N_FoF = len(set(FoF)) - 1

    if verbose:
        print 'initialize halos from FoF group IDs...'
        sys.stdout.flush()

    halos = [ Halo(s[FoF==i],calc=calc) for i in xrange(N_FoF) ]
    if exclude:
        halos = [ h for h in halos if not exclude(h,s) ]

    if verbose:
        print 'initialized %d halos (excluded %d).' % (
                len(halos), N_FoF-len(halos))
        sys.stdout.flush()

    return halos

def _common_mass(h1, h2, s):
    # access private attribute for speed
    ID_both = set(h1._IDs) & set(h2._IDs)
    return s[IDMask(ID_both)]['mass'].sum()
def find_most_massive_progenitor(s, halos, h0):
    '''
    Find the halo with the most mass in common.

    Args:
        s (Snap):       The snapshot to which the halo catalogue `halos` belongs.
        halos (list):   A list of halos in which the one with the most common mass
                        with `h0` is searched for.
        h0 (Halo):      The halo for which to find the most massive progenitor in
                        `halos` (the one with the most mass in common).

    Returns:
        mmp (Halo):     The most massive progenitor (in `halos`) of `h0`.
    '''
    if len(halos) == 0:
        return None

    h0_mass = h0.mass.in_units_of(halos[0].mass.units, subs=s)

    # propably one of the closest halos (with at least 10% of the mass), find them
    closest = []
    min_mass = 0.1 * h0_mass
    min_mass.convert_to(halos[0].mass.units, subs=s)
    h0_com = h0.com.in_units_of(halos[0].com.units, subs=s)
    for i in xrange(3):
        close = None
        close_d = np.inf
        for h in halos:
            d = np.linalg.norm(h.com - h0_com)
            if d < close_d and h.mass > min_mass:
                for nh in closest:
                    if h is nh:
                        continue
                close = h
                close_d = d
        closest.append( close )

    # if any of them has more than 50% of the mass, we are done
    com_mass = []
    for h in closest:
        com_mass.append( _common_mass(h, h0, s) )
        if com_mass[-1] / h0_mass > 0.5:
            return h

    # if no other halo can have more mass than the most massive of the closest, it
    # is this one
    mm_closest, cm = max(zip(closest, com_mass), key=lambda p: p[1])
    if h0_mass-sum(com_mass) < cm:
        return mm_closest

    # iterate and find the most massive progenitor
    # not done in the beginning, since this requires the calculation of the common
    # mass for *all* halos and, hence, is slow
    mmp = max(halos,
              key=lambda h: _common_mass(h, h0, s))
    return mmp if _common_mass(mmp, h0, s)>0 else None

