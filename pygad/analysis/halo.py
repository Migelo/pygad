'''
A collection of analysis functions that are somewhat connected to halo properties.

Examples:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snapshot
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_320', physical=True)
    >>> center = shrinking_sphere(s.stars, center=[s.boxsize/2]*3,
    ...                           R=s.boxsize*np.sqrt(3)) # doctest: +ELLIPSIS
    load block pos... done.
    do a shrinking sphere...
      starting values:
        center = ...
        R      = ...
    load block mass... done.
    done.
    >>> if np.linalg.norm( center - UnitArr([33816.9, 34601.1, 32681.0], 'kpc') ) > 1.0:
    ...     print(center)

    >>> R200, M200 = virial_info(s, center)
    >>> if abs(R200 - '177 kpc') > 3 or abs(M200 - '1e12 Msol') / '1e12 Msol' > 0.1:
    ...     print(R200, M200)
    >>> Translation(-center).apply(s)
    apply Translation to "pos" of "snap_M1196_4x_320"... done.

    >>> l3 = 1500.*s.cosmology.rho_crit(s.redshift)/np.median(s.highres.dm['mass'])
    >>> FoF, N_FoF = find_FoF_groups(s.highres.dm,
    ...             l=l3**Fraction(-1,3),
    ...             dvmax='100 km/s') # doctest: +ELLIPSIS
    load block vel... done.
    perform a FoF search on 1,001,472 particles:
      l      = 2.2 [kpc]
      dv_max = 1e+02 [km s**-1]
      N     >= 100
    found 90 groups
    the 3 most massive ones are:
      group 0:   1.02e+11 [Msol]  @  [0.333, 0.255, 0.111] [kpc]
      group 1:    4.1e+10 [Msol]  @  [492, 941, 429] [kpc]
      group 2:   8.37e+09 [Msol]  @  [-1.57e+03, -1.4e+03, -1.11e+03] [kpc]

    # find galaxies (exclude those with almost only gas)
    >>> galaxies = generate_FoF_catalogue(s.baryons,
    ...             min_N=300,
    ...             dvmax='100 km/s', #max_halos=5,
    ...             progressbar=False,
    ...             exclude=lambda g,s: g.Mgas/g.mass>0.9)  # doctest: +ELLIPSIS
    perform a FoF search on 1,001,472 particles:
      l      = 1.3 [kpc]
      dv_max = 1e+02 [km s**-1]
      N     >= 300
    found 4 groups
    the 3 most massive ones are:
      group 0:   3.12e+10 [Msol]  @  [0.0693, 0.218, 0.0566] [kpc]
      group 1:   6.94e+09 [Msol]  @  [492, 940, 428] [kpc]
      group 2:   3.36e+09 [Msol]  @  [871, 1.36e+03, 729] [kpc]
    initialize halos from FoF group IDs...
    load block ID... done.
    load block Z... done.
    derive block elements... done.
    derive block H... done.
    load block ne... done.
    load block u... done.
    derive block temp... done.
    load block rho... done.
    load block form_time... done.
    derive block age... done.
    initialized 3 halos.
    >>> galaxies[0] # doctest: +ELLIPSIS
    <Halo @0x..., N = 55,..., M = 3.1e+10 [Msol]>
    >>> gal = s[galaxies[0]]
    >>> assert len(gal) == len(galaxies[0])
    >>> assert set(gal['ID']) == set(galaxies[0].IDs)
    >>> assert np.all(gal.parts == np.array(galaxies[0].parts))
    >>> assert gal['mass'].sum() == galaxies[0].props['mass']
    >>> assert gal.stars['mass'].sum() == galaxies[0].Mstars
    >>> gal = galaxies[0]
    >>> gal.calc_prop('ssc', root=s)    # doctest:+ELLIPSIS
    UnitArr([...], units="kpc")
    >>> if np.linalg.norm(gal.ssc - gal.com) > 1000:
    ...     print(gal.ssc)
    ...     print(gal.com)
    >>> assert 'Rmax' in Halo.calculable_props()
    >>> Halo.prop_descr('M200_com')
    'spherical M200 with `virial_info` with com as center'

    Test pickling:
    >>> import pickle as pickle
    >>> pkld_gal = pickle.dumps(gal)
    >>> gal_2 = pickle.loads(pkld_gal)
    >>> assert np.all(gal_2.IDs == gal.IDs)
    >>> assert set(gal_2.props.keys()) == set(gal.props.keys())
    >>> assert np.all(gal_2.com == gal.com)
'''
__all__ = ['shrinking_sphere', 'virial_info', 'find_FoF_groups',
           'NO_FOF_GROUP_ID', 'Rockstar_halo_field_names',
           'Rockstar_particle_field_names', 'RockstarHeader',
           'read_Rockstar_file', 'generate_Rockstar_halos', 'Halo',
           'nxt_ngb_dist_perc', 'generate_FoF_catalogue',
           'find_most_massive_progenitor']

import numpy as np
from .. import utils
from ..units import *
from ..utils import *
import sys, os
from ..transformation import *
from .properties import *
from ..snapshot import *
from .. import environment
from .. import C


def shrinking_sphere(s, center, R, periodic=True, shrink_factor=0.93,
                     stop_N=10, verbose=None):
    '''
    Find the densest point by shrinking sphere technique.

    Technique is described in Power et al. (2003).

    Args:
        s (Snap):               The (sub-)snapshot to find the densest point for.
        center (array-like):    The center to start with.
        R (float, UnitArr, str):The initial radius.
        periodic (bool):        Whether to assume a periodic box (with the
                                sidelength of snap.props['boxsize']) or not.
        shrink_factor (float):  The factor to shrink the sphere in each step.
        stop_N (int):           If so many or less particles are left in the
                                sphere, stop.
        verbose (int):          Verbosity level. Default: the gobal pygad
                                verbosity level.

    Returns:
        center (UnitArr):       The center.
    '''
    if verbose is None:
        verbose = environment.verbose
    center0 = UnitQty(center, s['pos'].units, subs=s, dtype=np.float64)
    R = UnitScalar(R, s['pos'].units, subs=s)

    if verbose >= environment.VERBOSE_NORMAL:
        print('do a shrinking sphere...')
        print('  starting values:')
        print('    center = %s' % center0)
        print('    R      = %s' % R)
        sys.stdout.flush()

    if not 0 < shrink_factor < 1:
        raise ValueError('"shrink_factor" must be in the interval (0,1)!')
    if not 0 < stop_N:
        raise ValueError('"stop_N" must be positive!')

    pos = s['pos'].astype(np.float64)
    mass = s['mass'].astype(np.float64)
    assert len(pos) == len(mass)
    boxsize = float(s.boxsize.in_units_of(s['pos'].units))

    # needed since C does not know about stridings
    if pos.base is not None:
        pos = pos.copy()
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

    if verbose >= environment.VERBOSE_NORMAL:
        print('done.')
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
        center = [0, 0, 0]
    center = UnitQty(center, s['pos'].units, subs=s).view(np.ndarray)

    rho_crit = s.cosmology.rho_crit(z=s.redshift)
    rho_crit = rho_crit.in_units_of(s['mass'].units / s['pos'].units ** 3, subs=s)

    # make use of the potentially precalculated derived block r
    if np.all(center == 0):
        r = s['r']
    else:
        r = dist(s['pos'], center)

    mass = s['mass'].astype(np.float64)
    r = r.astype(np.float64)
    if mass.base is not None:
        mass = mass.copy()
        r = r.copy()
    info = np.empty((2,), dtype=np.float64)
    C.cpygad.virial_info(C.c_size_t(len(r)),
                         C.c_void_p(mass.ctypes.data),
                         C.c_void_p(r.ctypes.data),
                         C.c_double(odens * rho_crit),
                         C.c_size_t(N_min),
                         C.c_void_p(info.ctypes.data),
                         )
    if info[0] == 0.0:
        info[:] = np.nan
    return UnitArr(info[0], s['pos'].units), \
           UnitArr(info[1], s['mass'].units)


NO_FOF_GROUP_ID = int(np.array(-1, np.uintp))


def find_FoF_groups(s, l, dvmax=np.inf, min_N=100, sort=True, periodic_boundary=2, boxsize_manual=None, verbose=None):
    '''
    Perform a friends-of-friends search on a (sub-)snapshot.

    Args:
        s (Snap):           The (sub-)snapshot to perform the FoF finder on.
        l (UnitScalar):     The linking length to use for the FoF finder (for dark
                            matter / all matter it turned out that
                            ( 1500. * rho_crit / median(mass) )^(-1/3) is a
                            reasonable value).
        dvmax (UnitScalar): The "linking velocity" to use for the FoF finder.
        min_N (int):        The minimum number of particles in a FoF group to
                            actually define it as such.
        sort (bool):        Whether to sort the groups by mass. If True, the group
                            with ID 0 will be the most massive one and the in
                            descending order.
        periodic_boundary (int): Whether the simulation uses periodic boundary
                                 conditions or not:
                                     - 0: no periodic boundary
                                     - 1: periodic boundary turned on
                                     - 2: automatic determination, turned on for s.cosmological == True
                                     - 3: periodic boundary turned on with boxsize from boxsize_manual argument
        boxsize_manual (UnitScalar) : Boxsize for periodic boundary conditions.
        verbose (int):      Verbosity level. Default: the gobal pygad verbosity
                            level.

    Returns:
        FoF (np.ndarray):   A block of FoF group IDs for the particles of `s`.
                            (IDs are ordered in mass, if `sort=True`). Particles
                            that are in no FoF group have ID = NO_FOF_GROUP_ID =
                            np.array(-1,np.uintp).
        N_FoF (int):        The number of FoF groups found.
    '''
    if verbose is None:
        verbose = environment.verbose
    l = UnitScalar(l, s['pos'].units, subs=s, dtype=float)
    dvmax = UnitScalar(dvmax, s['vel'].units, subs=s, dtype=float)
    sort = bool(sort)
    min_N = int(min_N)

    if verbose >= environment.VERBOSE_NORMAL:
        print('perform a FoF search on %s particles:' % nice_big_num_str(len(s)))
        print('  l      = %.2g %s' % (l, l.units))
        if dvmax != np.inf:
            print('  dv_max = %.2g %s' % (dvmax, dvmax.units))
        print('  N     >= %g' % (min_N))
        sys.stdout.flush()

    pos = s['pos'].astype(np.float64)
    vel = s['vel'].astype(np.float64)
    mass = s['mass'].astype(np.float64)
    if pos.base is not None:
        pos = pos.copy()
    if vel.base is not None:
        vel = vel.copy()
    if mass.base is not None:
        mass = mass.copy()
    FoF = np.empty(len(s), dtype=np.uintp)

    # check setting of periodic boundaries
    periodic = False
    if periodic_boundary == 0:
        periodic = False
    elif periodic_boundary == 1:
        periodic = True
    elif periodic_boundary == 2:
        if s.cosmological:
            periodic = True
        else:
            periodic = False

    if periodic_boundary == 3:
        assert(boxsize_manual is not None)
        boxsize = float(boxsize_manual)
    else:
        if periodic:
            boxsize = float(s.boxsize.in_units_of(s['pos'].units))
        else:
            boxsize = float(s['pos'].in_units_of(s['pos'].units).max() * 2)

    C.cpygad.find_fof_groups(C.c_size_t(len(s)),
                             C.c_void_p(pos.ctypes.data),
                             C.c_void_p(vel.ctypes.data),
                             C.c_void_p(mass.ctypes.data),
                             C.c_double(l),
                             C.c_double(dvmax),
                             C.c_size_t(min_N),
                             C.c_int(int(sort)),
                             C.c_void_p(FoF.ctypes.data),
                             C.c_double(boxsize),
                             None,  # build new tree
                             )

    # do not count the particles with no halo!
    N_FoF = len(set(FoF)) - 1

    if verbose >= environment.VERBOSE_NORMAL:
        print('found %d groups' % N_FoF)
        N_list = min(N_FoF, 3)
        if N_list:
            if N_list == 1:
                print('the most massive one is:')
            else:
                print('the %d most massive ones are:' % N_list)
            for i in range(N_list):
                FoF_group = s[FoF == i]
                com = center_of_mass(FoF_group)
                M = FoF_group['mass'].sum()
                print('  group %d:   %8.3g %s  @  [%.3g, %.3g, %.3g] %s' % (
                    i, M, M.units, com[0], com[1], com[2], com.units))
        sys.stdout.flush()

    return FoF, N_FoF


_ROCKSTAR_HALO_DTYPES = [
    ('id', 'i'), ('internal_id', 'i'), ('num_p', 'i'),
    ('mvir', 'f'), ('mbound_vir', 'f'), ('rvir', 'f'), ('vmax', 'f'),
    ('rvmax', 'f'), ('vrms', 'f'), ('x', 'f'), ('y', 'f'), ('z', 'f'),
    ('vx', 'f'), ('vy', 'f'), ('vz', 'f'), ('Jx', 'f'), ('Jy', 'f'),
    ('Jz', 'f'), ('energy', 'f'), ('spin', 'f')
]
_ROCKSTAR_PART_DTYPES = [
    ('x', 'f'), ('y', 'f'), ('z', 'f'), ('vx', 'f'), ('vy', 'f'), ('vz', 'f'),
    ('particle_id', 'i'), ('assigned_internal_haloid', 'i'),
    ('internal_haloid', 'i'), ('external_haloid', 'i')
]


def Rockstar_halo_field_names():
    return [name for name, t in _ROCKSTAR_HALO_DTYPES]


def Rockstar_particle_field_names():
    return [name for name, t in _ROCKSTAR_PART_DTYPES]


class RockstarHeader(object):
    def __init__(self, txt):
        self._txt = txt

        props = {}
        for line in txt.split('\n'):
            for info in line.split(';'):
                info = info.split(':' if ':' in info else '=')
                if len(info) != 2 or not info[-1]:
                    # not a info with ':' or '=', or with an empty value
                    continue
                name, value = info
                name = name.strip('#')
                if name == 'Units':
                    try:
                        try:
                            name, value = value.split(' in ')
                        except:
                            name, value = value.split(' are ')
                    except:
                        continue
                props[name.strip()] = value.strip()
        self._props = props

    @property
    def txt(self):
        return self._txt

    @property
    def props(self):
        return self._props.copy()

    def get(self, key, default=None):
        return self._props.get(key, default)

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        return key in self._props


def read_Rockstar_file(fname):
    '''
    Read in a Rockstar file.

    Note:
        Rockstar had to be run with the option 'FULL_PARTICLE_CHUNKS = 1', in
        order to output the whole particle data, which is needed here. This is
        also the file which filename needs to be specified.

    Args:
        fname (str):                The path to the particle(!) Rockstar output.

    Returns:
        header (RockstarHeader):    The header from the Rockstar file.
        halos (np.ndarray):         A numpy array with the halo table.
        particles (np.ndarray):     A numpy array with the particle table.
    '''
    from io import StringIO
    import codecs
    BEFORE_HALO_TBL = '#Halo table begins here:'
    BEFORE_PART_TBL = '#Particle table begins here:'

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('read Rockstar file "%s"' % fname)
    with codecs.open(fname, 'r', "utf-8") as f:
        rs_file = f.read()

    halo_start = rs_file.find(BEFORE_HALO_TBL)
    particle_start = rs_file.find(BEFORE_PART_TBL)

    # read the header
    header = RockstarHeader(rs_file[:halo_start])

    # read the halo table
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('  read halo table')
    # get rid of the leading '#'
    halo_tbl = '\n'.join(line[1:] \
                         for line in
                         rs_file[halo_start + len(BEFORE_HALO_TBL) + 1:particle_start].split('\n'))
    halos = np.loadtxt(StringIO(halo_tbl), dtype=_ROCKSTAR_HALO_DTYPES)

    # read the particle table
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('  read particle table')
    particles = np.loadtxt(
        StringIO(rs_file[particle_start + len(BEFORE_PART_TBL) + 1:]),
        dtype=_ROCKSTAR_PART_DTYPES)

    return header, halos, particles


def generate_Rockstar_halos(fname, snap, exclude=None, nosubs=True, data=None,
                            ignore_inconsistency=False, **kwargs):
    '''
    Create a list of halos from a Rockstar particle file.

    Args:
        fname (str):        The path to the particle(!) Rockstar output. (Cf. also
                            `read_Rockstar_file`!)
                            This argument is ignored, if `data` contains 'halos'
                            and 'particle'.
        snap (Snap):        The corresponding snapshot. Can be None, if used in
                            neither `exclude` nor calculation of the Halo classes
                            are requested.
        exclude (function): Exclude all halos h for which `exclude(h,snap)`
                            returns a true value. Here `h` is the halo information
                            from the Rockstar file, i.e. a np.ndarray with named
                            fields (for instance: `h['num_p']` is the number of
                            particles in that halos).
                            For all fields see `halo.Rockstar_halo_field_names()`.
        nosubs (bool):      TODO
        data (dict):        If a dictionary is passed, it will hold the Rockstar
                            tables for the halos and particles after the function
                            call.
                            Furthermore, if it already contains these two entries,
                            `fname` is ignored and this data is taken for the
                            generation.
        ignore_inconsistency (bool):
                            Just warn about inconsistencies between the Rockstar
                            file and the snapshot and do not raise exceptions.
        **kwargs            Further arguments are passed to `Halo.__init__`.

    Returns:
        halos (list):       A list of Halo class instances created from the
                            Rockstar output.
    '''
    # get the data / create dictionary to return them
    if data is not None and 'halos' in data and 'particles' in data:
        pass
    else:
        if data is None:
            data = {}
        # read the Rockstar file
        import time
        start_time = time.time()
        data['header'], data['halos'], data['particles'] = read_Rockstar_file(fname)
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('loaded in %.2f sec' % (time.time() - start_time))

    # check for consistency between Rockstar file and snapshot
    try:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('checking for consistency...')
        h = data['header']
        if 'a' in h and abs(float(h['a']) - snap.scale_factor) > 0.01:
            raise RuntimeError("Scale factors of snapshot and Rockstar file do "
                               "not match: %s vs. %s" % (
                                   h['a'], snap.scale_factor))
        cosmo = snap.cosmology
        for name, attr in [('Om', 'Omega_m'), ('Ol', 'Omega_Lambda'), ('h', 'h_0')]:
            if name in h and abs(float(h[name]) - getattr(cosmo, attr)) > 0.0001:
                raise RuntimeError("%s of snapshot and Rockstar file " % (name, attr) +
                                   "do not match: %s vs. %s" % (
                                       h[name], getattr(cosmo, attr)))
        if 'Box size' in h:
            # it seems Rockstar supresses the information that the units are
            # comoving fot the box size
            bz = UnitScalar(str(h['Box size']).replace('h', 'h_0'),
                            snap.boxsize.units,
                            subs={'a': 1.0, 'z': 0.0, 'h_0': cosmo.h_0})
            if abs(float(bz / snap.boxsize) - 1) > 0.01:
                raise RuntimeError("box sizes of snapshot and Rockstar file " +
                                   "do not match: %s vs. %s" % (bz, snap.boxsize))
    except RuntimeError as e:
        if ignore_inconsistency:
            print('WARNING:', e.message)
        else:
            raise

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('create halo list from the Rockstar data')
    halo_classes = []
    parts = data['particles']

    halos = data['halos']
    import time
    start_time = time.time()
    # exclude non-physical halos and those excluded specifically
    halos = [h for h in data['halos']
             if ((h['id'] != -1) and not (exclude and exclude(h, snap)))]
    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('pre-selection from %s down to %s halos' % (
            len(data['halos']), len(halos)))

    # load blocks needed in order to avoid cluttering the progress bar
    if snap is not None:
        snap['ID']
        snap['pos']
        snap['mass']

    if 'calc' not in kwargs:
        kwargs['calc'] = None
    if 'properties' in kwargs:
        properties = kwargs['properties']
        del kwargs['properties']
    else:
        properties = {}
    start_time = time.time()
    with ProgressBar(halos, label='initialize halos', show_eta=False,
                     show_percent=False) as pbar:
        for h in pbar:
            # create mask for the particle data
            # 1st requirement: restrict to chosen halo
            # 2nd requirement: exclude substructures
            mask = (parts['external_haloid'] == h['id'])
            if nosubs:
                mask &= (parts['internal_haloid'] ==
                         parts['assigned_internal_haloid'])

            # enrich the halo properties (e.g. by units)
            prop = {dt[0]: val for dt, val in zip(_ROCKSTAR_HALO_DTYPES, h)}
            for name in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Jx', 'Jy', 'Jz']:
                del prop[name]
            prop.update({
                'mvir': UnitArr(h['mvir'], 'Msol/h_0'),
                'mbound_vir': UnitArr(h['mbound_vir'], 'Msol/h_0'),
                'center': UnitArr([h['x'], h['y'], h['z']], 'cMpc/h_0'),
                'vel': UnitArr([h['vx'], h['vy'], h['vz']], 'km/s'),
                'vmax': UnitArr(h['vmax'], 'km/s'),
                'rvmax': UnitArr(h['rvmax'], 'km/s'),
                'vrms': UnitArr(h['vrms'], 'km/s'),
                'J': UnitArr([h['Jx'], h['Jy'], h['Jz']],
                             'Msol/h_0 * Mpc/h_0 * km/s'),
                'energy': UnitArr(h['energy'], 'Msol/h_0 * (km/s)**2'),
            })
            # TODO: are these quantities those?
            # prop['mass'] = prop['mvir']
            # prop['com']  = prop['center']
            # (over-)write specified properties
            prop.update(properties)

            halo = Halo(IDs=parts['particle_id'][mask], root=snap,
                        properties=prop, **kwargs)
            halo_classes.append(halo)
    duration = time.time() - start_time

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('created a list of %s halos in %s' % (
            nice_big_num_str(len(halo_classes)), sec_to_nice_str(duration)))

    return halo_classes


class Halo(object):
    '''
    A class representing an halo or some group of an snapshot represented by IDs.

    Internally the defining IDs are stored and on creation some properties (such
    as total mass and R200) are calculated.

    Args:
        IDs (set, array-like):      The IDs that define the halo.
        halo (Snap):                A sub-snapshot that defines the halo. Only
                                    used if `IDs` is None.
        calc (set, list, tuple):    A list of the properties to calculate at
                                    instantiation.
                                    If calc='all', all defined properties are
                                    calculated.
                                    For an overview of the available properties
                                    see `Halo.calculable_props()` (also see
                                    `Halo.prop_descr(pro_name)`.
        root (Snap):                Some (sub-)snapshot that entirely contains the
                                    halo-defining IDs. It is only needed in case
                                    properties are about to calculate and `halo`
                                    is not given, otherwise it will be set to
                                    `halo.root`.
        properties (dict):          Properties to be set. This is done before the
                                    property calculation, meaning if a property
                                    would be calculated otherwise, it will not be
                                    if specified here.
        testing (bool):             Perform some (rather compute intensive)
                                    consistency checks.
    '''
    __long_name_prop__ = {
        'mass': 'total mass',
        'Mstars': 'total stellar mass',
        'Mgas': 'total gas mass',
        'Mdm': 'total dark matter mass',
        'parts': 'number of particles per species',
        'com': 'center of mass',
        'ssc': 'shrinking sphere center',
        'vel': 'mass-weighted velocity',
        'vel_sigma': 'mass-weighted velocity dispersion',
        'Rmax': 'maximum distance of a particle from com',
        'lowres_part': 'number of low-resolution particles',
        'lowres_mass': 'mass of low-resolution particles',
        'avg_gas_temp': 'mass weighted average gas temperature',
        'rho_gas_avg': 'mass weighted average gas density',
        'mean_stellar_age': 'mean stellar age',
        'gas_half_mass_radius_from_com': '3d half mass radius of gas from com',
        'gas_half_mass_radius_from_ssc': '3d half mass radius of gas from ssc',
        'stars_half_mass_radius_from_com': '3d half mass radius of stars from com',
        'stars_half_mass_radius_from_ssc': '3d half mass radius of stars from ssc'

    }

    for odens in ['vir', '200', '500']:
        prop = 'R' + odens
        __long_name_prop__[prop + '_FoF'] = \
            '%s (Mo+ 2002): (3*M / (4*pi * %s * rho_c))**(1/3.)' % (
                prop, '18*pi**2' if odens == 'vir' else odens)
        if odens == 'vir':
            continue
        for qty in ['R', 'M']:
            prop = qty + odens
            __long_name_prop__[prop + '_ssc'] = \
                'spherical %s with `virial_info` with ssc as center' % prop
            __long_name_prop__[prop + '_com'] = \
                'spherical %s with `virial_info` with com as center' % prop
        del prop, qty
    del odens

    @staticmethod
    def calculable_props():
        '''A list of the properties that can be calculated.'''
        return list(Halo.__long_name_prop__.keys())

    @staticmethod
    def prop_descr(name):
        '''Long description of a property.'''
        return Halo.__long_name_prop__.get(name, '<unknown>')

    def __init__(self, IDs=None, halo=None,
                 calc=None, root=None,
                 properties=None, testing=False):
        if IDs is None:
            if halo is None:
                raise ValueError('Halo needs to be defined by either an ID list '
                                 '(`IDs`) or a sub-snapshot (`halo`)!')
            IDs = halo['ID']

        # the defining property:
        self._IDs = np.array(IDs, copy=True)
        # property dictionary
        self._props = {}

        # add the predefined properties
        if properties is not None:
            for prop, val in properties.items():
                self._props[prop.replace(' ', '_')] = val

        # some testing for consistency
        if testing:
            IDset = set(self._IDs)
            if len(IDset) != len(self._IDs):
                raise ValueError('ID list is not unique!')
            if halo is None and root is not None:
                halo = root[IDMask(self._IDs)]
            if root is None and halo is not None:
                root = halo.root
            if root is not halo.root:
                raise ValueError('`halo` and `root` are inconsistent!')
            if len(IDset - set(root['ID'])) > 0:
                print('WARNING: Not all IDs are in the ' + \
                      'root snapshot!', file=sys.stderr)
        if halo is not None and len(halo) != len(self._IDs):
            print('WARNING: The given halo does not ' + \
                  'contain all specified IDs!', file=sys.stderr)

        # calculate the properties asked for

        if calc is None:
            return
        if halo is None:
            if root is None:
                raise ValueError('Need `halo` or `root` to calculate properties!')
            halo = root[IDMask(self._IDs)]
            if len(halo) != len(self._IDs):
                print('WARNING: The given snapshot does not ' + \
                      'contain all specified IDs!', file=sys.stderr)
        elif root is not None and testing:
            if root is not halo.root:
                print('WARNING: `halo` and `root` are ' + \
                      'inconsistent!', file=sys.stderr)
        if root is None:
            root = halo.root

        if calc == 'all':
            calc = Halo.calculable_props()
        for prop in calc:
            self.calc_prop(prop, halo=halo, root=root, recompute=False)

    @property
    def IDs(self):
        '''The defining IDs.'''
        return self._IDs

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

    def __dir__(self):
        return list(self.__dict__.keys()) + list(self._props.keys()) + \
               [k for k in list(self.__class__.__dict__.keys())
                if not k.startswith('_')]

    def __getattr__(self, name):
        if name.startswith('__'):
            return super(Halo, self).__getattr__(name)
        attr = self._props.get(name, None)
        if attr is not None:
            return attr
        raise AttributeError('%s has no attribute "%s"!' % (self, name))

    def __repr__(self):
        r = '<Halo @%s' % hex(id(self))
        try:
            r += ', N = %s' % nice_big_num_str(len(self))
        except:
            pass
        try:
            r += ', M = %.2g %s' % (self.mass, self.mass.units)
        except:
            pass
        r += '>'
        return r

    def __len__(self):
        if 'parts' in self._props:
            return sum(self._props['parts'])
        else:
            return None

    def _get(self, prop, halo=None, root=None, recompute=False):
        if recompute or prop not in self._props:
            self.calc_prop(prop, halo=halo, root=root)
        return self._props[prop]

    def calc_prop(self, prop='all', halo=None, root=None, recompute=True):
        '''
        Calculate a (defined) property.

        Args:
            prop (str):         The property's name to compute. If it is 'all',
                                all properties in `Halo.calculable_props()` are
                                computed.
            halo (Snap):        The (sub-)snapshot of this halo, i.e. `root[self]`
                                where `root` is the root snapshot.
            root (Snap):        The root snapshot of this halo. Does not need to
                                be specified, but if `halo` is None, it will be
                                set by masking this root with the IDMask of this
                                halo.
            recompute (bool):   Recompute this quantity even if it already is
                                stored in the properties dictionary.

        Returns:
            val (...):      The value of the property calculated (or just taken
                            from the properties dictionary, if recompute=False).
                            In case of calc='all', the entire property dictionary
                            is returned.
        '''
        # preparation
        if not recompute and (isinstance(prop, str) and prop != 'all'
                              and prop in self._props):
            return self._props[prop]
        if halo is None:
            halo = root[self.mask]
        elif root is None:
            root = halo.root
        args = {'halo': halo, 'root': root, 'recompute': recompute}

        # handle special cases
        if prop == 'all':
            for p in Halo.calculable_props():
                self.calc_prop(p, **args)
            return self.props
        if isinstance(prop, (list, tuple, np.ndarray)):
            for p in prop:
                self.calc_prop(p, **args)
            return self.props

        # compute the requested
        if prop == 'mass':
            val = halo['mass'].sum()
        elif prop == 'com':
            val = center_of_mass(halo)
        elif prop == 'parts':
            val = tuple(halo.parts)  # shall not change!
        elif prop in ['Mstars', 'Mgas', 'Mdm']:
            sub = getattr(halo, prop[1:], None)
            if sub is None:
                val = UnitScalar(0.0, halo['mass'].units)
            else:
                val = sub['mass'].sum()
        elif prop == 'vel':
            val = mass_weighted_mean(halo, 'vel')
        elif prop == 'vel_sigma':
            v0 = mass_weighted_mean(halo, 'vel')
            val = np.sqrt(np.sum(mass_weighted_mean(halo, 'vel**2') - v0 ** 2))
        elif prop == 'ssc':
            com = self._get('com', **args)
            R_max = np.percentile(periodic_distance_to(halo['pos'],
                                                       com,
                                                       halo.boxsize),
                                  90)
            val = shrinking_sphere(halo, center=com,
                                   R=R_max, verbose=False)
        elif prop == 'Rmax':
            val = periodic_distance_to(halo['pos'], self._get('com', **args),
                                       halo.boxsize).max()
        elif prop[0] in ['R', 'M'] and len(prop) > 4 and \
                (prop[1:4] == 'vir' or prop[1:4].isdigit()) and prop[4] == '_':
            qty = prop[0]
            odens_n = prop[1:4]
            scheme = prop[5:]
            odens = 18. * np.pi ** 2 if odens_n == 'vir' else float(odens_n)
            if scheme == 'FoF':
                rho_crit = halo.cosmology.rho_crit(z=halo.redshift)
                rho_crit.convert_to(halo['mass'].units / halo['pos'].units ** 3,
                                    subs=halo)
                mass = self._get('mass', **args)
                r3 = 3.0 * mass / (4.0 * np.pi * odens * rho_crit)
                val = r3 ** Fraction(1, 3)
            else:
                if scheme not in ['com', 'ssc']:
                    raise ValueError('Unknown scheme for property "%s"!' % prop)
                center = self._get(scheme, **args)
                Rodens, Modens = virial_info(root, center=center, odens=odens)
                val = Rodens if qty == 'R' else Modens
                # don't waste the additional information!
                for qty in ['R', 'M']:
                    name = qty + odens_n + '_' + scheme
                    # if we recompute some property, it might be requested to keep
                    # old values...
                    if not recompute and name not in self._props:
                        self._props[name] = Rodens if qty == 'R' else Modens
        elif prop == 'lowres_part':
            low = getattr(halo, 'lowres', None)
            val = 0 if low is None else len(low)
        elif prop == 'lowres_mass':
            low = getattr(halo, 'lowres', None)
            if low is None:
                val = UnitScalar(0.0, halo['mass'].units)
            else:
                val = low['mass'].sum()
        elif prop == 'avg_gas_temp':
            val = mass_weighted_mean(halo.gas, 'temp')
        elif prop == 'rho_gas_avg':
            val = mass_weighted_mean(halo.gas, 'rho')
        elif prop == 'mean_stellar_age':
            # only if there are stars in the halo
            if halo.stars['mass'].size > 0:
                val = halo.stars['age'].mean()
            else:
                val = np.nan
        elif prop == 'gas_half_mass_radius_from_com':
            val = half_mass_radius(halo.gas, M=halo.gas['mass'].sum(), center=self._get('com', **args), proj=None)
        elif prop == 'gas_half_mass_radius_from_ssc':
            val = half_mass_radius(halo.gas, M=halo.gas['mass'].sum(), center=self._get('ssc', **args), proj=None)
        elif prop == 'stars_half_mass_radius_from_com':
            if halo.stars['mass'].size > 0:
                val = half_mass_radius(halo.stars, M=halo.stars['mass'].sum(), center=self._get('com', **args), proj=None)
            else:
                val = np.nan

        elif prop == 'stars_half_mass_radius_from_ssc':
            if halo.stars['mass'].size > 0:
                val = half_mass_radius(halo.stars, M=halo.stars['mass'].sum(), center=self._get('ssc', **args), proj=None)
            else:
                val = np.nan

        else:
            raise ValueError('Unknown property "%s"!' % prop)

        self._props[prop] = val
        return val


def nxt_ngb_dist_perc(s, q, N=1000, tree=None, ret_sample=False, verbose=None):
    '''
    Estime the percentile distance to the the next neighbour within the given snapshot.

    Args:
        s (Snap):           The snapshot to use.
        q (int,float):      The percentile to ask for.
        N (int):            The sample size to estimate the distance from.
        tree (cOctree):     The octree class to use, if already present. Will be
                            generated on the fly otherwise.
        ret_sample (bool):  Also return the entire sample drawn.
        verbose (int):      Verbosity level. Default: the gobal pygad verbosity
                            level.
    Returns:
        d (UnitArr):        The q-percentile distance to the next neighbour.
       [dists (UnitArr):    The sample drawn.]
    '''
    if verbose is None:
        verbose = environment.verbose
    from .. import octree
    pos = s['pos'].view(np.ndarray)
    boxsize = s.boxsize.in_units_of(s['pos'].units)
    if tree is None:
        if verbose >= environment.VERBOSE_NORMAL:
            print('building the octree...')
        tree = octree.cOctree(pos)
    if verbose >= environment.VERBOSE_NORMAL:
        print('preparing...')
    d = np.empty(N, dtype=float)
    cond = np.ones(len(s), dtype=np.int32)
    with ProgressBar(np.random.randint(len(s), size=N), label='sampling') as pbar:
        for i in pbar:
            cond[i] = 0
            i_next = tree.find_next_ngb(pos[i], pos, periodic=boxsize, cond=cond)
            cond[i] = 1
            d[pbar.iteration - 1] = dist(pos[i], pos[i_next])
    if ret_sample:
        return UnitArr(np.percentile(d, q), s['pos'].units), \
               UnitArr(d, s['pos'].units)
    else:
        return UnitArr(np.percentile(d, q), s['pos'].units)


def generate_FoF_catalogue(s, l=None, calc='all', FoF=None, exclude=None,
                           max_halos=None, ret_FoFs=False, verbose=None,
                           progressbar=True, **kwargs):
    '''
    Generate a list of Halos defined by FoF groups.

    Args:
        s (Snap):           The snapshot to generate the FoF groups for.
        l (UnitScalar):     The linking length for the FoF groups (for dark
                            matter / all matter it turned out that
                            ( 1500. * rho_crit / median(mass) )^(-1/3) is a
                            reasonable value, which will be used as default, when
                            l=None).
        calc (set, list, tuple):
                            A list of the properties to calculate at instantiation
                            of the Halo instances. (Cf. also `Halo`.)
        FoF (np.array):     An array with group IDs for each particle in the
                            snapshot `s`. If given, `l` is ignored and the FoF
                            groups are not calculated within this function.
        exclude (function): Exclude all halos h for which `exclude(h,s)` returns
                            a true value (note: s[h] gives the halo as a
                            sub-snapshot).
        max_halos (int):    Limit the number of returned halos to the `max_halos`
                            most massive halos. If None, all halos are returned.
        ret_FoFs (bool):    Also return the array with the FoF group indices
                            (sorted by mass).
        verbose (bool):     Verbosity level. Default: the gobal pygad verbosity
                            level.
        progressbar (bool): Whether to show a progress bar for initialising the
                            halo classes (not for generating the FoF groups,
                            though!).
        **kwargs:           Other keywords are passed to `find_FoF_groups` (e.g.
                            `dvmax`).

    Returns:
        halos (list):       A list of all the groups as Halo instances. (Sorted in
                            mass, it not `sort=False` in the `kwargs`.)

        if ret_FoFs==True:
        FoF (np.array):     The FoF group IDs for each particle; particles with no
                            group have ID = NO_FOF_GROUP_ID (cf. `FoF` argument
                            and/or function `find_FoF_groups`).
        N_FoF (int):        The number of FoF groups in `FoF`.
    '''
    if verbose is None:
        verbose = environment.verbose
    if FoF is None:
        if l is None:
            l = (1500. * s.cosmology.rho_crit(s.redshift)
                 / np.median(s['mass'])) ** Fraction(-1, 3)
        FoF, N_FoF = find_FoF_groups(s, l=l, verbose=verbose, **kwargs)
    else:
        N_FoF = len(set(FoF)) - 1

    from ..utils import ProgressBar, DevNull
    if verbose >= environment.VERBOSE_NORMAL and progressbar:
        outfile = sys.stdout
    else:
        outfile = DevNull()
    halos = []
    with ProgressBar(
            range(min(N_FoF, max_halos) if exclude is None else N_FoF),
            show_eta=False,
            show_percent=False,
            label='initialize halos',
            file=outfile) as pbar:
        if not progressbar:
            print('initialize halos from FoF group IDs...')
            sys.stdout.flush()
        for i in pbar:
            h = Halo(halo=s[FoF == i], root=s, calc=calc)
            h.linking_length = l
            if exclude is None or not exclude(h, s):
                halos.append(h)
            if len(halos) == max_halos:
                break

    if verbose >= environment.VERBOSE_NORMAL:
        print('initialized %d halos.' % (len(halos)))
        sys.stdout.flush()

    if ret_FoFs:
        return halos, FoF, N_FoF
    else:
        del FoF
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
    for i in range(3):
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
        if close is not None:
            closest.append(close)

    # if any of them has more than 50% of the mass, we are done
    com_mass = []
    for h in closest:
        com_mass.append(_common_mass(h, h0, s))
        if com_mass[-1] / h0_mass > 0.5:
            return h

    # if no other halo can have more mass than the most massive of the closest, it
    # is this one
    if closest:
        mm_closest, cm = max(list(zip(closest, com_mass)), key=lambda p: p[1])
        if h0_mass - sum(com_mass) < cm:
            return mm_closest

    # iterate and find the most massive progenitor
    # not done in the beginning, since this requires the calculation of the common
    # mass for *all* halos and, hence, is slow
    mmp = max(halos,
              key=lambda h: _common_mass(h, h0, s))
    return mmp if _common_mass(mmp, h0, s) > 0 else None

