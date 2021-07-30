'''
A module to deal with Gadget snapshots. Amongst other objects, it also defines a
class Snap to hold a snapshot.

Example:
    Make sure that the Gadget snapshot used here exists and the destination
    snapshot file does not yet exist. This example copies the snapshot by first
    reading it in and then writing it do the destination file (amongst other
    things). Finally it is tested whether the copy is equivalent to the source
    in the sense that if it is read all blocks are the same. Later on some basic
    sub-snapshots are created and the concept of derived arrays is briefly
    demonstrated.

    >>> snap_tmp_file = 'test.gdt'
    >>> s = Snapshot(module_dir+'snaps/AqA_ICs_C02_200_nm.gdt', physical=False,
    ...          gad_units={'LENGTH':'cMpc/h_0'})
    >>> s
    <Snap "AqA_ICs_C02_200_nm.gdt.0-8"; N=2,951,686; z=127.000>
    >>> s['pos'].units
    load block pos... done.
    Unit("cMpc h_0**-1")
    >>> s.gadget_units
    {'LENGTH': 'cMpc/h_0', 'VELOCITY': 'a**(1/2) km / s', 'MASS': '1e10 Msol / h_0'}
    >>> print('current age of the universe: %s' % s.cosmic_time().in_units_of('Myr'))
    current age of the universe: 12.869734401309785 [Myr]
    >>> s.loadable_blocks()
    ['pos', 'vel', 'ID', 'mass']
    >>> if not set(s.deriveable_blocks()) >= set('Epot vx jzjc vy vz rcyl momentum angmom E dV vrad Ekin temp vcirc r jcirc y x z'.split()):
    ...     print(' '.join(sorted(s.deriveable_blocks())))
    >>> assert set(s.all_blocks()) == set(s.loadable_blocks() + s.deriveable_blocks())
    >>> mwgt_pos = np.tensordot(s['mass'], s['pos'], axes=1).view(UnitArr)
    load block mass... done.
    >>> mwgt_pos.units = s['mass'].units * s['pos'].units
    >>> com = mwgt_pos / s['mass'].sum()
    >>> np.linalg.norm(com - UnitArr([50.2]*3, 'cMpc/h_0')) < 0.5
    True

    And the physical distance between the center of mass and the unweighted mean
    of the positions is:
    (Conversion from 'ckpc/h_0' to 'kpc' is done automatically: the values for 'a'
    and 'h_0' are taken from the associated snapshot and substitued.)
    >>> np.sqrt(np.sum( (com - s['pos'].mean(axis=0))**2 )).in_units_of('kpc', subs=s) < 10.0
    UnitArr(True)

    Whereas the physical dimensions of the simulation's box are:
    >>> s.boxsize
    SimArr(100.0, units="cMpc h_0**-1", snap="AqA_ICs_C02_200_nm.gdt.0-8")
    >>> s['pos'].max(axis=0) - s['pos'].min(axis=0)
    UnitArr([97.78572, 97.81533, 97.83026], dtype=float32, units="cMpc h_0**-1")

    >>> s.load_all_blocks()
    load block vel... done.
    load block ID... done.

    It is also possible to slice entire snapshot, e.g. to access single families
    of the snapshot (gas, stars, dm, bh, baryons) or to mask them with a
    np.ndarray of bools (for more information see SubSnap).
    In fact, in order to access blocks that are only for certains families, one has
    to restrict the snapshot to appropiately.
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_470', physical=False)
    >>> s.gas['rho']
    load block rho... done.
    SimArr([2.1393036e-03, 8.1749255e-04, 9.4836479e-04, ..., 4.6801919e-08,
            3.9165805e-08, 3.4469060e-08],
           dtype=float32, units="1e+10 Msol ckpc**-3 h_0**2", snap="snap_M1196_4x_470":gas)

    The derived block R (radius) is updated automatically, if the block pos it
    depends on changes:
    >>> s['r']
    load block pos... done.
    derive block r... done.
    SimArr([59671.72500919, 59671.40716648, 59671.54775578, ...,
            59672.52310468, 59671.98484173, 59671.74536916],
           units="ckpc h_0**-1", snap="snap_M1196_4x_470")
    >>> assert np.all( s['r'] == dist(s['pos']) )
    >>> s['pos'] -= UnitArr([34622.7,35522.7,33180.5], 'ckpc/h_0')
    >>> s['r']
    derive block r... done.
    SimArr([ 9.41927424,  9.69844804,  9.6864571 , ..., 25.08852488,
            23.95322772, 24.6890346 ],
           units="ckpc h_0**-1", snap="snap_M1196_4x_470")
    >>> s.cache_derived = False
    >>> s.delete_blocks(derived=True)
    >>> s['r'].units
    derive block r... done.
    Unit("ckpc h_0**-1")
    >>> s['r'].units    # now gets re-derived
    derive block r... done.
    Unit("ckpc h_0**-1")
    >>> s.stars['age'].units    # this block is always cached! (see `derived.cfg`)
    load block form_time... done.
    derive block age... done.
    Unit("Gyr")
    >>> s.stars['age'].units
    Unit("Gyr")
    >>> s.cache_derived = True

    >>> s.to_physical_units()
    >>> sub = s[s['r'] < UnitScalar('30 kpc')]
    derive block r... done.
    >>> sub
    <Snap "snap_M1196_4x_470":masked; N=121,013; z=0.000>
    >>> sub.gas
    <Snap "snap_M1196_4x_470":masked:gas; N=11,191; z=0.000>

    # You can also get (almost) arbitrary combinations of the blocks
    >>> s.gas.get('dist(pos)**2 / rho')
    SimArr([1.54323637e-05, 4.28144806e-05, 3.68149753e-05, ...,
            2.25019852e+04, 2.70823020e+04, 3.08840657e+04],
           units="Msol**-1 kpc**5", snap="snap_M1196_4x_470":gas)
    >>> assert np.max(np.abs( (s.get('dist(pos)') - s['r']) / s['r'] )) < 1e-6
    >>> del s['pos']
    >>> s['r']
    SimArr([13.08232505, 13.47006664, 13.45341258, ..., 34.84517185,
            33.268371  , 34.2903246 ], units="kpc", snap="snap_M1196_4x_470")
    >>> s['pos']
    load block pos... done.
    SimArr([[48074.324, 49335.855, 46081.395],
            [48074.023, 49335.785, 46080.992],
            [48073.973, 49335.945, 46081.223],
            ...,
            [48067.195, 49359.18 , 46065.844],
            [48067.496, 49357.258, 46066.246],
            [48066.184, 49357.805, 46066.43 ]],
           dtype=float32, units="kpc", snap="snap_M1196_4x_470")
    >>> s['r']
    derive block r... done.
    SimArr([82877.39261021, 82876.95257089, 82877.14659861, ...,
            82878.50049942, 82877.75400349, 82877.42052568],
           units="kpc", snap="snap_M1196_4x_470")

    One can test for available blocks and families by the 'in' opertator:
    >>> 'r' in s
    True
    >>> if not set(s.available_blocks()) >= set('Epot pot pos jzjc vx vy vz rcyl momentum mass vel angmom jcirc E vrad ID Ekin vcirc r y x z'.split()):
    ...     print(' '.join(sorted(s.available_blocks())))
    >>> s.delete_blocks(derived=True)
    >>> 'r' in s
    True
    >>> 'stars' in s
    True

    New custom blocks can be set easily (though have to fit the (sub-)snapshot):
    >>> sub = s.baryons[s.baryons['metallicity']>1e-3].stars
    load block Z... done.
    derive block elements... done.
    derive block H... done.
    derive block He... done.
    derive block metals... done.
    derive block metallicity... done.
    >>> s.stars['new'] = np.ones(len(s.stars))
    >>> sub['new']
    SimArr([1., 1., 1., ..., 1., 1., 1.], snap="snap_M1196_4x_470":stars)

    >>> dest_file = module_dir+'test.gad'
    >>> assert not os.path.exists(dest_file)
    >>> gadget.write(sub, dest_file, blocks=['pos', 'ID', 'r'])
    load block vel... done.
    load block ID... done.
    load block mass... done.
    derive block r... done.
    writing block POS  (dtype=float32, units=[ckpc h_0**-1])... done.
    writing block VEL  (dtype=float32, units=[a**1/2 km s**-1])... done.
    writing block ID   (dtype=uint32, units=[1])... done.
    writing block MASS (dtype=float32, units=[1e+10 Msol h_0**-1])... done.
    writing block R    (dtype=float32, units=[kpc])... done.
    >>> sub_copy = Snapshot(dest_file, physical=True)
    >>> sub_copy['r'].units = 'kpc' # notice: block r gets *loaded*!
    ...                             # the units are those from before!
    ...                             # and are not in the config!
    load block r... done.
    >>> assert sub.parts == sub_copy.parts
    >>> assert np.max(np.abs((sub['pos'] - sub_copy['pos']) / sub['pos'])) < 1e-6
    load block pos... done.
    >>> assert np.max(np.abs((sub['r'] - sub_copy['r']) / sub['r'])) < 1e-6
    >>> assert np.abs((sub.boxsize - sub_copy.boxsize) / sub.boxsize) < 1e-6
    >>> import os
    >>> os.remove(dest_file)
    >>> del sub, sub_copy
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_470', physical=False)
    >>> gadget.write(s, dest_file)  # doctest:+ELLIPSIS
    load block pos... done.
    load block vel... done.
    load block ID... done.
    load block mass... done.
    load block u... done.
    ...
    writing block POS  (dtype=float32, units=[ckpc h_0**-1])... done.
    writing block VEL  (dtype=float32, units=[a**1/2 km s**-1])... done.
    writing block ID   (dtype=uint32, units=[1])... done.
    writing block MASS (dtype=float32, units=[1e+10 Msol h_0**-1])... done.
    writing block U    (dtype=float32, units=[a**2 km**2 s**-2])... done.
    ...
    >>> s2 = Snapshot(dest_file, physical=False)
    >>> s2.load_all_blocks()    # doctest:+ELLIPSIS
    load block ... done.
    ...
    >>> for name in s.loadable_blocks():
    ...     b  = s.get_host_subsnap(name)[name]
    ...     b2 = s2.get_host_subsnap(name)[name]
    ...     err = np.max( np.abs((b - b2) / b) )
    ...     if err[np.isfinite(err)] > 1e-6:
    ...         print(name, b, b2)
    >>> os.remove(dest_file)

    >>> dest_file_hdf5 = module_dir+'test.hdf5'
    >>> assert not os.path.exists(dest_file_hdf5)
    >>> gadget.write(s, dest_file_hdf5) # doctest:+ELLIPSIS
    ...
    writing block POS  (dtype=float32, units=[ckpc h_0**-1])... done.
    writing block VEL  (dtype=float32, units=[a**1/2 km s**-1])... done.
    writing block ID   (dtype=uint32, units=[1])... done.
    writing block MASS (dtype=float32, units=[1e+10 Msol h_0**-1])... done.
    writing block U    (dtype=float32, units=[a**2 km**2 s**-2])... done.
    ...
    >>> tmp = Snapshot(dest_file_hdf5, physical=False)
    >>> gadget.write(tmp, dest_file)    # doctest:+ELLIPSIS
    load block pos... done.
    load block vel... done.
    load block ID... done.
    load block mass... done.
    load block u... done.
    ...
    writing block POS  (dtype=float32, units=[ckpc h_0**-1])... done.
    writing block VEL  (dtype=float32, units=[a**1/2 km s**-1])... done.
    writing block ID   (dtype=uint32, units=[1])... done.
    writing block MASS (dtype=float32, units=[1e+10 Msol h_0**-1])... done.
    writing block U    (dtype=float32, units=[a**2 km**2 s**-2])... done.
    ...
    >>> del tmp
    >>> os.remove(dest_file_hdf5)
    >>> s2 = Snapshot(dest_file, physical=False)
    >>> for name in s.loadable_blocks():    # doctest:+ELLIPSIS
    ...     b  = s.get_host_subsnap(name)[name]
    ...     b2 = s2.get_host_subsnap(name)[name]
    ...     assert np.all( b == b2 )
    load block ...
    >>> assert s.redshift == s2.redshift
    >>> for name, prop in s.properties.items():
    ...     if not prop == s2.properties[name]:
    ...         print(name, prop, s2.properties[name])
    >>> os.remove(dest_file)
    >>> del s2

    Some basic testing for (reading) HDF5 snapshots:
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_conv_470.hdf5', physical=False)
    >>> s
    <Snap "snap_M1196_4x_conv_470.hdf5"; N=2,079,055; z=0.000>
    >>> s.cosmology
    FLRWCosmo(h_0=0.72, O_Lambda=0.74, O_m=0.26, O_b=0.0416, sigma_8=None, n_s=None)
    >>> s['pos']
    load block pos... done.
    SimArr([[34613.516, 35521.816, 33178.605],
            [34613.297, 35521.766, 33178.316],
            [34613.26 , 35521.883, 33178.48 ],
            ...,
            [34608.383, 35538.61 , 33167.41 ],
            [34608.598, 35537.227, 33167.7  ],
            [34607.652, 35537.62 , 33167.832]],
           dtype=float32, units="ckpc h_0**-1", snap="snap_M1196_4x_conv_470.hdf5")
    >>> s['r']
    derive block r... done.
    SimArr([59671.72500919, 59671.40716648, 59671.54775578, ...,
            59672.52310468, 59671.98484173, 59671.74536916],
           units="ckpc h_0**-1", snap="snap_M1196_4x_conv_470.hdf5")
    >>> s.parts
    [921708, 1001472, 56796, 19315, 79764, 0]
    >>> sub = s.SubSnap([0,1,2,4,5])
    >>> sub
    <Snap "snap_M1196_4x_conv_470.hdf5":pts=[0,1,2,4,5]; N=2,059,740; z=0.000>
    >>> sub.parts
    [921708, 1001472, 56796, 0, 79764, 0]
    >>> s.SubSnap([0,2,4]).parts
    [921708, 0, 56796, 0, 79764, 0]
'''
__all__ = ['write', 'Snapshot', 'Snap', 'SubSnapshot']

import sys
import os.path
from .. import gadget
from ..gadget import write
from .. import physics
from .. import utils
from ..units import *
from ..utils import dist
from .. import environment
from ..environment import module_dir
import numpy as np
import warnings
import ast
import weakref
from . import derived
import fnmatch


class Snapshot(object):
    '''
    A class holding a Gadget snapshot.

    Args:
        physical (bool):    Whether to convert to physical units on loading.
        cosmological (bool):Whether the simulation was a cosmological one.
        gad_units (dict):   Alternative base units (LENGTH, VELOCITY, MASS) for
                            this snapshot.
    '''

    def _initsnap(self, filename, base, suffix, physical=False, load_double_prec=False, cosmological=None,
                  gad_units=None, unclear_blocks=None, H_neutral_only=None):
        '''
        Create a snapshot from file (without loading the blocks, yet).

        Args:
            filename (str):         The path to the snapshot. If it is distributed
                                    over several files, you shall omit the trailing
                                    (of inbetween in case of an HDF5 file) '.0'.
            physical (bool):        Whether to convert to physical units on loading.
            load_double_prec (bool):Force to load all blocks in double precision.
                                    Equivalent with setting the snapshots attribute.
            cosmological (bool):    Explicitly tell if the simulation is a
                                    cosmological one.
            gad_units (dict):       Alternative base units (LENGTH, VELOCITY, MASS)
                                    for this snapshot. The default base units units
                                    are updated, meaning one can also just change one
                                    them.
            unclear_blocks (str):   What to do the blocks for which the block info is
                                    unclear (cannot be infered). Possible modes are:
                                    * exception:    raise an IOError
                                    * warning:      print a warning to the stderr
                                    * ignore:       guess what
                                    If it is None, the value from the `gadget.cfg` is
                                    taken.

        Raises:
            IOError:            If the snapshot does not exist.
            RuntimeError:       If the information could not be infered or if the
                                given dtype of the given family is unknown.
        '''

        from .sim_arr import SimArr
        s = self
        s._filename = os.path.abspath(base + suffix)
        s._descriptor = os.path.basename(base) + suffix
        # need first (maybe only) file for basic information
        greader = gadget.FileReader(filename, unclear_blocks=unclear_blocks)
        s._file_handlers = [greader]

        s._block_avail = {
            block.name: block.ptypes
            for block in greader.infos()
            if block.dtype is not None}

        s._N_part = list(map(int, greader.header['N_part_all']))
        s._time = greader.header['time']
        s._redshift = greader.header['redshift']
        s._boxsize = SimArr(greader.header['boxsize'],
                            units=s._gad_units['LENGTH'],
                            snap=s)
        s._cosmology = physics.FLRWCosmo(  # Note: Omega_b at default!
            h_0=greader.header['h_0'],
            Omega_Lambda=greader.header['Omega_Lambda'],
            Omega_m=greader.header['Omega_m'])
        s._properties = {k: v for k, v in greader.header.items()
                         if k not in ['N_part', 'mass', 'time', 'redshift', 'N_part_all',
                                      'N_files', 'h_0', 'Omega_Lambda', 'Omega_m', 'boxsize',
                                      'unused']}
        if s._cosmological is None:
            s._cosmological = abs(s.scale_factor - s.time) < 1e-6
        s._load_double_prec = bool(load_double_prec)

        if physical:
            s._boxsize.convert_to(s._boxsize.units.free_of_factors(['a', 'h_0']),
                                  subs=s)

        if greader.header['N_files'] > 1:
            s._descriptor += '.0-' + str(greader.header['N_files'])
            # ensure Python int's to avoid overflows
            N_part = list(map(int, greader.header['N_part']))
            for n in range(1, greader.header['N_files']):  # first already done
                filename = base + '.' + str(n) + suffix
                greader = gadget.FileReader(filename, unclear_blocks=unclear_blocks)
                s._file_handlers.append(greader)
                # ensure Python int's to avoid overflows
                for i in range(6): N_part[i] += int(greader.header['N_part'][i])
                # update loadable blocks:
                for block in greader.infos():
                    if block.name in s._block_avail:
                        s._block_avail[block.name] = [(o or n) for o, n \
                                                      in zip(s._block_avail[block.name], block.ptypes)]
                    else:
                        s._block_avail[block.name] = block.ptypes
            if N_part != s._N_part:
                # more particles than fit into a native int
                s._N_part = N_part

        # Process block names: make standard names lower case (except ID) and replace
        # spaces with underscores for HDF5 names. Also strip names
        s._load_name = {}
        for name, block in list(s._block_avail.items()):
            if s._file_handlers[0]._format == 3 \
                    and '%-4s' % name not in gadget.std_name_to_HDF5:
                new_name = name.strip()
            else:
                new_name = name.strip().lower()

            # some renaming
            if new_name in ['id', 'z']:
                new_name = new_name.upper()
            elif new_name == 'age':
                new_name = 'form_time'

            s._load_name[new_name] = name
            s._block_avail[new_name] = s._block_avail[name]
            if name != new_name:
                if environment.verbose >= environment.VERBOSE_TALKY \
                        and new_name.lower() != name.strip().lower():
                    print('renamed block "%s" to %s' % (name, new_name))
                del s._block_avail[name]  # blocks should not appear twice
        # now the mass block is named 'mass' for all cases (HDF5 or other)
        s._block_avail['mass'] = [n > 0 for n in s._N_part]

        s.fill_derived_rules()

        s._descriptor = '"' + s._descriptor + '"'

        return s

    def __init__(self, filename, physical=False, load_double_prec=False, cosmological=None,
                                 gad_units=None, unclear_blocks=None, H_neutral_only=None):

        filename = os.path.expandvars(filename)
        filename = os.path.expanduser(filename)
        # handle different filenames, e.g. cases where the snapshot is distributed
        # over severale files
        base, suffix = os.path.splitext(filename)
        if suffix != '.hdf5':
            base, suffix = filename, ''
        if not os.path.exists(filename):
            filename = base + '.0' + suffix
            if not os.path.exists(filename):
                raise IOError('Snapshot "%s%s" does not exist!' % (base, suffix))

        self._filename              = '<none>'
        self._descriptor            = 'new'
        self._file_handlers         = []
        self._gad_units             = gadget.default_gadget_units.copy()
        if gad_units:
            self._gad_units.update( gad_units )
        self._load_name             = {}    # from attribute name to actual block
                                            # name as to use for loading
        self._block_avail           = {}    # keys are lower case, stripped names
                                            # or the hdf5 names with underscores
        self._derive_rule_deps      = {}
        self._blocks                = {}
        self._cosmological          = cosmological
        self._N_part                = [0] * 6
        self._time                  = 1.0
        self._redshift              = 0.0
        self._boxsize               = 0.0
        self._cosmology             = physics.Planck2013()
        self._properties            = {}
        self._load_double_prec      = False
        self._phys_units_requested  = bool(physical)
        self._trans_at_load         = []
        self._root                  = self
        self._base                  = None
        self._cache_derived         = derived.general['cache_derived']
        self._always_cache          = set() # _derive_rule_deps is empty; gets
                                            # filled in Snap() with
                                            # derived.general['always_cache']
        # Horst: copy configuration to snapshot-property
        if H_neutral_only is None:
            self._H_neutral_only        = gadget.config.general['H_neutral_only']
        else:
            self._H_neutral_only        = H_neutral_only

        # Actual initialization is done in the factory function Snap. Just do some
        # basic setting of the attributes to ensure that even snapshot created by
        # just _Snap are somewhat functioning.
        self._initsnap(filename, base, suffix, physical=physical, load_double_prec=load_double_prec, cosmological=cosmological,
                  gad_units=gad_units, unclear_blocks=unclear_blocks, H_neutral_only=H_neutral_only)

    @property
    def filename(self):
        '''The absolut path of the snapshot. For multiple files, omitting the
        numbers in the filename, so that it can be reused for loading.'''
        return self._root._filename

    @property
    def descriptor(self):
        '''A short description of the origin of the (sub-)snapshot.'''
        return self._descriptor

    @property
    def gadget_units(self):
        return self._root._gad_units.copy()

    @property
    def cosmological(self):
        '''Is the simulation cosmological?'''
        return self._root._cosmological

    @property
    def N_files(self):
        '''The number of file the snapshot is distributed over.'''
        return len(self._root._file_handlers)

    # Horst: new property-method to query neutral-H setting
    @property
    def H_neutral_only(self):
        '''The configuration property for neutral H.'''
        return self._root._H_neutral_only

    def __len__(self):
        return sum(self._N_part)

    @property
    def parts(self):
        '''A list of the number of particles of each particle type in the
        snapshot.'''
        return self._N_part[:]

    @property
    def time(self):
        '''The time variable of the snapshot.'''
        return self._root._time

    @property
    def redshift(self):
        '''The redshift of the snapshot.'''
        return self._root._redshift

    @property
    def boxsize(self):
        '''The side length of the simulation box.'''
        return self._root._boxsize

    @property
    def cosmology(self):
        '''
        A copy of the simulations cosmology as taken from the header.

        The baryon fraction is the default (of the matter fraction) as
        physics.FLRWCosmo yields.
        '''
        return self._root._cosmology.copy()

    @property
    def properties(self):
        '''A copy of some more properties from the header(s).'''
        return self._root._properties.copy()

    @property
    def cache_derived(self):
        '''Whether to cache derived blocks (or to calculate them every time).'''
        return self._root._cache_derived

    @cache_derived.setter
    def cache_derived(self, sd):
        self._root._cache_derived = bool(sd)

    @property
    def always_cached_derived(self):
        '''List of blocks that are always changed.'''
        return list(self._root._always_cache)

    @property
    def load_double_prec(self):
        '''Whether to convert all blocks with floating point data to double
        precision on load.'''
        return self._root._load_double_prec

    @load_double_prec.setter
    def load_double_prec(self, do):
        self._root._load_double_prec = bool(do)

    @property
    def phys_units_requested(self):
        '''Whether to convert all blocks to physical units on load.'''
        return self._root._phys_units_requested

    @phys_units_requested.setter
    def phys_units_requested(self, do):
        self._root._phys_units_requested = bool(do)

    @property
    def root(self):
        '''The underlying snapshot of this (sub-)snapshot (self for root
        itself).'''
        return self._root

    @property
    def base(self):
        '''The snapshot this one is sliced/masked from (None for root).'''
        return self._base

    @property
    def scale_factor(self):
        '''
        The scale factor of the universe at the time of the snapshot.

        This is *not* simply the time, but calculated from the redshift, in order
        to also be correct in the case of non-cosmological simulations.
        '''
        return physics.z2a(self.redshift)

    def SubSnap(self, mask):
        '''
        A factory function for creating masked/sliced sub-snapshots.

        Sub-snapshots should always be instantiated with this function or the bracket
        notation (or FamilySubSnap).

        Args:
            base (Snap):    The snapshot to create the sub-snapshot from.

            mask:           The mask to use. It can be:

                            * slice:
                                Slice the entire snapshot as one would do with
                                np.ndarray's. Blocks are then slices of the base
                                snapshot. It is taken care of those which are now
                                available, but were not for the base.

                            * np.ndarray[bool]:
                                Has to have the length of the snapshot. The
                                sub-snapshot then consists of all the particles for
                                which the entry is True. Otherwise the same as for a
                                slice.

                            * particle types (list):
                                Create a sub-snapshot of the specified particle types
                                only.

                            * mask class (SnapMask):
                                Create a sub-snapshot according to the mask.

                            * mask class (Halo):
                                Create a sub-snapshot according to the mask of the
                                halo (an IDMask).

                            * index list (np.ndarray[int]):
                                This can in fact also be a tuple with one element,
                                which is such a index list, as returned by np.where.
                                This is similar to passing a boolean mask, but here
                                the indices (not the IDs) are passed explicitly.

                                With contrast to passing this directly to the block
                                arrays, the particles are *not reordered* here and
                                every particle can only *occur once*, since in the
                                back, this converts the mask to a boolean one with the
                                use of sets. I did this for a combination of little
                                effort in writing code (reusing) and speed (using
                                sets).
                                TODO:
                                    - Change this behaviour?
                                    - Really keep this option?

        Returns:
            sub (_SubSnap):     The sub-snapshot.

        Raises:
            KeyError:           If the mask was not understood.
        '''
        from .masks import SnapMask
        from ..analysis.halo import Halo

        base = self

        if isinstance(mask, slice) \
                or (isinstance(mask, np.ndarray) and mask.dtype == bool):
            # slices and masks are handled directly by _SubSnap
            return SubSnapshot(base, mask)

        elif isinstance(mask, list):
            ptypes = sorted(set(mask))
            # precalculating N_part is faster than the standard way in _SubSnap
            N_part = [(base._N_part[pt] if pt in ptypes else 0) for pt in range(6)]
            if utils.is_consecutive(ptypes):
                # slicing is faster than masking!
                sub = slice(sum(base._N_part[:ptypes[0]]),
                            sum(base._N_part[:ptypes[-1] + 1]))
            else:
                l = [(np.ones(base._N_part[pt], bool) if pt in ptypes
                      else np.zeros(base._N_part[pt], bool)) for pt in range(6)]
                sub = np.concatenate(l)
            sub = SubSnapshot(base, sub, N_part)
            sub._descriptor = base._descriptor + ':pts=' + str(ptypes).replace(' ', '')
            return sub

        elif isinstance(mask, SnapMask) or isinstance(mask, Halo):
            if isinstance(mask, Halo):
                mask = mask.mask
            sub = SubSnapshot(base, mask.get_mask_for(base))
            sub._descriptor = base._descriptor + ':' + str(mask)
            return sub

        elif isinstance(mask, np.ndarray) and mask.dtype.kind == 'i':
            # I probably do not want to keep this. Increases backward compability,
            # though.
            warnings.warn('Consider using the faster boolean masks!')
            # is convering into boolean array the best choice?
            warnings.warn('Indexed snapshot masking does not reorder!')
            idx_set = set(mask)
            if len(idx_set) < len(mask):
                print("WARNING: lost %d" % (len(mask) - len(idx_set)) + \
                      " particles in snapshot masking!", file=sys.stderr)
            mask = np.array([(i in idx_set) for i in range(len(base))])
            return SubSnapshot(base, mask)

        elif isinstance(mask, tuple) and len(mask) == 1 \
                and isinstance(mask[0], np.ndarray) and mask[0].dtype.kind == 'i':
            return SubSnapshot(base, mask[0])

        else:
            raise KeyError('Mask of type %s not understood.' % type(mask).__name__)

    def write(self, filename, **kwargs):
        '''
        Write this (sub-)snapshot to the given file.

        For more information see `gadget.write` as this function just calls it.

        Args:
            filename (str):     The path to write the snapshot to.
            **kwargs:           Further (keyword) arguments are simply passed to
                                `gadget.write` (by default it is
                                `gad_units=self.root._gad_units`).
        '''
        return gadget.write(self, filename, gad_units=self.root._gad_units, **kwargs)

    def headers(self, idx=None):
        '''
        A list of the header(s) as dict's. For common features see `properties`.

        Args:
            idx (int):      If not None, do not return the entire list, but just
                            the header of the file of given number.
        '''
        if idx is None:
            return [reader.header.copy() for reader in self._root._file_handlers]
        else:
            return self._root._file_handlers[idx].header.copy()

    def families(self):
        '''Return the names of the particle families (at least partly) present in
        this snapshot.'''
        families = set()
        for name, ptypes in gadget.families.items():
            if sum(self._N_part[pt] for pt in ptypes):
                families.add(name)
        return list(families)

    def available_blocks(self):
        '''The names of the blocks that are available for all particle types of
        this snapshot.'''
        return [ name for name, ptypes
                in self._root._block_avail.items()
                if all([(Np==0 or s) for s,Np in zip(ptypes,self._N_part)]) ]

    def loadable_blocks(self):
        '''The names of all blocks of this snapshot.'''
        return list(self._root._load_name.keys())

    def deriveable_blocks(self):
        '''The names of all blocks of this snapshot.'''
        return list(self._root._derive_rule_deps.keys())

    def all_blocks(self):
        '''The names of all blocks of this snapshot.'''
        return list(self._root._block_avail.keys())

    def cosmic_time(self):
        '''The cosmic time (i.e. the current universe age).'''
        return self.cosmology.cosmic_time(self.redshift)

    def __repr__(self):
        return '<Snap %s; N=%s; z=%.3f>' % (
                self._descriptor,
                utils.nice_big_num_str(len(self)),
                self._root.redshift)

    def fill_derived_rules(self, rules=None, clear_old=False):
        '''
        Fill the derived rules for (the root of) this snapshot.

        Args:
            rules (dict):       A dictionary of the rules to add. The keys are the
                                names of the blocks (blocks that can be loaded
                                will not be added!). The entries shall be strings
                                with the rule.
                                Defaults to the list loaded from `derived.cfg`.
            clear_old (dict):   Whether to first clear any already existing rules.
        '''
        if not self is self.root:
            self.root.fill_derived_rules(rules, clear_old)
            return

        if rules is None:
            rules = derived._rules
        rules = rules.copy()

        if clear_old:
            for name in self._derive_rule_deps.keys():
                self._block_avail.pop(name,None)
            self._derive_rule_deps = {}

        # calculate the dependencies and particle types of the derived blocks
        # define 'dV' by 'vol_def_x' if not explicitly given
        if 'dV' not in rules:
            x = gadget.config.general['vol_def_x']
            if x != '<undefined>':
                rules['dV'] = '%s / kernel_weighted(gas,%s)' % (x,x)

        # remove derived blocks that can be loaded
        for name in list(rules.keys()):
            if name in self._load_name:
                del rules[name]

        # calculate the dependencies
        changed = True
        while changed:
            changed = False
            for name, rule in rules.items():
                if name in self._load_name:
                    continue    # this derived block can actually be loaded
                ptypes, deps = derived.ptypes_and_deps(rule, self)
                if name in self._block_avail:
                    if ptypes!=self._block_avail[name] \
                            or deps!=self._derive_rule_deps[name][1]:
                       changed = True
                else:
                   changed = True
                self._block_avail[name] = ptypes
                self._derive_rule_deps[name] = (rule, deps)

        # now remove those derived blocks, that depend on non-existent blocks
        not_available = set()
        removed = -1
        while removed != len(not_available):
            removed = len(not_available)
            for name, (rule, deps) in self._derive_rule_deps.items():
                if not (any(self._block_avail[name]) and (deps-not_available)):
                    not_available.add(name)
            for name in not_available:
                self._block_avail.pop(name,None)
                self._derive_rule_deps.pop(name,None)

        self._always_cache = set(self._derive_rule_deps.keys()) \
                            & derived.general['always_cache']

    def __dir__(self):
        # Add the families available such that they occure in tab completion in
        # iPython, for instance.
        # The other list added here seems to include excatly the properties.
        return [k for k in list(self.__dict__.keys())
                        if not k.startswith('_')] + \
                [k for k in list(self._root.__class__.__dict__.keys())
                        if not k.startswith('_')] + \
                self.families()

    def get_host_subsnap(self, block_name):
        '''
        Find the (sub-)snapshot, that stores a given block.

        The family sub-snapshot of the root that can use the block unsliced/-mask
        is the one to store it. Needless to say, if the block is available for all
        particle types, it is the root.

        Args:
            block_name (str):   The (attribute) name of the block.

        Returns:
            host (Snap):        The host (sub-)snapshot.
        '''
        root = self._root
        parts = root._N_part
        # It might happen, that a block is tagged as present for some particle
        # types that does not exist at all. Hence, the following:
        avail = root._block_avail[block_name]
        ptypes = [pt for pt in range(6) if (avail[pt] and parts[pt])]

        # block available for all families? -> root
        if not any( [parts[pt] for pt in range(6) if pt not in ptypes] ):
            return root

        # find biggest family sub-snapshot (that is an attribute of root) which has
        # the block
        for fam, fptypes in gadget.families.items():
            if fptypes == ptypes:
                return getattr(root, fam)

        return None

    def iter_blocks(self, present=True, block=True, name=False, forthis=True):
        '''
        Iterate all blocks, even those that are not for all particles.

        If present==False and block==True, this loads / derives all the not yet
        present (loaded/derived) blocks.

        Args:
            present (bool): If set, only iterate over the already loaded / derived
                            blocks.
            block (bool):   Whether to yield the (full) block itself.
            name (bool):    Whether to yield the block name.
            forthis (bool): Only include blocks that are available for this entire
                            (sub-)snapshot.

        Yields:
            block (UnitArr):    The full block, if only this was requested,
            name (str):         the name of the block, if only that was requested,
            OR
            (block, name):      both as a tuple, if both are requested.
            (And just nothing, if neither was requested.)
        '''
        done = set()
        if forthis:
            names = iter(self._blocks.keys()) if present else self.available_blocks()
        else:
            names = iter(self._root._block_avail.keys())
        for block_name in names:
            if block_name in done:
                continue
            host = self.get_host_subsnap(block_name)
            if host is None:
                continue
            if not present or block_name in host._blocks:
                if block and name:
                    yield host[block_name], block_name
                elif block:
                    yield host[block_name]
                elif name:
                    yield block_name

    def _get_block(self, name):
        '''
        This gets called, if a block is not yet set in self._blocks.

        Overwritten for _SubSnap, where care is taken for sub-snapshots that are
        not directly sliced/mased from the block's host. Here, for the root, it
        simply calls self._host_get_block (and asserts that the block is
        available).
        '''
        assert name in self.available_blocks()
        assert name not in self._blocks
        block = self._host_get_block(name)
        return block

    def __getattr__(self, name):
        # Create a family sub-snapshot and set it as attribute.
        if name in gadget.families:
            fam_snap = _FamilySubSnap(self, name)
            setattr(self, name, fam_snap)
            return fam_snap
        # Explicitly set attributes (like self.redshift and self.gas), member
        # functions, and properties are already handles before __getattr__ is
        # called, hence, we now have an attribute error:
        raise AttributeError('%r object has no ' % type(self).__name__ + \
                             'attribute %r' % name)

    def __getitem__(self, key):
        # strings are block names
        if isinstance(key, str):
            block = self._blocks.get(key,None)
            if block is not None:
                return block
            if key in self.available_blocks():
                # TODO: don't need to pass it though all the functions...
                return self._get_block(key)
            elif key in self._root._block_avail:
                raise KeyError('Block "%s" is not available for all ' % key +
                               'particle types of this (sub-)snapshot.')
            else:
                raise KeyError('(Sub-)Snapshot %r has no block "%s".' % (
                                    self, key))
        # postprone the import to here to speed up the access to blocks
        from .masks import SnapMask
        from ..analysis.halo import Halo
        if isinstance(key, (slice,np.ndarray,list,SnapMask,Halo,tuple)):
            # Handling of the index is fully done by the factory function.
            return self.SubSnap(key)
        else:
            raise KeyError(repr(key))

    def __setitem__(self, key, value):
        if key in self.available_blocks():
            old = self._blocks.get(key,None)
            if old is None:
                warnings.warn('Need to load/derive an available blocks before ' +
                              'overwriting it to some custom value!')
                old = self[key]
            from .sim_arr import SimArr
            if not isinstance(value, SimArr):
                value = SimArr(value, snap=self)
            if value.shape != old.shape:
                raise RuntimeError('Already existing blocks must not change ' +
                                   'their shape (here: "%s" with %r)!' % (
                                       key,old.shape))
            value.dependencies.update( old.dependencies )
            value.invalidate_dependencies()
            self._blocks[key] = value
        else:
            self._add_custom_block(value, key)

    def __delitem__(self, name):
        if name in self.available_blocks():
            try:
                host = self.get_host_subsnap(name)
                del host._blocks[name]
                if name not in self._root._load_name \
                        and name not in self._root._derive_rule_deps:
                    # delete custom blocks entirely
                    del self._root._block_avail[name]
            except KeyError:
                pass
        else:
            raise KeyError('(Sub-)Snapshot %r has no block "%s".' % (
                                    self, name))

    # iterate over all available loaded blocks (including custom and derived
    # blocks)
    def __iter__(self):
        for name, block in self.iter_blocks(present=True, block=True, name=True,
                                            forthis=True):
            yield name, block

    # whether a block of familiy is in the (sub-)snapshot (by name)
    def __contains__(self, el):
        return el in self.available_blocks() or el in self.families()

    def _host_load_block(self, name):
        '''
        Load a block and add it to its host.

        Note:
            For getting blocks -- also derived ones -- call _host_get_block!

        Args:
            name (str): The stripped name of the block (as it is in
                        self._root._load_name.keys()).

        Returns:
            block (SimArr):     The (entire) block.
        '''
        from .sim_arr import SimArr

        root = self._root
        if not root._file_handlers:
            raise RuntimeError('No file(s) to load from.')
        if not name in root._load_name:
            raise ValueError("There is no block '%s' to load!" % name)

        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('load block %s%s...' % ('"%s" as '%root._load_name[name]
                    if environment.verbose >= environment.VERBOSE_TALKY else '',
                    name), end=' ')
            sys.stdout.flush()

        block_name = root._load_name[name]
        if len(root._file_handlers) == 1:
            block = root._file_handlers[0].read_block(block_name,
                                                      gad_units=root._gad_units)
        else:
            first = True
            for reader in root._file_handlers:
                # getting masses should always return something - even if there is
                # no such block, but the masses are given in the header only
                if not reader.has_block(block_name) and block_name!='mass':
                    continue
                read = reader.read_block(block_name, gad_units=root._gad_units)
                units = read.units
                if first:
                    blocks = [read[int(np.sum(reader.header['N_part'][:pt]))
                                    :np.sum(reader.header['N_part'][:pt+1])] \
                                for pt in range(6)]
                    first = False
                else:
                    for pt in range(6):
                        if not reader.header['N_part'][pt]:
                            continue
                        read_pt = read[int(np.sum(reader.header['N_part'][:pt]))
                                       :np.sum(reader.header['N_part'][:pt+1])]
                        blocks[pt] = np.concatenate((blocks[pt], read_pt),
                                                    axis=0).view(UnitArr)
                        blocks[pt].units = units
            assert not first
            block = np.concatenate(blocks, axis=0).view(UnitArr)
            block.units = units
        block = block.view(SimArr)
        block._snap = weakref.ref(self)
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('done.')
            sys.stdout.flush()

        if self.load_double_prec and block.dtype.kind == 'f' \
                and block.dtype != 'float64':
            if environment.verbose >= environment.VERBOSE_TALKY:
                print('convert to double precision...', end=' ')
                sys.stdout.flush()
            block = block.astype(np.float64)
            if environment.verbose >= environment.VERBOSE_TALKY:
                print('done.')
                sys.stdout.flush()

        if root._phys_units_requested:
            self._convert_block_to_physical_units(block, name)

        host = self.get_host_subsnap(name)
        host._blocks[name] = block

        for trans in root._trans_at_load:
            if name in trans._change:
                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print('apply stored %s to block %s...' % (
                        trans.__class__.__name__, name), end=' ')
                    sys.stdout.flush()
                trans.apply_to_block(name,host)
                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print('done.')
                    sys.stdout.flush()

        # if there are still dependent blocks, add them as dependencies and
        # invalidate them
        for derived_name, (rule, deps) in root._derive_rule_deps.items():
            if name in deps:
                host = self.get_host_subsnap(derived_name)
                if derived_name in host._blocks:
                    block.dependencies.add(derived_name)
        block.invalidate_dependencies()

        return block

    def _host_derive_block(self, name):
        '''
        Derive a block from the stored rules (and add it to its host).

        Note:
            For getting blocks call _host_get_block!

        Args:
            name (str):             The name of the derived block (as it is in
                                    self._root._derive_rule_deps).

        Returns:
            block (SimArr):     The (entire) block.
        '''
        root = self._root
        if not name in root._derive_rule_deps:
            raise ValueError("There is no block '%s' to derive!" % name)

        host = self.get_host_subsnap(name)
        rule, deps = root._derive_rule_deps[name]

        orig_caching_state = self._root._cache_derived
        for cache_name in self._root._always_cache:
            # in order to ensure that the automatic updating works (the dependencies
            # connect from the loading blocks to the always cached derived block),
            # temporarily turn on _cache_derived globally!
            if fnmatch.fnmatch(name, cache_name):
                self._root._cache_derived = True
                break

        # pre-load all needed (and not yet loaded) blocks in order not to mess up
        # the output (too much)
        dep_blocks = {}
        for dep in sorted(deps):
            if self._root._cache_derived:
                host[dep]
            else:
                # re-use stored blocks
                dep_blocks[dep] = host.get(dep, None)
                if dep_blocks[dep] is None:
                    # ... only calculate others, but do not store them
                    dep_blocks[dep] = host._host_derive_block(dep, False)

        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('derive block %s%s...' % (name, ' := "%s"'%rule \
                    if environment.verbose >= environment.VERBOSE_TALKY else ''), end=' ')
            sys.stdout.flush()
        block = host.get(rule, namespace=None if self._root._cache_derived else dep_blocks)
        if self._root._cache_derived:
            for dep in deps:
                host[dep].dependencies.add(name)
            host._blocks[name] = block
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print('done.')
            sys.stdout.flush()

        self._root._cache_derived = orig_caching_state

        return block

    def _add_custom_block(self, data, name):
        '''
        Add a new custom block to the snapshot.

        Args:
            data (array-like):  The new block (of appropiate size, i.e. length of
                                this (sub-)snapshot).
            name (str):         The name of the new block.

        Returns:
            block (SimArr):     The reference to the new block.

        Raises:
            RuntimeError:       If the length of the data does not fit the length
                                of the (sub-)snapshot; or if the latter is not the
                                correct "host" of the new block.
            KeyError:           If there already exists a block of that name.
        '''
        if len(data) == len(self):
            ptypes = [bool(N) for N in self.parts]
        else:
            raise RuntimeError('Data length does not fit the (sub-)snapshot!')

        if name in self._root._block_avail:
            raise NotImplementedError('Block name "%s" already exists ' % name +
                                      'for other parts of the snapshot!\n' +
                                      'Combining is not implemented.')

        self._root._block_avail[name] = list(ptypes)
        host = self.get_host_subsnap(name)
        if host is not self:
            print("name:", name, file=sys.stderr)
            print("for particle types:", self._root._block_avail[name], file=sys.stderr)
            print("self:", self, file=sys.stderr)
            print("host:", host, file=sys.stderr)
            del self._root._block_avail[name]
            raise RuntimeError('Can only set new blocks for entire family '
                               'sub-snapshots or the root.')

        from .sim_arr import SimArr
        host._blocks[name] = SimArr(data,snap=host)

    def _host_get_block(self, name):
        '''
        Load or derive a block of the given name and set it as attribute of its
        host.

        Args:
            name (str):         The stripped name of the block (as it is in
                                self._root._block_avail).

        Returns:
            block (SimArr):     The (entire) block.
        '''
        if name in self._root._load_name:
            return self._host_load_block(name)
        elif name in self._root._derive_rule_deps:
            return self._host_derive_block(name)
        else:
            raise ValueError("These is no block '%s' than can be " % name +
                             "loaded or derived!")

    def _convert_block_to_physical_units(self, block, name):
        '''
        Helper function to convert a block into its physical units, i.e. removing
        'a' and 'h_0' from the units.
        '''
        if block.units is None:
            return

        phys_units = block.units.free_of_factors(['a','h_0'])

        if phys_units != block.units:
            if environment.verbose >= environment.VERBOSE_TALKY or (
                    environment.verbose >= environment.VERBOSE_NORMAL
                    and name=='age'):
                print('convert block %s to physical units...' % name, end=' ')
                sys.stdout.flush()
            block.convert_to(phys_units, subs=self)
            if environment.verbose >= environment.VERBOSE_TALKY:
                print('done.')
                sys.stdout.flush()

    def to_physical_units(self, on_load=True):
        '''
        Convert all blocks and the boxsize to physical units.

        Convert all loaded blocks to units without scale factors and Hubble
        parameters and without numerical factors (e.g. 'a kpc / h_0' -> 'kpc' and
        '1e10 Msol / h_0' -> 'Msol').

        Note:
            There is no function to conveniently reverse this. You would have to
            unload (delete) all blocks, set phys_units_requested to False, and the
            reload the blocks (which is done automatically):
            --> for name, block in self:
            ...     del self[name]
            --> self.phys_units_requested = False

        Args:
            on_load (bool):     If True also convert newly loaded blocks
                                automatically.
        '''
        root = self._root

        # remember whether to convert to blocks or not
        root._phys_units_requested = bool(on_load)

        # convert all loaded blocks
        for block, name in self.iter_blocks(present=True, block=True, name=True,
                                            forthis=False):
            self._convert_block_to_physical_units(block, name)

        # convert the boxsize
        if environment.verbose >= environment.VERBOSE_TALKY:
            print('convert boxsize to physical units...', end=' ')
            sys.stdout.flush()
        root._boxsize.convert_to(
                root._boxsize.units.free_of_factors(['a','h_0']), subs=root)
        if environment.verbose >= environment.VERBOSE_TALKY:
            print('done.')
            sys.stdout.flush()

    def load_all_blocks(self):
        '''Load all blocks from file. (No block deriving, though.)'''
        for name in self._root._load_name:
            host = self.get_host_subsnap(name)
            host[name]

    def delete_blocks(self, derived=None, loaded=None):
        '''
        Delete all blocks.

        Args:
            derived, loaded (bools):
                        If both are None (the default), all blocks are deleted.
                        Otherwise only the blocks for which the argument is set to
                        True are deleted.
        '''
        if derived is None and loaded is None:
            derived = True
            loaded = True

        if derived:
            for name, (rule, deps) in self._root._derive_rule_deps.items():
                host = self.get_host_subsnap(name)
                if name in host._blocks:
                    del host[name]
        if loaded:
            for name in self._root._load_name:
                host = self.get_host_subsnap(name)
                if name in host._blocks:
                    del host[name]

    def get(self, expr, units=None, namespace=None):
        '''
        Evaluate the expression for a new block (e.g.: 'log10(Fe/mass)').

        The evaluation is done with a utils.Evaluator with numpy for math and the
        snapshot's blocks as well as some chosen objects (e.g. 'dist', 'Unit',
        'UnitArr', 'UnitQty', 'inter_bc_qty', 'solar', 'G') as namespace.

        Args:
            expr (str):         An python expression that knows about the blocks
                                in this snapshot.
            units (str, Unit):  If set, convert the array to this units before
                                returning.
            namespace (dict):   Additional namespace to use for calculating the
                                defined block.

        Returns:
            res (SimArr):       The result.
        '''
        from .sim_arr import SimArr
        from ..ssp import inter_bc_qty, lum_to_mag

        # prepare evaluator
        from numpy.core.umath_tests import inner1d
        from .. import analysis
        namespace = {} if namespace is None else namespace.copy()
        namespace.update( {'dist':dist, 'Unit':Unit, 'Units':Units,
                           'UnitArr':UnitArr, 'UnitQty':UnitQty,
                           'UnitScalar':UnitScalar, 'inner1d':inner1d,
                           'inter_bc_qty':inter_bc_qty, 'lum_to_mag':lum_to_mag,
                           'perm_inv':utils.perm_inv,
                           'solar':physics.solar, 'WMAP7':physics.WMAP7,
                           'Planck2013':physics.Planck2013,
                           'FLRWCosmo':physics.FLRWCosmo, 'a2z':physics.a2z,
                           'z2a':physics.z2a,
                           'kernel_weighted':analysis.kernel_weighted,
                           'len':len, 'module_dir':module_dir}
        )
        from . import derive_rules
        for n,obj in [(n,getattr(derive_rules,n)) for n in dir(derive_rules)]:
            if hasattr(obj, '__call__'):
                namespace[n] = obj
        for n,obj in [(n,getattr(physics,n)) for n in dir(physics)]:
            if isinstance(obj, UnitArr):
                namespace[n] = obj
        e = utils.Evaluator(namespace, my_math=np)
        # load properties etc. from this snapshot only if needed in the expression
        namespace = {'self':self}
        for name, el in utils.iter_idents_in_expr(expr, True):
            if name not in e.namespace and name not in namespace:
                try:
                    try:
                        namespace[name] = getattr(self,name)
                    except AttributeError:
                        namespace[name] = self[name]
                except KeyError:
                    if not isinstance(el, ast.Attribute):
                        raise ValueError("A unknown name ('%s') " % name +
                                         "appeared in expression for a new" +
                                         "block: '%s'" % expr)
        res = e.eval(expr, namespace)
        res = res.view(SimArr)
        res._snap = weakref.ref(self)
        if units is not None:
            res.convert_to(units)

        return res

    def IDs_unique(self):
        '''Check whether the IDs (of this (sub-)snapshot) are unique.'''
        return len(np.unique(self['ID'])) == len(self)


class SubSnapshot(Snapshot):
    '''
    A class for "sub-snapshots", that slice or mask an underlying (sub-)snapshot
    as a whole.

    The slice / mask is basically propagated to the arrays of the base. To
    instantiate this class, one should use the factory function SubSnap.

    Note:
        Sub-snapshots are convenient, but can become quite expensive in memory
        (and CPU time). If you encounter problems, consider taking the needed
        sub-blocks from it only and deleting the sub-snapshots -- or masking the
        blocks directly. Making copies can eliminate the mask which might occupy a
        lot of memory.

    Args:
        base (Snap):    The base (sub-)snapshot.
        mask (slice, np.ndarray[bool]):
                        A slice or a mask in form of a boolean np.ndarray.
        N_part (list):  The number of particles per type to take. Can be faster
                        than calculating them by the standard functions.
                        ATTENTION: you are responsible for this being correct!

    Doctests:
        >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_470', physical=False)
        >>> if any(s.gas.parts[1:]): print(s.gas.parts)
        >>> if any(s.stars.parts[:4]) or any(s.stars.parts[5:]): print(s.gas.parts)
        >>> s[123:len(s):100]
        <Snap "snap_M1196_4x_470"[123:2079055:100]; N=20,790; z=0.000>
        >>> s[123:len(s):100].parts
        [9216, 10015, 568, 193, 798, 0]
        >>> s['pos'] -= [34623, 35522, 33181]
        load block pos... done.
        >>> mask = abs(s['pos']).max(axis=-1) < 100
        >>> s[mask]
        <Snap "snap_M1196_4x_470":masked; N=355,218; z=0.000>
        >>> s[mask][::100].parts
        [431, 2459, 0, 0, 663, 0]
        >>> s[mask][::100].gas
        <Snap "snap_M1196_4x_470":masked[::100]:gas; N=431; z=0.000>
        >>> slim = s[::100]
        >>> slim.parts
        [9218, 10014, 568, 193, 798, 0]
        >>> np.sum(slim.gas['mass'] >= 0.0002)
        load block mass... done.
        UnitArr(1)
        >>> slim_mask = slim['mass'] < 0.0002   # exludes one gas particle...
        >>> slim[slim_mask].parts
        [9217, 0, 0, 0, 798, 0]
        >>> slim[slim_mask][100:9000:3]['rho']
        load block rho... done.
        SimArr([3.1859734e-10, 8.6869852e-11, 2.5398775e-10, ..., 9.3606489e-10,
                1.1147246e-09, 4.2680767e-10],
               dtype=float32, units="1e+10 Msol ckpc**-3 h_0**2", snap="snap_M1196_4x_470":gas)
        >>> len(slim[slim_mask][100:9000:3]['rho'])
        2967
        >>> mask = np.zeros(len(s), dtype=bool)
        >>> for pt in gadget.families['baryons']:
        ...     mask[sum(s.parts[:pt]):sum(s.parts[:pt+1])] = True

        s[mask], however, is not host (!)
        >>> assert np.all( s[mask]['elements'] == s.baryons['elements'] )
        load block Z... done.
        derive block elements... done.
        >>> len(slim[slim_mask]['elements'])
        10015
        >>> assert s[2:-123:10].stars['form_time'].shape[0] == len(s[2:-123:10].stars)
        load block form_time... done.
        >>> s[2:-123:10].stars['form_time']
        SimArr([0.7739953 , 0.950091  , 0.81303227, ..., 0.2623567 , 0.40703428,
                0.28800005],
               dtype=float32, units="a_form", snap="snap_M1196_4x_470":stars)
        >>> assert slim[slim_mask].stars['form_time'].shape[0] == len(slim[slim_mask].stars)
        >>> slim[slim_mask].stars['form_time'][::100]
        SimArr([0.9974333 , 0.30829304, 0.6479812 , 0.6046823 , 0.18474203,
                0.5375404 , 0.18224117, 0.26500258],
               dtype=float32, units="a_form", snap="snap_M1196_4x_470":stars)

        # index list (+family) tests (see comments in SubSnap)
        >>> r = np.sqrt(np.sum(s['pos']**2, axis=1))
        >>> assert np.all( s[r<123]['pos'] == s[np.where(r<123, True, False)]['pos'] )
        >>> assert np.all( s[s['pos'][:,0]>0]['pos']
        ...             == s[np.where(s['pos'][:,0]>0, True, False)]['pos'] )
        >>> assert np.all( s.stars[s.stars['inim']>s.stars['mass'].mean()]['pos'] ==
        ...         s.stars[np.where(s.stars['inim']>s.stars['mass'].mean(), True, False)]['pos'] )
        load block inim... done.
    '''
    def _calc_N_part_from_slice(self, N_part_base, _slice):
        # work with positive stepping (and without None's):
        _slice = utils.sane_slice(_slice, int(sum(N_part_base)))
        start, stop, step = _slice.start, _slice.stop, _slice.step

        N_part = [0]*6
        cur_pos = start
        N_cum_base = np.cumsum(N_part_base)
        pt = 0
        # goto first particle type that is in the sub-snapshot
        while N_cum_base[pt] <= cur_pos:
            pt += 1
        # count particles of the different types
        while pt < 6 and cur_pos < stop:
            remain = int( min(stop, N_cum_base[pt]) - cur_pos )
            N_part[pt] = remain // step
            if remain % step:
                N_part[pt] += 1
            cur_pos += N_part[pt] * step
            pt += 1

        return N_part

    def _calc_N_part_from_mask(self, N_part_base, mask):
        N_cum_base = np.cumsum(N_part_base)
        return [np.sum(mask[s:e]) for s,e in zip([0]+list(N_cum_base[:-1]),
                                                 N_cum_base)]

    def __init__(self, base, mask, N_part=None):
        #Snapshot.__init__(self, gad_units=None, physical=False, cosmological=None)
        self._base      = base
        self._root      = base._root
        self._blocks    = {}

        if isinstance(mask, slice):
            if mask.step == 0:
                raise ValueError('Slice step cannot be 0!')

            self._mask = slice(mask.start, mask.stop, mask.step) # make a copy
            if N_part is None:
                self._N_part = self._calc_N_part_from_slice(base._N_part, mask)
            else:
                self._N_part = N_part[:]

            self._descriptor = base._descriptor + '['
            if mask.start is not None:
                self._descriptor += str(mask.start)
            self._descriptor += ':'
            if mask.stop is not None:
                self._descriptor += str(mask.stop)
            if mask.step is not None:
                self._descriptor += ':' + str(mask.step)
            self._descriptor += ']'
        elif isinstance(mask, np.ndarray) and mask.dtype == bool:
            mask = mask.view(np.ndarray)
            if len(mask) != len(base):
                raise ValueError('Mask has to have the same length as the ' + \
                                 'base snapshot!')

            self._mask = mask.copy()
            if N_part is None:
                self._N_part = self._calc_N_part_from_mask(base._N_part, mask)
            else:
                self._N_part = N_part[:]

            self._descriptor = base._descriptor + ':masked'
        else:
            raise TypeError('Need either a slice or a boolean np.ndarray ' + \
                            'for sub-snapshots!')

    def _restrict_mask_to_ptypes(self, ptypes):
        '''
        Get the sub-snapshot's mask restricted to the given particle types.

        Args:
            ptypes (list):  A (sorted) list of the particle types to restrict to.
        '''
        base = self._base

        if isinstance(self._mask, slice):
            # work with positive stepping (and without None's):
            _slice = utils.sane_slice(self._mask, len(base))
            start, stop, step = _slice.start, _slice.stop, _slice.step
            offset = sum( base._N_part[:ptypes[0]] )
            if utils.is_consecutive(ptypes):
                if start < sum(base._N_part[:ptypes[0]]):
                    diff = sum(base._N_part[:ptypes[0]]) - start
                    start += diff // step * step
                    if diff % step:
                        start += step
                stop = min(stop, sum(base._N_part[:ptypes[-1]+1]))
                return slice(start-offset, stop-offset, step)
            else:
                # slicing is not sufficient, need masking
                fullmask = np.zeros(len(base), dtype=bool)
                fullmask[start:stop:step] = True
                pmasks = []
                for pt in ptypes:
                    start = int(sum(base._N_part[:pt]))
                    end   = start + base._N_part[pt]
                    pmasks.append( fullmask[start:end] )
                mask = np.concatenate(pmasks)
                return mask
        else:
            assert isinstance(self._mask, np.ndarray) and self._mask.dtype==bool
            if utils.is_consecutive(ptypes):
                start = sum(base._N_part[:ptypes[0]])
                end   = sum(base._N_part[:ptypes[-1]+1])
                return self._mask[start:end]
            else:
                pmasks = []
                for pt in ptypes:
                    start = int(sum(base._N_part[:pt]))
                    end   = start + base._N_part[pt]
                    pmasks.append( self._mask[start:end] )
                mask = np.concatenate(pmasks)
                return mask

    def _get_block(self, name):
        '''
        This gets called, if a block is not yet set in self._blocks.

        Here for a _SubSnap, one has to handle the case where the block is not
        available for the base. It has to be stored appropiately then.
        '''
        if name in self._base.available_blocks():
            # do not store the masked snapshot, but only the base one
            return self._base[name][self._mask]
        else:
            # Due to calling by __getitem__, the following should always be
            # fulfilled:
            assert name in self.available_blocks()
            assert name not in self._blocks
            host = self.get_host_subsnap(name)
            if host is self:
                # no slicing/masking needed!
                # (caching of blocks is done within this function)
                block = host._host_get_block(name)
                return block
            else:
                # delegate the loading to the host
                block = host[name]
                #################################################################
                # The general situation here:
                #
                #   root -- sub[1] -- ... -- sub[N-1] -- sub[N] (==self)
                #       \
                #       host
                #
                # where sub[1] != host and sub[N-1] does not have the requested
                # block.
                # Now restrict all the masks of sub[1] though sub[N] and self to
                # the particle types of the host and apply the masks to the block.
                #################################################################
                ptypes = [pt for pt in range(6) if host._N_part[pt]]
                masks = []
                sub = self
                subs = []
                while sub != sub._root:
                    masks.append( sub._restrict_mask_to_ptypes(ptypes) )
                    subs.append( sub )
                    sub = sub._base
                for mask in reversed(masks):
                    block = block[mask]
                # do not store the masked snapshot, but only the base one
                return block


def _FamilySubSnap(base, fam):
    '''
    A factory function for creating a sub-snapshot of all the particles of the
    given family of the base snapshot.

    Sub-snapshots for families should always be instantiated with this function or
    by accessing attributes. The latter caches the sub-snapshot.

    Args:
        base (Snap):        The snapshot to create the family sub-snapshot from.
        fam (str):          The name of the family (as in `gadget.families`) to
                            restrict to.

    Returns:
        fam_subsnap (Snap): The sub-snapshot of `base` that is restricted to the
                            specified family.
    '''
    if fam not in gadget.families:
        raise ValueError('Unknown familiy "%s"!' % fam)

    # family sub-snapshot
    family_s = base.SubSnap(gadget.families[fam])
    family_s._descriptor = base._descriptor + ':' + fam
    return family_s


def Snap(filename, physical=False, load_double_prec=False, cosmological=None,
             gad_units=None, unclear_blocks=None, H_neutral_only=None):
    from warnings import warn
    warn("\nfunction Snap(...) has been renamed to Snapshot class\nplease use Snapshot(..) to create a snapshot object", DeprecationWarning, stacklevel=2)
    s = Snapshot(filename, physical, load_double_prec, cosmological, gad_units, unclear_blocks, H_neutral_only)
    return s