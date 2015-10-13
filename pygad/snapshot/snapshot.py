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

    >>> from ..environment import module_dir
    >>> snap_tmp_file = 'test.gdt'
    >>> s = Snap(module_dir+'../snaps/AqA_ICs_C02_200_nm.gdt', physical=False,
    ...          gad_units={'LENGTH':'cMpc/h_0'})
    >>> s
    <Snap "AqA_ICs_C02_200_nm.gdt.0-8"; N=2,951,686; z=127.000>
    >>> s.pos.units
    load block pos... done.
    Unit("cMpc h_0**-1")
    >>> s.gadget_units
    {'VELOCITY': 'km / s', 'LENGTH': 'cMpc/h_0', 'MASS': '1e10 Msol / h_0'}
    >>> print 'current age of the universe: %s' % s.cosmic_time().in_units_of('Myr')
    current age of the universe: 12.8697344013 [Myr]
    >>> s.loadable_blocks()
    ['vel', 'mass', 'ID', 'pos']
    >>> for d in s.deriveable_blocks(): print d,
    lum_j lum_k lum_h lum_i Epot lum_b Ne metals jzjc RemainingElements lum_r rcyl lum_v lum_u mag_j alpha_el r Fe mag_v lum momentum He Mg E mag_b H mag_i mag_h O N mag_u S angmom mag_k mag mag_r Z Ekin C Ca vcirc Si vrad jcirc
    >>> assert set(s.all_blocks()) == set(s.loadable_blocks() + s.deriveable_blocks())
    >>> mwgt_pos = np.tensordot(s.mass, s.pos, axes=1).view(UnitArr)
    load block mass... done.
    >>> mwgt_pos.units = s.mass.units * s.pos.units
    >>> com = mwgt_pos / s.mass.sum()
    >>> com
    UnitArr([ 50.24965286,  50.23574448,  50.130867  ],
            dtype=float32, units="cMpc h_0**-1")

    And the physical distance between the center of mass and the unweighted mean
    of the positions is:
    (Conversion from 'ckpc/h_0' to 'kpc' is done automatically: the values for 'a'
    and 'h_0' are taken from the associated snapshot and substitued.)
    >>> np.sqrt(np.sum( (com - s.pos.mean(axis=0))**2 )).in_units_of('Mpc', subs=s)
    UnitArr(0.0118607562035, dtype=float32, units="Mpc")

    Whereas the physical dimensions of the simulation's box are:
    >>> s.boxsize
    SimArr(100.0, units="cMpc h_0**-1", snap="AqA_ICs_C02_200_nm.gdt.0-8")
    >>> s.pos.max(axis=0) - s.pos.min(axis=0)
    UnitArr([ 97.78572083,  97.81533051,  97.83026123],
            dtype=float32, units="cMpc h_0**-1")

    >>> s.load_all_blocks()
    load block vel... done.
    load block ID... done.
    
    It is also possible to slice entire snapshot, e.g. to access single families
    of the snapshot (gas, stars, dm, bh, baryons) or to mask them with a
    np.ndarray of bools (for more information see SubSnap).
    In fact, in order to access blocks that are only for certains families, one has
    to restrict the snapshot to appropiately.
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_470', physical=False)
    >>> s.gas.rho
    load block rho... done.
    SimArr([  2.13930360e-03,   8.17492546e-04,   9.48364788e-04, ...,
              4.68019188e-08,   3.91658048e-08,   3.44690605e-08],
           dtype=float32, units="1e+10 Msol ckpc**-3 h_0**2", snap="snap_M1196_4x_470":gas)

    The derived block R (radius) is updated automatically, if the block pos it
    depends on changes:
    >>> s.r
    load block pos... done.
    derive block r... done.
    SimArr([ 59671.72500919,  59671.40716648,  59671.54775578, ...,
             59672.52310468,  59671.98484173,  59671.74536916],
           units="ckpc h_0**-1", snap="snap_M1196_4x_470")
    >>> np.all( s.r == dist(s.pos) ).value
    True
    >>> s.pos -= UnitArr([34622.7,35522.7,33180.5], 'ckpc/h_0')
    >>> s.r
    derive block r... done.
    SimArr([  9.41927424,   9.69844804,   9.6864571 , ...,  25.08852488,
             23.95322772,  24.6890346 ],
           units="ckpc h_0**-1", snap="snap_M1196_4x_470")

    >>> s.to_physical_units()
    convert block pos to physical units... done.
    convert block r to physical units... done.
    convert block rho to physical units... done.
    convert boxsize to physical units... done.
    >>> sub = s[s.r < UnitScalar('30 kpc')]
    >>> sub
    <Snap "snap_M1196_4x_470":masked; N=121,013; z=0.000>
    >>> sub.gas
    <Snap "snap_M1196_4x_470":masked:gas; N=11,191; z=0.000>

    # You can also get (almost) arbitrary combinations of the blocks
    >>> s.gas.get('dist(pos)**2 / rho')
    SimArr([  1.54323637e-05,   4.28144806e-05,   3.68149753e-05, ...,
              2.25019852e+04,   2.70823020e+04,   3.08840657e+04],
           units="Msol**-1 kpc**5", snap="snap_M1196_4x_470":gas)
    >>> assert np.max(np.abs( (s.get('dist(pos)') - s.r) / s.r )) < 1e-6
    >>> del s.pos
    >>> s.r
    SimArr([ 13.08232533,  13.47006673,  13.45341264, ...,  34.84517345,
             33.26837183,  34.29032583], units="kpc", snap="snap_M1196_4x_470")
    >>> s.pos
    load block pos... done.
    convert block pos to physical units... done.
    SimArr([[ 48074.32421875,  49335.85546875,  46081.39453125],
            [ 48074.0234375 ,  49335.78515625,  46080.9921875 ],
            [ 48073.97265625,  49335.9453125 ,  46081.22265625],
            ..., 
            [ 48067.1953125 ,  49359.1796875 ,  46065.84375   ],
            [ 48067.49609375,  49357.2578125 ,  46066.24609375],
            [ 48066.18359375,  49357.8046875 ,  46066.4296875 ]],
           dtype=float32, units="kpc", snap="snap_M1196_4x_470")
    >>> s.r
    derive block r... done.
    SimArr([ 82877.39261021,  82876.95257089,  82877.14659861, ...,
             82878.50049942,  82877.75400349,  82877.42052568],
           units="kpc", snap="snap_M1196_4x_470")

    One can test for available blocks and families by the 'in' opertator:
    >>> 'r' in s
    True
    >>> for a in s.available_blocks(): print a,
    Epot pot pos jzjc rcyl r mass vel momentum E vrad ID Ekin vcirc angmom jcirc
    >>> s.delete_blocks(derived=True)
    >>> 'r' in s
    True
    >>> 'stars' in s
    True

    New custom blocks can be set via add_custom_block:
    >>> sub = s.baryons[s.baryons.Z>1e-3].stars
    load block elements... done.
    convert block elements to physical units... done.
    derive block H... done.
    derive block He... done.
    derive block metals... done.
    derive block Z... done.
    >>> new = np.ones(len(s.stars))
    >>> s.stars.add_custom_block(new, 'new')
    >>> sub.new
    SimArr([ 1.,  1.,  1., ...,  1.,  1.,  1.], snap="snap_M1196_4x_470":stars)

    >>> dest_file = module_dir+'test.gad'
    >>> assert not os.path.exists(dest_file)
    >>> gadget.write(sub, dest_file, blocks=['pos', 'ID', 'r'])
    load block vel... done.
    load block ID... done.
    load block mass... done.
    convert block mass to physical units... done.
    derive block r... done.
    writing block POS  (dtype=float32, units=[ckpc h_0**-1])... done.
    writing block VEL  (dtype=float32, units=[s**-1 km])... done.
    writing block ID   (dtype=uint32, units=[1])... done.
    writing block MASS (dtype=float32, units=[1e+10 Msol h_0**-1])... done.
    writing block R    (dtype=float32, units=[kpc])... done.
    >>> sub_copy = Snap(dest_file, physical=True)
    >>> sub_copy.r.units = 'kpc'    # notice: block r gets *loaded*!
    ...                             # the units are those from before!
    ...                             # and are not in the config!
    load block r... done.
    >>> assert sub.parts == sub_copy.parts
    >>> assert np.max(np.abs((sub.pos - sub_copy.pos) / sub.pos)) < 1e-6
    load block pos... done.
    convert block pos to physical units... done.
    >>> assert np.max(np.abs((sub.r - sub_copy.r) / sub.r)) < 1e-6
    >>> assert np.abs((sub.boxsize - sub_copy.boxsize) / sub.boxsize) < 1e-6
    >>> import os
    >>> os.remove(dest_file)
    >>> del sub, sub_copy
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_470', physical=False)
    >>> gadget.write(s, dest_file)  # doctest:+ELLIPSIS
    load block pos... done.
    load block vel... done.
    load block ID... done.
    load block mass... done.
    load block nh... done.
    ...
    writing block POS  (dtype=float32, units=[ckpc h_0**-1])... done.
    writing block VEL  (dtype=float32, units=[s**-1 km])... done.
    writing block ID   (dtype=uint32, units=[1])... done.
    writing block MASS (dtype=float32, units=[1e+10 Msol h_0**-1])... done.
    writing block NH   (dtype=float32, units=[1])... done.
    ...
    >>> s2 = Snap(dest_file, physical=False)
    >>> s2.load_all_blocks()    # doctest:+ELLIPSIS
    load block nh... done.
    load block elements... done.
    load block hsml... done.
    load block age... done.
    load block inim... done.
    ...
    >>> for name in s.loadable_blocks():
    ...     b  = getattr(s.get_host_subsnap(name), name)
    ...     b2 = getattr(s2.get_host_subsnap(name), name)
    ...     err = np.max( np.abs((b - b2) / b) )
    ...     if err[np.isfinite(err)] > 1e-6:
    ...         print name, b, b2
    >>> os.remove(dest_file)

    >>> dest_file_hdf5 = module_dir+'test.hdf5'
    >>> assert not os.path.exists(dest_file_hdf5)
    >>> gadget.write(s, dest_file_hdf5) # doctest:+ELLIPSIS
    ...
    writing block POS  (dtype=float32, units=[ckpc h_0**-1])... done.
    writing block VEL  (dtype=float32, units=[s**-1 km])... done.
    writing block ID   (dtype=uint32, units=[1])... done.
    writing block MASS (dtype=float32, units=[1e+10 Msol h_0**-1])... done.
    writing block NH   (dtype=float32, units=[1])... done.
    ...
    >>> tmp = Snap(dest_file_hdf5, physical=False)
    >>> gadget.write(tmp, dest_file)    # doctest:+ELLIPSIS
    load block pos... done.
    load block vel... done.
    load block ID... done.
    load block mass... done.
    load block nh... done.
    ...
    writing block POS  (dtype=float32, units=[ckpc h_0**-1])... done.
    writing block VEL  (dtype=float32, units=[s**-1 km])... done.
    writing block ID   (dtype=uint32, units=[1])... done.
    writing block MASS (dtype=float32, units=[1e+10 Msol h_0**-1])... done.
    writing block NH   (dtype=float32, units=[1])... done.
    ...
    >>> del tmp
    >>> os.remove(dest_file_hdf5)
    >>> s2 = Snap(dest_file, physical=False)
    >>> for name in s.loadable_blocks():    # doctest:+ELLIPSIS
    ...     b  = getattr(s.get_host_subsnap(name), name)
    ...     b2 = getattr(s2.get_host_subsnap(name), name)
    ...     assert np.all( b == b2 )
    load block pos... done.
    load block nh... done.
    load block elements... done.
    load block hsml... done.
    load block pot... done.
    ...
    >>> assert s.redshift == s2.redshift
    >>> for name, prop in s.properties.iteritems():
    ...     if not prop == s2.properties[name]:
    ...         print name, prop, s2.properties[name]
    >>> os.remove(dest_file)
    >>> del s2

    Some basic testing for (reading) HDF5 snapshots:
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_conv_470.hdf5', physical=False)
    >>> s
    <Snap "snap_M1196_4x_conv_470.hdf5"; N=2,079,055; z=0.000>
    >>> s.cosmology
    FLRWCosmo(h_0=0.72, O_Lambda=0.74, O_m=0.26, O_b=0.0416, sigma_8=None, n_s=None)
    >>> s.pos
    load block pos... done.
    SimArr([[ 34613.515625  ,  35521.81640625,  33178.60546875],
            [ 34613.296875  ,  35521.765625  ,  33178.31640625],
            [ 34613.26171875,  35521.8828125 ,  33178.48046875],
            ..., 
            [ 34608.3828125 ,  35538.609375  ,  33167.41015625],
            [ 34608.59765625,  35537.2265625 ,  33167.69921875],
            [ 34607.65234375,  35537.62109375,  33167.83203125]],
           dtype=float32, units="ckpc h_0**-1", snap="snap_M1196_4x_conv_470.hdf5")
    >>> s.r
    derive block r... done.
    SimArr([ 59671.72500919,  59671.40716648,  59671.54775578, ...,
             59672.52310468,  59671.98484173,  59671.74536916],
           units="ckpc h_0**-1", snap="snap_M1196_4x_conv_470.hdf5")
    >>> s.parts
    [921708, 1001472, 56796, 19315, 79764, 0]
    >>> sub = SubSnap(s, [0,1,2,4,5])
    >>> sub
    <Snap "snap_M1196_4x_conv_470.hdf5":pts=[0,1,2,4,5]; N=2,059,740; z=0.000>
    >>> sub.parts
    [921708, 1001472, 56796, 0, 79764, 0]
    >>> SubSnap(s, [0,2,4]).parts
    [921708, 0, 56796, 0, 79764, 0]
'''
__all__ = ['Snap', 'SubSnap', 'write']

import sys
import os.path
from .. import gadget
from ..gadget import write
from .. import physics
from ..units import *
from .. import utils
from .. import environment
import numpy as np
import warnings
import ast

def Snap(filename, physical=False, cosmological=None, gad_units=None):
    '''
    Create a snapshot from file (without loading the blocks, yet).

    Args:
        filename (str):     The path to the snapshot. If it is distributed over
                            several files, you shall omit the trailing (of
                            inbetween in case of an HDF5 file) '.0'.
        physical (bool):    Whether to convert to physical units on loading.
        cosmological (bool):Explicitly tell if the simulation is a cosmological
                            one.
        gad_units (dict):   Alternative base units (LENGTH, VELOCITY, MASS) for
                            this snapshot. The default base units units are
                            updated, meaning one can also just change one them.
    '''
    from sim_arr import SimArr
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
            raise RuntimeError('Snapshot "%s%s" does not exist!' % (base, suffix))

    s = _Snap(gad_units=gad_units, physical=physical, cosmological=cosmological)
    s._filename   = os.path.abspath(base+suffix)
    s._descriptor = os.path.basename(base)+suffix
    # need first (maybe only) file for basic information
    greader = gadget.FileReader(filename)
    s._file_handlers = [greader]

    s._block_avail = { block.name:block.ptypes for block in greader.infos() }

    s._N_part   = list(greader.header['N_part_all'])
    s._time     = greader.header['time']
    s._redshift = greader.header['redshift']
    s._boxsize  = SimArr(greader.header['boxsize'],
                         units=s._gad_units['LENGTH'],
                         snap=s)
    s._cosmology = physics.FLRWCosmo(              # Note: Omega_b at default!
                    h_0          = greader.header['h_0'],
                    Omega_Lambda = greader.header['Omega_Lambda'],
                    Omega_m      = greader.header['Omega_m'])
    s._properties = { k:v for k,v in greader.header.iteritems()
            if k not in ['N_part', 'mass', 'time', 'redshift', 'N_part_all',
                         'N_files', 'h_0', 'Omega_Lambda', 'Omega_m', 'boxsize',
                         'unused'] }
    if s._cosmological is None:
        s._cosmological = abs(s.scale_factor-s.time) < 1e-6

    if physical:
        s._boxsize.convert_to(s._boxsize.units.free_of_factors(['a','h_0']),
                              subs=s)


    if greader.header['N_files'] > 1:
        s._descriptor += '.0-'+str(greader.header['N_files'])
        if greader.format == 3:
            # TODO
            raise NotImplementedError('Reading HDF5 snapshots spread over ' +
                                      'multiple files is not yet supported.')
        else:
            # enshure Python int's to avoid overflows
            N_part = list( greader.header['N_part'] )
            for n in xrange(1, greader.header['N_files']): # first already done
                filename = base + '.' + str(n) + suffix
                greader = gadget.FileReader(filename)
                s._file_handlers.append( greader )
                # enshure Python int's to avoid overflows
                for i in xrange(6): N_part[i] += greader.header['N_part'][i]
                # update loadable blocks:
                for block in greader.infos():
                    if block.name in s._block_avail:
                        s._block_avail[block.name] = [ (o or n) for o,n \
                                in zip(s._block_avail[block.name], block.ptypes)]
                    else:
                        s._block_avail[block.name] = block.ptypes
            if N_part != s._N_part:
                # more particles than fit into a native int
                s._N_part = N_part

    # Process block names: make standard names lower case (except ID) and replace
    # spaces with underscores for HDF5 names. Also strip names
    s._load_name = {}
    for name, block in s._block_avail.items():
        if s._file_handlers[0]._format == 3:
            if '%-4s'%name in gadget.std_name_to_HDF5:
                new_name = name.strip().lower()
            else:
                new_name = name.strip().replace(' ','_')
        else:
            new_name = name.strip().lower()
        if new_name in ['id']:
            new_name = new_name.upper()
        elif new_name == 'z':
            new_name = 'elements'
        s._load_name[new_name] = name
        s._block_avail[new_name] = s._block_avail[name]
        del s._block_avail[name]    # blocks should not appear twice, could
                                    # confuse in __dict__ (tab completion etc.)
    # now the mass block is named 'mass' for all cases (HDF5 or other)
    s._block_avail['mass'] = [n>0 for n in s._N_part]

    import derived
    changed = True
    while changed:
        changed = False
        for name, rule in derived._rules.iteritems():
            if name in s._load_name:
                continue
            ptypes, deps = derived.ptypes_and_deps(rule, s)
            if name in s._block_avail:
                if ptypes!=s._block_avail[name] \
                        or deps!=s._derive_rule_deps[name][1]:
                   changed = True
            else:
               changed = True
            s._block_avail[name] = ptypes
            s._derive_rule_deps[name] = (rule, deps)

    s._descriptor = '"' + s._descriptor + '"'

    return s


class _Snap(object):
    '''
    A class holding a Gadget snapshot.

    Args:
        physical (bool):    Whether to convert to physical units on loading.
        cosmological (bool):Whether the simulation was a cosmological one.
        gad_units (dict):   Alternative base units (LENGTH, VELOCITY, MASS) for
                            this snapshot.
    '''
    def __init__(self, physical, cosmological, gad_units=None):
        # Actual initialization is done in the factory function Snap. Just do some
        # basic setting of the attributes to enshure that even snapshot created by
        # just _Snap are somewhat functioning.
        self._filename              = '<none>'
        self._descriptor            = 'new'
        self._file_handlers         = []
        self._gad_units             = gadget.default_gadget_units.copy()
        if gad_units:
            self._gad_units.update( gad_units )
        self._load_name             = {}    # from attribute name to actual block
                                            # name as to use for loading
        self._block_avail           = {}    # keys are lower case, stripped names
                                            # or the HDF5 names with underscores
        self._derive_rule_deps      = {}
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

    def families(self):
        '''Return the names of the particle families (at least partly) present in
        this snapshot.'''
        families = set()
        for name, ptypes in gadget.families.iteritems():
            if sum(self._N_part[pt] for pt in ptypes):
                families.add(name)
        return list(families)

    def available_blocks(self):
        '''The names of the blocks that are available for all particle types of
        this snapshot.'''
        return [ name for name, ptypes
                in self._root._block_avail.iteritems()
                if all([(Np==0 or s) for s,Np in zip(ptypes,self._N_part)]) ]

    def loadable_blocks(self):
        '''The names of all blocks of this snapshot.'''
        return self._root._load_name.keys()

    def deriveable_blocks(self):
        '''The names of all blocks of this snapshot.'''
        return self._root._derive_rule_deps.keys()

    def all_blocks(self):
        '''The names of all blocks of this snapshot.'''
        return self._root._block_avail.keys()

    def cosmic_time(self):
        '''The cosmic time (i.e. the current universe age).'''
        return self.cosmology.cosmic_time(self.redshift)

    def __repr__(self):
        return '<Snap %s; N=%s; z=%.3f>' % (
                self._descriptor,
                utils.nice_big_num_str(len(self)),
                self._root.redshift)

    def __dir__(self):
        # Add the block names available for all particle types,
        # such that they occure in tab completion in iPython, for instance.
        # The second list seems to include excatly the properties.
        return self.__dict__.keys() + \
                [k for k in self.__class__.__dict__.keys()
                        if not k.startswith('_')] + \
                self.available_blocks() + \
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
        ptypes = [pt for pt in xrange(6) if (avail[pt] and parts[pt])]

        # block available for all families? -> root
        if not any( [parts[pt] for pt in xrange(6) if pt not in ptypes] ):
            return root

        # find biggest family sub-snapshot (that is an attribute of root) which has
        # the block
        for fam, fptypes in gadget.families.iteritems():
            if fptypes == ptypes:
                return getattr(root, fam)

        return None

    def iter_blocks(self, present=True, block=True, name=False):
        '''
        Iterate all blocks, even those that are not for all particles.

        If present==False and block==True, this loads / derives all the not yet
        present (loaded/derived) blocks.

        Args:
            present (bool): If set, only iterate over the already loaded / derived
                            blocks.
            block (bool):   Whether to yield the (full) block itself.
            name (bool):    Whether to yield the block name.

        Yields:
            block (UnitArr):    The full block, if only this was requested,
            name (str):         the name of the block, if only that was requested,
            OR
            (block, name):      both as a tuple, if both are requested.
            (And just nothing, if neither was requested.)
        '''
        root = self._root
        done = set()
        for block_name in root._block_avail.keys():
            if block_name in done:
                continue
            host = self.get_host_subsnap(block_name)
            if not present or block_name in host.__dict__:
                if block and name:
                    yield getattr(host, block_name), block_name
                elif block:
                    yield getattr(host, block_name)
                elif name:
                    yield block_name

    def _set_block_attr(self, name):
        '''
        This gets called, if the attribute for a block is not yet set.

        Overwritten for _SubSnap, where care is taken for sub-snapshots that are
        not directly sliced/mased from the block's host. Here, for the root, it
        simply calls self._get_block (and asserts that the block is available).
        '''
        assert name in self.available_blocks()
        return self._get_block(name)

    def __getattr__(self, name):
        # Load block, if available for all particle types.
        if name in self.available_blocks():
            return self._set_block_attr(name)
        # Create a family sub-snapshot and set it as attribute.
        elif name in gadget.families:
            fam_snap = SubSnap(self, name)
            setattr(self, name, fam_snap)
            return fam_snap
        # There is such a block, but it is not available (for all particle types).
        elif name in self._root._block_avail:
            raise AttributeError("Block '%s' is not available for all " % name +
                                 "particle types of this (sub-)snapshot.")
        # Explicitly set attributes (like self.redshift and self.gas), member
        # functions, and properties are already handles before __getattr__ is
        # called, hence, we now have an attribute error:
        raise AttributeError("%r object has no " % type(self).__name__ + \
                             "attribute %r" % name)

    def __getitem__(self, i):
        # Handling of the index is fully done by the factory function.
        return SubSnap(self, i)

    def __contains__(self, el):
        return el in self.available_blocks() or el in self.families()

    def _load_block(self, name):
        '''
        Load a block and add it to its host.

        Note:
            For getting blocks -- also derived ones -- call _get_block!

        Args:
            name (str): The stripped name of the block (as it is in
                        self._root._load_name.keys()).

        Returns:
            block (SimArr):     The (entire) block.
        '''
        from sim_arr import SimArr

        root = self._root
        if not root._file_handlers:
            raise RuntimeError('No file(s) to load from.')
        if not name in root._load_name:
            raise ValueError("There is no block '%s' to load!" % name)

        if environment.verbose: print 'load block %s...' % name,
        sys.stdout.flush()

        block_name = root._load_name[name]
        if len(root._file_handlers) == 1:
            block = root._file_handlers[0].read_block(block_name,
                                                      gad_units=root._gad_units)
        else:
            first = True
            for reader in root._file_handlers:
                if not reader.has_block(block_name) and block_name!='mass':
                    continue
                read = reader.read_block(block_name, gad_units=root._gad_units)
                if first:
                    block = read
                    first = False
                else:
                    block = np.concatenate((block, read), axis=0).view(UnitArr)
                    block.units = read.units
            assert not first
        block = block.view(SimArr)
        block._snap = self
        if environment.verbose: print 'done.'
        sys.stdout.flush()

        if self.load_double_prec and block.dtype.kind == 'f' \
                and block.dtype != 'float64':
            if environment.verbose: print 'convert to double precision...',
            sys.stdout.flush()
            block = block.astype(np.float64)
            if environment.verbose: print 'done.'
            sys.stdout.flush()

        if root._phys_units_requested:
            self._convert_block_to_physical_units(block, name)

        host = self.get_host_subsnap(name)
        setattr(host, name, block)

        for trans in root._trans_at_load:
            if name in trans._change:
                if environment.verbose: print 'apply stored %s to block %s...' % (
                        trans.__class__.__name__, name),
                sys.stdout.flush()
                trans.apply_to_block(name,host)
                if environment.verbose: print 'done.'
                sys.stdout.flush()

        # if there are still dependent blocks, add them as dependencies and
        # invalidate them
        for derived_name, (rule, deps) in root._derive_rule_deps.iteritems():
            if name in deps:
                host = self.get_host_subsnap(derived_name)
                if derived_name in host.__dict__:
                    block.dependencies.add(derived_name)
        block.invalidate_dependencies()

        return block

    def _derive_block(self, name):
        '''
        Derive a block from the stored rules and add it to its host.

        Note:
            For getting blocks call _get_block!

        Args:
            name (str): The name of the derived block (as it is in
                        self._root._derive_rule_deps).

        Returns:
            block (SimArr):     The (entire) block.
        '''
        root = self._root
        if not name in root._derive_rule_deps:
            raise ValueError("These is no block '%s' to derive!" % name)

        host = self.get_host_subsnap(name)
        rule, deps = root._derive_rule_deps[name]

        # pre-load all needed (and not yet loaded) blocks in order not to mess up
        # the output
        for dep in deps:
            getattr(host, dep)

        if environment.verbose: print 'derive block %s...' % name,
        sys.stdout.flush()
        block = host.get(rule)
        for dep in deps:
            getattr(host,dep).dependencies.add(name)
        if environment.verbose: print 'done.'
        sys.stdout.flush()

        setattr(host, name, block)
        return block

    def add_custom_block(self, data, name):
        '''
        Add a new custom block to the snapshot.

        Args:
            data (array-like):  The new block (of appropiate size, i.e. length of
                                this (sub-)snapshot).
            name (str):         The name of the new block.
        '''
        if len(data) == len(self):
            ptypes = [bool(N) for N in self.parts]
        else:
            raise RuntimeError('Data length does not fit the (sub-)snapshot!')

        root = self._root
        if name in root._block_avail:
            raise KeyError('Block "%s" already exists!' % name)

        root._block_avail[name] = list(ptypes)
        host = self.get_host_subsnap(name)
        if host is not self:
            del root._block_avail[name]
            raise RuntimeError('Can only set new blocks for entire family '
                               'sub-snapshots or the root.')

        from sim_arr import SimArr
        setattr( host, name, SimArr(data,snap=host) )

    def refresh(self):
        '''Update sliced/masked blocks of this snapshot.'''
        # TODO: can be removed once the problem with the unit attribute of
        # UnitArr's is solved
        for name in self.iter_blocks(present=True, block=False, name=True):
            if name in self.__dict__ and self is not self.get_host_subsnap(name):
                delattr(self, name)

    def _get_block(self, name):
        '''
        Load or derive a block of the given name and set it as attribute of its
        host.

        Args:
            name (str): The stripped name of the block (as it is in
                        self._root._block_avail).

        Returns:
            block (SimArr):     The (entire) block.
        '''
        if name in self._root._load_name:
            return self._load_block(name)
        elif name in self._root._derive_rule_deps:
            return self._derive_block(name)
        else:
            raise ValueError("These is no block '%s' than can be " % name +
                             "loaded or derived!")

    def _convert_block_to_physical_units(self, block, name):
        '''
        Helper function to convert a block into its physical units.

        Attention:
            For the block pot it actualy subtracts the cosmological part
            (-1/2 Omega_0 H_0 x**2). Here it requires that all stored
            transformions are already performed on the block pos.

            pot(from file) = -G sum_k m_k g |x_i-x_k| - 1/2 Omega_0 H_0^2 x_i^2
                             \__________  __________/
                                        \/
                                the Newtonian part
        '''
        """
        This is only for non-periodic cosmological simulations...
        if self.cosmological and name == 'pot':
            # block pos might have already different units, but all the stored
            # transformations should be applied to it
            root = self.root    # 'pos' and 'pot' should be available for all
                                # particles
            pos = root.pos
            # need original, comoving block
            from ..transformation import Translation
            for trans in root._trans_at_load:
                Tinv = trans.inverse()
                if isinstance(Tinv,Translation):
                    Tinv._trans.convert_to(pos.units, subs=root)
                pos = Tinv.apply_to_block(pos)
            print 'convert block pot to physical...',
            sys.stdout.flush()
            O_0 = root.cosmology.Omega_tot
            H_0 = root.cosmology.H(0)
            # brackets might avoid mutliple multilplications with the entire array
            hubble_pot = (0.5 * O_0 * H_0**2) * np.sum(pos**2,axis=1)
            hubble_pot.convert_to(block.units, subs=root)
            block -= hubble_pot
            print 'done.'
            sys.stdout.flush()
            return
        """

        if block.units in ['a_form', 'z_form']:
            phys_units = 'Gyr'
        else:
            phys_units = block.units.free_of_factors(['a','h_0'])

        if phys_units != block.units:
            if environment.verbose:
                print 'convert block %s to physical units...' % name,
            sys.stdout.flush()
            block.convert_to(phys_units, subs=self)
            if environment.verbose: print 'done.'
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
            --> for bn in self.iter_blocks(block=False, name=True):
            ...     delattr(self, bn)
            --> self.phys_units_requested = False

        Args:
            on_load (bool):     If True also convert newly loaded blocks
                                automatically.
        '''
        root = self._root

        # remember whether to convert to blocks or not
        root._phys_units_requested = bool(on_load)

        # convert all loaded blocks
        for block, name in self.iter_blocks(name=True):
            self._convert_block_to_physical_units(block, name)

        # convert the boxsize
        if environment.verbose: print 'convert boxsize to physical units...',
        sys.stdout.flush()
        root._boxsize.convert_to(
                root._boxsize.units.free_of_factors(['a','h_0']), subs=root)
        if environment.verbose: print 'done.'
        sys.stdout.flush()

    def load_all_blocks(self):
        '''Load all blocks from file. (No block deriving, though.)'''
        for name in self._root._load_name:
            host = self.get_host_subsnap(name)
            getattr(host, name)

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
            for name, (rule, deps) in self._root._derive_rule_deps.iteritems():
                host = self.get_host_subsnap(name)
                if name in host.__dict__:
                    delattr(host, name)
        if loaded:
            for name in self._root._load_name:
                host = self.get_host_subsnap(name)
                if name in host.__dict__:
                    delattr(host, name)

    def get(self, expr, units=None):
        '''
        Evaluate the expression for a new block (e.g.: 'log10(Fe/mass)').

        The evaluation is done with a utils.Evaluator with numpy for math and the
        snapshot's blocks as well as some chosen objects (e.g. 'dist', 'Unit',
        'UnitArr', 'UnitQty', 'calc_mags', 'solar', 'G') as namespace.

        Args:
            expr (str):         An python expression that knows about the blocks
                                in this snapshot.
            units (str, Unit):  If set, convert the array to this units before
                                returning.

        Returns:
            res (SimArr):       The result.
        '''
        from sim_arr import SimArr
        from ..luminosities import calc_mags

        # prepare evaluator
        from numpy.core.umath_tests import inner1d
        namespace = {'dist':dist, 'Unit':Unit, 'Units':Units, 'UnitArr':UnitArr,
                     'UnitQty':UnitQty, 'UnitScalar':UnitScalar,
                     'inner1d':inner1d, 'calc_mags':calc_mags,
                     'perm_inv':utils.perm_inv, 'solar':physics.solar,
                     'G':physics.G, 'm_p':physics.m_p}
        e = utils.Evaluator(namespace, my_math=np)
        # load properties etc. from this snapshot only if needed in the expression
        namespace = {}
        for name, el in utils.iter_idents_in_expr(expr, True):
            if name not in e.namespace and name not in namespace:
                try:
                    namespace[name] = getattr(self,name)
                except AttributeError:
                    if not isinstance(el, ast.Attribute):
                        raise ValueError("A unknown name ('%s') " % name +
                                         "appeared in expression for a new" +
                                         "block: '%s'" % expr)
                except:
                    raise
        res = e.eval(expr, namespace)
        res = res.view(SimArr)
        res._snap = self
        if units is not None:
            res.convert_to(units)

        return res

    def IDs_unique(self):
        '''Check whether the IDs (of this (sub-)snapshot) are unique.'''
        num_unique_IDs = len(set( self.ID ))
        return num_unique_IDs == len(self.ID)

class _SubSnap(_Snap):
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
        >>> from ..environment import module_dir
        >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_470', physical=False)
        >>> if any(s.gas.parts[1:]): print s.gas.parts
        >>> if any(s.stars.parts[:4]) or any(s.stars.parts[5:]): print s.gas.parts
        >>> s[123:len(s):100]
        <Snap "snap_M1196_4x_470"[123:2079055:100]; N=20,790; z=0.000>
        >>> s[123:len(s):100].parts
        [9216, 10015, 568, 193, 798, 0]
        >>> s.pos -= [34623, 35522, 33181]
        load block pos... done.
        >>> mask = abs(s.pos).max(axis=-1) < 100
        >>> s[mask]
        <Snap "snap_M1196_4x_470":masked; N=355,218; z=0.000>
        >>> s[mask][::100].parts
        [431, 2459, 0, 0, 663, 0]
        >>> s[mask][::100].gas
        <Snap "snap_M1196_4x_470":masked[::100]:gas; N=431; z=0.000>
        >>> slim = s[::100]
        >>> slim.parts
        [9218, 10014, 568, 193, 798, 0]
        >>> np.sum(slim.gas.mass >= 0.0002)
        load block mass... done.
        UnitArr(1)
        >>> slim_mask = slim.mass < 0.0002  # exludes one gas particle...
        >>> slim[slim_mask].parts
        [9217, 0, 0, 0, 798, 0]
        >>> slim[slim_mask][100:9000:3].rho
        load block rho... done.
        SimArr([  3.18597343e-10,   8.68698516e-11,   2.53987748e-10, ...,
                  9.36064892e-10,   1.11472465e-09,   4.26807673e-10],
               dtype=float32, units="1e+10 Msol ckpc**-3 h_0**2", snap="snap_M1196_4x_470":gas)
        >>> len(slim[slim_mask][100:9000:3].rho)
        2967
        >>> mask = np.zeros(len(s), dtype=bool)
        >>> for pt in gadget.families['baryons']:
        ...     mask[sum(s.parts[:pt]):sum(s.parts[:pt+1])] = True

        s[mask], however, is not host (!)
        >>> np.all( s[mask].elements == s.baryons.elements )
        load block elements... done.
        UnitArr(True, dtype=bool)
        >>> len(slim[slim_mask].elements)
        10015
        >>> assert s[2:-123:10].stars.age.shape[0] == len(s[2:-123:10].stars)
        load block age... done.
        >>> s[2:-123:10].stars.age
        SimArr([ 0.77399528,  0.950091  ,  0.81303227, ...,  0.2623567 ,
                 0.40703428,  0.28800005],
               dtype=float32, units="a_form", snap="snap_M1196_4x_470":stars)
        >>> assert slim[slim_mask].stars.age.shape[0] == len(slim[slim_mask].stars)
        >>> slim[slim_mask].stars.age[::100]
        SimArr([ 0.9974333 ,  0.30829304,  0.64798123,  0.60468233,  0.18474203,
                 0.53754038,  0.18224117,  0.26500258],
               dtype=float32, units="a_form", snap="snap_M1196_4x_470":stars)

        # index list (+family) tests (see comments in SubSnap)
        >>> r = np.sqrt(np.sum(s.pos**2, axis=1))
        >>> assert np.all( s[r<123].pos == s[np.where(r<123)].pos )
        >>> assert np.all( s[s.pos[:,0]>0].pos == s[np.where(s.pos[:,0]>0)].pos )
        >>> assert np.all( s.stars[s.stars.inim>s.stars.mass.mean()].pos ==
        ...         s.stars[np.where(s.stars.inim>s.stars.mass.mean())].pos )
        load block inim... done.
    '''
    def _calc_N_part_from_slice(self, N_part_base, _slice):
        # work with positive stepping (and without None's):
        _slice = utils.positive_simple_slice(_slice, sum(N_part_base))
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
            remain = min(stop, N_cum_base[pt]) - cur_pos
            N_part[pt] = remain / step
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
        self._base = base
        self._root = base._root

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
        elif isinstance(mask, np.ndarray) and mask.dtype == 'bool':
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
            _slice = utils.positive_simple_slice(self._mask, len(base))
            start, stop, step = _slice.start, _slice.stop, _slice.step
            offset = sum( base._N_part[:ptypes[0]] )
            if utils.is_consecutive(ptypes):
                if start < sum(base._N_part[:ptypes[0]]):
                    diff = sum(base._N_part[:ptypes[0]]) - start
                    start += diff / step * step
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
            assert isinstance(self._mask, np.ndarray) and self._mask.dtype == 'bool'
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

    def _set_block_attr(self, name):
        '''
        This gets called, if the attribute for a block is not yet set.

        Here for a _SubSnap, one has to handle the case where the block is not
        available for the base. It has to be stored appropiately then.
        '''
        if hasattr(self._base,name):
            return getattr(self._base,name)[self._mask]
        else:
            # Due to calling by __getattr__, the following should always be
            # fulfilled:
            assert name in self.available_blocks()
            host = self.get_host_subsnap(name)
            if host is self:
                block = self._get_block(name)  # no slicing/masking needed!
                return block
            else:
                # delegate the loading to the host
                block = getattr(host, name)
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
                ptypes = [pt for pt in xrange(6) if host._N_part[pt]]
                masks = []
                sub = self
                subs = []
                while sub != sub._root:
                    masks.append( sub._restrict_mask_to_ptypes(ptypes) )
                    subs.append( sub )
                    sub = sub._base
                for mask in reversed(masks):
                    block = block[mask]
                setattr(self, name, block)
                return block

def SubSnap(base, mask):
    '''
    A factory function for creating sub-snapshots.

    Sub-snapshots should always be instantiated with this function.

    Args:
        base (Snap):    The snapshot to create a sub-snapshot from.

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

                        * family name (str):
                            Creates a sub-snapshot of all the particles of the
                            given family of the base snapshot.

                        * particle types (list):
                            Create a sub-snapshot of the specified particle types
                            only.

                        * mask class (SnapMask):
                            Create a sub-snapshot according to the mask.

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
    from masks import SnapMask

    if isinstance(mask,slice) \
            or (isinstance(mask,np.ndarray) and mask.dtype=='bool'):
        # slices and masks are handled directly by _SubSnap
        return _SubSnap(base, mask)

    elif isinstance(mask,str) and mask in gadget.families:
        # family sub-snapshot
        family_s = SubSnap(base, gadget.families[mask])
        family_s._descriptor = base._descriptor + ':' + mask
        return family_s

    elif isinstance(mask,list):
        ptypes = sorted(set(mask))
        # precalculating N_part is faster than the standard way in _SubSnap
        N_part = [(base._N_part[pt] if pt in ptypes else 0) for pt in xrange(6)]
        if utils.is_consecutive(ptypes):
            # slicing is faster than masking!
            sub = slice(sum(base._N_part[:ptypes[0]   ]),
                        sum(base._N_part[:ptypes[-1]+1]))
        else:
            l = [(np.ones(base._N_part[pt],bool) if pt in ptypes
                    else np.zeros(base._N_part[pt],bool)) for pt in xrange(6)]
            sub = np.concatenate(l)
        sub = _SubSnap(base, sub, N_part)
        sub._descriptor = base._descriptor + ':pts=' + str(ptypes).replace(' ','')
        return sub

    elif isinstance(mask,SnapMask):
        sub = SubSnap(base, mask.get_mask_for(base))
        sub._descriptor = base._descriptor + ':' + str(mask)
        return sub

    elif isinstance(mask,np.ndarray) and mask.dtype.kind=='i':
        # I probably do not want to keep this. Increases backward compability,
        # though.
        warnings.warn('Consider using the faster boolean masks!')
        # is convering into boolean array the best choice?
        warnings.warn('Indexed snapshot masking does not reorder!')
        idx_set = set(mask)
        if len(idx_set) < len(mask):
            print >> sys.stderr, "WARNING: lost %d" % (len(mask)-len(idx_set)) + \
                                 " particles in snapshot masking!"
        mask = np.array( [ (i in idx_set) for i in xrange(len(base)) ] )
        return _SubSnap(base, mask)

    elif isinstance(mask,tuple) and len(mask)==1 \
            and isinstance(mask[0],np.ndarray) and mask[0].dtype.kind=='i':
        return SubSnap(base, mask[0])

    else:
        raise KeyError('Mask of type %s not understood.' % type(mask).__name__)

