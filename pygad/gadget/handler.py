'''
High level access for (single) Gadget files.

Doctesting is implicity done in the snapshot module.
'''
__all__ = ['FileReader', 'write']

import os
from .. import environment
from ..environment import secure_get_h5py
h5py = secure_get_h5py()
from .lowlevel_file import *
from ..units import *
from . import config
import numpy as np
import warnings
import sys

class FileReader(object):
    '''
    A handler for reading Gadget files of all formats.

    Args:
        filename (str):         The path to the Gadget file.
        unclear_blocks (str):   What to do the blocks for which the block info is
                                unclear (cannot be infered). Possible modes are:
                                * exception:    raise an IOError
                                * warning:      print a warning to the stderr
                                * ignore:       guess what
                                If it is None, the value from the `gadget.cfg` is
                                taken.

    Raises:
        RuntimeError:       If the information could not be infered or if the
                            given dtype of the given family is unknown.
    '''
    def __init__(self, filename, unclear_blocks=None):
        if not os.path.exists(os.path.expanduser(filename)):
            raise IOError('"%s" does not exist!' % filename)
        self._filename = filename
        if unclear_blocks is None:
            unclear_blocks = config.general['unclear_blocks']
        if h5py.is_hdf5(self._filename):
            self._format, self._endianness = 3, None
            with h5py.File(self._filename, 'r') as gfile:
                self._header = read_header(gfile, self._format,
                                           self._endianness)
                self._info = get_block_info(gfile, self._format,
                                            self._endianness, self._header,
                                            unclear_blocks)
        else:
            with open(os.path.expanduser(self._filename), 'rb') as gfile:
                self._format, self._endianness = \
                            get_format_and_endianness(gfile)
                self._header = read_header(gfile, self._format,
                                           self._endianness)
                self._info = get_block_info(gfile, self._format,
                                            self._endianness, self._header,
                                            unclear_blocks)

    @property
    def filename(self):
        '''The (basic) filename.'''
        return self._filename

    @property
    def format(self):
        '''The Gadget file format.'''
        return self._format

    @property
    def endianness(self):
        '''The endianess of the file.'''
        return self._endianness

    @property
    def header(self):
        '''The file header in form of a dictionary.'''
        return self._header

    def infos(self):
        '''Return a sorted list of the block informations.'''
        try:
            erg = sorted(list(self._info.values()), key=lambda e: e.start_pos)
        except Exception as e:
            erg = list(self._info.values())
        return erg

    def block_names(self):
        '''Return a list of all available blocks.'''
        return list(self._info.keys())

    def has_block(self, name):
        '''Check whether there is a block with the asked name.'''
        return name in self._info

    def read_block(self, name, gad_units=None, unit=None):
        '''
        Args:
            name (str):         The name of the block. For HDF5 files: If not the
                                actual block name is passed, it is tried to
                                convert from the standard Gadget block name to the
                                HDF5 one via 'gadget.config.std_name_to_HDF5'.
            gad_units (dict):   The basic gadget units. If None, take the units
                                defined in the `gadget.cfg`.
            unit (Unit):        The unit to give the data. If it is None, use a
                                default one if defined, Unit(1) otherwise.

        Returns:
            block (UnitArr):    The loaded block.
        '''
        block = self._info[name]
        N = sum(self._header['N_part'][pt] for pt in range(6)
                if block.ptypes[pt])

        if self._format == 3:
            with h5py.File(self._filename, 'r') as gfile:
                HDF5_name = config.std_name_to_HDF5.get(name,name)
                data = np.empty(N*block.dimension, dtype=block.dtype)
                if block.dimension > 1:
                    data = data.reshape( (N,block.dimension) )
                off = 0
                for pt in range(6):
                    if block.ptypes[pt]:
                        ds = gfile['PartType%d' % pt][HDF5_name]
                        data[off:off+ds.shape[0]] = ds[:]   # actually load data
                        off += len(ds)
        else:
            filename = os.path.expanduser(self._filename)
            with open(filename, 'r') as gfile:
                gfile.seek(block.start_pos)
                data = np.fromfile(gfile,
                                   dtype=self._endianness+block.type_descr,
                                   count=N*block.dimension)
                if block.dimension > 1:
                    data = data.reshape( (N,block.dimension) )

        if block.name == 'MASS':
            pos = 0
            for pt in range(6):
                if self._header['mass'][pt]:
                    data = np.insert(data,
                                     pos,
                                     np.ones(self._header['N_part'][pt],
                                             dtype=block.dtype) * \
                                        self._header['mass'][pt])
                    """
                    for md, mh in [(data[pos], self._header['mass'][pt]),
                                   (data[pos+self._header['N_part'][pt]-1],
                                        self._header['mass'][pt])]:
                        assert (md-mh)/mh < 1e-6
                    """
                pos += self._header['N_part'][pt]

        data = data.view(UnitArr)
        if unit is not None:
            data.units = unit
        else:
            try:
                data.units = config.get_block_units(block.name,gad_units)
            except KeyError:
                data.units = 1
            except:
                raise

        return data

def _create_header(s, gad_units=None, double_prec=None):
    '''
    Create a header dictionary as needed by the low-level file routines.

    Up to now: Just one file and no mass section...
    TODO: change that!?

    Args:
        s (Snap):           The (sub-)snapshot to create the header from.
        gad_units (dict):   The Gadget unit system to use. If None, use the one
                            stored in the snapshot.
        double_prec (bool): The value for the flag for double precision. If it is
                            None, the value from the snapshot properties is taken.

    Returns:
        header (dict):  The header dictionary (including the unused section).
    '''
    header = dict()

    if gad_units is None:
        gad_units = s.gadget_units
    else:
        tmp, gad_units = gad_units, s.gadget_units.copy()
        gad_units.update( tmp )

    header['N_part']        = s.parts
    header['mass']          = [0.0]*6
    header['time']          = s.time
    header['redshift']      = s.redshift
    header['flg_sfr']       = int(bool(s.properties['flg_sfr']))
    header['flg_feedback']  = int(bool(s.properties['flg_feedback']))
    header['N_part_all']    = s.parts
    header['flg_cool']      = int(bool(s.properties['flg_cool']))
    header['N_files']       = 1
    header['boxsize']       = float(s.boxsize.in_units_of(gad_units['LENGTH']))
    header['Omega_m']       = s.cosmology.Omega_m
    header['Omega_Lambda']  = s.cosmology.Omega_Lambda
    header['h_0']           = s.cosmology.h_0
    header['flg_age']       = int(bool(s.properties['flg_age']))
    header['flg_metals']    = int(bool(s.properties['flg_metals']))
    header['flg_entropy_instead_u'] = int(bool(
                                        s.properties['flg_entropy_instead_u']))
    if double_prec is None:
        header['flg_doubleprecision'] = s.properties['flg_doubleprecision']
    else:
        header['flg_doubleprecision'] = int(bool(double_prec))
    header['flg_ic_info']   = int(bool(s.properties['flg_ic_info']))
    header['lpt_scalingfactor'] = \
            0.0 if s.properties['lpt_scalingfactor'] is None \
            else float(s.properties['lpt_scalingfactor'])
    header['unused']        = ' '*68

    return header

def write(snap, filename, blocks=None, gformat=2, endianness='native',
          infoblock=True, double_prec=False, gad_units=None, overwrite=False):
    '''
    Write a snapshot.

    TODO:
        * implement putting masses to header if they are all equal for a particle
          type
        * implement writing to multiple files (needed?)

    Args:
        snap (Snap):        The (sub-)snapshot to write.
        filename (str):     The path to write to.
        blocks (list):      A list of the names of the blocks to write (as in the
                            attributes, not the Gadget names, i.e. 'pos' rather
                            than 'POS '). The blocks are written in the order
                            given in this list. The standard blocks 'pos', 'vel',
                            'ID', and 'mass, however, are always written and
                            always in this order at the beginning.
                            If None, all loadable blocks (but no derived ones) of
                            the snapshots are written. The order is random, except
                            that the first blocks are those specified in
                            gadget.config.block_order (non-available blocks are
                            skipped).
        gformat (int):      The file format (either 1, 2, or 3 (HDF5)).
        endianness (str):   The endianess ('='/'native', '<'/'little', '>'/'big').
        infoblock (bool):   Whether to write an infoblock.
        double_prec (bool): Whether to write in double precision.
        gad_units (dict):   The Gadget unit system to use. If None, use the one
                            stored in the snapshot.
        overwrite (bool):   Allow to overwrite, if the file already exists.
    '''
    if os.path.exists(os.path.expanduser(filename)) and not overwrite:
        raise IOError('The file "%s" already exists. ' % filename +
                      'Consider "overwrite=True"!')
    if blocks is None:
        blocks = snap.loadable_blocks()
        tmp = []
        for name in config.block_order:
            for attrname, lname in snap._root._load_name.items():
                if name == lname:
                    name = attrname
                    break
            if name in blocks:
                blocks.remove(name)
                tmp.append(name)
        blocks = tmp + blocks
    else:
        std_blocks = ['pos', 'vel', 'ID', 'mass']
        blocks = std_blocks + [name for name in blocks if name not in std_blocks]
    if gformat not in [1,2,3]:
        raise ValueError('Unknow format %s!' % gformat)
    endianness = {'native':'=',
                  'little':'<',
                  'big':'>'}.get(endianness,endianness)
    if endianness not in ['=','<','>']:
        raise ValueError('Unknown endianness %s!' % endianness)
    if gad_units is None:
        gad_units = snap.gadget_units
    else:
        tmp, gad_units = gad_units, snap.gadget_units.copy()
        gad_units.update( tmp )

    # prepare
    header = _create_header(snap, gad_units=gad_units, double_prec=double_prec)
    data = {}
    info = {}
    for name in blocks:
        # do not use block hosts, since we want to be able to write sub-snapshots,
        # as well
        for sub in [snap]+[getattr(snap,fam) for fam in config.families]:
            if len(sub)>0 and name in sub.available_blocks():
                ptypes = [bool(N) for N in sub.parts]
                if name in data:
                    # only update, if we have the bigger block
                    if sum(ptypes) <= sum(info[name].ptypes):
                        continue
                block = sub[name]
                size = block.dtype.itemsize*np.prod(block.shape)
                dtype = block.dtype.base
                if dtype.kind == 'f':
                    if double_prec and dtype != 'float64':
                        size = 8 * size / dtype.itemsize
                        dtype = np.dtype('float64')
                    elif not double_prec and dtype != 'float32':
                        size = 4 * size / dtype.itemsize
                        dtype = np.dtype('float32')
                block_name = snap._root._load_name.get(name, '%-4s'%name.upper())
                info[name] = BlockInfo(name=block_name, dtype=dtype,
                                       dimension=np.prod(block.shape[1:]),
                                       ptypes=ptypes, start_pos=None, size=size)
                if block_name in config.block_units:
                    units = config.get_block_units(block_name, gad_units)
                else:
                    units = sub[name].units
                    warnings.warn('Blocks that do not have default units, are '
                                  'stored in their current units -- other blocks '
                                  'are converted!')
                if gformat == 3:
                    data[name] = [None] * 6
                    for pt in range(6):
                        if sub._N_part[pt]:
                            data[name][pt] = sub[[pt]][name] \
                                                .astype(info[name].dtype) \
                                                .in_units_of(units, subs=snap)
                else:
                    data[name] = sub[name] \
                                    .astype(info[name].dtype) \
                                    .in_units_of(units, subs=snap)

    # it might happen, that a sub-snapshot shall be written for which not all
    # blocks are actually present (e.g. it does not contain gas though the root
    # does)
    for name in set(blocks)-set(data.keys()):
        print('WARNING: block "%s" is not present for ' % name + \
                             'this (sub-)snapshot and, hence, not written!', file=sys.stderr)
        blocks.remove(name)

    if gformat == 3:
        with h5py.File(filename, 'w') as gfile:
            write_header(gfile, header, gformat, endianness)
            for pt in range(6):
                gfile.create_group('PartType%d' % pt)

            for name in blocks:
                d = data[name]
                for pt in range(6):
                    if d[pt] is not None:
                        units = d[pt].units
                hdf5name = snap._root._load_name.get(name)
                if hdf5name in config.std_name_to_HDF5:
                    hdf5name = config.std_name_to_HDF5[hdf5name]
                if hdf5name is None:
                    hdf5name = name

                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print('writing block', hdf5name, end=' ')
                    print('(dtype=%s, units=%s)...' % (info[name].dtype, units), end=' ')
                    sys.stdout.flush()
                for pt in range(6):
                    if not snap._root._block_avail[name][pt]:
                        continue
                    group = gfile['PartType%d' % pt]
                    gf_block = group.create_dataset(
                            hdf5name,
                            d[pt].shape,
                            d[pt].dtype)
                    gf_block[...] = d[pt]
                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print('done.')
                    sys.stdout.flush()
    else:
        with open(os.path.expanduser(filename), 'wb') as gfile:
            write_header(gfile, header, gformat, endianness)
            for name in blocks:
                block_name = info[name].name
                info[name].start_pos = gfile.tell()+4
                if gformat == 2:
                    info[name].start_pos += 4+8+4
                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print('writing block', block_name, end=' ')
                    print('(dtype=%s, units=%s)...' % (info[name].dtype,
                            data[name].units), end=' ')
                    sys.stdout.flush()
                sys.stdout.flush()
                write_block(gfile, block_name, data[name], gformat, endianness)
                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print('done.')
                    sys.stdout.flush()
            info = sorted(list(info.values()), key=lambda x: x.start_pos)
            write_info(gfile, info, gformat, endianness)
