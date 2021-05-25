'''
Module for low-level access to Gadget files of format 1, 2, and 3 (HDF5).

Writing snapshots is not yet fully supported and HDF5 files cannot be written at
all, yet.

Example:
    This tests format 2 with info block only.
    TODO: add other doctests!? Some are implicitly done in snapshot module.
    >>> from ..environment import module_dir
    >>> orig_file = module_dir+'snaps/snap_M1196_4x_470'
    >>> dest_file = module_dir+'test.gad'
    >>> blocks = {}
    >>> with open(orig_file, 'rb') as gfile:
    ...     gformat, endianness = get_format_and_endianness(gfile)
    ...     header = read_header(gfile, gformat, endianness)
    ...     info = get_block_info(gfile, gformat, endianness, header, 'exception')
    ...     for block in info.values():
    ...         N = sum(header['N_part'][i] for i in range(6) if block.ptypes[i])
    ...         gfile.seek(block.start_pos)
    ...         data = np.fromfile(gfile,
    ...                             dtype=endianness+block.type_descr,
    ...                             count=N*block.dimension)
    ...         if block.dimension > 1:
    ...             data = data.reshape( (N,block.dimension) )
    ...         blocks[block.name] = data
    300
    24948984
    49897668
    58213912
    66530156
    70217012
    73903868
    77590724
    81277580
    84964436
    88651292
    88970372
    89289452
    89608532
    137679212
    145995456
    >>> import os
    >>> assert not os.path.exists(dest_file)
    >>> info = sorted(info.values(), key=lambda x: x.start_pos)
    >>> with open(dest_file, 'wb') as gfile:
    ...     write_header(gfile, header, gformat, endianness)
    ...     for block in info:
    ...         write_block(gfile, block.name, blocks[block.name], gformat,
    ...                     endianness)
    ...     write_info(gfile, info, gformat, endianness)
    >>> import filecmp
    >>> assert filecmp.cmp(orig_file, dest_file)
    >>> os.remove(dest_file)
'''
__all__ = ['get_format_and_endianness', 'read_header', 'write_header',
           'BlockInfo', 'write_info', 'get_block_info', 'write_block']

import sys
from ..environment import secure_get_h5py
h5py = secure_get_h5py()
import struct
from . import config
import numpy as np
from itertools import combinations

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2

def get_format_and_endianness(gfile):
    '''
    Get format and endianness of the already opened gadget file.

    Args:
        gfile (file):       The already in binary read mode opened Gadget file or
                            a HDF5 file (which then is detected as format 3).

    Returns:
        gformat (int):      The format; either 1, 2, or 3.
        endianness (str):   Either '=' for native endianness or '<' / '>' for
                            non-native little / big endianness. None in case of a
                            HDF5 file.
    '''
    if isinstance(gfile, h5py.File):
        return 3, None

    gfile.seek(0)
    size, = struct.unpack('=i', gfile.read(4))
    if size == 256:
        gformat    = 1
        endianness = '='
    elif size == 8:
        gformat    = 2
        endianness = '='
        assert struct.unpack('=4s', gfile.read(4))[0].decode('ascii') == 'HEAD'
        assert struct.unpack('=i', gfile.read(4)) == (264,)
        assert struct.unpack('=i', gfile.read(4)) == (8,)
        size = struct.unpack('=i', gfile.read(4))
    elif size == 65536:     # header size (256) with endian swap
        gformat    = 1
        # the non-native endianness
        endianness = '>' if sys.byteorder == 'little' else '<'
    elif size == 134217728: # header pre-block size (8) with endian swap
        gformat    = 2
        # the non-native endianness
        endianness = '>' if sys.byteorder == 'little' else '<'
        assert struct.unpack(endianness + '4s', gfile.read(4))[0].decode('ascii') == 'HEAD'
        assert struct.unpack(endianness + 'i', gfile.read(4)) == (264,)
        assert struct.unpack(endianness + 'i', gfile.read(4)) == (8,)
        size = struct.unpack(endianness + 'i', gfile.read(4))
    else:
        raise RuntimeError('Encountered unexpected bytes (%d) ' % size + \
                           'at beginning of Gadget file.')
    return gformat, endianness

def read_header(gfile, gformat, endianness):
    '''
    Read and return the header of a single Gadget file.

    Args:
        gfile (file):       The already in binary read mode opened Gadget file or
                            with h5py open HDF5 Gadget file.
        gformat (int):      Gadget file format (either 1, 2, or 3 (HDF5)).
        endianness (str):   The endianness of the file (either native '=' or
                            non-native '<' (little) or '>' (big)). Ignored for
                            HDF5 (format 3) files.

    Returns:
        header (dict):      The header class containting everything of the real
                            header.
    '''
    if gformat not in [1,2,3]:
        raise ValueError('Only formats 1, 2, and 3 (HDF5) are known!')

    header = dict()

    if gformat == 3:
        hattrs = gfile['Header'].attrs

        header['N_part'] = list(hattrs['NumPart_ThisFile'])
        header['mass'] = list(hattrs['MassTable'])
        header['time'] = float(hattrs['Time'])
        header['redshift'] = float(hattrs['Redshift'])
        header['flg_sfr'] = int(hattrs['Flag_Sfr'])
        header['flg_feedback'] = int(hattrs['Flag_Feedback'])
        header['N_part_all'] = list(hattrs['NumPart_Total'].astype(np.uint64) + \
                 (hattrs['NumPart_Total_HighWord'].astype(np.uint64)<<32))
        header['flg_cool'] = int(hattrs['Flag_Cooling'])
        header['N_files'] = int(hattrs['NumFilesPerSnapshot'])
        header['boxsize'] = float(hattrs['BoxSize'])
        header['Omega_m'] = float(hattrs['Omega0'])
        header['Omega_Lambda'] = float(hattrs['OmegaLambda'])
        header['h_0'] = float(hattrs['HubbleParam'])
        header['flg_age'] = int(hattrs['Flag_StellarAge'])
        header['flg_metals'] = int(hattrs['Flag_Metals'])
        header['flg_entropy_instead_u'] = None #int(hattrs['???'])
        header['flg_doubleprecision'] = int(hattrs['Flag_DoublePrecision'])
        header['flg_ic_info'] = int(hattrs['Flag_IC_Info']) \
                if 'Flag_IC_Info' in hattrs else None
        header['lpt_scalingfactor'] = None #float(hattrs['???'])
        header['unused'] = ' '*68  # for consistency
    else:
        gfile.seek(4 if gformat == 1 else 4+8+4+4)

        header['N_part'] = list(struct.unpack(endianness + '6I', gfile.read(6*4)))
        header['mass'] = list(struct.unpack(endianness + '6d', gfile.read(6*8)))
        header['time'], header['redshift'], header['flg_sfr'], \
            header['flg_feedback'] \
                = struct.unpack(endianness + 'd d i i', gfile.read(8+8+4+4))
        header['N_part_all'] = list(struct.unpack(endianness + '6I', gfile.read(6*4)))
        header['flg_cool'], header['N_files'], header['boxsize'], \
            header['Omega_m'], header['Omega_Lambda'], header['h_0'], \
            header['flg_age'], header['flg_metals'], \
            header['flg_entropy_instead_u'], header['flg_doubleprecision'], \
            header['flg_ic_info'], header['lpt_scalingfactor'] \
                = struct.unpack(endianness + 'i i 4d 5i d',
                                gfile.read(2*4+4*8+5*4+8))
        header['unused'] = gfile.read(68).decode('ascii')

        assert struct.unpack(endianness + 'i', gfile.read(4)) == (256,)

    return header

def write_header(gfile, header, gformat, endianness):
    '''
    Write header to the (Gadget-)file gfile with given format and endianness.

    Args:
        gfile (file):       The already in binary write mode opened Gadget file.
        header (dict):      The Gadget header to write.
        gformat (int):      Gadget file format (either 1, 2, or 3 (HDF5)).
        endianness (str):   The endianness of the file (either native '=' or
                            non-native '<' (little) or '>' (big)).
    '''
    if gformat not in [1,2,3]:
        raise ValueError('Only formats 1, 2, and 3 (HDF5) are known!')

    if gformat == 3:
        hat = gfile.create_group('Header').attrs
        hat['NumPart_ThisFile'] = np.array(header['N_part'], dtype=np.int32)
        hat['MassTable'] = np.array(header['mass'], dtype=np.float64)
        hat['Time'] = header['time']
        hat['Redshift'] = header['redshift']
        hat['Flag_Sfr'] = header['flg_sfr']
        hat['Flag_Feedback'] = header['flg_feedback']
        hat['NumPart_Total'] = np.array(header['N_part_all'], dtype=np.uint32)
        hat['NumPart_Total_HighWord'] = np.array(np.array(header['N_part_all'],
                                                           dtype=np.uint64) >> 32,
                                                  np.uint32)
        hat['Flag_Cooling'] = header['flg_cool']
        hat['NumFilesPerSnapshot'] = header['N_files']
        hat['BoxSize'] = header['boxsize']
        hat['Omega0'] = header['Omega_m']
        hat['OmegaLambda'] = header['Omega_Lambda']
        hat['HubbleParam'] = header['h_0']
        hat['Flag_StellarAge'] = header['flg_age']
        hat['Flag_Metals'] = header['flg_metals']
        #hat['???'] = header['flg_entropy_instead_u']
        hat['Flag_DoublePrecision'] = header['flg_doubleprecision']
        if header['flg_ic_info'] is not None:
            hat['Flag_IC_Info'] = header['flg_ic_info']
        #hat['???'] = header['lpt_scalingfactor']
        #header['unused']
    else:
        size = 256
        if gformat == 2:
            _write_format2_leading_block(gfile, 'HEAD', size, endianness)
        gfile.write(struct.pack(endianness + 'i', size))
        start_pos = gfile.tell()

        gfile.write(struct.pack(endianness + '6i', *header['N_part']))
        gfile.write(struct.pack(endianness + '6d', *header['mass']))
        gfile.write(struct.pack(endianness + 'd d i i', header['time'],
                header['redshift'], header['flg_sfr'], header['flg_feedback']))
        gfile.write(struct.pack(endianness + '6i', *header['N_part_all']))
        gfile.write(struct.pack(endianness + 'i i 4d 5i d',
                header['flg_cool'], header['N_files'], header['boxsize'],
                header['Omega_m'], header['Omega_Lambda'], header['h_0'],
                header['flg_age'], header['flg_metals'],
                header['flg_entropy_instead_u'], header['flg_doubleprecision'],
                header['flg_ic_info'], header['lpt_scalingfactor']))
        if isinstance(header['unused'], str):
            gfile.write(header['unused'].encode('ascii'))
        else:
            gfile.write(header['unused'])


        assert gfile.tell() - start_pos == size
        gfile.write(struct.pack(endianness + 'i', size))

class BlockInfo(object):
    '''
    Simple class that holds information of a Gadget file block.

    Args:
        name (str):             The name of the block.
        dtype (str, np.dtype):  The type of the data in form of a
                                numpy.dtype.name, a Gadget style name or a
                                np.dtype.
        dimension (int):        The number of data elements per particle.
        ptypes (list):          A list of booleans telling whether the data is
                                present for a particle type.
        start_pos (int):        The binary start position of the data block on the
                                file or None if it is a HDF5 file.
        size (int):             The size of the data block on the file. None if it
                                is s HDF5 file.
    '''
    _np_type_name = {'LONG':'uint32', 'LLONG':'uint64', 'FLOAT':'float32',
                     'DOUBLE':'float64'}

    def __init__(self, name, dtype, dimension, ptypes, start_pos, size):
        if isinstance(dtype, str) and dtype[-1] == 'N' and dimension < 2:
            raise ValueError('Dimension has to be larger one, if the Gadget ' +
                             'typename ends with "N"!')
        self.name      = name
        self.dtype     = dtype
        if dimension is not None:
            self.dimension = int(dimension)
        else:
            self.dimension = dimension
        self.ptypes    = ptypes
        self.start_pos = start_pos
        self.size      = size

    def is_filled(self, check_type):
        non_none = (self.name, self.start_pos, self.size)
        if check_type:
            non_none += (self.dtype, self.dimension, self.ptypes)
        return not any((prop is None) for prop in non_none)

    @property
    def dtype(self):
        return self._dtype

    @property
    def type_descr(self):
        return self._dtype.kind + str(self._dtype.itemsize)

    @property
    def Gadget_type_name(self):
        Gadget_name = {n:g for g,n
                        in BlockInfo._np_type_name.items()}[self._dtype.name]
        if self.dimension > 1:
            Gadget_name += 'N'
        return Gadget_name

    @dtype.setter
    def dtype(self, dt):
        if dt is None:
            self._dtype = None
            return
        if isinstance(dt, str):
            dt = BlockInfo._np_type_name.get(dt.rstrip('N'), dt)
        dt = np.dtype(dt)
        self._dtype = dt.base

def _read_info(gfile, header, start_pos, block_sizes, info_pos, endianness):
    '''
    Read and return the INFO block from gfile that is positioned at info_pos.

    In practical situations use 'get_block_info' to get the block informations
    from a file. That also works if there is no INFO block.

    Args:
        gfile (file):       The already in binary read mode opened Gadget file.
        header (dict):      The header of the corresponding Gadget file for
                            checking (or to provide hints for deductions).
        start_pos (list):   The start positions of all blocks.
        block_sizes (list): The block sizes of all blocks (pure, the number
                            before the actual block in the file).
        info_pos (int):     The position of the info block (not the leading size,
                            but the actual block).
        endianness (str):   The endianness of the file (either native '=' or
                            non-native '<' (little) or '>' (big)).

    Returns:
        info (dict):        A dictionary with BlockInfo classes as values,
                            containing all information of the blocks.
    '''
    info = {}
    gfile.seek(info_pos)
    for i in range(len(start_pos)):
        if i == 0:                      # HEAD
            pass
        elif start_pos[i] == info_pos:  # INFO block itself
            pass
        else:                           # read info for this block
            name        = struct.unpack(endianness + '4s', gfile.read(4))[0].decode('ascii')
            type_name   = struct.unpack(endianness + '8s', gfile.read(8))[0].strip().decode('ascii')
            dimension   = struct.unpack(endianness + 'i',  gfile.read(4))[0]
            ptypes      = [ n==1 for n in
                    struct.unpack(endianness + '6i', gfile.read(6*4)) ]
            info[name] = BlockInfo(name=name,
                                   dtype=type_name,
                                   dimension=dimension,
                                   ptypes=ptypes,
                                   start_pos=start_pos[i],
                                   size=block_sizes[i])
    return info

def write_info(gfile, info, gformat, endianness):
    '''
    Write the INFO block to the (Gadget-)file gfile with given format and
    endianness.

    Write the Gadget INFO block from the iterable info with BlockInfo elements.
    Here name, type, dimension, and particle types are written. Potentially
    existing blocks in info named HEAD or INFO are skipped.

    Args:
        gfile (file):       The already in binary write mode opened Gadget file.
        info (iterable):    The info to write. The elements have to be BlockInfo
                            classes, which are then going to be written in the
                            order in the iterable, *not* using
                            BlockInfo.start_pos (it is ignored).
        gformat (int):      Gadget file format (either 1 or 2 -- format 3 / HDF5
                            does not have a info block).
        endianness (str):   The endianness of the file (either native '=' or
                            non-native '<' (little) or '>' (big)).
    '''
    if gformat not in [1,2,3]:
        raise ValueError('Only formats 1, 2, and 3 (HDF5) are known!')
    if gformat == 3:
        raise ValueError('HDF5 files (format 3) do not have info blocks!')

    # information of each block is 40 bytes large
    size = len(info) * 40
    if gformat == 2:
        _write_format2_leading_block(gfile, 'INFO', size, endianness)
    gfile.write(struct.pack(endianness + 'i', size))
    start_pos = gfile.tell()

    for block in info:
        gfile.write(struct.pack(endianness + '4s', block.name.encode('ascii')))
        tn = '%-8s' % block.Gadget_type_name
        gfile.write(struct.pack(endianness + '8s', tn.encode('ascii')))
        gfile.write(struct.pack(endianness + 'i',  block.dimension))
        gfile.write(struct.pack(endianness + '6i', *list(map(int,block.ptypes))))

    assert gfile.tell() - start_pos == size
    gfile.write(struct.pack(endianness + 'i', size))

def _fill_block_with_known(block, header):
    '''
    Fill block with the known properties of the block from the config file.
    Also fill the pytes for the MASS block, if not given in the config file.

    Args:
        block (BlockInfo):  The block info class to be filled (if possible).
        header (dict):      The header of the corresponding Gadget file for
                            knowing the number of particle per type.

    Returns:
        type_hint (str):    A hint for the type (e.g. 'float': floating point
                            variable, but unknown whether it is single or double
                            precision).

    Raises:
        RuntimeError:       If the given dtype of the given family is unknown.
    '''
    # read information from config file
    if block.name in config.block_infos:
        info = config.block_infos[block.name]
        block.dimension = info[0]
        if info[1] in ['int', 'uint', 'unsigned', 'float']:
            # type without explicit size -- it's just a hint
            type_hint = info[1]
        else:
            try:
                block.dtype = info[1]
            except TypeError:
                raise RuntimeError('unknown type "%s" for block ' % info[1] +
                                   '"%s"!' % block.name)
            type_hint = None
        if info[2] is None:
            block.dtype = None
        else:
            try:
                fam = list(range(6)) if info[2]=='all' else config.families[info[2]]
                block.ptypes = [(i in fam and header['N_part'][i]>0)
                                    for i in range(6)]
            except KeyError:
                raise RuntimeError('unknown family "%s" for block ' % info[2] +
                                   '"%s"!' % block.name)
    else:
        type_hint = None

    # special treatment of the mass block's ptypes if not given in the config
    # file:
    if block.name == 'MASS' and block.ptypes is None:
        block.ptypes = [(n>0 and m==0) for n,m in
                        zip(header['N_part'],header['mass'])]
        type_hint = 'float'

    return type_hint

def _block_inferring(block, header):
    '''
    Infere those of `dtype`, `dimension`, and `ptypes` that are None for the block
    using its filled properties and combinatorics.

    Args:
        block (BlockInfo):  The block info class to be filled (if possible).
        header (dict):      The header of the corresponding Gadget file to
                            provide hints for deductions.

    Raises:
        RuntimeError:       If the information could not be infered or if the
                            given dtype of the given family is unknown.
                            If, however, there are mutliple combinations ptypes
                            that allow the data type size to be a mutliple of 4
                            bit, do not raise the exception and return the
                            possible combinations.
    '''
    type_hint = _fill_block_with_known(block, header)

    if block.ptypes is None:
        """
        Just do combinatorics: Try all possible combinations of ptypes and collect
        those for which the given block with its size (and its dimension and/or
        dtype, if known) could exist.
        If the dtype is not known, assume its size to be mutliples of 4 (as int32,
        int64, float32, and float64). Furthermore assume that the dimension is not
        larger than 20.
        If there is just one single such combination, we are done. If there is
        more than one, return the combinations, otherwise raise a RuntimeError.
        """
        if block.dimension is not None and block.dtype is not None:
            el_size == block.dimension * block.dtype.itemsize
        else:
            el_size = None
            if block.dimension is not None:
                divisor = 4 * block.dimension
                max_el_size = block.dimension * 8
            elif block.dtype is not None:
                divisor = block.dtype.itemsize
                max_el_size = 20 * block.dtype.itemsize
            else:
                divisor, max_el_size = 4, 20*8
        possible_ptype_combi = []
        for num_ptypes in range(1,7):
            for combi in combinations(list(range(6)), num_ptypes):
                if any(header['N_part'][i] == 0 for i in combi):
                    # avoid N == 0 and multiple combinations with and without
                    # particle types that are not present in this file
                    continue
                N = sum(header['N_part'][i] for i in combi)
                if el_size is not None:
                    if el_size * N == block.size:
                        possible_ptype_combi.append(combi)
                else:
                    element_size = block.size / N
                    if element_size * N == block.size \
                            and element_size % divisor == 0 \
                            and element_size < max_el_size:
                        possible_ptype_combi.append(combi)
        # if the combinatorics yield a definit combination, store it
        if len(possible_ptype_combi) == 1:
            block.ptypes = [(i in possible_ptype_combi[0]) for i in range(6)]
        elif len(possible_ptype_combi) > 1:
            return possible_ptype_combi
        else:
            raise RuntimeError('Could not infere information for block ' +
                               '"%s"!' % block.name)
    # now block.ptypes are known

    if block.dimension is not None and block.dtype is not None:
        return  # all known

    N = sum(header['N_part'][i] for i in range(6) if block.ptypes[i])
    if N == 0:
        # most likely a empty mass block (with masses in header)
        return 'delete'
    element_size = block.size / N
    if block.dtype is not None:
        block.dimension = int(element_size / block.dtype.itemsize)
        return  # all known

    assert block.dtype is None
    # dtype is unknown (and dimension might be as well)...
    if block.dimension is not None:
        itemsize = int(element_size / block.dimension)
        if type_hint == 'int':
            block.dtype = 'i' + str(itemsize)
        elif type_hint in ['uint', 'unsigned']:
            block.dtype = 'u' + str(itemsize)
        else:
            assert type_hint is None or type_hint == 'float'
            # assume float
            block.dtype = 'f' + str(itemsize)
        return  # all known

    # both, dtype and dimension, are unknown...
    if type_hint in ['int', 'uint', 'unsigned']:
        if element_size*8 in [8,16,32,64]:
            block.dtype = type_hint[0] + element_size*8
            block.dimension = 1
        elif element_size % np.dtype(int).itemsize == 0:
            # could be multi-dimensional native
            block.dtype = np.dtype(int)
            block.dimension = int(element_size / block.dtype.itemsize)
        else:
            raise RuntimeError('Could not infere information for block ' +
                               '"%s"!' % block.name)
        assert element_size == block.dimension * block.dtype.itemsize
        return  # all known
    assert type_hint is None or type_hint == 'float'
    # assume float
    block.dtype = 'float64' if header['flg_doubleprecision'] else 'float32'
    block.dimension = int(element_size / block.dtype.itemsize)
    if element_size != block.dimension * block.dtype.itemsize:
        block.dtype = 'float32'
        block.dimension = int(element_size / block.dtype.itemsize)
        if element_size != block.dimension * block.dtype.itemsize:
            block.dtype = None
            block.dimension = None
            raise RuntimeError('Could not infere information for block ' +
                               '"%s"!' % block.name)
    return  # all known

def _infer_info(gfile, header, gformat, endianness, start_pos, block_sizes,
                unclear_blocks):
    '''
    Try to infere block info, if no INFO block is present.

    In practical situations use 'get_block_info' to get the block informations
    from a file.

    Args:
        gfile (file):           The already in binary read mode opened Gadget
                                file.
        header (dict):          The file's header. Used to infer information of
                                the blocks.
        gformat (int):          Gadget file format (either 1 or 2 -- for format 3
                                (HDF5) this function is never needed).
        endianness (str):       The endianness of the file (either native '=' or
                                non-native '<' (little) or '>' (big)).
        start_pos (list):       The start positions of all blocks.
        block_sizes (list):     The block sizes of all blocks (pure, the number
                                before the actual block in the file).
        unclear_blocks (str):   What to do the blocks for which the block info is
                                unclear (cannot be infered). Possible modes are:
                                * exception:    raise an IOError
                                * warning:      print a warning to the stderr
                                * ignore:       guess what

    Returns:
        info (dict):        A dictionary with BlockInfo classes as values,
                            containing all inferred information if the blocks.

    Raises:
        RuntimeError:       If the information could not be infered or if the
                            given dtype of the given family is unknown.
    '''
    if gformat not in [1,2,3]:
        raise ValueError('Only formats 1, 2, and 3 (HDF5) are known!')
    if gformat == 3:
        raise ValueError('HDF5 files (format 3) are not handled here; and this ' +
                         'function is not needed for those.')

    info = {}
    for i in range(len(start_pos)):
        if i == 0:      # HEAD
            continue
        elif block_sizes[i] == (len(block_sizes)-2)*40:
            continue    # INFO
        else:           # actual data block
            if gformat == 1:
                name = config.block_order[i-1] if \
                    i <= len(config.block_order) else '?%03d' % i
            elif gformat == 2:  # can read name
                gfile.seek(start_pos[i]-4-4-8)
                name = struct.unpack(endianness + '4s', gfile.read(4))[0].decode('ascii')
            block = BlockInfo(name=name, dtype=None, dimension=None,
                              ptypes=None, start_pos=start_pos[i],
                              size=block_sizes[i])
            # try to infer type, dimension, and particle types
            combis = _block_inferring(block, header)
            if combis == 'delete':
                continue
            if not block.is_filled(check_type=False):
                assert False    # should never happen
            if not block.is_filled(check_type=True):
                if unclear_blocks == 'exception':
                    raise IOError('Cannot infere info for block "%s"!' %
                            block.name)
                elif unclear_blocks == 'warning':
                    print('WARNING: cannot infere info for ' + \
                                         'block "%s"!\n' % block.name + \
                                         '  possible combinations for the ' + \
                                         'particles types are: ' + \
                                         ', '.join(str(combi) for combi in combis), file=sys.stderr)
                elif unclear_blocks == 'ignore':
                    pass
                else:
                    raise RuntimeError('Unkown block mode ' +
                                       '"%s" is not ' % unclear_blocks +
                                       'understood!')
            info[block.name] = block
    return info

def get_block_info(gfile, gformat, endianness, header, unclear_blocks):
    '''
    Get info (like position in file, if not HDF5) about all blocks in a single
    Gadget file.

    This function utilizes the INFO block of a Gadget file, if (format 1 or 2 and)
    there is a INFO block.
    Otherwise it locates all blocks, does some guessing on the names, if it is a
    format 1 file, and infers the particles for which it is present as well as
    the elment size by combinatorics.

    For HDF5 files, it reads all information available.

    Args:
        gfile (file):           The already in binary read mode opened Gadget
                                file.
        gformat (int):          Gadget file format (either 1, 2, or 3 (HDF5)).
        endianness (str):       The endianness of the file (either native '=' or
                                non-native '<' (little) or '>' (big)).
        header (dict):          The file's header. Used if there is no INFO block
                                and information of the blocks has to inferred.
        unclear_blocks (str):   What to do the blocks for which the block info is
                                unclear (cannot be infered). Possible modes are:
                                * exception:    raise an IOError
                                * warning:      print a warning to the stderr
                                * ignore:       guess what

    Returns:
        info (dict):        A dictionary with BlockInfo classes as values,
                            containing all inferred information if the blocks.

    Raises:
        RuntimeError:       If the information could not be infered or if the
                            given dtype of the given family is unknown.
    '''
    if gformat not in [1,2,3]:
        raise ValueError('Only formats 1, 2, and 3 (HDF5) are known!')

    if gformat == 3:
        info = {}
        for name, group in gfile.items():
            if not name.startswith('PartType'):
                continue
            ptype = int(name[-1])   # name is e.g. 'PartType3'
            # We are accessing the dataset classes, but do not load the actual
            # data! For instance, ds.dtype does not require to load the data (a
            # h5py datastructure is not a np.ndarray!), only actually accessing
            # data within the array (like ds[:] or ds[0]) would load it.
            for ds_name, ds in group.items():
                # for the EAGLE simulations, it happens that there are
                # sub-groups (e.g. for the different element abundancies), hence:
                if not isinstance(ds, h5py.Dataset):
                    # TODO: read the EAGLE elements!
                    continue
                block_name = config.HDF5_to_std_name.get(ds_name, None)
                if block_name is None:
                    block_name = str(ds_name)
                if block_name not in info:
                    info[block_name] = BlockInfo(
                            name=block_name,
                            dtype=ds.dtype.base.name,
                            dimension=ds.shape[1] if len(ds.shape) == 2 else 1,
                            ptypes=[False]*6,
                            start_pos=None,
                            size=None)
                info[block_name].ptypes[ptype] = True
    else:
        # Count blocks and remember start positions.
        # If format 2: Also remember if there is a INFO block and where it is.
        start_pos = []
        block_sizes = []
        if gformat == 2:
            info_pos = None
        gfile.seek(0)
        size = gfile.read(4)
        while size:
            if gformat == 2:
                name = struct.unpack(endianness + '4s', gfile.read(4))[0].decode('ascii')
                gfile.seek(4+4, SEEK_CUR)
                size = gfile.read(4)
            # remember start position and size of block
            start_pos.append(gfile.tell())
            if gformat == 2 and name == 'INFO':
                info_pos = start_pos[-1]
            size, = struct.unpack(endianness + 'i', size)
            block_sizes.append(size)
            # jump over actual block
            gfile.seek(size, SEEK_CUR)
            assert struct.unpack(endianness + 'i', gfile.read(4)) == (size,)
            # try to read size of next block
            size = gfile.read(4)

        if gformat == 1 and block_sizes[-1] == (len(block_sizes)-2)*40:
            # If there is a INFO block, it has to be at the end and of size
            # (#blocks-2)*40.
            info = _read_info(gfile, header, start_pos, block_sizes,
                              start_pos[-1], endianness)
        elif gformat == 2 and info_pos:
            info = _read_info(gfile, header, start_pos, block_sizes, info_pos,
                              endianness)

        else:
            # no info block -> infer as much as possible (using block names in
            # format 2 files)
            info = _infer_info(gfile, header, gformat, endianness, start_pos,
                               block_sizes, unclear_blocks=unclear_blocks)

    if 'MASS' not in info:
        # just put it after ID block as usual
        if gformat==3:
            pos = None
        else:
            pos = info['ID  '].start_pos + info['ID  '].size + \
                    (info['ID  '].start_pos -
                            info['VEL '].start_pos + info['VEL '].size)
        info['MASS'] = BlockInfo(
                name='MASS',
                dtype='float64' if header['flg_doubleprecision']
                            else 'float32',
                dimension=1,
                ptypes=[False]*6,
                start_pos=pos,
                size=0)

    return info

def _write_format2_leading_block(gfile, name, size, endianness):
    '''Little helper function with speaking name, that writes the small leading
    blocks for format 2 Gadget files.'''
    gfile.write(struct.pack(endianness + ' i 4s i i', 8, name.encode('ascii'), size+8, 8))

def write_block(gfile, block_name, data, gformat, endianness='=',
                gad_units=None):
    '''
    Write a block to the (Gadget-)file gfile with given format and endianness.

    Args:
        gfile (file):       The already in binary write mode opened Gadget file.
        block_name (str):   The block name for the block to write.
        data (...):         The data to write. A UnitArr (or simplye a
                            numpy.array) for regular blocks, a header dict for
                            the HEAD block and an iterable with BlockInfo classes
                            as elements for the INFO block.
        gformat (int):      Gadget file format (either 1 or 2 -- format 3 (HDF5)
                            should be written explicitly).
        endianness (str):   The endianness of the file (either native '=' or
                            non-native '<' (little) or '>' (big)).
        gad_units (dict):   The basic gadget units. If None, take the units
                            defined in the `gadget.cfg`.
    '''
    if gformat not in [1,2,3]:
        raise ValueError('Only formats 1, 2, and 3 (HDF5) are known!')
    if gformat == 3:
        raise NotImplementedError('Writing HDF5 blocks shall not be done by ' + \
                                  'this function!')

    if gad_units is None:
        gad_units = config.default_gadget_units

    # warn if the data has units but these are not the default units
    if hasattr(data,'unit') and block_name in config._block_units \
            and data.unit != config.get_block_units(block_name,gad_units):
        warn_msg = "Your are writing the block '%s' in other " % block_name + \
                   "units ('%s') than the default units " % data.unit + \
                   "('%s')!" % config.get_block_units(block_name,gad_units)
        print('WARNING:', warn_msg, file=sys.stderr)
    # reduce data to the numpy array
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    size = data.nbytes
    if gformat == 2:
        _write_format2_leading_block(gfile, block_name, size, endianness)
    gfile.write(struct.pack(endianness + 'i', size))
    start_pos = gfile.tell()

    data.tofile(gfile)

    assert gfile.tell() - start_pos == size
    gfile.write(struct.pack(endianness + 'i', size))

