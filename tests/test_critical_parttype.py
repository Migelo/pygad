"""Regression test for CRITICAL #5 (review/REVIEW.md).

pygad/gadget/lowlevel_file.py parsed the particle type of HDF5 groups via
``int(name[-1])``, i.e. only the *last* character of 'PartTypeN'. Multi-digit
groups such as 'PartType10' were hence misattributed: type 10 was silently
merged into type 0, type 11 into type 1, etc.

Run with the project venv:

    .venv/bin/python -m pytest tests/test_critical_parttype.py
"""
import h5py
import numpy as np

from pygad.gadget.handler import FileReader
from pygad.gadget.lowlevel_file import get_block_info


def _write_hdf5_snapshot(path, part_types):
    """Create a minimal HDF5 snapshot with one small block per PartTypeN."""
    with h5py.File(str(path), 'w') as f:
        for pt in part_types:
            grp = f.create_group('PartType%d' % pt)
            grp.create_dataset('Coordinates',
                               data=np.zeros((2, 3), dtype=np.float64))


def _block_ptypes(path, block='POS '):
    # get_block_info always synthesizes a MASS BlockInfo if the file lacks
    # one, which needs the header's double-precision flag
    header = {'flg_doubleprecision': 0}
    with h5py.File(str(path), 'r') as f:
        info = get_block_info(f, 3, None, header, 'exception')
    return info[block].ptypes


def test_parttype10_not_misattributed_to_type0(tmp_path):
    path = tmp_path / 'snap_pt10.hdf5'
    _write_hdf5_snapshot(path, [10])
    ptypes = _block_ptypes(path)
    # pre-fix 'PartType10' was parsed as type 0, silently merging its blocks
    # into particle type 0
    assert ptypes[0] == False, 'PartType10 blocks misattributed to type 0'
    assert len(ptypes) > 10 and ptypes[10], 'PartType10 not recorded'
    assert sum(ptypes) == 1


def test_parttype11_does_not_collide_with_type1(tmp_path):
    path = tmp_path / 'snap_pt11.hdf5'
    _write_hdf5_snapshot(path, [1, 11])
    ptypes = _block_ptypes(path)
    # pre-fix 'PartType11' was parsed as type 1, colliding with 'PartType1'
    expected = [False, True] + [False] * 9 + [True]
    assert list(ptypes) == expected


def test_single_digit_parttypes_unchanged(tmp_path):
    path = tmp_path / 'snap_std.hdf5'
    _write_hdf5_snapshot(path, range(6))
    ptypes = _block_ptypes(path)
    assert list(ptypes) == [True] * 6


def test_filereader_end_to_end(tmp_path):
    # the full FileReader path (read_header + get_block_info) must attribute
    # PartType10 blocks to type 10 and still read the standard types
    path = tmp_path / 'snap_full.hdf5'
    coords0 = np.array([[1., 2., 3.], [4., 5., 6.]])
    coords10 = np.array([[7., 8., 9.]])
    with h5py.File(str(path), 'w') as f:
        h = f.create_group('Header')
        h.attrs['NumPart_ThisFile'] = [2, 0, 0, 0, 0, 0]
        h.attrs['NumPart_Total'] = [2, 0, 0, 0, 0, 0]
        h.attrs['NumPart_Total_HighWord'] = [0] * 6
        h.attrs['MassTable'] = [0.] * 6
        h.attrs['Time'] = 1.0
        h.attrs['Redshift'] = 0.0
        h.attrs['Flag_Sfr'] = 0
        h.attrs['Flag_Feedback'] = 0
        h.attrs['Flag_Cooling'] = 0
        h.attrs['NumFilesPerSnapshot'] = 1
        h.attrs['BoxSize'] = 1000.0
        h.attrs['Omega0'] = 0.3
        h.attrs['OmegaLambda'] = 0.7
        h.attrs['HubbleParam'] = 0.7
        h.attrs['Flag_StellarAge'] = 0
        h.attrs['Flag_Metals'] = 0
        h.attrs['Flag_DoublePrecision'] = 1
        f.create_group('PartType0').create_dataset('Coordinates', data=coords0)
        f.create_group('PartType10').create_dataset('Coordinates', data=coords10)
    reader = FileReader(str(path))
    info = {block.name: block for block in reader.infos()}
    ptypes = info['POS '].ptypes
    assert len(ptypes) > 10 and ptypes[10], 'PartType10 not recorded'
    assert ptypes[0]
    # the standard 6-type read path must still return the type-0 data
    assert np.allclose(np.asarray(reader.read_block('POS ')), coords0)
