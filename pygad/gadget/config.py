'''
Module for reading config files for Gadget parameters (not actual Gadget
configs!).

Example:
    >>> from ..environment import module_dir
    >>> read_config([module_dir+'config/gadget.cfg'])
    reading config file "gadget.cfg"
    >>> block_order
    ['POS ', 'VEL ', 'ID  ', 'MASS']
    >>> families.keys()
    dict_keys(['gas', 'stars', 'dm', 'bh', 'gands', 'sandbh', 'baryons', 'highres', 'lowres'])
    >>> families['dm']
    [1, 2, 3]
    >>> block_infos['MASS'], block_infos['HSML']
    ((1, 'float', None), (1, 'float', 'gas'))
    >>> get_block_units('RHO ').standardize()
    Unit("6.76991e-16 a**-3 g h_0**2 m**-3")
    >>> HDF5_to_std_name['Coordinates'], HDF5_to_std_name['ParticleIDs']
    ('POS ', 'ID  ')
    >>> std_name_to_HDF5['POS '], std_name_to_HDF5['ID  ']
    ('Coordinates', 'ParticleIDs')
    >>> elements[:3]
    ['He', 'C', 'Mg']
'''
__all__ = ['families', 'elements', 'default_gadget_units', 'block_units',
           'std_name_to_HDF5', 'HDF5_to_std_name', 'read_config',
           'get_block_units', 'general', 'block_infos']

from configparser import SafeConfigParser
from ..units import *
from os.path import exists, expanduser
from .. import environment
from .. import kernels
# already with some basic default values
families = {'gas':[0], 'stars':[4], 'dm':[1,2,3], 'bh':[5], 'baryons':[0,4,5]}
block_order = []
elements = []
# Horst: H_neutral_only added
general = {
    'kernel': '<undefined>',
    'vol_def_x': '<undefined>',
    'UVB': '<undefined>',
    'IMF': '<undefined>',
    'SSP_dir': '<undefined>',
    'unclear_blocks': 'warning',
    'Profile_dir': '',
    'H_neutral_only': False,
    }
block_infos = {}
# def. units have to be strings - they are used as replacements
default_gadget_units = {
    'LENGTH':   'ckpc/h_0',
    'VELOCITY': 'km/s',
    'MASS':     '1e10 Msol/h_0',
    }
block_units = {}
std_name_to_HDF5 = {}
HDF5_to_std_name = {}

def read_config(config):
    '''
    Reading some Gadget file definitions from a config file.

    The config file can have the following sections (the first two are required):

    families:           A definition of the families in terms of the particle
                        types. It must define gas, stars, dm (dark matter),
                        baryons, and bh (black holes).
    general:            A definition of the block ordering for format 1 files
                        (without info block), a list of the elements in block 'Z',
                        IFM definition, and more.
    base units:         The (default) Gadget base units (length, velocity, mass).
    block units:        The units for the different blocks.
    hdf5 names:         Name correspondences blocks in HDF5 files. (From HDF5 to
                        standard ones)

    Args:
        config (list):  list of possible filenames for the config file.
    '''
    global families, block_order, elements, default_gadget_units, block_units, \
           std_nameto_HDF5, HDF5_to_std_name, general, block_infos

    def test_section(cfg, section, entries):
        if not cfg.has_section(section):
            raise KeyError('Section "%s" is required in Gadget ' % section +
                           'config file.')
        if set(cfg.options(section)) < set(entries):
            raise ValueError('Section "%s" must have the ' % section +
                             'following entries: ' + str(entries))

    for filename in config:
        if exists(expanduser(filename)):
            break
    else:
        raise IOError('Config file "%s" does not exist!' % config)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        filenamep = os.path.split(filename)[1]
        print('reading config file "%s"' % filenamep)

    # The SafeConfigParser class has been renamed to ConfigParser in Python 3.2, not yet removed for compatibility
    cfg = SafeConfigParser(allow_no_value=True,
                           inline_comment_prefixes = ('#', ';') ) # new to python3. ignores comments at end of values
    cfg.optionxform = str
    cfg.read(filename)

    test_section(cfg, 'general', ['kernel', 'vol_def_x', 'IMF'])
    test_section(cfg, 'families', ['gas', 'stars', 'dm', 'bh', 'baryons'])
    test_section(cfg, 'base units', ['LENGTH', 'VELOCITY', 'MASS'])

    families.clear()
    for family in cfg.options('families'):
        if family in cfg.defaults():
            continue
        families[family] = sorted([int(t) for t
                in cfg.get('families',family).split(',')])

    while block_order: block_order.pop()
    if cfg.has_option('general', 'block order'):
        block_order += ['%-4s'%s.strip() for s in cfg.get('general','block order').split(',')]
    while elements: elements.pop()
    if cfg.has_option('general', 'elements'):
        elements += list(map(str.strip, cfg.get('general', 'elements').split(',')))
    kernel = cfg.get('general', 'kernel')
    if kernel not in kernels.kernels:
        raise ValueError('Kernel "%s" is unknown!' % kernel)
    general['kernel'] = kernel
    x = cfg.get('general', 'vol_def_x')
    if x == '1':
        x = 'ones(len(gas))'
    general['vol_def_x'] = x
    general['UVB'] = cfg.get('general', 'UVB')
    IMF = cfg.get('general', 'IMF')
    if IMF not in ['Kroupa', 'Salpeter', 'Chabrier']:
        raise ValueError('IMF "%s" is unknown!' % IMF)
    general['IMF'] = IMF
    if cfg.has_option('general', 'SSP_dir'):
        general['SSP_dir'] = cfg.get('general', 'SSP_dir',
                                     vars={'PYGAD_DIR':environment.module_dir})
    unclear_blocks = cfg.get('general', 'unclear_blocks')
    if unclear_blocks not in ['exception', 'warning', 'ignore']:
        raise ValueError('Unclear block mode "%s" is unknown!' % unclear_blocks)
    general['unclear_blocks'] = unclear_blocks

    if cfg.has_option('general', 'Profile_dir'):
        general['Profile_dir'] = cfg.get('general', 'Profile_dir')

    # Horst: read config-file into configuration
    if cfg.has_option('general', 'H_neutral_only'):
        general['H_neutral_only'] = cfg.getboolean('general','H_neutral_only')
    else:
        general['H_neutral_only'] = False # Gadget default value

    default_gadget_units.clear()
    default_gadget_units.update( cfg.items('base units') )

    block_infos.clear()
    if cfg.has_section('block infos'):
        from ast import literal_eval
        block_infos.update( { '%-4s'%n:literal_eval(u)
                                for n,u in cfg.items('block infos') } )

    block_units.clear()
    if cfg.has_section('block units'):
        block_units.update( { '%-4s'%n:u for n,u in cfg.items('block units') } )

    std_name_to_HDF5.clear()
    if cfg.has_section('hdf5 names'):
        std_name_to_HDF5.update( { '%-4s'%std:HDF5 for std,HDF5 \
                in cfg.items('hdf5 names') } )
    HDF5_to_std_name.clear()
    HDF5_to_std_name.update( { HDF5:std for std,HDF5 \
            in std_name_to_HDF5.items() } )

def get_block_units(block, gad_units=None):
    '''
    Return the default units (i.e. the units Gadget stores the block in) for a
    given block.

    Args:
        block (str):        The name of the block (e.g. "MASS").
        gad_units (dict):   The basic gadget units to use.

    Returns:
        unit (Unit):        The default units of this block.

    Raises:
        KeyError:           If the units are not known.
    '''
    if block not in block_units:
        raise KeyError('Units of block "%s" are not known' % block)
    if gad_units is None:
        gad_units = default_gadget_units
    gad_units = gad_units.copy()
    gad_units['TIME'] = gad_units['LENGTH'] + ' / (' + gad_units['VELOCITY'] + ')'

    u = block_units[block]
    for dimension, unit in gad_units.items():
        if isinstance(unit, str):
            u = u.replace(dimension, '('+str(unit)+')')
        else:
            u = u.replace(dimension, '('+str(Unit(unit))[1:-1]+')')
    return Unit(u).gather()
