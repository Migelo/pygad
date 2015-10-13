'''
A module to prepare derived arrays for the snapshot.

Examples:
    >>> from ..environment import module_dir
    >>> rules = read_cfg([module_dir+'snapshot/derived.cfg'])
    reading config file "pygad/snapshot/derived.cfg"
    >>> assert rules == _rules
    >>> rules['r'], rules['Z']
    ('dist(pos)', 'metals/elements.sum(axis=1)')
    >>> from snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_470')
    >>> ptypes_and_deps(rules['r'], s)
    ([True, True, True, True, True, False], set(['pos']))
    >>> ptypes_and_deps(rules['Z'], s)  # derived from mass (for all particles
    ...                                 # types) and elements (baryons only)
    ([True, False, False, False, True, False], set(['elements', 'metals']))
'''
__all__ = ['ptypes_and_deps', 'read_cfg', 'calc_temps']

from ConfigParser import SafeConfigParser
from .. import utils
from .. import gadget
from .. import environment

_rules = {}

def ptypes_and_deps(defi, snap):
    '''
    Get the ptypes and dependecies of a block definition.

    The ptypes are the greates common set of particle types of all the
    dependencies.

    Args:
        defi (str):     The definition of the block in form of a expression (that
                        is interpretable by Snap.get).
        snap (Snap):    The snapshot the block shall be added to.

    Returns:
        ptypes (list):  A list of length 6 with booleans for the particle types.
        deps (set):     The set of the names of the blocks it depends on.

    Raises:
        ValueError
    '''
    ptypes = [True] * 6    # gets restricted in the following
    deps = set()
    root = snap.root
    for name in utils.iter_idents_in_expr(defi):
        if name in root._block_avail:
            ptypes = [(pt and avail) for pt,avail
                        in zip(ptypes,root._block_avail[name])]
            deps.add(name)
        elif name in gadget.families:
            fam = gadget.families[name]
            ptypes = [(ptypes[i] and (i in fam)) for i in xrange(6)]
    return ptypes, deps

def read_cfg(config, store_as_default=True, delete_old=False):
    '''
    Read rules for derived blocks from a config file.

    The config file consists of a single section 'rules' that has names and their
    corresponding rules (Python expressions, that have the namespace of the
    snapshot and numpy -- see Snap.get for more information).

    Args:
        config (list):              list of possible filenames for the config file.
        store_as_default (bool):    Wether to store them as global defaults.
        delete_old (bool):          If store_as_default==True, delete old
                                    definition before
    '''
    global _rules
    if store_as_default:
        rules = _rules
    else:
        rules = {}

    from os.path import exists
    for filename in config:
        if exists(filename):
            break
    else:
        raise RuntimeError('Config file "%s" does not exist!' % config)

    if environment.verbose:
        print 'reading config file "%s"' % filename

    cfg = SafeConfigParser(allow_no_value=True)
    cfg.optionxform = str
    cfg.read(filename)

    if not cfg.has_section('rules'):
        raise KeyError('Section "rules" is required in derived config file.')

    if delete_old:
        _rules.clear()

    for i,el in enumerate(gadget.elements):
        rules[el] = 'elements[:,%d]' % i
    for band in ['u','b','v','r','i','j','h','k']:
        rules['mag_'+band] = "calc_mags(stars,'%s')" % band
        rules['lum_'+band] = "UnitQty(10**(-0.4*(mag_%s-solar.abs_mag)),'Lsol')" % band

    rules.update( cfg.items('rules') )

    return rules

def calc_temps(s, gamma=5./3.):
    '''
    Calculate the block of temperatures form internal energy.

    This function calculates the temperatures from the internal energy using the
    ideal gas law:

        U = f/2 N k_b T

    TODO:
        What to do about ionisation states? What about the different elements? How
        do they affect the degrees of freedom f?

    Args:
        s (Snap):       The snapshot to calculate the temperatures for.

    Returns:
        T (UnitArr):    The temperatures for the particles of s.
    '''
    if s.properties['flg_entropy_instead_u']:
        raise NotImplementedError('Temperatures cannot be calculated from ' + 
                                  'entropy, yet. However, '
                                  'flg_entropy_instead_u is True.')

    from .. import physics

    # roughly the average weight for primordial gas
    av_particle_weight = (0.76*1. + 0.24*4.) * physics.m_u
    # roughly the average degrees of freedom for primordial gas (no molecules)
    f = 3

    # the internal energy in Gadget actually is the specific internal energy
    # TODO: check!
    T = s.u * s.mass / (f/2. * (s.mass/av_particle_weight) * physics.kB)
    T.convert_to('K', subs=s)
    return T

