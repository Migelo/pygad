'''
A module to prepare derived arrays for the snapshot.

Examples:
    >>> from ..environment import module_dir
    >>> rules = read_derived_rules([module_dir+'snapshot/derived.cfg'])
    reading config file "pygad/snapshot/derived.cfg"
    >>> assert rules == _rules
    >>> rules['r'], rules['metallicity']
    ('dist(pos)', 'metals/elements.sum(axis=1)')
    >>> from snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_470')
    >>> ptypes_and_deps(rules['r'], s)
    ([True, True, True, True, True, False], set(['pos']))
    >>> ptypes_and_deps(rules['metallicity'], s) # derived from mass (for all
    ...                                          # particles types) and elements
    ...                                          # (baryons only)
    ([True, False, False, False, True, False], set(['elements', 'metals']))

'''
__all__ = ['ptypes_and_deps', 'read_derived_rules', 'general']

from ConfigParser import SafeConfigParser
from .. import utils
from .. import gadget
from .. import environment
import re
import warnings
import derive_rules

_rules = {}
general = {
        'cache_derived': True,
        'always_cache': set(),
}

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
        elif hasattr(derive_rules, name):
            func = getattr(derive_rules, name)
            if not hasattr(func,'_deps'):
                warnings.warn('The derived block defining function ' +
                              '"%s" has not attribute `_deps` ' % name +
                              'defining its dependencies! -- Assume no ' +
                              'dependencies.')
            for dep in getattr(func,'_deps',set()):
                if dep in root._block_avail:
                    ptypes = [(pt and avail) for pt,avail
                                in zip(ptypes,root._block_avail[dep])]
                    deps.add(dep)
                else:
                    # all dependencies are needed -> we can shortcut
                    return [False]*6, set()
    return ptypes, deps

def read_derived_rules(config, store_as_default=True, delete_old=False):
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
    global _rules, general
    if store_as_default:
        rules = _rules
    else:
        rules = {}

    def test_section(cfg, section, entries):
        if not cfg.has_section(section):
            raise KeyError('Section "%s" is required in config ' % section +
                           'file for derived blocks.')
        if set(cfg.options(section)) < set(entries):
            raise ValueError('Section "%s" must have the ' % section +
                             'following entries: ' + str(entries))

    from os.path import exists, expanduser
    for filename in config:
        if exists(expanduser(filename)):
            break
    else:
        raise IOError('Config file "%s" does not exist!' % config)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'reading config file "%s"' % filename

    cfg = SafeConfigParser(allow_no_value=True)
    cfg.optionxform = str
    cfg.read(filename)

    test_section(cfg, 'general', ['cache_derived'])
    test_section(cfg, 'rules', [])

    general['cache_derived'] = cfg.getboolean('general', 'cache_derived')
    if cfg.has_option('general', 'always_cache'):
        general['always_cache'] = set(block.strip() for block
                in cfg.get('general','always_cache').split(','))

    if delete_old:
        _rules.clear()

    for i,el in enumerate(gadget.elements):
        rules[el] = 'elements[:,%d]' % i
    rules.update( cfg.items('rules') )

    for derived_name in rules.keys():
        if derived_name=='mag':
            mag, lum = 'mag', 'lum'
        elif re.match('mag_[a-zA-Z]', derived_name):
            mag, lum = derived_name, 'lum_'+derived_name[-1]
        else:
            continue
        rules[lum] = "UnitQty(10**(-0.4*(%s-solar.abs_mag)),'Lsol')" % mag

    return rules

