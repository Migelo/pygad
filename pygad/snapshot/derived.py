'''
A module to prepare derived arrays for the snapshot.

Examples:
    >>> from ..environment import module_dir
    >>> rules = read_derived_rules([module_dir+'snapshot/derived.cfg'])
    reading config file "pygad/snapshot/derived.cfg"
    >>> assert rules == _rules
    >>> rules['r'], rules['Z']
    ('dist(pos)', 'metals/elements.sum(axis=1)')
    >>> general
    {'always_cache': set(['Ekin', 'temp', 'age', 'mag*', 'angmom', 'LX', 'jcirc']), 'cache_derived': True}
    >>> iontable
    {'ions': ['H I', 'He I', 'He II', 'C II', 'C III', 'C IV', 'N V', 'O VI', 'Mg II', 'Si II', 'Si III', 'Si IV', 'P IV', 'S VI'], 'selfshield': True, 'pattern': 'lt<z>f10', 'tabledir': 'pygad//../iontbls/', 'flux_factor': 1.0, 'T_vals': [2.5, 0.05, 150.0], 'nH_vals': [-8.0, 0.05, 160.0]}

    >>> from snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_470')
    >>> ptypes_and_deps(rules['r'], s)
    ([True, True, True, True, True, False], set(['pos']))
    >>> ptypes_and_deps(rules['Z'], s)  # derived from mass (for all particles
    ...                                 # types) and elements (baryons only)
    ([True, False, False, False, True, False], set(['elements', 'metals']))
    >>> s.gas['CIV'] # doctest: +ELLIPSIS
    load block elements... done.
    derive block H... done.
    load block mass... done.
    load block rho... done.
    load block u... done.
    load block ne... done.
    derive block temp... done.
    derive block CIV... load tables:
      "pygad//../iontbls/lt00f10" (z=0.000)
      "pygad//../iontbls/lt01f10" (z=0.100)
    derive block C... done.
    done.
    SimArr([...],
           units="1e+10 Msol h_0**-1", snap="snap_M1196_4x_470":gas)

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
iontable = {
        'tabledir':     None,
        'pattern':      'lt<z>f10',
        'ions':         [],
        'nH_vals':      [-8  , 0.05, 160],
        'T_vals':       [ 2.5, 0.05, 140],
        'selfshield':   True,
        'flux_factor':  1.0
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

def read_derived_rules(config, delete_old=False):
    '''
    Read rules for derived blocks from a config file.

    The config file consists of a single section 'rules' that has names and their
    corresponding rules (Python expressions, that have the namespace of the
    snapshot and numpy -- see Snap.get for more information).

    Args:
        config (list):              list of possible filenames for the config file.
        delete_old (bool):          Delete old definition before add the new ones.
    '''
    global _rules, general, iontable

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
        general.clear()
        general['cache_derived'] = True
        general['always_cache'] = set()
        iontable.clear()
        iontable['tabledir'] = None
        iontable['pattern'] = 'lt<z>f10'
        iontable['ions'] = []
        iontable['nH_vals'] = [-8  , 0.05, 160]
        iontable['T_vals']  = [ 2.5, 0.05, 140]
        iontable['selfshield'] = True
        iontable['flux_factor'] = 1.0

    if cfg.has_section('iontable'):
        test_section(cfg, 'iontable', ['tabledir', 'ions', 'nH_vals', 'T_vals'])
        iontable['tabledir'] = cfg.get('iontable', 'tabledir',
                                       vars={'PYGAD_DIR':environment.module_dir})
        if cfg.has_option('iontable', 'pattern'):
            iontable['pattern'] = cfg.get('iontable', 'pattern',
                                          vars={'PYGAD_DIR':environment.module_dir})
        iontable['ions'] = [ion.strip()
                for ion in cfg.get('iontable','ions').split(',')]
        for vals in ['nH_vals', 'T_vals']:
            iontable[vals] = [float(v.strip())
                    for v in cfg.get('iontable',vals).split(',')]
        if cfg.has_option('iontable', 'selfshield'):
            iontable['selfshield'] = cfg.getboolean('iontable', 'selfshield')
        if cfg.has_option('iontable', 'flux_factor'):
            iontable['flux_factor'] = cfg.getfloat('iontable', 'flux_factor')

    for i,el in enumerate(gadget.elements):
        _rules[el] = 'elements[:,%d]' % i
    _rules.update( cfg.items('rules') )

    for derived_name in _rules.keys():
        if derived_name=='mag':
            mag, lum = 'mag', 'lum'
        elif re.match('mag_[a-zA-Z]', derived_name):
            mag, lum = derived_name, 'lum_'+derived_name[-1]
        else:
            continue
        _rules[lum] = "UnitQty(10**(-0.4*(%s-solar.abs_mag)),'Lsol')" % mag

    # add the ions from the Cloudy table as derived blocks
    for ion in iontable['ions']:
        el, ionisation = ion.split()
        ion = el + ionisation   # getting rid of the white space
        if ion in _rules:
            continue
        _rules[ion] = "calc_ion_mass(gas, '%s', '%s', selfshield=%s)" % (
                                el, ionisation, iontable['selfshield'])

    return _rules

