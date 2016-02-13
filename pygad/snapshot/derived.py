'''
A module to prepare derived arrays for the snapshot.

Examples:
    >>> from ..environment import module_dir
    >>> rules = read_derived_rules([module_dir+'snapshot/derived.cfg'])
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

    >>> cosmo = physics.Planck2013()
    >>> age_from_form(UnitArr([0.001, 0.1, 0.5, 0.9], 'a_form'),
    ...               subs={'a':0.9, 'z':physics.a2z(0.9)},
    ...               cosmo=cosmo)
    UnitArr([ 12.30493079,  11.75687265,   6.44278561,   0.        ], units="Gyr")
    >>> age_from_form(UnitArr([10.0, 1.0, 0.5, 0.1], 'z_form'),
    ...               subs={'a':0.9, 'z':physics.a2z(0.9)},
    ...               cosmo=cosmo)
    UnitArr([ 11.82987008,   6.44278561,   3.70734788,  -0.13772577], units="Gyr")
    >>> age_from_form(UnitArr([-2.0, 0.0, 1.0], '(ckpc h_0**-1) / (km/s)'),
    ...               cosmic_time='2.1 Gyr',
    ...               subs={'a':0.9, 'z':physics.a2z(0.9), 'h_0':cosmo.h_0},
    ...               cosmo=cosmo,
    ...               units='Myr')
    UnitArr([ 4697.05769369,  2100.        ,   801.47115316], units="Myr")
'''
__all__ = ['ptypes_and_deps', 'read_derived_rules', 'calc_temps', 'age_from_form']

from ConfigParser import SafeConfigParser
import warnings
import numpy as np
from .. import utils
from .. import gadget
from .. import environment
import re
from .. import physics
from ..units import UnitArr, UnitScalar
from multiprocessing import Pool, cpu_count

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
        raise IOError('Config file "%s" does not exist!' % config)

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

    # roughly the average weight for primordial gas
    av_particle_weight = (0.76*1. + 0.24*4.) * physics.m_u
    # roughly the average degrees of freedom for primordial gas (no molecules)
    f = 3

    # the internal energy in Gadget actually is the specific internal energy
    # TODO: check!
    T = s.u * s.mass / (f/2. * (s.mass/av_particle_weight) * physics.kB)
    T.convert_to('K', subs=s)
    return T

"""
def _Gyr2z_vec(arr, cosmo):
    '''Needed to pickle cosmo.lookback_time_2z for Pool().apply_async.'''
    return np.vectorize(lambda t: cosmo.lookback_time_2z(t))(arr)
"""

def _z2Gyr_vec(arr, cosmo):
    '''Needed to pickle cosmo.lookback_time_in_Gyr for Pool().apply_async.'''
    return np.vectorize(cosmo.lookback_time_in_Gyr)(arr)


def age_from_form(form, subs, cosmic_time=None, cosmo=None, units='Gyr', parallel=None):
    '''
    Calculate ages from formation time.

    Args:
        form (UnitArr):     The formation times to convert. Has to be UnitArr with
                            appropiate units, i.e. '*_form' or a time unit.
        subs (dict, Snap):  Subsitution for unit convertions. See e.g.
                            `UnitArr.convert_to` for more information.
        cosmic_time (UnitScalar):
                            The current cosmic time. If None and subs is a
                            snapshot, it defaults to subs.time.
        cosmo (FLRWCosmo):  A cosmology to use for conversions. If None and subs
                            is a snapshot, the cosmology of that snapshot is used.
        units (str, Unit):  The units to return the ages in. If None, the return
                            value still has correct units, you just do not have
                            control over them.
        parallel (bool):    If units are converted from Gyr (or some other time
                            unit) to z_form / a_form, one can choose to use
                            multiple threads. By default, the function chooses
                            automatically whether to perform in parallel or not.

    Returns:
        ages (UnitArr):     The ages.
    '''
    from ..snapshot.snapshot import _Snap
    if subs is None:
        subs = {}
    elif isinstance(subs, _Snap):
        snap, subs = subs, {}
        subs['a'] = snap.scale_factor
        subs['z'] = snap.redshift
        subs['h_0'] = snap.cosmology.h_0
        if cosmo is None:
            cosmo = snap.cosmology
        if cosmic_time is None:
            cosmic_time = UnitScalar(snap.time, gadget.get_block_units('AGE '),
                                     subs=subs)

    form = form.copy().view(UnitArr)

    if str(form.units).endswith('_form]'):
        # (a ->) z -> Gyr (-> time_units)
        if form.units == 'a_form':
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                form.setfield(np.vectorize(physics.a2z)(form), dtype=form.dtype)
        form.units = 'z_form'

        if environment.allow_parallel_conversion and (
                parallel or (parallel is None and len(form) > 1000)):
            N_threads = cpu_count()
            chunk = [[i*len(form)/N_threads, (i+1)*len(form)/N_threads]
                        for i in xrange(N_threads)]
            p = Pool(N_threads)
            res = [None] * N_threads
            with warnings.catch_warnings():
                # warnings.catch_warnings doesn't work in parallel
                # environment...
                warnings.simplefilter("ignore") # for _z2Gyr_vec
                for i in xrange(N_threads):
                    res[i] = p.apply_async(_z2Gyr_vec,
                                (form[chunk[i][0]:chunk[i][1]], cosmo))
            for i in xrange(N_threads):
                form[chunk[i][0]:chunk[i][1]] = res[i].get()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # for _z2Gyr_vec
                form.setfield(_z2Gyr_vec(form,cosmo), dtype=form.dtype)
        form.units = 'Gyr'
        # from present day ages (lookback time) to actual current ages
        form -= cosmo.lookback_time(subs['z'])

    else:
        # 't_form' -> actual age
        cosmic_time = UnitScalar(cosmic_time, form.units, subs=subs)
        form = cosmic_time - form

    if units is not None:
        form.convert_to(units, subs=subs)

    return form
