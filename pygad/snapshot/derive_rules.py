'''
Definition of more complicated derived blocks. May be changed bu user.

A function for a derived block shall return the derived block with units as a
`UnitArr` and shall have an attribute `_deps` that is a set of the names of its
direct dependencies, i.e. the blocks it needs directly to calculate the derived
one from.
'''
__all__ = ['calc_temps', 'age_from_form']

from .. import environment
from ..units import UnitArr, UnitScalar
import numpy as np
from .. import physics
from .. import gadget
from multiprocessing import Pool, cpu_count
import warnings

def calc_temps(gas, gamma=5./3.):
    '''
    Calculate the block of temperatures form internal energy.

    This function calculates the temperatures from the internal energy using the
    ideal gas law:

        U = f/2 N k_b T

    TODO:
        What to do about ionisation states? What about the different elements? How
        do they affect the degrees of freedom f?

    Args:
        gas (Snap):     The (sub-)snapshot to calculate the temperature for.

    Returns:
        T (UnitArr):    The temperatures for the particles of `gas` (in K).
    '''
    if gas.properties['flg_entropy_instead_u']:
        raise NotImplementedError('Calculating temperatures from entropy is ' +
                                  'not implemented, yet, but ' +
                                  '"flg_entropy_instead_u" is set.')

    # roughly the average weight for primordial gas
    av_particle_weight = (0.76*1. + 0.24*4.) * physics.m_u
    # roughly the average degrees of freedom for primordial gas (no molecules)
    f = 3

    # the internal energy in Gadget actually is the specific internal energy
    # TODO: check!
    T = gas['u'] * gas['mass'] / (f/2. * (gas['mass']/av_particle_weight) * physics.kB)
    T.convert_to('K', subs=gas)
    return T
calc_temps._deps = set(['u', 'mass'])

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

    Examples:
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
age_from_form._deps = set(['form_time'])

