'''
Definition of more complicated derived blocks. May be changed bu user.

A function for a derived block shall return the derived block with units as a
`UnitArr` and shall have an attribute `_deps` that is a set of the names of its
direct dependencies, i.e. the blocks it needs directly to calculate the derived
one from.
'''
__all__ = ['calc_temps', 'age_from_form', 'calc_x_ray_lum', 'calc_HI_mass',
           'calc_ion_mass', 'calc_cooling_rates', 'get_luminosities']

from .. import environment
from ..units import UnitArr, UnitScalar, UnitQty, UnitError
import numpy as np
from .. import physics
from .. import gadget
from .. import cloudy
from . import derived
from fractions import Fraction
from multiprocessing import Pool, cpu_count
import warnings
import gc
import sys


def calc_cooling_rates(s, tbl='CoolingTables/z_0.000.hdf5'):
    '''
    Calculate the total cooling rates for the given snapshot.
    '''
    tbl = physics.cooling.Wiersma_CoolingTable(tbl)
    Lambda = tbl.get_cooling(s, units='erg cm**3 s**-1')
    return Lambda


# TODO: add the elements individually (not just 'elements')
calc_cooling_rates._deps = set(['elements', 'mass', 'Z', 'temp', 'rho'])


def calc_temps(u, XH=0.76, ne=0.0, XZ=None, f=3, subs=None):
    '''
    Calculate the block of temperatures form internal energy.

    This function calculates the temperatures from the internal energy using the
    ideal gas law,
        U = f/2 N k_b T
    and calculating an average particle mass from XH, ne, and XZ.

    Args:
        u (UnitQty):            The (mass-)specific internal energies.
        XH (float, array-like): The Hydrogen *mass* fraction(s). It can be either
                                a constant value or one for each particle
                                (H/mass).
        ne (float, array-like): The number of electrons per atom(!).
        XZ (iterable):          The metal mass fraction(s) and their atomic weight
                                in atomic units. Each element of this iterable
                                shall be a tuple of mass fraction (which can be an
                                array with fractions for each particle) at first
                                position and atomic weight (a single float) at
                                second postion.
                                The Helium mass fractions is always the remaining
                                mass:  1.0 - XH - sum(XZ[:,0]).
        f (float):              The (effective) degrees of freedom.
        subs (dict, Snap):      Substitutions used for conversion into Kelvin.
                                (c.f. `UnitArr.convert_to`)

    Returns:
        T (UnitArr):            The temperatures for the particles (in K if
                                possible).
    '''
    u = UnitQty(u, 'km**2/s**2', subs=subs)
    XH = np.array(XH)
    ne = np.array(ne)

    tmp = XH / 1.008
    XHe = 1.0 - XH
    if XZ is not None:
        for X, m in XZ:
            tmp += X / float(m)
            XHe -= X
    tmp += XHe / 4.003

    # assuming `ne` is the mean number of electrons per atom:
    # av_m = physics.m_u / (tmp * (1.0 + ne))
    # as in Gadget (where XZ=None, though): `ne` are the electrons per Hydrogen
    # atom:
    av_m = physics.m_u / (tmp + ne * XH)

    # solving:  U = f/2 N k_b T
    # which is:  u = f/2 N/M k_b T
    T = u / (f / 2.) * av_m / physics.kB
    try:
        T.convert_to('K', subs=subs)
    except UnitError as ue:
        print('WARNING: in "calc_temps":\n%s' % ue, file=sys.stderr)
    gc.collect()
    return T


# (additional) dependencies
calc_temps._deps = set()


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
    from ..snapshot.snapshot import Snapshot
    if subs is None:
        subs = {}
    elif isinstance(subs, Snapshot):
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
            chunk = [[i * len(form) // N_threads, (i + 1) * len(form) // N_threads]
                     for i in range(N_threads)]
            p = Pool(N_threads)
            res = [None] * N_threads
            with warnings.catch_warnings():
                # warnings.catch_warnings doesn't work in parallel
                # environment...
                warnings.simplefilter("ignore")  # for _z2Gyr_vec
                for i in range(N_threads):
                    res[i] = p.apply_async(_z2Gyr_vec,
                                           (form[chunk[i][0]:chunk[i][1]], cosmo))
            for i in range(N_threads):
                form[chunk[i][0]:chunk[i][1]] = res[i].get()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # for _z2Gyr_vec
                form.setfield(_z2Gyr_vec(form, cosmo), dtype=form.dtype)
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


def calc_x_ray_lum(s, lumtable, **kwargs):
    '''
    Wrapping `x_ray_luminosity` for derived blocks.

    Args:
        s (Snap):           The snapshot to use.
        lumtable (str):     The filename of the XSPEC emission table to use.
        **kwargs:           Further arguments passed to `x_ray_luminosity`.

    Returns:
        lx (UnitArr):       X-ray luminosities of the gas particles.
    '''
    from ..analysis import x_ray_luminosity
    return x_ray_luminosity(s, lumtable=lumtable, **kwargs)


calc_x_ray_lum._deps = set(['metallicity', 'ne', 'H', 'rho', 'mass', 'temp'])


def calc_HI_mass(s, UVB=gadget.general['UVB'], flux_factor=None):
    '''
    Estimate the HI mass with the fitting formula from Rahmati et al. (2013).

    It just calls `cloudy.Rahmati_HI_mass`, but this function has `_deps`.

    Args:
        s (Snap):       The (gas-particles sub-)snapshot to use.
        UVB (str):      The name of the UV background as named in `cloudy.UVB`.
                        Defaults to the value of the UVB in the `gadget.cfg`.
        flux_factor (float):
                        Adjust the UVB by this factor (assume a optically thin
                        limit and scale down the densities during the look-up by
                        this factor).
                        (Note: `sigHI` used in the Rahmati fitting formula is not
                        adjusted!)

    Returns:
        HI (UnitArr):   The HI mass block for the gas (within `s`).
    '''
    return cloudy.Rahmati_HI_mass(s, UVB, flux_factor=flux_factor)


calc_HI_mass._deps = set(['H', 'temp', 'rho', 'mass'])


def calc_ion_mass(s, el, ionisation, selfshield=True, iontbl=None,
                  warn_outofbounds=True):
    '''
    Calculate the mass of the given ion from Cloudy tables.

    Args:
        s (Snap):           The (gas-particles sub-)snapshot to use.
        el (str):           The name of the ion-element. Needs also to be a block
                            name (e.g. as one of the 'element' block).
        ionisation (str):   The ionisation state. The concatenation of `el` and
                            this must result in an ion name in the Cloudy table.
        selfshield (bool):  Whether to account for self-shielding by assuming the
                            attenuation of the UVB as experienced by HI using the
                            Rahmati+ (2013) prescription (formula (14) of the
                            paper). Note that this is just a rough approximation!
        iontbl (IonisationTable):
                            The ionisation table to use for the table
                            interpolation. Default to the one given by
                            `derived.cfg` for the redshift of the (sub-)snapshot.
        warn_outofbounds (bool):
                            Whether to print out a warning if particles are
                            out of the bounds of the ionisation tables.

    Returns:
        ion_mass (UnitArr): The mass block of the ion (per particle) in units if
                            the block given by `el`.
    '''
    # if there is some ion table specified in the config, use it as default
    iontbl = cloudy.config_ion_table(s.redshift) if iontbl is None else iontbl
    f_ion = 10. ** iontbl.interp_snap(el + ' ' + ionisation, s.gas,
                                      selfshield=selfshield,
                                      warn_outofbounds=warn_outofbounds)
    return f_ion * s.gas.get(el)


# this is not super correct: typically the element is taken from the block
# 'elements', but could in principle also be defined seperately!
calc_ion_mass._deps = set(['H', 'mass', 'rho', 'temp', 'elements'])


def get_luminosities(stars, band='bolometric', IMF=None):
    '''
    Get the luminosities for a stellar sub-snapshot for a given band.

    This function calls `ssp.inter_bc_qty` and, hence, is using

    Args:
        stars (Snap):   The stellar snapshot to interpolate the luminosities for.
        band (str):     The band in which to interpolate the luminosities for.
                        Possible choices are:
                        'bolometric', 'U', 'B', 'V', 'R', and 'K',
        IMF (str):      The name of the IMF to use, if folder and base is not
                        given (undefined behaviour otherwise). By default the one
                        given in `gadget.cfg` is used. Values that are not in
                        `table_base_name.keys()` are invalid.

    Returns:
        lum (UnitArr):  The luminosities of the star particles.
    '''
    from ..ssp import inter_bc_qty
    from ..physics import solar

    if band == 'bolometric':
        qty = 'Mbol'
    else:
        qty = '%smag' % band.upper()

    mag_per_Msol = inter_bc_qty(stars['age'], stars['metallicity'], qty=qty,
                                units='mag', IMF=IMF)
    lum_per_Msol = UnitQty(10. ** (-0.4 * (mag_per_Msol - solar.abs_mag)), 'Lsol')
    return (stars['mass'] / UnitScalar('1 Msol')).in_units_of(1, subs=stars) * lum_per_Msol


get_luminosities._deps = set(['age', 'metallicity', 'mass'])

