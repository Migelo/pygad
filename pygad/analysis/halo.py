'''
A collection of analysis functions that are somewhat connected to halo properties.

Examples:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_320', physical=True)
    >>> center = shrinking_sphere(s.stars, center=[s.boxsize/2]*3,
    ...                           R=s.boxsize*np.sqrt(3)) # doctest: +ELLIPSIS
    load block pos... done.
    convert block pos to physical units... done.
    do a shrinking sphere...
      starting values:
        center = ...
        R      = ...
    load block mass... done.
    convert block mass to physical units... done.
    done.
    >>> if np.linalg.norm( center - UnitArr([33816.9, 34601.1, 32681.0], 'kpc') ) > 1.0:
    ...     print center
    >>> R200, M200 = virial_info(s, center)
    >>> if abs(R200 - '177 kpc') > 3 or abs(M200 - '1e12 Msol') / '1e12 Msol' > 0.1:
    ...     print R200, M200
    >>> Translation(-center).apply(s)
    apply Translation to "pos" of "snap_M1196_4x_320"... done.
    >>> sub = s[s['r'] < 0.10*R200]
    derive block r... done.
    >>> if abs(half_mass_radius(sub.stars) - '4.3 kpc') > '0.1 kpc':
    ...     print half_mass_radius(sub.stars)
    >>> if abs(eff_radius(sub, 'V', proj=None) - '3.3 kpc') > '0.3 kpc':
    ...     print eff_radius(sub, 'V', proj=None)
    load block form_time... done.
    derive block age... done.
    load block elements... done.
    convert block elements to physical units... done.
    derive block H... done.
    derive block He... done.
    derive block metals... done.
    derive block Z... done.
    derive block mag_v... interpolate SSP tables for qty "Vmag"...
    read tables...
    table limits:
      age [yr]:    1.00e+05 - 2.00e+10
      metallicity: 1.00e-04 - 5.00e-02
    interpolate in age...
    interpolate in metallicity...
    done.
    derive block lum_v... done.
    >>> if abs(eff_radius(sub, 'V', proj=2) - '2.9 kpc') > '0.2 kpc':
    ...     print eff_radius(sub, 'V', proj=2)
    derive block rcyl... done.
    >>> if abs(half_qty_radius(sub.stars, qty='mass', proj=2) - '3.77 kpc') > '0.1 kpc':
    ...     print half_qty_radius(sub.stars, qty='mass', proj=2)
    >>> ifr, ofr = flow_rates(s, '50 kpc')
    load block vel... done.
    derive block vrad... done.
    >>> if abs(ifr - '421 Msol/yr') > '5 Msol/yr' or abs(ofr - '370 Msol/yr') > '5 Msol/yr':
    ...     print ifr, ofr
    >>> if abs(shell_flow_rates(s.gas, UnitArr([48,52],'kpc')) - '-7.1 Msol/yr') > '0.1 Msol/yr':
    ...     print shell_flow_rates(s.gas, UnitArr([48,52],'kpc'))
    >>> if abs(shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), 'in') - '-14.7 Msol/yr') > '0.2 Msol/yr':
    ...     print shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), 'in')
    >>> if abs(shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), 'out') - '7.5 Msol/yr') > '0.1 Msol/yr':
    ...     print shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), 'out')
    >>> ifr, ofr = flow_rates(s.gas, '50 kpc')

    >>> if abs(ifr - '11.0 Msol/yr') > '1.0 Msol/yr' or abs(ofr - '7.1 Msol/yr') > '0.1 Msol/yr':
    ...     print ifr, ofr
    >>> eta = ofr / s.gas['sfr'].sum()
    load block sfr... done.
    >>> if abs(eta - 1.546) > 0.01:
    ...     print 'mass loading:', eta

    >>> FoF, N_FoF = find_FoFs(s.highres.dm, '2.5 kpc')
    perfrom a FoF search (l = 2.5 [kpc], N >= 50)...
    found 201 groups
    the 3 most massive ones are:
      group 0:   3.57e+11 [Msol]  @  [1.02, -1.02, 1.1] [kpc]
      group 1:   8.05e+10 [Msol]  @  [491, 941, 428] [kpc]
      group 2:   2.52e+10 [Msol]  @  [872, 1.36e+03, 729] [kpc]
    >>> halo0 = s.highres.dm[FoF==0]
    >>> if abs(halo0['mass'].sum() - '3.57e11 Msol') > '0.02e11 Msol':
    ...     print halo0['mass'].sum()
    >>> FoF, N_FoF = find_FoFs(s.baryons, '2.5 kpc')
    perfrom a FoF search (l = 2.5 [kpc], N >= 50)...
    found 15 groups
    the 3 most massive ones are:
      group 0:   3.93e+10 [Msol]  @  [-0.116, 0.55, 0.0737] [kpc]
      group 1:   7.14e+09 [Msol]  @  [492, 940, 428] [kpc]
      group 2:   4.83e+09 [Msol]  @  [871, 1.36e+03, 729] [kpc]
    >>> gal0 = s.baryons[FoF==0]
    >>> if abs(gal0['mass'].sum() - '3.93e10 Msol') > '0.02e10 Msol':
    ...     print gal0['mass'].sum()
'''
__all__ = ['shrinking_sphere', 'virial_info', 'half_qty_radius',
           'half_mass_radius', 'eff_radius', 'shell_flow_rates', 'flow_rates',
           'find_FoFs']

import numpy as np
from .. import utils
from ..units import *
import sys
from ..transformation import *
from .. import gadget
from snap_props import *
from ..snapshot import *
from .. import environment
from .. import C

def shrinking_sphere(s, center, R, periodic=True, shrink_factor=0.93,
                     stop_N=10):
    '''
    Find the densest point by shrinking sphere technique.

    Cf. TODO: cite!!!

    Args:
        s (Snap):               The (sub-)snapshot to find the densest point for.
        center (array-like):    The center to start with.
        R (float, UnitArr, str):The initial radius.
        periodic (bool):        Whether to assume a periodic box (with the
                                sidelength of snap.props['boxsize']) or not.
        shrink_factor (float):  The factor to shrink the sphere in each step.
        stop_N (int):           If so many or less particles are left in the
                                sphere, stop.

    Returns:
        center (UnitArr):       The center.
    '''
    center0 = UnitQty(center,s['pos'].units,subs=s,dtype=np.float64)
    R = UnitScalar(R,s['pos'].units,subs=s)

    if environment.verbose:
        print 'do a shrinking sphere...'
        print '  starting values:'
        print '    center = %s' % center0
        print '    R      = %s' % R
        sys.stdout.flush()

    if not 0 < shrink_factor < 1:
        raise ValueError('"shrink_factor" must be in the interval (0,1)!')
    if not 0 < stop_N:
        raise ValueError('"stop_N" must be positive!')

    pos  = s['pos'].astype(np.float64)
    mass = s['mass'].astype(np.float64)
    assert len(pos) == len(mass)
    boxsize = float( s.boxsize.in_units_of(s['pos'].units) )

    # needed since C does not know about stridings
    if pos.base is not None:
        pos  = pos.copy()
        mass = mass.copy()

    center = np.empty((3,), dtype=np.float64)
    if periodic:
        C.cpygad.shrinking_sphere_periodic(
                                  C.c_void_p(center.ctypes.data),
                                  C.c_size_t(len(pos)),
                                  C.c_void_p(pos.ctypes.data),
                                  C.c_void_p(mass.ctypes.data),
                                  C.c_void_p(center0.ctypes.data), C.c_double(R),
                                  C.c_double(shrink_factor), C.c_size_t(stop_N),
                                  C.c_double(boxsize))
    else:
        C.cpygad.shrinking_sphere_nonperiodic(
                                  C.c_void_p(center.ctypes.data),
                                  C.c_size_t(len(pos)),
                                  C.c_void_p(pos.ctypes.data),
                                  C.c_void_p(mass.ctypes.data),
                                  C.c_void_p(center0.ctypes.data), C.c_double(R),
                                  C.c_double(shrink_factor), C.c_size_t(stop_N))
    center = center.view(UnitArr)
    center.units = s['pos'].units

    if environment.verbose:
        print 'done.'
        sys.stdout.flush()

    return center

def virial_info(s, center=None, odens=200.0, N_min=10):
    '''
    Return the virial radius (R_odens) and the virial mass (M_odens) of the
    structure at 'center'.
    
    The virial radius is the radius, where the average density within that radius
    is <odens> times the critical density of the universe. If the inner <N_min>
    particles do not exceed a density of <odens> times the critical density,
    R_<odens> and M_<odens> are both returned as zero.

    Args:
        s (Snap):               The (sub-)snapshot to use.
        center (array-like):    The center of the structure to calculate the
                                properties for. (default: (0,0,0))
        odens (float):          Overdensity parameter (rho/rho_crit) used as
                                threshold for calculated radius and enclosed
                                mass to be returned.
        N_min (int):            The minimum number of particles required to form
                                the halo (have a density higher than
                                odens * rho_crit). If not reached, the return
                                values are NaN.

    Returns:
        R_odens, M_odens (both UnitArr):
                                The virial radius & mass (or corresponding
                                R_odens, M_odens)
    '''
    if center is None:
        center = [0,0,0]
    center = UnitQty(center,s['pos'].units,subs=s).view(np.ndarray)

    rho_crit = s.cosmology.rho_crit(z=s.redshift)
    rho_crit = rho_crit.in_units_of(s['mass'].units/s['pos'].units**3, subs=s)

    # make use of the potentially precalculated derived block r
    if np.all(center==0):
        r = s['r']
    else:
        r = dist(s['pos'], center)

    mass = s['mass'].astype(np.float64)
    r    = r.astype(np.float64)
    if mass.base is not None:
        mass = mass.copy()
        r    = r.copy()
    info = np.empty((2,), dtype=np.float64)
    C.cpygad.virial_info(C.c_size_t(len(r)),
                         C.c_void_p(mass.ctypes.data),
                         C.c_void_p(r.ctypes.data),
                         C.c_double(odens*rho_crit),
                         C.c_size_t(N_min),
                         C.c_void_p(info.ctypes.data),
    )
    if info[0] == 0.0:
        info[:] = np.nan
    return UnitArr(info[0],s['pos'].units), \
           UnitArr(info[1],s['mass'].units)

def half_qty_radius(s, qty, Qtot=None, center=None, proj=None):
    '''
    Calculate the radius at which half of a quantity is confined in.

    Args:
        s (Snap):               The (sub-)snapshot to use.
        qty (str, UnitQty):     The quantity to calculate the half-X-radius for.
        Qtot (float, UnitArr):  The total amount of the quantity. If not given,
                                the total quantity of the (sub-)snapshot is used.
        center (array-like):    The center of the structure to calculate the
                                properties for. (default: [0,0,0])
        proj (int):             If set, do the calculation for the projection
                                along the specified axis (0=x, 1=y, 2=z).

    Returns:
        r12 (UnitArr):          The half-X-radius.
    '''
    if len(s) == 0:
        # leads to exceptions otherwise
        return UnitArr(0.0, s['pos'].units)

    if center is None:
        center = UnitQty([0]*3)
    else:
        center = UnitQty(center)
    center = center.in_units_of(s['pos'].units,subs=s)

    if isinstance(proj,int):
        proj_mask = tuple([i for i in xrange(3) if i!=proj])
        if len(center)==3:
            center = center[(proj_mask,)]

    if np.all(center==0):
        if isinstance(proj,int):
            r = s['rcyl'] if proj==2 else dist(s['pos'][:,proj_mask])
        else:
            r = s['r']
    else:
        r = dist(s['pos'][:,proj_mask],center) if isinstance(proj,int) \
                else dist(s['pos'],center)
    r_ind = r.argsort()

    if isinstance(qty,str):
        qty = s.get(qty)
    else:
        qty = UnitQty(qty)

    Q = np.cumsum(qty[r_ind])
    if Qtot is None:
        Qtot = Q[-1]
    else:
        Qtot = UnitScalar(Qtot,qty.units,subs=s,dtype=float)

    Q_half_ind = np.abs(Q - Qtot/2.).argmin()
    if Q_half_ind == len(Q)-1:
        print >> sys.stderr, 'WARNING: The half-qty radius is larger than ' + \
                             'the (sub-)snapshot passed!'
    elif Q_half_ind < 10:
        print >> sys.stderr, 'WARNING: The half-qty radius is not resolved ' + \
                             'for %s!' % s
    return UnitArr(r[r_ind][Q_half_ind], s['pos'].units)

def half_mass_radius(s, M=None, center=None, proj=None):
    '''
    Calculate the (by default 3D) half-mass-radius of the structure at the center.

    For more details see analysis.half_qty_radius.

    Args:
        s (Snap):               The (sub-)snapshot to use.
        M (float, UnitArr):     The total mass.
        center (array-like):    The center of the structure. (default: [0,0,0])
        proj (int):             If set, do the calculation for the projection
                                along the specified axis (0=x, 1=y, 2=z).

    Returns:
        r12 (UnitArr):          The half-mass-radius.
    '''
    return half_qty_radius(s, qty='mass', Qtot=M, center=center)

def eff_radius(s, band=None, L=None, center=None, proj=2):
    '''
    Calculate the (by default 2D) half-light-radius of the structure at the
    center.

    For more details see analysis.half_qty_radius.

    Args:
        s (Snap):               The (sub-)snapshot to use.
        band (str):             The band in which the effective radius is
                                calculated. If it is None or 'bol', the bolometric
                                luminosity is taken.
        L (float, UnitArr):     The total luminosity.
        center (array-like):    The center of the structure. (default: [0,0,0])
        proj (int):             The axis to project along.

    Returns:
        r12 (UnitArr):          The half-light-radius.
    '''
    qty = 'lum'
    if band is not None and band != 'bol':
        qty += '_' + band.lower()
    return half_qty_radius(s.stars, qty=qty, Qtot=L, center=center, proj=proj)

def shell_flow_rates(s, Rlim, direction='both', units='Msol/yr'):
    '''
    Estimate flow rate in spherical shell.
    
    The estimation is done by caluculating m*v/d on a particle base, where d is
    the thickness of the shell.

    Args:
        s (Snap):               The (sub-)snapshot to use.
        Rlim (UnitQty):         The inner and outer radius of the shell.
        direction ('in', 'out', 'both'):
                                The direction to take into account. If 'in', only
                                take onflowing gas particles into the calculation;
                                analog for 'out'; and do not restrict to particles
                                with 'both'.
        units (Unit, str):      The units in which to return the flow rate.

    Returns:
        flow (UnitArr):         The estimated flow rate.
    '''
    Rlim = UnitQty(Rlim, s['r'].units)
    if not Rlim.shape == (2,):
        raise ValueError('Rlim must have shape (2,)!')
    shell = s[(Rlim[0]<s['r']) & (s['r']<Rlim[1])]

    if direction=='both':
        pass
    elif direction=='in':
        shell = shell[shell['vrad'] < 0]
    elif direction=='out':
        shell = shell[shell['vrad'] > 0]
    else:
        raise RuntimeError('unknown direction %s!' % direction)

    flow = np.sum(shell['mass'] * shell['vrad'] / UnitArr(Rlim[1]-Rlim[0], Rlim.units))
    flow.convert_to(units)
    return flow

def flow_rates(s, R, dt='3 Myr'):
    #TODO: extend to discs!
    '''
    Estimate in- and outflow rates through a given radius.
    
    The estimation is done by propagating the positions with constant current
    velocities, i.e. pos_new = pos_old + vel*dt. Then counting the mass that
    passed the shell of radius R.

    Args:
        s (Snap):               The (sub-)snapshot to use.
        R (UnitScalar):         The radius of the shell through which the flow is
                                estimated.
        dt (UnitScalar):        The time used for the linear extrapolation of the
                                current positions (if it is a plain number without
                                units, they are assumed to be 'Myr').

    Returns:
        ifr (UnitArr):          The estimated inflow rate.
        ofr (UnitArr):          The estimated outflow rate.
    '''
    R = UnitScalar(R, units=s['pos'].units, subs=s)
    dt = UnitScalar(dt, units='Myr', subs=s)

    # predicted particle distances from center in dt
    dt.convert_to(s['r'].units/s['vel'].units,subs=s)   # avoid conversions of
                                                        # entire arrays
    rpred = s['r'] + s['vrad']*dt
    of_mass = s['mass'][(s['r'] < R) & (rpred >= R)]
    if_mass = s['mass'][(s['r'] >= R) & (rpred < R)]

    dt.convert_to('yr',subs=s)  # to more intuitive units again
    ofr = np.sum(of_mass) / dt
    ifr = np.sum(if_mass) / dt

    return ifr, ofr

def find_FoFs(s, l, min_parts=50, sort=True):
    '''
    Perform a friends-of-friends search on a (sub-)snapshot.

    Args:
        s (Snap):           The (sub-)snapshot to perform the FoF finder on.
        l (UnitScalar):     The linking length to use for the FoF finder.
        min_parts (int):    The minimum number of particles in a FoF group to
                            actually define it as such.
        sort (bool):        Whether to sort the groups by mass. If True, the group
                            with ID 0 will be the most massive one and the in
                            descending order.

    Returns:
        FoF (np.ndarray):   A block of FoF group IDs for the particles of `s`.
        N_FoF (int):        The number of FoF groups found.
    '''
    l = UnitScalar(l, s['pos'].units, subs=s)
    sort = bool(sort)
    min_parts = int(min_parts)

    if environment.verbose:
        print 'perfrom a FoF search (l = %.2g %s, N >= %d)...' % (
                l, l.units, min_parts)
        sys.stdout.flush()

    pos = s['pos'].astype(np.float64)
    mass = s['mass'].astype(np.float64)
    if pos.base is not None:
        pos = pos.copy()
    if mass.base is not None:
        mass = mass.copy()
    FoF = np.empty(len(s), dtype=np.uintp)
    boxsize = float(s.boxsize.in_units_of(s['pos'].units))

    C.cpygad.find_fof_groups(C.c_size_t(len(s)),
                             C.c_void_p(pos.ctypes.data),
                             C.c_void_p(mass.ctypes.data),
                             C.c_double(l),
                             C.c_size_t(min_parts),
                             C.c_int(int(sort)),
                             C.c_void_p(FoF.ctypes.data),
                             C.c_double(boxsize),
                             None,  # build new tree
    )

    # do not count the particles with no halo!
    N_FoF = len(set(FoF)) - 1

    if environment.verbose:
        print 'found %d groups' % N_FoF
        N_list = min(N_FoF, 3)
        if N_list:
            if N_list==1:
                print 'the most massive one is:'
            else:
                print 'the %d most massive ones are:' % N_list
            for i in xrange(N_list):
                FoF_group = s[FoF==i]
                com = center_of_mass(FoF_group)
                M = FoF_group['mass'].sum()
                print '  group %d:   %8.3g %s  @  [%.3g, %.3g, %.3g] %s' % (
                        i, M, M.units, com[0], com[1], com[2], com.units)
        sys.stdout.flush()

    return FoF, N_FoF

