'''
A collection of analysis functions that are somewhat connected to halo properties.

Examples:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_320', physical=True)
    >>> center = shrinking_sphere(s.stars, center=[s.boxsize/2]*3,
    ...                           R=s.boxsize*np.sqrt(3))
    load block pos... done.
    convert block pos to physical units... done.
    do a shrinking sphere...
      starting values:
        center = [ 34999.99976045  34999.99976045  34999.99976045] [kpc]
        R      = 1.212436e+05 [kpc]
    load block mass... done.
    convert block mass to physical units... done.
    done.
    >>> center
    UnitArr([ 33816.9017345 ,  34601.11199658,  32681.01209451], units="kpc")
    >>> R200, M200 = virial_info(s, center)
    >>> print R200, M200
    177.000365334 [kpc] 1.001023e+12 [Msol]
    >>> Translation(-center).apply(s)
    apply Translation to "pos" of "snap_M1196_4x_320"... done.
    >>> sub = s[s['r'] < 0.10*R200]
    derive block r... done.
    >>> half_mass_radius(sub.stars)
    UnitArr(4.31200747373, units="kpc")
    >>> eff_radius(sub, 'V', proj=None)
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
    UnitArr(3.27053620882, units="kpc")
    >>> eff_radius(sub, 'V', proj=2)
    derive block rcyl... done.
    UnitArr(2.92146861909, units="kpc")
    >>> half_qty_radius(sub.stars, qty='mass', proj=2)
    UnitArr(3.7688323337, units="kpc")
    >>> print map(str,flow_rates(s, '50 kpc'))
    load block vel... done.
    derive block vrad... done.
    ['419.673898667 [Msol yr**-1]', '371.26784 [Msol yr**-1]']
    >>> shell_flow_rates(s.gas, UnitArr([48,52],'kpc'))
    UnitArr(-7.097851e+00, units="Msol yr**-1")
    >>> shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), 'in')
    UnitArr(-1.463570e+01, units="Msol yr**-1")
    >>> shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), 'out')
    UnitArr(7.53784771595, units="Msol yr**-1")
    >>> ifr, ofr = flow_rates(s.gas, '50 kpc')
    >>> print ifr
    11.5465066667 [Msol yr**-1]
    >>> print ofr
    7.09649333333 [Msol yr**-1]
    >>> eta = ofr / s.gas['sfr'].sum()
    load block sfr... done.
    >>> print 'mass loading:', eta
    mass loading: 1.54604865966
'''
__all__ = ['shrinking_sphere', 'virial_info', 'half_qty_radius',
           'half_mass_radius', 'eff_radius', 'shell_flow_rates', 'flow_rates']

import numpy as np
from .. import utils
from ..units import *
import sys
from ..transformation import *
from .. import gadget
from snap_props import *
from ..octree import *
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
    boxsize = s.boxsize.in_units_of(s['pos'].units)

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

def virial_info(s, center=None, odens=200.0, N_min=50):
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
                                odens * rho_crit).

    Returns:
        R_odens, M_odens (both UnitArr):
                                The virial radius & mass (or corresponding
                                R_odens, M_odens)

    Raises:
        RuntimeError:           If the average density of the entire
                                (sub-)snapshot is larger than <odens> times the
                                critical density.
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
    r_ind = r.argsort()
    M_enc = np.cumsum(s['mass'][r_ind])[utils.perm_inv(r_ind)]
    rho = M_enc / ((4.0 / 3.0 * np.pi) * r**3)

    if rho[r_ind[N_min]] < odens*rho_crit:
        print >> sys.stderr, 'WARNING: the mean density within the inner ' + \
                             '%d particles (' % N_min + \
                             '%.3g' % (rho[r_ind[N_min]]/rho_crit) + \
                             '*rho_crit) does not exceed %g*rho_crit ' % odens + \
                             '(rho_crit = %s)!' % UnitArr(rho_crit,
                                     (s['mass'].units/s['pos'].units**3).gather())
        return UnitArr(0.0,s['pos'].units), UnitArr(0.0,s['mass'].units)
    if rho[r_ind[-1]] > odens*rho_crit:
        raise RuntimeError('The mean density within all particles is larger ' + \
                           'than %d rho_crit (%gx)!' % (odens,
                                                        rho[r_ind[-1]]/rho_crit))

    ind_odens = np.abs(rho - odens*rho_crit).argmin()
    return UnitArr(r[ind_odens],s['pos'].units), \
           UnitArr(M_enc[ind_odens],s['mass'].units)

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

