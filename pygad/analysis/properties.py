'''
A collection of (sub-)snapshot wide analysis functions.

Example:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snapshot
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_320', physical=False)
    >>> if np.linalg.norm(mass_weighted_mean(s,'pos') - center_of_mass(s)) > 1e-3:
    ...     print(mass_weighted_mean(s,'pos'))
    ...     print(center_of_mass(s,))
    load block pos... done.
    load block mass... done.

    Center of mass is *not* center of galaxy / halo!
    >>> Translation([-34792.2, -35584.8, -33617.9]).apply(s)
    apply Translation to "pos" of "snap_M1196_4x_320"... done.
    >>> s.to_physical_units()
    >>> sub = s[s['r'] < '20 kpc']
    derive block r... done.
    >>> orientate_at(sub, 'L', total=True)
    load block vel... done.
    derive block momentum... done.
    derive block angmom... done.
    apply Rotation to "pos" of "snap_M1196_4x_320"... done.
    apply Rotation to "vel" of "snap_M1196_4x_320"... done.
    >>> if np.linalg.norm(sub['angmom'].sum(axis=0) -
    ...         UnitArr([3.18e+08,-5.05e+07,1.24e+14],'kpc Msol km/s')) > 1e12:
    ...     print(sub['angmom'].sum(axis=0))
    derive block momentum... done.
    derive block angmom... done.
    >>> redI = reduced_inertia_tensor(sub.baryons)
    derive block r... done.
    >>> np.linalg.norm(redI - np.matrix([[ 2.54e10, -3.28e9, 2.74e9],
    ...                                  [-3.28e9,   2.75e9, 5.16e8],
    ...                                  [ 2.74e9,   5.16e8, 5.84e9]])) < 1e10
    True
    >>> orientate_at(s, 'red I', redI)
    apply Rotation to "pos" of "snap_M1196_4x_320"... done.
    apply Rotation to "vel" of "snap_M1196_4x_320"... done.
    >>> redI = reduced_inertia_tensor(sub.baryons)
    derive block r... done.
    >>> np.linalg.norm(redI - np.matrix([[ 2.62e10,  0.00   , 0.00   ],
    ...                                  [ 0.00   ,  5.73e+9, 0.00   ],
    ...                                  [ 0.00   ,  0.00   , 2.05e9]])) < 1e8
    True
    >>> if abs( los_velocity_dispersion(sub) - '170 km/s' ) > '5 km/s':
    ...     print(round(float(los_velocity_dispersion(sub)),4))
    119.8176
    >>> if abs(half_mass_radius(sub.stars) - '11 kpc') > '0.8 kpc':
    ...     print(half_mass_radius(sub.stars))
    >>> if abs(eff_radius(sub, 'V', proj=None) - '11 kpc') > '0.8 kpc':
    ...     print(eff_radius(sub, 'V', proj=None))
    load block form_time... done.
    derive block age... done.
    load block Z... done.
    derive block elements... done.
    derive block H... done.
    derive block He... done.
    derive block metals... done.
    derive block metallicity... done.
    derive block lum_v... done.
    >>> if abs(eff_radius(sub, 'V', proj=2) - '10.7 kpc') > '0.2 kpc':
    ...     print(eff_radius(sub, 'V', proj=2))
    derive block rcyl... done.
    >>> if abs(half_mass_radius(sub.stars, proj=2) - '10.8 kpc') > '0.2 kpc':
    ...     print(half_mass_radius(sub.stars))
    >>> ifr, ofr = flow_rates(s, '50 kpc')
    derive block vrad... done.
    >>> if abs(ifr - '355 Msol/yr') > '10 Msol/yr' or abs(ofr - '337 Msol/yr') > '10 Msol/yr':
    ...     print(ifr, ofr)
    >>> if abs(shell_flow_rates(s.gas, UnitArr([48,52],'kpc')) - '-4.6 Msol/yr') > '0.2 Msol/yr':
    ...     print(shell_flow_rates(s.gas, UnitArr([48,52],'kpc')))
    >>> if abs(shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), qty='metals') - '-0.014 Msol/yr') > '0.002 Msol/yr':
    ...     print(shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), qty='metals'))
    >>> if abs(shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), direction='in') - '-12.2 Msol/yr') > '0.2 Msol/yr':
    ...     print(shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), direction='in'))
    >>> if abs(shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), direction='out') - '7.6 Msol/yr') > '0.2 Msol/yr':
    ...     print(shell_flow_rates(s.gas, UnitArr([48,52],'kpc'), direction='out'))
    >>> ifr, ofr = flow_rates(s.gas, '50 kpc')

    >>> if abs(ifr - '13.0 Msol/yr') > '1.0 Msol/yr' or abs(ofr - '9.1 Msol/yr') > '0.1 Msol/yr':
    ...     print(ifr, ofr)
    >>> eta = ofr / s.gas['sfr'].sum()
    load block sfr... done.
    >>> if abs(eta - 2.0) > 0.2:
    ...     print('mass loading:', eta)

    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_470', load_double_prec=True)
    >>> s.gas['lx'] = x_ray_luminosity(s, lumtable=module_dir+'snaps/em.dat')
    load block Z... done.
    derive block elements... done.
    derive block H... done.
    derive block He... done.
    derive block metals... done.
    derive block metallicity... done.
    load block ne... done.
    load block rho... done.
    load block mass... done.
    load block u... done.
    derive block temp... done.
    >>> s.gas['lx'].units
    Unit("erg s**-1")
    >>> if abs(np.mean(s.gas['lx']) - '1.68e33 erg/s') > '0.05e33 erg/s':
    ...     print(np.mean(s.gas['lx']))
    >>> if abs(s.gas['lx'].sum() - '1.55e39 erg/s') > '0.05e39 erg/s':
    ...     print(s.gas['lx'].sum())
'''
__all__ = ['mass_weighted_mean', 'center_of_mass', 'reduced_inertia_tensor',
           'orientate_at', 'half_qty_radius', 'half_mass_radius', 'eff_radius',
           'shell_flow_rates', 'flow_rates', 'los_velocity_dispersion',
           'x_ray_luminosity']

import numpy as np
from ..units import *
from ..utils import dist
from ..transformation import *
from pygad import physics
import sys


def mass_weighted_mean(s, qty, mass='mass'):
    '''
    Calculate the mass weighted mean of some quantity, i.e.
    sum(mass[i]*qty[i])/sum(mass).

    Args:
        s (Snap):           The (sub-)snapshot the quantity belongs to (the masses
                            are taken from it, too).
        qty (str, UnitArr): The quantity to average. It can either be the block
                            itself (make shure it already has the appropiate
                            shape), or a expression that can be passed to Snap.get
                            (e.g. simply a name of a block).
        mass (str, UnitArr):The mass block.

    Returns:
        mean (UnitArr):     The mass weighted mean.
    '''
    if isinstance(qty, str):
        qty = s.get(qty)
    else:
        qty = UnitArr(qty)
    if len(s) == 0:
        return UnitArr([0] * qty.shape[-1], units=qty.units, dtype=qty.dtype)
    if isinstance(mass, str):
        mass = s.get(mass)
    else:
        mass = UnitArr(mass)
    # only using the np.ndarray views does not speed up
    mwgt = np.tensordot(mass, qty, axes=1)
    normalized_mwgt = mwgt / float(mass.sum())
    return UnitArr(normalized_mwgt, qty.units)


def center_of_mass(snap):
    '''Calculate and return the center of mass of this snapshot.'''
    return mass_weighted_mean(snap, 'pos')


def reduced_inertia_tensor(s):
    '''
    Calculate the 'reduced' inertia tensor by Gerhard (1983) / Bailin & Steinmetz
    (2005) of this ensemble.

    $I_ij = \\sum_k m_k \\frac{r_{k,i} r_{k,j}}{r_k^2}$
    I_ij = sum_k m_k (r_ki r_kj) / r_k^2

    Args:
        s (Snap):   The (sub-)snapshot to calculate the reduced inertia tensor of.

    Returns:
        I (np.matrix):  The reduced inertia tensor. (Without units, but they would
                        be s['mass'].units/s['pos'].units.)
    '''
    # a bit faster with the np.ndarray views
    r2 = (s['r'] ** 2).view(np.ndarray)
    m = s['mass'].view(np.ndarray)
    pos = s['pos'].view(np.ndarray)

    # do not divide by zero
    rzero = (r2 == 0)
    if np.any(rzero):
        rzero = ~rzero
        r2 = r2[rzero]
        m = m[rzero]
        pos = pos[rzero]

    I_xx = np.sum(m * pos[:, 0] ** 2 / r2)
    I_yy = np.sum(m * pos[:, 1] ** 2 / r2)
    I_zz = np.sum(m * pos[:, 2] ** 2 / r2)
    I_xy = np.sum(m * pos[:, 0] * pos[:, 1] / r2)
    I_xz = np.sum(m * pos[:, 0] * pos[:, 2] / r2)
    I_yz = np.sum(m * pos[:, 1] * pos[:, 2] / r2)
    I = np.matrix([[I_xx, I_xy, I_xz], \
                   [I_xy, I_yy, I_yz], \
                   [I_xz, I_yz, I_zz]], dtype=np.float64)
    return I


def orientate_at(s, mode, qty=None, total=False, remember=True):
    '''
    Orientate the (sub-)snapshot at a given quantity.

    Possible modes:
        'vec'/'L':          Orientate such that the given vector alignes with the
                            z-axis. If no vector is given and the mode is 'L', the
                            angular momentum is used.
        'tensor'/'red I':   Orientate at the eigenvectors of a tensor. The
                            eigenvector with the smallest eigenvalue is
                            orientated along the z-axis and the one with the
                            largest eigenvalue along the x-axis. (Hence, the
                            orientation to 'red I' is similar to the orientation
                            to 'L'). If no tensor is given and the mode is
                            'red I', the reduced inertia tensor is used.

    Args:
        s (Snap):       The snapshot to orientate
        mode (str):     The mode of orientation. See above.
        qty (...):      The quantity to use for orientation. If it is None, it is
                        calculated on the fly (for the passed (sub-)snapshot).
        total (bool):   Whether to apply the transformation to the entire snapshot
                        or just the passed sub-snapshot. (Cf. Transformation.apply
                        for more information!)
        remember (bool):
                        Remember the transformation for blocks loaded later. (Cf.
                        Transformation.apply for more information!)
    '''
    if qty is None:
        if mode == 'L':
            qty = s['angmom'].sum(axis=0)
        elif mode == 'red I':
            qty = reduced_inertia_tensor(s)
        else:
            raise ValueError('No quantity passed to orientate at!')

    if mode in ['vec', 'L']:
        T = rot_to_z(qty)
    elif mode in ['tensor', 'red I']:
        qty = np.matrix(qty)
        if np.max(np.abs(qty.H - qty)) > 1e-6:
            raise ValueError('The matrix passed as qty has to be Hermitian!')
        vals, vecs = np.linalg.eigh(qty)
        i = np.argsort(vals)[::-1]
        try:
            T = Rotation(vecs[:, i].T)
        except ValueError:
            # probably not a proper rotation... (not right-handed)
            vecs[:, i[1]] *= -1
            T = Rotation(vecs[:, i].T)
        except:
            raise
    else:
        raise ValueError('unknown orientation mode \'%s\'' % what)

    T.apply(s, total=total, remember=remember)


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
        center = UnitQty([0] * 3)
    center = UnitQty(center, s['pos'].units, subs=s)

    if isinstance(proj, int):
        proj_mask = [i for i in range(3) if i != proj]
        if len(center) == 3:
            center = center[proj_mask]

    if np.all(center == 0):
        if isinstance(proj, int):
            r = s['rcyl'] if proj == 2 else dist(s['pos'][:, proj_mask])
        else:
            r = s['r']
    else:
        r = dist(s['pos'][:, proj_mask], center) if isinstance(proj, int) \
            else dist(s['pos'], center)
    r_ind = r.argsort()

    if isinstance(qty, str):
        qty = s.get(qty)
    else:
        qty = UnitQty(qty)

    Q = np.cumsum(qty[r_ind])
    if Qtot is None:
        Qtot = Q[-1]
    else:
        Qtot = UnitScalar(Qtot, qty.units, subs=s, dtype=float)

    Q_half_ind = np.abs(Q - Qtot / 2.).argmin()
    if Q_half_ind == len(Q) - 1:
        print('WARNING: The half-qty radius is larger than ' + \
              'the (sub-)snapshot passed!', file=sys.stderr)
    elif Q_half_ind < 10:
        print('WARNING: The half-qty radius is not resolved ' + \
              'for %s!' % s, file=sys.stderr)
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
    return half_qty_radius(s, qty='mass', Qtot=M, center=center, proj=proj)


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


def shell_flow_rates(s, Rlim, qty='mass', direction='both', units='Msol/yr'):
    '''
    Estimate flow rate in spherical shell.

    The estimation is done by caluculating m*v/d on a particle base, where d is
    the thickness of the shell.

    Args:
        s (Snap):               The (sub-)snapshot to use.
        Rlim (UnitQty):         The inner and outer radius of the shell.
        qty (str, UnitQty):     The (mass) quantity to calculate the flow rates of.
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
    shell = s[(Rlim[0] < s['r']) & (s['r'] < Rlim[1])]

    if direction == 'both':
        pass
    elif direction == 'in':
        shell = shell[shell['vrad'] < 0]
    elif direction == 'out':
        shell = shell[shell['vrad'] > 0]
    else:
        raise RuntimeError('unknown direction %s!' % direction)

    if isinstance(qty, str):
        qty = shell.get(qty)
    else:
        qty = UnitArr(qty, 'Msol', subs=s)

    flow = np.sum(qty * shell['vrad'] / UnitArr(Rlim[1] - Rlim[0], Rlim.units))
    flow.convert_to(units, subs=s)
    return flow


def flow_rates(s, R, qty='mass', dt='3 Myr'):
    '''
    Estimate in- and outflow rates of a given quantity (default: mass)
    through a given radius.

    The estimation is done by propagating the positions with constant current
    velocities, i.e. pos_new = pos_old + vel*dt. Then counting the mass that
    passed the shell of radius R.

    Args:
        s (Snap):               The (sub-)snapshot to use.
        qty (str):              The quantity to calculate the flow-rates of
                                (default: 'mass')
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
    dt.convert_to(s['r'].units / s['vel'].units, subs=s)  # avoid conversions of
    # entire arrays
    rpred = s['r'] + s['vrad'] * dt
    of_mass = s[qty][(s['r'] < R) & (rpred >= R)]
    if_mass = s[qty][(s['r'] >= R) & (rpred < R)]

    dt.convert_to('yr', subs=s)  # to more intuitive units again
    ofr = np.sum(of_mass) / dt
    ifr = np.sum(if_mass) / dt

    return ifr, ofr


def los_velocity_dispersion(s, proj=2):
    '''
    Calculate (mass-weighted) line-of-sight velocity dispersion.

    Args:
        s (Snap):       The (sub-)snapshot to use.
        proj (int):     The line of sight is along this axis (0=x, 1=y, 2=z).
    '''
    # array of los velocities
    v = s['vel'][:, proj].ravel()
    av_v = mass_weighted_mean(s, v)
    sigma_v = np.sqrt(mass_weighted_mean(s, (v - av_v) ** 2))

    return sigma_v


def x_ray_luminosity(s, lumtable='em.dat', tempbin=None, lx0bin=None, dlxbin=None,
                     Zref=0.4, z_table=0.001):
    '''
    Calculate X-ray luminosity of gas particles using a prepared emission table
    from XSPEC.

    Args:
        s (Snap):               The snapshot to use.
        lumtable (str):         The filename of the XSPEC emission table to use
                                (default: 'em.dat' in current directory)
        tempbin, lx0bin, dlxbin (array-like):
                                Temperature, Lx0 and dLx bins: Can be passed
                                instead of lumtable file (e.g. to avoid reading
                                in the same file multiple times in a loop)
        Zref (float):           The reference metallicity used for the XSPEC
                                table.
        z_table (float):        The redshift assumed for the XSPEC table.

    Returns:
        lx (UnitArr):           X-ray luminosities of the gas particles
    '''
    if abs(s.redshift - z_table) > 1e-2:
        # from pygad-dsorini
        print('WARNING: Snapshot\'s redshift (%.3g) does not ' % s.redshift +
                           'match the table\'s redshift (%.3g)!' % z_table)
        # raise RuntimeError('Snapshot\'s redshift (%.3g) does not ' % s.redshift +
        #                    'match the table\'s redshift (%.3g)!' % z_table)

    # Read in temperature bins and corresponding Lx0(T,Z=Zref) and (dLx/dZ)(T)
    # (both in 1e44 erg/s (per Zsol))
    if tempbin == None:
        tempbin, lx0bin, dlxbin = np.loadtxt(lumtable, usecols=(0, 3, 5), unpack=True)
    else:
        tempbin = np.asarray(tempbin)

    tlow = tempbin[0] - 0.5 * (tempbin[1] - tempbin[0])  # lower temperature bin limit
    Z = s.gas['metallicity'] / physics.solar.Z()  # metallicity in solar units
    mp = physics.m_p.in_units_of('g')  # proton mass

    # Horst: configurability of H-property
    if s.H_neutral_only:
        # Romeel 28/5/2018: can't use s.gas['H'] in our Gizmo sims, this corresponds to *neutral* H; have to get H mass from total-helium-metals
        mhydr = np.float64(s.gas['mass']).in_units_of('g')*(1.-Z*physics.solar.Z())-np.float64(s.gas['He']).in_units_of('g') # H mass in g
    else:
        mhydr = np.float64(s.gas['H']).in_units_of('g')

    #print (1.-Z*physics.solar.Z()),'H in g',mhydr[Z>0]
    # emission measure of gas particles (n_e * n_H * V)
    # Horst: configurability of H-property
    #em = np.float64(s.gas['ne']) * np.float64(s.gas['H']).in_units_of('g') ** 2 * \
    em = np.float64(s.gas['ne']) * mhydr**2 * \
         np.float64(s.gas['rho']).in_units_of('g/cm**3') / \
         (np.float64(s.gas['mass']).in_units_of('g') * mp ** 2)
    # rescaling factor for precomputed luminosities
    Da = s.cosmology.angular_diameter_distance(z_table, 'cm')
    norm = UnitArr(1e-14, units='cm**5') * em / (4 * np.pi * (Da * (1 + z_table)) ** 2)
    norm = norm.view(np.ndarray)
    lx = np.zeros(s.gas['rho'].shape[0])  # array for X-ray luminosity
    kB_T = (s.gas['temp'] * physics.kB).in_units_of('keV').view(np.ndarray)
    indices = np.zeros(s.gas['rho'].shape[0])  # array for fitting tempbin
    # indices for gas particles
    dtemp = np.zeros(s.gas['rho'].shape[0]) + 1e30  # minimal differences of gas
    # particle temperature and binned temperatures

    # loop over tempbin array to find nearest tempbin for all gas particle
    # temperatures
    for i in range(0, tempbin.shape[0]):
        dtemp[np.where(np.abs(kB_T - tempbin[i]) < dtemp)] = \
            np.abs(kB_T[np.where(np.abs(kB_T - tempbin[i]) < dtemp)] - tempbin[i])
        indices[np.where(np.abs(kB_T - tempbin[i]) == dtemp)] = i

    # calculate X-ray luminosities for all gas particles
    for i in range(0, tempbin.shape[0]):
        lx[np.where(indices == i)] = lx0bin[i] + \
                                     (Z[np.where(indices == i)] - Zref) * dlxbin[i]
    lx[np.where(kB_T < tlow)] = 0  # particles below threshold temperature do not
    # contribute to Lx
    return UnitArr(lx * norm * 1e44, 'erg/s')  # luminosities of all gas particles [erg/s]
