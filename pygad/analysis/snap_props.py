'''
A collection of (sub-)snapshot wide analysis functions.

Example:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_320', physical=False)
    >>> if np.linalg.norm(mass_weighted_mean(s,'pos') - center_of_mass(s)) > 1e-3:
    ...     print mass_weighted_mean(s,'pos')
    ...     print center_of_mass(s,)
    load block pos... done.
    load block mass... done.

    Center of mass is *not* center of galaxy / halo!
    >>> Translation([-34792.2, -35584.8, -33617.9]).apply(s)
    apply Translation to "pos" of "snap_M1196_4x_320"... done.
    >>> sub = s[s['r'] < '20 kpc']
    derive block r... done.
    >>> orientate_at(sub, 'L', total=True)
    load block vel... done.
    derive block momentum... done.
    derive block angmom... done.
    apply Rotation to "vel" of "snap_M1196_4x_320"... done.
    apply Rotation to "pos" of "snap_M1196_4x_320"... done.
    >>> sub['angmom'].sum(axis=0)
    derive block momentum... done.
    derive block angmom... done.
    UnitArr([  3.35956104e-02,  -1.74680334e-02,   1.10114893e+04],
            dtype=float32, units="1e+10 ckpc h_0**-1 Msol h_0**-1 km s**-1")
    >>> redI = reduced_inertia_tensor(sub.baryons)
    derive block r... done.
    >>> redI
    matrix([[ 1.82747424, -0.23583228,  0.19712902],
            [-0.23583228,  0.19789133,  0.0371508 ],
            [ 0.19712902,  0.0371508 ,  0.42077191]])
    >>> orientate_at(s, 'red I', redI)
    apply Rotation to "vel" of "snap_M1196_4x_320"... done.
    apply Rotation to "pos" of "snap_M1196_4x_320"... done.
    >>> redI = reduced_inertia_tensor(sub.baryons)
    derive block r... done.
    >>> redI[np.abs(redI)<1e-9] = 0.0
    >>> redI
    matrix([[ 0.14794385,  0.        ,  0.        ],
            [ 0.        ,  0.41260123,  0.        ],
            [ 0.        ,  0.        ,  1.8855924 ]])
    >>> if abs( los_velocity_dispersion(sub) - '170 km/s' ) > '5 k/s':
    ...     print los_velocity_dispersion(sub)

    >>> s.to_physical_units()
    convert block pos to physical units... done.
    convert block mass to physical units... done.
    convert block r to physical units... done.
    convert boxsize to physical units... done.
    >>> SPH_qty_at(s.gas, 'rho', [0,0,0])
    load block rho... done.
    convert block rho to physical units... done.
    load block hsml... done.
    convert block hsml to physical units... done.
    derive block dV... done.
    UnitArr(1.740805e+05, units="Msol kpc**-3")
    >>> SPH_qty_at(s.gas, 'rho', s.gas['pos'][331798])
    UnitArr(5.330261e+04, units="Msol kpc**-3")
    >>> s.gas['rho'][331798]
    56831.004

    #to slow...!
    #>>> scatter_gas_qty_to_stars(s, 'Z', name='gas_Z')
    load block elements... done.
    derive block H... done.
    derive block He... done.
    derive block metals... done.
    derive block Z... done.
    load block hsml... done.
    build tree...
    done.
    scatter property of 64,158 star particles...
    done.
    didn't found neighbours 30440 times
    UnitArr([  6.62012251e-09,   2.28143236e-09,   1.77712193e-14, ...,
               0.00000000e+00,   0.00000000e+00,   0.00000000e+00])
       
    >>> s.gas['lx'] = x_ray_luminosity(s, lumtable='em.dat')
    load block elements... done.
    convert block elements to physical units... done.
    derive block H... done.
    derive block He... done.
    derive block metals... done.
    derive block Z... done.
    load block ne... done.
    load block temp... done.
    >>> s.gas['lx']
    SimArr([ 0.,  0.,  0., ...,  0.,  0.,  0.],
           units="erg s**-1", snap="snap_M1196_4x_320":gas)
    >>> s.gas['lx'].sum()
    UnitArr(1.631726e+41, units="erg s**-1")
'''
__all__ = ['mass_weighted_mean', 'center_of_mass', 'reduced_inertia_tensor',
           'orientate_at', 'los_velocity_dispersion',
           'kernel_weighted', 'SPH_qty_at', 'scatter_gas_qty_to_stars',
           'x_ray_luminosity'
          ]

import numpy as np
from ..units import *
from ..transformation import *
from pygad import physics

def mass_weighted_mean(s, qty):
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

    Returns:
        mean (UnitArr):     The mass weighted mean.
    '''
    if isinstance(qty, str):
        qty = s.get(qty)
    else:
        qty = UnitArr(qty)
    if len(s) == 0:
        return UnitArr([0]*qty.shape[-1], units=qty.units, dtype=qty.dtype)
    # only using the np.ndarray views does not speed up
    mwgt = np.tensordot(s['mass'], qty, axes=1).view(UnitArr)
    mwgt.units = s['mass'].units * qty.units
    normalized_mwgt = mwgt / s['mass'].sum()
    return normalized_mwgt

def center_of_mass(snap):
    '''Calculate and return the center of mass of this snapshot.'''
    return mass_weighted_mean(snap, 'pos')

def reduced_inertia_tensor(s):
    '''
    Calculate the 'reduced' inertia tensor by Raini and Steinmetz (2005) of this
    ensemble.

    $I_ij = \\sum_k m_k \\frac{r_{k,i} r_{k,j}}{r_k^2}$
    I_ij = sum_k m_k (r_ki r_kj) / r_k^2

    Args:
        s (Snap):   The (sub-)snapshot to calculate the reduced inertia tensor of.

    Returns:
        I (np.matrix):  The reduced inertia tensor. (Without units, but they would
                        be s['mass'].units/s['pos'].units.)
    '''
    # a bit faster with the np.ndarray views
    r2 = (s['r']**2).view(np.ndarray)
    m = s['mass'].view(np.ndarray)
    pos = s['pos'].view(np.ndarray)
    I_xx = np.sum(m * pos[:,0]**2 / r2)
    I_yy = np.sum(m * pos[:,1]**2 / r2)
    I_zz = np.sum(m * pos[:,2]**2 / r2)
    I_xy = np.sum(m * pos[:,0]*pos[:,1] / r2)
    I_xz = np.sum(m * pos[:,0]*pos[:,2] / r2)
    I_yz = np.sum(m * pos[:,1]*pos[:,2] / r2)
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
                            largest eigenvalue along the x-axis. If no tensor is
                            given and the mode is 'red I', the reduced inertia
                            tensor is used.

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
        T = transformation.rot_to_z(qty)
    elif mode in ['tensor', 'red I']:
        if np.max(np.abs(qty.H-qty)) > 1e-6:
            raise ValueError('The matrix passed as qty has to be Hermitian!')
        vals, vecs = np.linalg.eigh(qty)
        i = np.argsort(vals)
        try:
            T = transformation.Rotation(vecs[:,i].T)
        except ValueError:
            # probably not a proper rotation... (not right-handed)
            vecs[:,i[1]] *= -1
            T = transformation.Rotation(vecs[:,i].T)
        except:
            raise
    else:
        raise ValueError('unknown orientation mode \'%s\'' % what)

    T.apply(s, total=total, remember=remember)

def los_velocity_dispersion(s, proj=2):
    '''
    Calculate (mass-weighted) line-of-sight velocity dispersion.

    Args:
        s (Snap):       The (sub-)snapshot to use.
        proj (int):     The line of sight is along this axis (0=x, 1=y, 2=z).
    '''
    # array of los velocities
    v = s['vel'][:,proj].ravel()
    av_v = mass_weighted_mean(s, v)
    sigma_v = np.sqrt( mass_weighted_mean(s, (v-av_v)**2) )

    return sigma_v

def _calc_ys(qty, all_gas_pos, gas_pos, hsml, kernel):
    y = np.empty( (len(gas_pos),)+qty.shape[1:] )
    for i in xrange(len(gas_pos)):
        hi = hsml[i]
        d = dist(all_gas_pos, gas_pos[i]) / hi
        mask = d < 1.0
        y[i] = np.sum((qty[mask].T * (kernel(d[mask]) / hi**3)).T,
                      axis=0)
    return y
def kernel_weighted(s, qty, units=None, kernel=None, parallel=None):
    '''
    Calculate a kernel weighted SPH quantity for all the gas particles:
             N
            ___         /  \
      y  =  \    x  W  | h  |
       i    /__   j  ij \ i/
            j+1

    Args:
        s (Snap):               The (sub-)snapshot to take the gas (quantity)
                                from.
        qty (str, array-like):  The name of the gas quantity or a array-like
                                object with length of the gas.
        units (str, Unit):      The units to convert the result in. If None, no
                                conversion is done.
        kernel (str):           The kernel to use. The default is to take the
                                kernel given in the `gadget.cfg`.

    Returns:
        y (UnitArr):            The kernel weigthed SPH property for all the gas
                                particles.
    '''
    # TODO: find ways to speed it up! -- parallelisation very inefficient
    gas = s.gas
    if isinstance(qty, str):
        qty = gas.get(qty)
    else:
        if len(qty) != len(gas):
            from ..utils import nice_big_num_str
            raise RuntimeError('The length of the quantity ' + \
                               '(%s) does not ' % nice_big_num_str(len(qty)) + \
                               'match the number of gas ' + \
                               '(%s)!' % nice_big_num_str(len(gas)))
    units = getattr(qty,'units',None) if units is None else Unit(units)
    qty = np.asarray(qty)

    gas_pos = gas['pos'].view(np.ndarray)
    hsml = gas['hsml'].in_units_of(s['pos'].units,subs=s).view(np.ndarray)

    if kernel is None:
        from ..gadget import config
        kernel = config.general['kernel']
    from ..kernels import vector_kernels
    kernel = vector_kernels[kernel]

    if parallel is None:
        parallel = len(gas) > 1000
    if parallel:
        from multiprocessing import Pool, cpu_count
        N_threads = cpu_count()
        pool = Pool(N_threads)
        res = [None] * N_threads
        chunk = [[i*len(gas)/N_threads, (i+1)*len(gas)/N_threads]
                    for i in xrange(N_threads)]
        import warnings
        with warnings.catch_warnings():
            # warnings.catch_warnings doesn't work in parallel
            # environment...
            warnings.simplefilter("ignore") # for _z2Gyr_vec
            for i in xrange(N_threads):
                res[i] = pool.apply_async(_calc_ys,
                        (qty, gas_pos,
                         gas_pos[chunk[i][0]:chunk[i][1]],
                         hsml[chunk[i][0]:chunk[i][1]],
                         kernel))
        y = np.empty( (len(gas_pos),)+qty.shape[1:] )
        for i in xrange(N_threads):
            y[chunk[i][0]:chunk[i][1]] = res[i].get()
    else:
        y = _calc_ys(qty, gas_pos, hsml, kernel)

    return UnitArr(y, units)

def SPH_qty_at(s, qty, r, units=None, kernel=None, dV='dV'):
    '''
    Calculate a SPH quantity with the scatter approach at a given position.

    Args:
        s (Snap):               The (sub-)snapshot to take the gas (quantity)
                                from.
        qty (str, array-like):  The name of the gas quantity or a array-like
                                object with length of the gas.
        r (UnitQty):            The position(s) to evaluate the SPH quantity at.
        units (str, Unit):      The units to conver to. If None, no conversion is
                                done and the results is gioven in the units
                                encountered.
        kernel (str):           The kernel to use. The default is to take the
                                kernel given in the `gadget.cfg`.
        dV (str, array-like):   The volume measure of the SPH particles.

    Returns:
        Q (UnitArr):            The SPH property at the given position.
    '''
    # TODO: find ways to speed it up!
    # TODO: parallelize
    r = UnitQty(r, s['pos'].units, subs=s)
    if not (r.shape==(3,) or (r.shape[1:]==(3,) and len(r.shape)==2)):
        raise ValueError('Position `r` needs to have shape (3,) or (N,3)!')
    if isinstance(qty, str):
        qty = s.gas.get(qty)
    else:
        if len(qty) != len(s.gas):
            from ..utils import nice_big_num_str
            raise RuntimeError('The length of the quantity ' + \
                               '(%s) does not ' % nice_big_num_str(len(qty)) + \
                               'match the number of gas ' + \
                               '(%s)!' % nice_big_num_str(len(s.gas)))
    units = getattr(qty,'units',None) if units is None else Unit(units)
    qty = np.asarray(qty)

    r = r.view(np.ndarray)
    gas_pos = s.gas['pos'].view(np.ndarray)
    hsml = s.gas['hsml'].in_units_of(s['pos'].units,subs=s)
    dV_hsml3 = (s.get(dV) / hsml**3) \
            .in_units_of(1,subs=s).view(np.ndarray)
    hsml = hsml.view(np.ndarray)

    if kernel is None:
        from ..gadget import config
        kernel = config.general['kernel']
    from ..kernels import vector_kernels
    kernel = vector_kernels[kernel]

    if r.shape == (3,):
        d = dist(gas_pos, r) / hsml
        mask = d < 1.0
        Q = np.sum((qty[mask].T * (kernel(d[mask]) * dV_hsml3[mask])).T,
                   axis=0)
    else:
        Q = np.empty( (len(r),)+qty.shape[1:] )
        for i,x in enumerate(r):
            d = dist(gas_pos, x) / hsml
            mask = d < 1.0
            Q[i] = np.sum((qty[mask].T * (kernel(d[mask]) * dV_hsml3[mask])).T,
                          axis=0)

    return UnitArr(Q, units)

def scatter_gas_qty_to_stars(s, qty, name=None, units=None, kernel=None, dV='dV'):
    '''
    Calculate a gas property at the positions of the stars and store it as a
    stellar property.

    This function calculates the gas property at the positions of the star
    particles by the so-called scatter approach, i.e. by evaluating the gas
    property of every gas particle at the stellar positions, kernel-weighted by
    the kernel of the gas particled the property comes from. Finally these
    quantities are stored as a new block for the stars of the snapshot. (Make
    shure there is no such block yet.)

    Args:
        s (Snap):               The snapshot to spread the properties of.
        qty (array-like, str):  The name of the block or the block itself to
                                spread onto the neighbouring SPH (i.e. gas)
                                particles.
        name (str):             The name of the new SPH block. If it is None and
                                `qty` is a string, is taken as the name.
        units (str, Unit):      The units to store the property in. If None, no
                                conversion is done and the results is stored in
                                the units encountered.
        kernel (str):           The kernel to use. The default is to take the
                                kernel given in the `gadget.cfg`.
        dV (str, array-like):   The volume measure of the SPH particles.

    Returns:
        Q (UnitArr):            The new SPH block.

    Raises:
        RuntimeError:       From this function directly:
                                If no name was given, though the qty is not a
                                string; or if the length of `qty` does not match
                                the number of stars in `s`.
                            From setting the new block:
                                If the length of the data does not fit the length
                                of the (sub-)snapshot; or if the latter is not the
                                correct "host" of the new block.
        KeyError:           If there already exists a block of that name.
    '''
    if name is None:
        if isinstance(qty, str):
            name = qty
        else:
            raise RuntimeError('No name for the quantity is given!')

    s.stars[name] = SPH_qty_at(s, qty=qty, r=s.stars['pos'], units=units, kernel=kernel)
    return s.stars[name]

def x_ray_luminosity(s, lumtable='em.dat', tempbin=None, lx0bin=None, dlxbin=None):
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
    '''
    Zref = 0.4          # metallicity (in solar units) used for XSPEC calculations
    red = 0.001         # redshift assumed for XSPEC luminosity table
    Da = UnitArr(4.3*1e3, units='kpc').in_units_of('cm')   #angular diameter
                        # distance corresponding to redshift 0.001 in cm (would be
                        # better if directly calculated from redshift)

    # Read in temperature bins and corresponding Lx0(T,Z=Zref) and (dLx/dZ)(T)
    # (both in 1e44 erg/s (per Zsol))
    if tempbin == None:
        tempbin, lx0bin, dlxbin = np.loadtxt(lumtable, usecols=(0,3,5), unpack=True)

    tlow = tempbin[0] - 0.5*(tempbin[1]-tempbin[0]) # lower temperature bin limit
    Z = s.gas['Z'] / physics.solar.Z()              # metallicity in solar units
    mp = physics.m_p.in_units_of('g')               # proton mass
    # emission measure of gas particles (n_e * n_H * V)
    em = np.float64(s.gas['ne']) * np.float64(s.gas['H']).in_units_of('g')**2 * \
         np.float64(s.gas['rho']).in_units_of('g/cm**3') / \
         (np.float64(s.gas['mass']).in_units_of('g')*mp**2)
    # rescaling factor for precomputed luminosities
    norm = UnitArr(1e-14,units='cm**5') * em / (4 * np.pi * (Da*(1+red))**2)
    lx = np.zeros(s.gas['rho'].shape[0])         # array for X-ray luminosity
    temp = s.gas['temp']*1.3806e-16/1.6022e-9    # gas temperatures in keV
    indices = np.zeros(s.gas['rho'].shape[0])    # array for fitting tempbin
                                                 # indices for gas particles
    dtemp = np.zeros(s.gas['rho'].shape[0])+1e30 # minimal differences of gas
                                  # particle temperature and binned temperatures

    # loop over tempbin array to find nearest tempbin for all gas particle
    # temperatures
    for i in xrange(0,tempbin.shape[0]):
        dtemp[np.where(np.abs(temp-tempbin[i]) < dtemp)] = \
                np.abs(temp[np.where(np.abs(temp-tempbin[i])<dtemp)] - tempbin[i])
        indices[np.where(np.abs(temp-tempbin[i]) == dtemp)] = i

    # calculate X-ray luminosities for all gas particles
    for i in xrange(0,tempbin.shape[0]):
        lx[np.where(indices == i)] = lx0bin[i] + \
                (Z[np.where(indices==i)] - Zref) * dlxbin[i]
    lx = lx * norm * 1e44           # luminosities of all gas particles [erg/s]
    lx[np.where(temp < tlow)] = 0   # particles below threshold temperature do not
                                    # contribute to Lx
    lx.units = 'erg/s'
    return lx
