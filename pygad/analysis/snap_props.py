'''
A collection of (sub-)snapshot wide analysis functions.

Example:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_320', physical=False)
    >>> mass_weighted_mean(s, 'pos')
    load block pos... done.
    load block mass... done.
    UnitArr([ 34799.42578125,  33686.4765625 ,  32010.89453125],
            dtype=float32, units="ckpc h_0**-1")
    >>> center_of_mass(s)
    UnitArr([ 34799.42578125,  33686.4765625 ,  32010.89453125],
            dtype=float32, units="ckpc h_0**-1")

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
    >>> los_velocity_dispersion(sub)
    UnitArr(167.344421387, dtype=float32, units="km s**-1")

    >>> s.to_physical_units()
    convert block pos to physical units... done.
    convert block mass to physical units... done.
    convert block r to physical units... done.
    convert boxsize to physical units... done.
    >>> SPH_qty_at(s, 'rho', [0,0,0])
    load block rho... done.
    convert block rho to physical units... done.
    load block hsml... done.
    convert block hsml to physical units... done.
    derive block dV... done.
    UnitArr(1.740805e+05, units="Msol kpc**-3")
    >>> SPH_qty_at(s, 'rho', s.gas['pos'][331798])
    UnitArr(5.330261e+04, units="Msol kpc**-3")
    >>> s.gas['rho'][331798]
    56831.004
    >>> SPH_qty_at(s, 'rho', s.gas['pos'][:1000:100])
    UnitArr([  2.85217152e+08,   2.99014187e+01,   1.25738525e+01,
               8.15994930e+00,   1.05293274e+01,   1.17380829e+01,
               3.57516813e+00,   6.43977594e+00,   7.23333836e-01,
               3.83610177e+00], dtype=float32, units="Msol kpc**-3")
    >>> SPH_qty_at(s, 'rho', s.gas['pos'][:1000])[::100]
    UnitArr([  2.85217151e+08,   2.99014196e+01,   1.25738527e+01,
               8.15994893e+00,   1.05293272e+01,   1.17380827e+01,
               3.57516815e+00,   6.43977587e+00,   7.23333840e-01,
               3.83610195e+00], units="Msol kpc**-3")
    >>> scatter_gas_qty_to_stars(s, 'rho', name='gas_rho')
    SimArr([  3.39356901e+07,   3.74830586e+07,   2.57179940e+08, ...,
              1.03715005e+03,   1.06398579e+03,   2.16652432e+02],
           units="Msol kpc**-3", snap="snap_M1196_4x_320":stars)

    >>> SPH_qty_at(s, 'elements', s.gas['pos'][:200:100])
    load block elements... done.
    convert block elements to physical units... done.
    UnitArr([[  5.03078656e+05,   3.38629810e+03,   2.82325244e+03,
                2.00907266e+04,   2.60392407e+03,   1.98391785e+03,
                1.41223262e+06,   1.33162231e+03,   5.57730957e+03,
                9.51993042e+02,   1.20361496e+02,   4.72353955e+03],
             [  2.41064234e+05,   5.55662003e+01,   6.92608490e+01,
                5.19309509e+02,   4.00640755e+01,   3.92069435e+01,
                7.60207812e+05,   1.35450637e+00,   1.34345795e+02,
                1.83183250e+01,   3.22454572e+00,   1.57628281e+02]],
            dtype=float32, units="Msol")
    >>> SPH_qty_at(s, 'elements', s.gas['pos'][:200])[::100]
    UnitArr([[  5.03078655e+05,   3.38629804e+03,   2.82325237e+03,
                2.00907257e+04,   2.60392404e+03,   1.98391781e+03,
                1.41223264e+06,   1.33162234e+03,   5.57730973e+03,
                9.51993020e+02,   1.20361494e+02,   4.72353971e+03],
             [  2.41064242e+05,   5.55661972e+01,   6.92608447e+01,
                5.19309478e+02,   4.00640730e+01,   3.92069418e+01,
                7.60207788e+05,   1.35450636e+00,   1.34345793e+02,
                1.83183255e+01,   3.22454563e+00,   1.57628272e+02]], units="Msol")
'''
__all__ = ['mass_weighted_mean', 'center_of_mass', 'reduced_inertia_tensor',
           'orientate_at', 'los_velocity_dispersion',
           'kernel_weighted', 'SPH_qty_at', 'scatter_gas_qty_to_stars',
          ]

import numpy as np
from ..units import *
from ..transformation import *

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
    hsml = s.gas['hsml'].in_units_of(s['pos'].units,subs=s).view(np.ndarray)
    dV = s.gas.get(dV).in_units_of(s['pos'].units**3,subs=s).view(np.ndarray)

    if kernel is None:
        from ..gadget import config
        kernel = config.general['kernel']
    from ..kernels import vector_kernels
    kernel_func = vector_kernels[kernel]

    if r.shape == (3,):
        d = dist(gas_pos, r) / hsml
        mask = d < 1.0
        Q = np.sum((qty[mask].T * (kernel_func(d[mask]) * dV[mask] /
                                        hsml[mask]**3)).T,
                   axis=0)
    elif len(r) < 100:
        dV_hsml3 = dV / hsml**3
        Q = np.empty( (len(r),)+qty.shape[1:], dtype=qty.dtype )
        for i,x in enumerate(r):
            d = dist(gas_pos, x) / hsml
            mask = d < 1.0
            Q[i] = np.sum((qty[mask].T * (kernel_func(d[mask]) *
                                            dV_hsml3[mask])).T,
                          axis=0)
    else:
        from .. import C
        # C function expects doubles and cannot deal with views:
        r = r.astype(np.float64)
        gas_pos = gas_pos.astype(np.float64)
        hsml = hsml.astype(np.float64)
        dV = dV.astype(np.float64)
        qty = qty.astype(np.float64)
        if r.base is not None:
            r = r.copy()
        if gas_pos.base is not None:
            gas_pos = gas_pos.copy()
            hsml = hsml.copy()
            dV = dV.copy()
        #TODO: handle different types than double
        if len(qty.shape)==1:
            Q = np.empty( len(r), dtype=np.float64 )
            if qty.base is not None:
                qty = qty.copy()
            C.cpygad.eval_sph_at(
                    C.c_size_t(len(r)),
                    C.c_void_p(r.ctypes.data),
                    C.c_void_p(Q.ctypes.data),
                    C.c_size_t(len(gas_pos)),
                    C.c_void_p(gas_pos.ctypes.data),
                    C.c_void_p(hsml.ctypes.data),
                    C.c_void_p(dV.ctypes.data),
                    C.c_void_p(qty.ctypes.data),
                    C.create_string_buffer(kernel),
                    None    # build new tree
            )
        elif len(qty.shape)==2:
            # C function needs contiquous arrays and cannot deal with
            # mutli-dimenensional ones...
            Q = np.empty( (qty.shape[1],len(r)), dtype=np.float64 )
            qty = qty.T
            if qty.base is not None:
                qty = qty.copy()
            # avoid the reconstruction of the octree
            from ..octree import cOctree
            tree = cOctree(gas_pos, hsml)
            for k in xrange(Q.shape[0]):
                C.cpygad.eval_sph_at(
                        C.c_size_t(len(r)),
                        C.c_void_p(r.ctypes.data),
                        C.c_void_p(Q[k].ctypes.data),
                        C.c_size_t(len(gas_pos)),
                        C.c_void_p(gas_pos.ctypes.data),
                        C.c_void_p(hsml.ctypes.data),
                        C.c_void_p(dV.ctypes.data),
                        C.c_void_p(qty[k].ctypes.data),
                        C.create_string_buffer(kernel),
                        C.c_void_p(tree._cOctree__node_ptr),
                )
            Q = Q.T
        else:
            raise ValueError('Cannot handle more than two dimension in qty!')

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

