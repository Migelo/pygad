'''
Evaluate SPH properties.

Examples:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snapshot
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_320', physical=False)
    >>> from ..transformation import Translation
    >>> Translation([-34792.2, -35584.8, -33617.9]).apply(s)
    >>> s.to_physical_units()

    >>> if (abs(SPH_qty_at(s.gas, 'rho', [0,0,0]) - '1.74e5 Msol/kpc**3')
    ...             > '2e3 Msol/kpc**3'):
    ...     print(SPH_qty_at(s.gas, 'rho', [0,0,0]))
    load block pos... done.
    apply stored Translation to block pos... done.
    load block rho... done.
    load block hsml... done.
    load block mass... done.
    derive block dV... done.
    >>> if (abs(SPH_qty_at(s.gas, 'rho', s.gas['pos'][331798]) -
    ...         '5.33e4 Msol/kpc**3') > '2e2 Msol/kpc**3'):
    ...     print(SPH_qty_at(s.gas, 'rho', s.gas['pos'][331798]))
    >>> assert np.all(np.abs(SPH_qty_at(s, 'rho', s.gas['pos'][:1000:100]) -
    ...                      SPH_qty_at(s, 'rho', s.gas['pos'][:1000])[::100]) /
    ...                 SPH_qty_at(s, 'rho', s.gas['pos'][:1000:100]) < 1e-4)
    >>> scatter_gas_qty_to_stars(s, 'rho', name='gas_rho')  # doctest:+ELLIPSIS
    SimArr(...units="Msol kpc**-3", snap="snap_M1196_4x_320":stars)
    >>> if np.abs(s.stars['gas_rho'][0] - 3.5e7) > 0.2e7:
    ...     print(s.stars['gas_rho'][0])
    >>> if np.abs(s.stars['gas_rho'][-1] - 2.2e2) > 0.2e2:
    ...     print(s.stars['gas_rho'][-1])

    >>> v1 = SPH_qty_at(s, 'elements', s.gas['pos'][:200:10])
    load block Z... done.
    derive block elements... done.
    >>> v2 = SPH_qty_at(s, 'elements', s.gas['pos'][:200])[::10]
    >>> if np.max(np.abs(v1-v2)/v1) > 1e-6:
    ...     print(v1)
    ...     print(v2)

'''
__all__ = ['kernel_weighted', 'SPH_qty_at', 'scatter_gas_qty_to_stars']

import numpy as np
from ..units import *
from ..utils import dist


def _calc_ys(qty, all_gas_pos, gas_pos, hsml, kernel):
    y = np.empty((len(gas_pos),) + qty.shape[1:])
    for i in range(len(gas_pos)):
        hi = hsml[i]
        d = dist(all_gas_pos, gas_pos[i]) / hi
        mask = d < 1.0
        y[i] = np.sum((qty[mask].T * (kernel(d[mask]) / hi ** 3)).T,
                      axis=0)
    return y


def kernel_weighted(s, qty, units=None, kernel=None, parallel=None):
    '''
    Calculate a kernel weighted SPH quantity for all the gas particles:
             N
            ___         /  \
      y  =  \    x  W  | h  |
       i    /__   j  ij \ i/
            j=1

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
    units = getattr(qty, 'units', None) if units is None else Unit(units)
    qty = np.asarray(qty)

    gas_pos = gas['pos'].view(np.ndarray)
    hsml = gas['hsml'].in_units_of(s['pos'].units, subs=s).view(np.ndarray)

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
        chunk = [[i * len(gas) / N_threads, (i + 1) * len(gas) / N_threads]
                 for i in range(N_threads)]
        import warnings
        with warnings.catch_warnings():
            # warnings.catch_warnings doesn't work in parallel
            # environment...
            warnings.simplefilter("ignore")  # for _z2Gyr_vec
            for i in range(N_threads):
                res[i] = pool.apply_async(_calc_ys,
                                          (qty, gas_pos,
                                           gas_pos[chunk[i][0]:chunk[i][1]],
                                           hsml[chunk[i][0]:chunk[i][1]],
                                           kernel))
        y = np.empty((len(gas_pos),) + qty.shape[1:])
        for i in range(N_threads):
            y[chunk[i][0]:chunk[i][1]] = res[i].get()
    else:
        y = _calc_ys(qty, gas_pos, gas_pos, hsml, kernel)

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
    if not (r.shape == (3,) or (r.shape[1:] == (3,) and len(r.shape) == 2)):
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
    units = getattr(qty, 'units', None) if units is None else Unit(units)
    qty = np.asarray(qty)

    r = r.view(np.ndarray)
    gas_pos = s.gas['pos'].view(np.ndarray)
    hsml = s.gas['hsml'].in_units_of(s['pos'].units, subs=s).view(np.ndarray)
    dV = s.gas.get(dV).in_units_of(s['pos'].units ** 3, subs=s).view(np.ndarray)

    if kernel is None:
        from ..gadget import config
        kernel = config.general['kernel']
    from ..kernels import vector_kernels
    kernel_func = vector_kernels[kernel]

    if r.shape == (3,):
        d = dist(gas_pos, r) / hsml
        mask = d < 1.0
        Q = np.sum((qty[mask].T * (kernel_func(d[mask]) * dV[mask] /
                                   hsml[mask] ** 3)).T,
                   axis=0)
    elif len(r) < 100:
        dV_hsml3 = dV / hsml ** 3
        Q = np.empty((len(r),) + qty.shape[1:], dtype=qty.dtype)
        for i, x in enumerate(r):
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
        # TODO: handle different types than double
        if len(qty.shape) == 1:
            Q = np.empty(len(r), dtype=np.float64)
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
                C.create_string_buffer(kernel.encode('ascii')),
                None  # build new tree
            )
        elif len(qty.shape) == 2:
            # C function needs contiquous arrays and cannot deal with
            # mutli-dimenensional ones...
            Q = np.empty((qty.shape[1], len(r)), dtype=np.float64)
            qty = qty.T
            if qty.base is not None:
                qty = qty.copy()
            # avoid the reconstruction of the octree
            from ..octree import cOctree
            tree = cOctree(gas_pos, hsml)
            for k in range(Q.shape[0]):
                C.cpygad.eval_sph_at(
                    C.c_size_t(len(r)),
                    C.c_void_p(r.ctypes.data),
                    C.c_void_p(Q[k].ctypes.data),
                    C.c_size_t(len(gas_pos)),
                    C.c_void_p(gas_pos.ctypes.data),
                    C.c_void_p(hsml.ctypes.data),
                    C.c_void_p(dV.ctypes.data),
                    C.c_void_p(qty[k].ctypes.data),
                    C.create_string_buffer(kernel.encode('ascii')),
                    C.c_void_p(tree._cOctree__node_ptr),
                )
            del tree
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

