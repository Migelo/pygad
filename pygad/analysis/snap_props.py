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
    >>> Translation([-34792.2, -35584.8, -33617.9]).apply(s)
    apply Translation to "pos" of "snap_M1196_4x_320"... done.
    >>> sub = s[s.r < '20 kpc']
    derive block r... done.
    >>> orientate_at(sub, 'L', total=True)
    load block vel... done.
    derive block momentum... done.
    derive block angmom... done.
    apply Rotation to "vel" of "snap_M1196_4x_320"... done.
    apply Rotation to "pos" of "snap_M1196_4x_320"... done.
    >>> sub.angmom.sum(axis=0)
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

    >>> scatter_gas_to_stars(s, 'Z', name='gas_Z')
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
'''
__all__ = ['mass_weighted_mean', 'center_of_mass', 'reduced_inertia_tensor',
           'orientate_at', 'los_velocity_dispersion',
           'scatter_gas_to_stars',
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
        if hasattr(s,qty):
            qty = getattr(s, qty)
        else:
            qty = s.get(qty)
    else:
        qty = UnitArr(qty)
    if len(s) == 0:
        return UnitArr([0]*qty.shape[-1], units=qty.units, dtype=qty.dtype)
    # only using the np.ndarray views does not speed up
    mwgt = np.tensordot(s.mass, qty, axes=1).view(UnitArr)
    mwgt.units = s.mass.units * qty.units
    normalized_mwgt = mwgt / s.mass.sum()
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
                        be s.mass.units/s.pos.units.)
    '''
    # a bit faster with the np.ndarray views
    r2 = (s.r**2).view(np.ndarray)
    m = s.mass.view(np.ndarray)
    pos = s.pos.view(np.ndarray)
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
            qty = s.angmom.sum(axis=0)
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
    v = s.vel[:,proj].ravel()
    av_v = mass_weighted_mean(s, v)
    sigma_v = np.sqrt( mass_weighted_mean(s, (v-av_v)**2) )
    
    return sigma_v

def scatter_gas_to_stars(s, qty, name=None, units=None, kernel=None):
    '''
    Calculate a gas property at the positions of the stars and store it as a
    stellar property.

    This function calculates the gas property at the positions of the star
    particles by a so-called scatter approach, i.e. by evaluating the gas property
    of every gas particle at the stellar positions, kernel-weighted by the kernel
    of the gas particled the property comes from.
    Finally these quantities are stored as a new block for the stars of the
    snapshot. (Make shure there is no such block yet.)

    Args:
        s (Snap):               The snapshot to spread the properties of.
        qty (array-like, str):  The name of the block or the block itself to
                                spread onto the neighbouring SPH (i.e. gas)
                                particles.
        name (str):             The name of the new SPH block. If it is None and
                                `qty` is a string, is taken as the name.
        units (str, Unit):      The units to store the property in.
        kernel (str):           The kernel to use. The default is to take the
                                kernel given in the `gadget.cfg`.

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

    if isinstance(qty, str):
        qty = s.gas.get(qty)
    else:
        if len(qty) != len(s.gas):
            from ..utils import nice_big_num_str
            raise RuntimeError('The length of the quantity ' + \
                               '(%s) does not ' % nice_big_num_str(len(qty)) + \
                               'match the number of gas ' + \
                               '(%s)!' % nice_big_num_str(len(s.gas)))
    if units is None:
        units = getattr(qty, 'units', None)
    else:
        units = Unit(units)
    qty = np.asarray(qty)

    hsml = s.gas.hsml.in_units_of(s.pos.units).view(np.ndarray)
    gas_pos = s.gas.pos.view(np.ndarray)
    star_pos = s.stars.pos.view(np.ndarray)
    if len(qty.shape) > 1:
        Q = np.zeros((len(s.stars), qty.shape[1]))
    else:
        Q = np.zeros(len(s.stars))

    if kernel is None:
        from ..gadget import config
        kernel = config.general['kernel']
    from ..kernels import vector_kernels
    kernel = vector_kernels[kernel]

    for i in xrange(len(s.stars)):
        d = dist(gas_pos, star_pos[i]) / hsml
        mask = d < 1.0
        Q[i] = np.sum((qty[mask].T * kernel(d[mask])).T, axis=0)

    Q = UnitArr(Q, units)
    return s.stars.add_custom_block(Q, name)

