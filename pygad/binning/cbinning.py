'''
Interface to the C implementations of binning.

Testing:
    >>> from ..environment import module_dir
    >>> from ..snapshot import Snap
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_320', physical=True)
    >>> from ..transformation import Translation
    >>> Translation(-UnitArr([34.7828, 35.5898, 33.6147], 'cMpc/h_0')).apply(s)
    >>> extent = UnitArr([[-0.5,0.7],[-1.0,2.0],[-2.0,2.0]], 'Mpc')
    >>> sub = s[BoxMask(extent, sph_overlap=True)]
    load block pos... done.
    apply stored Translation to block pos... done.
    load block hsml... done.
    >>> Npx = np.array([ 30,  75, 100])
    >>> map2D, res = SPH_to_2Dgrid(sub.gas, extent=extent[:2], qty='rho', Npx=Npx[:2])
    create a 30 x 75 SPH-grid (1200 x 3000 [kpc])...
    load block rho... done.
    load block mass... done.
    derive block dV... done.
    done with SPH grid

    Consistency check of total mass:
    >>> mask = np.ones(len(sub.gas), dtype=bool)
    >>> pos, hsml = sub.gas['pos'], sub.gas['hsml']
    >>> for k in xrange(3):
    ...     mask &= (extent[k,0]<pos[:,k]-hsml) & (pos[:,k]+hsml<extent[k,1])
    >>> lower = sub.gas[mask]['mass'].sum()
    >>> upper = sub.gas['mass'].sum()
    >>> del mask
    >>> if not (lower <= map2D.sum()*np.prod(res) <= upper):
    ...     print lower, map2D.sum(), upper

    Consistency check between 3D and 2D:
    >>> map3D, res = SPH_to_3Dgrid(sub.gas, extent=extent, qty='rho', Npx=Npx)
    create a 30 x 75 x 100 SPH-grid (1200 x 3000 x 4000 [kpc])...
    done with SPH grid
    >>> map3D_proj = map3D.sum(axis=-1) * res.in_units_of('kpc')[2]*Unit('kpc')
    >>> tot_rel_err = (map3D_proj.sum() - map2D.sum()) / map2D.sum()
    >>> if tot_rel_err > 1e-3:
    ...     print tot_rel_err
    >>> px_rel_err = np.abs(map2D-map3D_proj)/map2D
    >>> if np.mean(np.abs(px_rel_err)) > 0.01:
    ...     print np.mean(np.abs(px_rel_err))
    >>> if np.percentile(np.abs(px_rel_err), [99]) > 0.1:
    ...     print np.percentile(np.abs(px_rel_err), [99])

    Quick consistency check of higher resolution
    >>> map2D_high, res = SPH_to_2Dgrid(sub.gas, extent=extent[:2], qty='rho',
    ...                                 Npx=Npx[:2]*4)
    create a 120 x 300 SPH-grid (1200 x 3000 [kpc])...
    done with SPH grid
    >>> tot_rel_err = np.sum(map3D_proj - map2D) / map2D.sum()
    >>> if tot_rel_err > 0.001:
    ...     print tot_rel_err
'''
__all__ = ['SPH_to_3Dgrid', 'SPH_to_2Dgrid', 'SPH_to_2Dgrid_by_particle']

import numpy as np
from ..kernels import *
from ..units import *
from .core import *
from ..utils import *
from .. import environment
from .. import gadget
from .. import C
from numbers import Number
from ..snapshot import BoxMask

def SPH_to_3Dgrid(s, qty, extent, Npx, kernel=None, dV='dV', hsml='hsml',
                  normed=True):
    '''
    Bin some SPH quantity onto a 3D grid, fully accouting for the smoothing
    lengths.

    This method also enshures that the integral over the region of the grid is
    conserved; meaning that no particles can "fall through the grid", not even
    partially.

    Args:
        s (Snap):               The gas-only (sub-)snapshot to bin from.
        qty (UnitQty, str):     The quantity to map. It can be a UnitArr of length
                                L=len(s) and dimension 1 (i.e. shape (L,)) or a
                                string that can be passed to s.get and returns
                                such an array.
        extent (UnitQty):       The extent of the map. It can be a scalar and then
                                is taken to be the total side length of a cube
                                around the origin or a sequence of the minima and
                                maxima of the directions:
                                [[xmin,xmax],[ymin,ymax],[zmin,zmax]].
        Npx (int, sequence):    The number of pixel per side. Either an integer
                                that is taken for all three sides or a 3-tuple of
                                such, each value for one direction.
        kernel (str):           The kernel to use for smoothing. (By default use
                                the kernel defined in `gadget.cfg`.)
        dV (str, UnitQty, Unit):The volume element to use. Can be a block name, a
                                block itself or a Unit that is taken as constant
                                volume for all particles.
        hsml (str, UnitQty, Unit):
                                The smoothing lengths to use. Defined analoguous
                                to dV.

    Returns:
        grid (UnitArr):     The binned SPH quantity.
        px_area (UnitArr):  The area of a pixel.
    '''
    # prepare arguments
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, dim=3)
    extent = UnitQty(extent, s['pos'].units, subs=s)
    res = UnitQty(res, s['pos'].units, subs=s)
    if kernel is None:
        kernel = gadget.general['kernel']

    if res.max()/res.min() > 5:
        import sys
        print >> sys.stderr, 'WARNING: 3D grid has very uneven ratio of ' + \
                             'smallest to largest resolution (ratio %.2g)!' % (
                                     res.max()/res.min())

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'create a %d x %d x %d' % tuple(Npx),
        print 'SPH-grid (%.4g x %.4g x %.4g' % tuple(extent[:,1]-extent[:,0]),
        print '%s)...' % extent.units

    # prepare (sub-)snapshot
    if isinstance(qty, str):
        qty = s.get(qty)
    qty_units = getattr(qty,'units',None).gather()
    if qty.shape!=(len(s),):
        raise ValueError('Quantity has to have shape (N,)!')
    if len(s) == 0:
        return UnitArr(np.zeros(tuple(Npx)), qty_units), res

    if len(s.gas) not in [0,len(s)]:
        raise NotImplementedError()
    sub = s[BoxMask(extent, sph_overlap=True)]

    pos = sub['pos'].view(np.ndarray).astype(np.float64)
    if isinstance(hsml, str):
        hsml = sub[hsml].in_units_of(s['pos'].units)
    elif isinstance(hsml, (Number,Unit)):
        hsml = UnitScalar(hsml,s['pos'].units)*np.ones(len(sub), dtype=np.float64)
    else:   # should be some array
        hsml = UnitQty(hsml,s['pos'].units,subs=s)[sub._mask]
    hsml = hsml.view(np.ndarray).astype(np.float64)
    if isinstance(dV, str):
        dV = sub[dV].in_units_of(s['pos'].units**3)
    elif dV is None:
        dV = (hsml/2.0)**3
    else:
        dV = UnitArr(dV[sub._mask], s['pos'].units**3)
    dV = dV.view(np.ndarray).astype(np.float64)
    qty = qty[sub._mask].view(np.ndarray).astype(np.float64)
    if pos.base is not None:
        pos.copy()
    if hsml.base is not None:
        hsml.copy()
    if dV.base is not None:
        dV.copy()
    if qty.base is not None:
        qty.copy()

    if np.percentile(hsml, 30) > np.min(Npx*res.in_units_of(s['pos'].units)):
        import sys
        print >> sys.stderr, 'WARNING: the 30-percentile (relevant) ' + \
                             'smoothing length is smaller than the smallest ' + \
                             'sidelength of the grid!'

    extent = extent.view(np.ndarray).astype(np.float64).copy()
    grid = np.empty(np.prod(Npx), dtype=np.float64)
    Npx = Npx.astype(np.intp)
    if normed:
        sph_bin = C.cpygad.sph_bin_3D
    else:
        sph_bin = C.cpygad.sph_bin_3D_nonorm
    sph_bin(C.c_size_t(len(sub)),
            C.c_void_p(pos.ctypes.data),
            C.c_void_p(hsml.ctypes.data),
            C.c_void_p(dV.ctypes.data),
            C.c_void_p(qty.ctypes.data),
            C.c_void_p(extent.ctypes.data),
            C.c_void_p(Npx.ctypes.data),
            C.c_void_p(grid.ctypes.data),
            C.create_string_buffer(kernel),
            C.c_double(s.boxsize.in_units_of(s['pos'].units)),
    )
    grid = UnitArr(grid.reshape(tuple(Npx)), qty_units)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'done with SPH grid'

    return grid, res

def SPH_to_2Dgrid(s, qty, extent, Npx, xaxis=0, yaxis=1, kernel=None, dV='dV',
                  hsml='hsml', normed=True):
    '''
    Bin some SPH quantity onto a 2D grid, fully accouting for the smoothing
    lengths.

    This method also enshures that the integral over the region of the grid is
    conserved; meaning that no particles can "fall through the grid", not even
    partially.

    Args:
        s (Snap):               The gas-only (sub-)snapshot to bin from.
        qty (UnitQty, str):     The quantity to map. It can be a UnitArr of length
                                L=len(s) and dimension 1 (i.e. shape (L,)) or a
                                string that can be passed to s.get and returns
                                such an array.
        extent (UnitQty):       The extent of the map. It can be a scalar and then
                                is taken to be the total side length of a square
                                around the origin or a sequence of the minima and
                                maxima of the directions:
                                [[xmin,xmax],[ymin,ymax]].
        Npx (int, sequence):    The number of pixel per side. Either an integer
                                that is taken for both sides or a 2-tuple of such,
                                each value for one direction.
        kernel (str):           The kernel to use for smoothing. (By default use
                                the kernel defined in `gadget.cfg`.)
        dV (str, UnitQty, Unit):The volume element to use. Can be a block name, a
                                block itself or a Unit that is taken as constant
                                volume for all particles.
        hsml (str, UnitQty, Unit):
                                The smoothing lengths to use. Defined analoguous
                                to dV.

    Returns:
        grid (UnitArr):     The binned SPH quantity.
        px_area (UnitArr):  The area of a pixel.
    '''
    # prepare arguments
    zaxis = (set([0,1,2]) - set([xaxis, yaxis])).pop()
    if set([xaxis, yaxis, zaxis]) != set([0,1,2]):
        raise ValueError('Illdefined axes (x=%s, y=%s)!' % (xaxis, yaxis))
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, dim=2)
    extent = UnitQty(extent, s['pos'].units, subs=s)
    res = UnitQty(res, s['pos'].units, subs=s)
    if kernel is None:
        kernel = gadget.general['kernel']

    if res.max()/res.min() > 5:
        import sys
        print >> sys.stderr, 'WARNING: 2D grid has very uneven ratio of ' + \
                             'smallest to largest resolution (ratio %.2g)' % (
                                     res.max()/res.min())

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'create a %d x %d' % tuple(Npx),
        print 'SPH-grid (%.4g x %.4g' % tuple(extent[:,1]-extent[:,0]),
        print '%s)...' % extent.units

    # prepare (sub-)snapshot
    if isinstance(qty, str):
        qty = s.get(qty)
    qty_units = (getattr(qty,'units',None) * s['pos'].units).gather()   # integrated!
    if qty.shape!=(len(s),):
        raise ValueError('Quantity has to have shape (N,)!')
    if len(s) == 0:
        return UnitArr(np.zeros(tuple(Npx)), qty_units), res

    if len(s.gas) not in [0,len(s)]:
        raise NotImplementedError()
    ext3D = UnitArr(np.empty((3,2)), extent.units)
    ext3D[xaxis] = extent[0]
    ext3D[yaxis] = extent[1]
    ext3D[zaxis] = [-np.inf, +np.inf]
    sub = s[BoxMask(ext3D, sph_overlap=True)]

    # TODO: why always need a copy? C vs Fortran alignment...!?
    pos = sub['pos'].view(np.ndarray)[:,(xaxis,yaxis)].astype(np.float64).copy()
    if isinstance(hsml, str):
        hsml = sub[hsml].in_units_of(s['pos'].units)
    elif isinstance(hsml, (Number,Unit)):
        hsml = UnitScalar(hsml,s['pos'].units)*np.ones(len(sub), dtype=np.float64)
    else:   # should be some array
        hsml = UnitQty(hsml,s['pos'].units,subs=s)[sub._mask]
    hsml = hsml.view(np.ndarray).astype(np.float64)
    if isinstance(dV, str):
        dV = sub[dV].in_units_of(s['pos'].units**3)
    elif dV is None:
        dV = (hsml/2.0)**3
    else:
        dV = UnitArr(dV[sub._mask], s['pos'].units**3)
    dV = dV.view(np.ndarray).astype(np.float64)
    qty = qty[sub._mask].view(np.ndarray).astype(np.float64)
    if hsml.base is not None:
        hsml.copy()
    if dV.base is not None:
        dV.copy()
    if qty.base is not None:
        qty.copy()

    extent = extent.view(np.ndarray).astype(np.float64).copy()
    grid = np.empty(np.prod(Npx), dtype=np.float64)
    Npx = Npx.astype(np.intp)
    if normed:
        sph_bin = C.cpygad.sph_3D_bin_2D
    else:
        sph_bin = C.cpygad.sph_3D_bin_2D_nonorm
    sph_bin(C.c_size_t(len(sub)),
            C.c_void_p(pos.ctypes.data),
            C.c_void_p(hsml.ctypes.data),
            C.c_void_p(dV.ctypes.data),
            C.c_void_p(qty.ctypes.data),
            C.c_void_p(extent.ctypes.data),
            C.c_void_p(Npx.ctypes.data),
            C.c_void_p(grid.ctypes.data),
            C.create_string_buffer(kernel),
            C.c_double(s.boxsize.in_units_of(s['pos'].units)),
    )
    grid = UnitArr(grid.reshape(tuple(Npx)), qty_units)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'done with SPH grid'

    return grid, res

def SPH_to_2Dgrid_by_particle(s, qty, extent, Npx, reduction, xaxis=0, yaxis=1,
                              kernel=None, av=None, dV='dV', hsml='hsml'):
    '''
    Reduce a specified quantity of all particles along the third axis within each
    2D pixel of a (projected) grid on a kernel-weighted particle basis.

    TODO

    Args:
        s (Snap):               The gas-only (sub-)snapshot to bin from.
        qty (UnitQty, str):     The quantity to map. It can be a UnitArr of length
                                L=len(s) and dimension 1 (i.e. shape (L,)) or a
                                string that can be passed to s.get and returns
                                such an array.
        extent (UnitQty):       The extent of the map. It can be a scalar and then
                                is taken to be the total side length of a square
                                around the origin or a sequence of the minima and
                                maxima of the directions:
                                [[xmin,xmax],[ymin,ymax]].
        reduction (str):        The reduction method. Available are:
                                * 'mean':   Take the mean of the partivcle
                                            property `qty`.
                                * 'stddev': Take the properties standard deviation
                                            along the line of sight.
        Npx (int, sequence):    The number of pixel per side. Either an integer
                                that is taken for both sides or a 2-tuple of such,
                                each value for one direction.
        kernel (str):           The kernel to use for smoothing. (By default use
                                the kernel defined in `gadget.cfg`.)
        av (UnitQty, str):      Take this quantity to average / weight the
                                peroperty `qty` with.
        dV (str, UnitQty, Unit):The volume element to use. Can be a block name, a
                                block itself or a Unit that is taken as constant
                                volume for all particles.
        hsml (str, UnitQty, Unit):
                                The smoothing lengths to use. Defined analoguous
                                to dV.

    Returns:
        grid (UnitArr):     The binned SPH quantity.
        px_area (UnitArr):  The area of a pixel.
    '''
    # prepare arguments
    zaxis = (set([0,1,2]) - set([xaxis, yaxis])).pop()
    if set([xaxis, yaxis, zaxis]) != set([0,1,2]):
        raise ValueError('Illdefined axes (x=%s, y=%s)!' % (xaxis, yaxis))
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, dim=2)
    extent = UnitQty(extent, s['pos'].units, subs=s)
    res = UnitQty(res, s['pos'].units, subs=s)
    if kernel is None:
        kernel = gadget.general['kernel']

    if res.max()/res.min() > 5:
        import sys
        print >> sys.stderr, 'WARNING: 2D grid has very uneven ratio of ' + \
                             'smallest to largest resolution (ratio %.2g)' % (
                                     res.max()/res.min())

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'create a particle %d x %d' % tuple(Npx),
        print 'SPH-grid (%.4g x %.4g' % tuple(extent[:,1]-extent[:,0]),
        print '%s) (reduction="%s")...' % (extent.units, reduction)

    # prepare (sub-)snapshot
    if isinstance(qty, str):
        qty = s.get(qty)
    qty_units = getattr(qty,'units',None)
    if qty.shape!=(len(s),):
        raise ValueError('Quantity has to have shape (N,)!')
    if isinstance(av, str):
        av = s.get(av)
    if av is not None and av.shape!=(len(s),):
        raise ValueError('Weights quantity (`av`) has to have shape (N,)!')
    if len(s) == 0:
        return UnitArr(np.zeros(tuple(Npx)), qty_units), res

    if len(s.gas) not in [0,len(s)]:
        raise NotImplementedError()
    ext3D = UnitArr(np.empty((3,2)), extent.units)
    ext3D[xaxis] = extent[0]
    ext3D[yaxis] = extent[1]
    ext3D[zaxis] = [-np.inf, +np.inf]
    sub = s[BoxMask(ext3D, sph_overlap=True)]

    # TODO: why always need a copy? C vs Fortran alignment...!?
    pos = sub['pos'].view(np.ndarray)[:,(xaxis,yaxis)].astype(np.float64).copy()
    if isinstance(hsml, str):
        hsml = sub[hsml].in_units_of(s['pos'].units)
    elif isinstance(hsml, (Number,Unit)):
        hsml = UnitScalar(hsml,s['pos'].units)*np.ones(len(sub), dtype=np.float64)
    else:   # should be some array
        hsml = UnitQty(hsml,s['pos'].units,subs=s)[sub._mask]
    hsml = hsml.view(np.ndarray).astype(np.float64)
    if isinstance(dV, str):
        dV = sub[dV].in_units_of(s['pos'].units**3)
    elif dV is None:
        dV = (hsml/2.0)**3
    else:
        dV = UnitArr(dV[sub._mask], s['pos'].units**3)
    dV = dV.view(np.ndarray).astype(np.float64)
    qty = qty[sub._mask].view(np.ndarray).astype(np.float64)
    av = None if av is None else av[sub._mask].view(np.ndarray).astype(np.float64)
    if hsml.base is not None:
        hsml.copy()
    if dV.base is not None:
        dV.copy()
    if qty.base is not None:
        qty.copy()
    if av is not None:
        if av.base is not None:
            av.copy()
    else:
        av = np.ones(len(sub), dtype=np.float64)

    extent = extent.view(np.ndarray).astype(np.float64).copy()
    grid = np.empty(np.prod(Npx), dtype=np.float64)
    Npx = Npx.astype(np.intp)

    if reduction == 'mean':
        bin_sph_reduction = C.cpygad.bin_sph_proj_mean
    elif reduction == 'stddev':
        bin_sph_reduction = C.cpygad.bin_sph_proj_stddev
    else:
        raise ValueError("Unknown reduction method '%s'!" % reduction)

    bin_sph_reduction(C.c_size_t(len(sub)),
                      C.c_void_p(pos.ctypes.data),
                      C.c_void_p(hsml.ctypes.data),
                      C.c_void_p(dV.ctypes.data),
                      C.c_void_p(qty.ctypes.data),
                      C.c_void_p(av.ctypes.data),
                      C.c_void_p(extent.ctypes.data),
                      C.c_void_p(Npx.ctypes.data),
                      C.c_void_p(grid.ctypes.data),
                      C.create_string_buffer(kernel),
                      C.c_double(s.boxsize.in_units_of(s['pos'].units)),
    )
    grid = UnitArr(grid.reshape(tuple(Npx)), qty_units)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'done with particle SPH grid'

    return grid, res

