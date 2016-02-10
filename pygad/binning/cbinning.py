'''
Interface to the C implementations of binning.

Testing:
    TODO! (test if 3D and 2D are consistent; mass conservation; uneven extensions)

TODO:
    * think about proper handing of non-SPH particles. how to give them hsml and
      dV? what if hsml==0 (would give empty map)
    * seperate function for general case?
    * implement pure binning routine without smoothing?
    * add warning about narrow grids (issue of particles that overlap with
      border...!)
    * think about infinitesimal slices
    * finally think about different methods of creating 2D maps: integration,
      projection, average, etc.
'''
__all__ = ['SPH_to_3Dgrid', 'SPH_to_2Dgrid']

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

def SPH_to_3Dgrid(s, qty, extent, Npx, kernel=None, dV='dV', hsml=None):
    '''
    TODO

    Args:
        s (Snap):           The gas-only (sub-)snapshot to bin from.
        qty (UnitQty, str): The quantity to map. It can be a UnitArr of length
                            L=len(s) and dimension 1 (i.e. shape (L,)) or a string
                            that can be passed to s.get and returns such an array.
        extent (UnitQty):   The extent of the map. It can be a scalar and then is
                            taken to be the total side length of a cube around
                            the origin or a sequence of the minima and maxima of
                            the directions: [[xmin,xmax],[ymin,ymax],[zmin,zmax]].
        Npx (int, sequence):The number of pixel per side. Either an integer that
                            is taken for all three sides or a 3-tuple of such,
                            each value for one direction.
        kernel (str):       The kernel to use for smoothing. (By default use the
                            kernel defined in `gadget.cfg`.)
        dV:                 TODO!
        hsml:               TODO!

    Returns:
        grid (UnitArr):     The binned SPH quantity.
        px_area (UnitArr):  The area of a pixel.
    '''
    # prepare arguments
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, res=None, d=3)
    extent = UnitQty(extent, s['pos'].units, subs=s)
    res = UnitQty(res, s['pos'].units, subs=s)
    if kernel is None:
        kernel = gadget.general['kernel']

    if environment.verbose:
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
    sub = s[BoxMask(extent)]

    pos = sub['pos'].view(np.ndarray).astype(np.float64)
    if hsml is None:
        hsml = sub['hsml'].in_units_of(s['pos'].units)
    elif isinstance(hsml, (Number,str,Unit)):
        hsml = UnitScalar(hsml,s['pos'].units)*np.ones(len(sub), dtype=np.float64)
    else:
        hsml = hsml[sub._mask]
    hsml = hsml.view(np.ndarray).astype(np.float64)
    if isinstance(dV, str):
        dV = sub[dV].in_units_of(s['pos'].units**3)
    elif dV is None:
        dV = hsml**3
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

    extent = extent.view(np.ndarray).astype(np.float64).copy()
    grid = np.empty(np.prod(Npx), dtype=np.float64)
    Npx = Npx.astype(np.intp)
    C.cpygad.sph_bin_3D(C.c_size_t(len(sub)),
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

    if environment.verbose:
        print 'done with SPH grid'

    return grid, res

def SPH_to_2Dgrid(s, qty, extent, Npx, xaxis=0, yaxis=1, kernel=None, dV='dV', hsml=None):
    '''
    TODO

    Args:
        s (Snap):           The gas-only (sub-)snapshot to bin from.
        qty (UnitQty, str): The quantity to map. It can be a UnitArr of length
                            L=len(s) and dimension 1 (i.e. shape (L,)) or a string
                            that can be passed to s.get and returns such an array.
        extent (UnitQty):   The extent of the map. It can be a scalar and then is
                            taken to be the total side length of a cube around
                            the origin or a sequence of the minima and maxima of
                            the directions: [[xmin,xmax],[ymin,ymax],[zmin,zmax]].
        Npx (int, sequence):The number of pixel per side. Either an integer that
                            is taken for all three sides or a 3-tuple of such,
                            each value for one direction.
        axis:               TODO!
        kernel (str):       The kernel to use for smoothing. (By default use the
                            kernel defined in `gadget.cfg`.)
        dV:                 TODO!
        hsml:               TODO!

    Returns:
        grid (UnitArr):     The binned SPH quantity.
        px_area (UnitArr):  The area of a pixel.
    '''
    # prepare arguments
    if len(set([0,1,2]) - set([xaxis, yaxis])) != 1:
        raise ValueError('Illdefined axes (x=%s, y=%s)!' % (xaxis, yaxis))
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, res=None, d=2)
    extent = UnitQty(extent, s['pos'].units, subs=s)
    res = UnitQty(res, s['pos'].units, subs=s)
    if kernel is None:
        kernel = gadget.general['kernel']

    if environment.verbose:
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
    ext3D[:2] = extent
    ext3D[2] = [-np.inf, +np.inf]
    sub = s[BoxMask(ext3D)]

    # TODO: why always need a copy? C vs Fortran alignment...!?
    pos = sub['pos'].view(np.ndarray)[:,(xaxis,yaxis)].astype(np.float64).copy()
    if hsml is None:
        hsml = sub['hsml'].in_units_of(s['pos'].units)
    elif isinstance(hsml, (Number,str,Unit)):
        hsml = UnitScalar(hsml,s['pos'].units)*np.ones(len(sub), dtype=np.float64)
    else:
        hsml = hsml[sub._mask]
    hsml = hsml.view(np.ndarray).astype(np.float64)
    if isinstance(dV, str):
        dV = sub[dV].in_units_of(s['pos'].units**3)
    elif dV is None:
        dV = hsml**3
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
    C.cpygad.sph_3D_bin_2D(C.c_size_t(len(sub)),
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

    if environment.verbose:
        print 'done with SPH grid'

    return grid, res

