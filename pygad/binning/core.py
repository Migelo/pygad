'''
General image manipulation routines.

Examples:
    >>> from ..environment import module_dir
    >>> from ..snapshot import *
    >>> from ..analysis import *
    >>> from ..transformation import *
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_470')
    >>> Translation(UnitArr([-48087.1,-49337.1,-46084.3],'kpc')).apply(s)
    >>> s['vel'] -= mass_weighted_mean(s[s['r']<'1 kpc'], 'vel')
    load block vel... done.
    load block pos... done.
    apply stored Translation to block pos... done.
    derive block r... done.
    load block mass... done.
    >>> orientate_at(s[s['r'] < '10 kpc'].baryons, 'L', total=True)
    derive block momentum... done.
    derive block angmom... done.
    apply Rotation to "vel" of "snap_M1196_4x_470"... done.
    apply Rotation to "pos" of "snap_M1196_4x_470"... done.
    >>> sub = s[BoxMask('100 kpc',fullsph=False)]
    >>> assert np.all(gridbin2d(sub['pos'][:,0],sub['pos'][:,1])
    ...             ==gridbin(sub['pos'][:,(0,1)]))
    >>> m1 = gridbin2d(sub['pos'][:,0], sub['pos'][:,1], sub['mass'], bins=100)
    >>> m2 = gridbin(sub['pos'][:,(0,1)], sub['mass'], bins=100)
    >>> assert np.all(m1==m2)
    >>> mcube = gridbin(sub['pos'], sub['mass'], bins=100)
    >>> assert np.all(m2 == mcube.sum(axis=2))
    >>> m1 = scale01(m1, np.percentile(m1, [5,95]))
    >>> assert m1.min() >= 0 and m1.max() <= 1
'''
__all__ = ['gridbin2d', 'gridbin', 'grid_props', 'scale01', 'smooth']

import numpy as np
from ..units import *
from scipy.stats import binned_statistic_dd
from scipy.ndimage.filters import convolve

def gridbin2d(x, y, vals=None, bins=50, extent=None, normed=False, stats=None,
              nanval=None):
    '''
    Bin data on a 2-dim. grid.

    This calls gridbin with x and y combined to pnts. See gridbin for more
    information!
    '''
    return gridbin(np.array([x,y]).T, vals=vals, bins=bins, extent=extent,
                   normed=normed, stats=stats, nanval=nanval)

def gridbin(pnts, vals=None, bins=50, extent=None, normed=False, stats=None,
            nanval=None):
    '''
    Bin data on a grid.
    
    It can used to speed-up scatter plots, for instance.

    It basically is a wrapper for binned_statistic_dd for a little more
    convenience and for adding units.

    Args:
        pnts (array-like):  An (N,D)-array of the points to bin.
        vals (UnitArr):     Values for the points to bin. If None, points are just
                            counted per bin.
        bins (int, array-like):
                            Number of bins per dimension.
        extent (secquence): The range for the grid. A squence of pairs of the
                            minimum and maximum for each dimension.
                            If both this and pnts are UnitArr's, it is taken care
                            of the units.
        normed (True):      Whether to norm the gridded quantity to one.
        stats (str, function):
                            A function to apply at the end. The predefined ones
                            are: 'count', 'sum', 'mean', 'median'. For more
                            information see binned_statistic_dd.
                            Default: 'count' if vals is None else 'sum'
        nanval (value):     All points where the grid is NaN are set to this
                            value.

    Returns:
        gridded (UnitArr):  The (N,...,N)-array of the binned data.
    '''
    known_stats = ['count', 'sum', 'mean', 'median']
    if isinstance(stats,str) and stats not in known_stats:
        raise ValueError('Unknown statistic. Choose from: %s' % known_stats)
    pnts = np.asanyarray(pnts)
    if len(pnts.shape) != 2:
        raise ValueError('The points array has to have shape (N,D)!')

    if vals is None:
        if stats is None or stats=='count':
            stats = 'count'
        else:
            vals = np.ones(len(pnts),int)
    else:
        if stats is None:
            stats = 'sum'
        vals = np.asanyarray(vals)

    if isinstance(extent,UnitArr) and isinstance(pnts,UnitArr):
        extent = extent.in_units_of(pnts.units)

    gridded, edges, binnum = binned_statistic_dd(pnts, vals, range=extent,
                                                 statistic=stats, bins=bins)

    gridded = UnitArr(gridded)
    # if the values to bin have units, the result should as well
    if isinstance(vals,UnitArr) and not stats=='count':
        if isinstance(stats,str):
            gridded.units = vals.units
        else:
            if stats in UnitArr._ufunc_registry:
                gridded.units = UnitArr._ufunc_registry[ufunc](vals)
            else:
                warnings.warn('Operation \'%s\' on units is ' % stats.__name__ + \
                              '*not* defined! Return normal numpy array.')
                gridded = gridded.view(np.ndarray)

    if normed:
        gridded /= gridded.sum()

    if nanval is not None:
        gridded[np.isnan(gridded)] = nanval

    return gridded

def grid_props(extent, Npx=256, res=None, dim=None):
    '''
    Calculate the grid properties from given values.

    Args:
        extent (UnitQty):   This can either be a scalar, it then defines the
                            (total) with of the grid; or a full extent of a
                            sequence of maximim and minimum for all coordinates:
                            [[x1min,x1max],[x2min,x2max],...].
        Npx (int, sequence):The number of pixels per side. It can either be a
                            single value that is taken for all sides or a tuple
                            with values for each direction.
        res (UnitQty):      The resolution / pixel side length. If this is given,
                            Npx is ignored. It can also be either the same for
                            all directions or a seperate values for each of them.
        dim (int):          The number of dimensions of the grid. If possible, it
                            will be inferred from the arguments, if it is None.

    Returns:
        extent (UnitQty):   The extent as a (dim,2)-array:
                            [[x1min,x1max],[x2min,x2max],...]
        Npx (sequence):     The number of pixels per side: [<in x1>,<in x2>,...].
        res (UnitQty):      The resolution in the different directions. It might
                            differ from the one passed (if passed), since the
                            numbers of pixels have to be intergers.
    '''
    extent = UnitQty(extent, dtype=float)
    Npx = np.array(Npx, dtype=int) if Npx is not None else None
    res = UnitQty(res, getattr(res,'units',None), dtype=float)
    if dim is None:
        if extent is not None and extent.shape != ():
            dim = len(extent)
        elif Npx is not None and Npx.shape != ():
            dim = len(Npx)
        elif res is not None and res.shape != ():
            dim = len(res)
        else:
            ValueError('Number of dimensions not given and cannot be inferred!')

    if extent.shape == ():
        widths = UnitQty([extent]*dim, getattr(extent,'units',None))
        w2 = extent / 2.
        extent = UnitArr([[-w2,w2]]*dim, getattr(w2,'units',None))
    else:
        extent = extent.reshape((dim,2,))
        widths = UnitArr(extent[:,1]-extent[:,0])

    if res is None:
        if Npx.shape == ():
            Npx = np.array([Npx]*dim)
    else:
        if res.shape==():
            res = np.array([res]*dim, res.units)
        Npx = np.ceil( widths / res ).astype(int)
    res = widths / Npx

    return extent, Npx, res

def scale01(arr, lims):
    '''
    Scale the values in an array to the interval [0,1].

    Linear map onto [0,1] with the given limits. Values beyond the limits are
    mapped to the borders.

    Args:
        arr (array-like):   The array to scale.
        lims (sequence):    The minimum and maximum value to scale to.

    Returns:
        arr (array-like):   The scaled array.
    '''
    arr = (arr - float(lims[0])) / (lims[1] - lims[0])
    arr[arr>1.0] = 1.0
    arr[arr<0.0] = 0.0
    return arr

def smooth(grid, sml, kernel, bndrymode='constant'):
    '''
    Smooth a gridded quantity (e.g. an image).

    Args:
        grid (array-like):      The grid to smooth. Can be a UnitArr and stays
                                one.
        sml (float):            The number of pixels to smooth over (a square
                                array is used for convolution).
        kernel (vector-function):
                                The kernel to use for smoothing. It is a function
                                of one argument (the radius in px/sml) and does
                                not have to be normed (is gets normed in here).
                                It has to be a vector function, i.e. be able to
                                operate on entire arrays.
        bndrymode (str):        How to handle the boundaries. See e.g.
                                scipy.ndimage.filters.convolve for more
                                information.

    Returns:
        smooth (array-like):    The smoothed grid. It preserves units.
    '''
    if sml < 0:
        raise ValueError('The smoothing length has to be positive or zero!')
    if sml == 0:
        return grid

    sml = float(sml)

    from ..kernels import kernels, vector_kernels

    pxs = int(2*np.ceil(sml)+1)
    x = np.linspace(-(pxs-1)/2., (pxs-1)/2., pxs) / sml
    D = len(grid.shape)
    x = np.meshgrid(*(x,)*D)
    x = np.array(x)
    dists = np.sqrt(np.sum(x**2,axis=0))
    conv_grid = kernel(dists.ravel())
    conv_grid = conv_grid.reshape((pxs,pxs))
    conv_grid /= np.sum(conv_grid)
    smooth = convolve(grid, conv_grid, mode=bndrymode)

    if isinstance(grid,UnitArr):
        smooth = UnitArr(smooth,grid.units)
    return smooth

