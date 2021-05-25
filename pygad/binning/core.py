'''
General image manipulation routines.

Examples:
    >>> for e in grid_props('100 kpc', Npx=100, dim=2): print(e)
    [[-50.  50.]
     [-50.  50.]] [kpc]
    [100 100]
    [1. 1.] [kpc]
    >>> for e in grid_props(UnitArr([[-5,5],[-10,3]],'kpc'), Npx=20): print(e)
    [[ -5.   5.]
     [-10.   3.]] [kpc]
    [20 20]
    [0.5  0.65] [kpc]
    >>> for e in grid_props(UnitArr([[-5,5],[-10,3]],'kpc'), Npx=[20,26]): print(e)
    [[ -5.   5.]
     [-10.   3.]] [kpc]
    [20 26]
    [0.5 0.5] [kpc]
    >>> for e in grid_props(UnitArr([5,10],'kpc'), Npx=100): print(e)
    [[-2.5  2.5]
     [-5.   5. ]] [kpc]
    [100 100]
    [0.05 0.1 ] [kpc]
    >>> for e in grid_props('1 Mpc', Npx=100, dim=3): print(e)
    [[-0.5  0.5]
     [-0.5  0.5]
     [-0.5  0.5]] [Mpc]
    [100 100 100]
    [0.01 0.01 0.01] [Mpc]
    >>> for e in grid_props([1,2], Npx=[100,50]): print(e)
    [[-0.5  0.5]
     [-1.   1. ]]
    [100  50]
    [0.01 0.04]
    >>> grid_props('1 Mpc', Npx=100)
    Traceback (most recent call last):
    ...
    ValueError: Number of dimensions not given and cannot be inferred!
    >>> grid_props([1,2], Npx=[100,50,80])
    Traceback (most recent call last):
    ...
    ValueError: Dimension mismatch of the parameters!

    >>> from ..environment import module_dir
    >>> from ..snapshot import *
    >>> from ..analysis import *
    >>> from ..transformation import *
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_470')
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
    apply Rotation to "pos" of "snap_M1196_4x_470"... done.
    apply Rotation to "vel" of "snap_M1196_4x_470"... done.
    >>> assert np.all(gridbin1d(s['pos'][:,0], qty=s['mass'], extent= [-100,100] )
    ...             == gridbin( s['pos'][:,0].reshape((len(s),1)),
    ...                                        qty=s['mass'], extent=[[-100,100]]))
    >>> sub = s[BoxMask('100 kpc',sph_overlap=False)]
    >>> assert np.all(gridbin2d(sub['pos'][:,0],sub['pos'][:,1])
    ...             == gridbin(sub['pos'][:,(0,1)]))
    >>> m1 = gridbin2d(sub['pos'][:,0], sub['pos'][:,1], sub['mass'], bins=100)
    >>> m2 = gridbin(sub['pos'][:,(0,1)], sub['mass'], bins=100)
    >>> assert np.all(m1==m2)
    >>> mcube = gridbin(sub['pos'], sub['mass'], bins=100)
    >>> assert np.all(m2 == mcube.sum(axis=2))
    >>> m1  # doctest: +ELLIPSIS
    <Map at 0x...; units="1e+10 Msol h_0**-1", Npx=(100, 100)>
    >>> m1.extent
    UnitArr([[-35.99917603,  35.99324417],
             [-35.99769974,  35.9996376 ]])
    >>> m1.Npx
    (100, 100)
    >>> m1.res()
    UnitArr([0.7199242 , 0.71997337])
    >>> m1.vol_voxel()
    UnitArr(0.518326256290704)
    >>> m1.vol_tot()
    UnitArr(5.183263e+03)
    >>> m1 = scale01(m1, np.percentile(m1, [5,95]))
    >>> assert m1.min() >= 0 and m1.max() <= 1
'''
__all__ = ['gridbin2d', 'gridbin1d', 'gridbin', 'grid_props', 'Map', 'scale01', 'smooth']

import numpy as np
from ..units import *
from scipy.stats import binned_statistic_dd
from scipy.ndimage.filters import convolve


def gridbin2d(x, y, qty=None, bins=50, extent=None, normed=False, stats=None,
              nanval=None):
    '''
    Bin data on a 2-dim. grid.

    This calls gridbin with x and y combined to pnts. See gridbin for more
    information!
    '''
    return gridbin(np.array([x, y]).T, qty=qty, bins=bins, extent=extent,
                   normed=normed, stats=stats, nanval=nanval)


def gridbin1d(x, qty=None, bins=50, extent=None, normed=False, stats=None,
              nanval=None):
    '''
    Bin data 1-dimensional.

    This calls gridbin. See gridbin for more information!
    '''
    extent = np.asarray(extent)
    if extent.shape == (2,):
        extent = extent.reshape((1, 2))
    return gridbin(np.array(x).reshape((len(x), 1)), qty=qty, bins=bins, extent=extent,
                   normed=normed, stats=stats, nanval=nanval)


def gridbin(pnts, qty=None, bins=50, extent=None, normed=False, stats=None,
            nanval=None):
    '''
    Bin data on a grid.

    It can used to speed-up scatter plots, for instance.

    It basically is a wrapper for binned_statistic_dd for a little more
    convenience and for adding units.

    Args:
        pnts (array-like):  An (N,D)-array of the points to bin.
        qty (UnitArr):      Values for the points to bin. If None, points are just
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
                            Default: 'count' if `qty` is None else 'sum'
        nanval (value):     All points where the grid is NaN are set to this
                            value.

    Returns:
        gridded (Map):      The (N,...,N)-array of the binned data.
    '''
    known_stats = ['count', 'sum', 'mean', 'median']
    if isinstance(stats, str) and stats not in known_stats:
        raise ValueError('Unknown statistic. Choose from: %s' % known_stats)
    pnts = np.asanyarray(pnts)
    if len(pnts.shape) != 2:
        raise ValueError('The points array has to have shape (N,D)!')

    if qty is None:
        if stats is None or stats == 'count':
            stats = 'count'
            qty = np.ones(len(pnts), int)
    else:
        if stats is None:
            stats = 'sum'
        qty = np.asanyarray(qty)

    if isinstance(extent, UnitArr) and isinstance(pnts, UnitArr):
        extent = extent.in_units_of(pnts.units)

    gridded, edges, binnum = binned_statistic_dd(pnts, qty, range=extent,
                                                 statistic=stats, bins=bins)
    if extent is None:
        extent = np.array([[e[0], e[-1]] for e in edges])

    gridded = UnitArr(gridded)
    # if the values to bin have units, the result should as well
    if isinstance(qty, UnitArr) and not stats == 'count':
        if isinstance(stats, str):
            gridded.units = qty.units
        else:
            if stats in UnitArr._ufunc_registry:
                gridded.units = UnitArr._ufunc_registry[ufunc](qty)
            else:
                import warnings
                warnings.warn('Operation \'%s\' on units is ' % stats.__name__ + \
                              '*not* defined! Return normal numpy array.')
                gridded = gridded.view(np.ndarray)

    if normed:
        gridded /= gridded.sum()

    if nanval is not None:
        gridded[np.isnan(gridded)] = nanval

    return Map(gridded, extent=extent)


def grid_props(extent, Npx=256, dim=None):
    '''
    Calculate the grid properties from given values.

    Args:
        extent (UnitQty):   This can either be a scalar, it then defines the
                            (total) with of the grid, a list of widths for each
                            dimension, or a full extent of a sequence of maximim
                            and minimum for all coordinates:
                            [[x1min,x1max],[x2min,x2max],...].
        Npx (int, sequence):The number of pixels per side. It can either be a
                            single value that is taken for all sides or a tuple
                            with values for each direction.
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
    if dim is None:
        if extent is not None and extent.shape != ():
            dim = len(extent)
        elif Npx is not None and Npx.shape != ():
            dim = len(Npx)
        else:
            raise ValueError('Number of dimensions not given and cannot be inferred!')
    for v in (extent, Npx):
        if v is not None and v.ndim != 0:
            if len(v) != dim:
                raise ValueError('Dimension mismatch of the parameters!')

    if extent.shape == ():
        widths = UnitQty([extent] * dim, getattr(extent, 'units', None))
        w2 = extent / 2.
        extent = UnitArr([[-w2, w2]] * dim, getattr(w2, 'units', None))
    elif extent.ndim == 1:
        widths = extent
        w2s = extent / 2.
        extent = UnitArr([[-w2, w2] for w2 in w2s], getattr(w2s, 'units', None))
    else:
        extent = extent.reshape((dim, 2,))
        widths = UnitArr(extent[:, 1] - extent[:, 0])
    # now extent is in final form and widths is defined

    if Npx.shape == ():
        Npx = np.array([Npx] * dim)
    # now also Npx is in final form (of shape (dim,))

    res = widths / Npx

    return extent, Npx, res


class Map(UnitArr):
    """
    An object of a grid with extent and resolution. It inherits from `UnitArr`.

    Args:
        grid (UnitQty):     The grid itself. Does not need to have the correct
                            shape yet.
        extent (UnitQty):   This can either be a scalar, it then defines the
                            (total) with of the grid, a list of widths for each
                            dimension, or a full extent of a sequence of maximim
                            and minimum for all coordinates:
                            [[x1min,x1max],[x2min,x2max],...].
        Npx (int, array-like):
                            If not None, the passed `grid` will be reshaped
                            accordingly. `extent` and `Npx` will be passed to
                            `grid_props`.
        keyword arguments:  Passed on to the array factory function of numpy
                            (np.array). By default this includes `copy=False`.
    """

    def __new__(cls, grid, extent, Npx=None, **kwargs):
        if 'copy' not in kwargs:
            kwargs['copy'] = False
        m = UnitArr(grid, **kwargs).view(cls)
        m._extent, Npx, res = grid_props(extent=extent,
                                         Npx=m.shape if Npx is None else Npx)

        Npx = tuple(Npx)
        if m.shape != Npx:
            m = m.reshape(Npx)

        return m

    def __array_finalize__(self, obj):
        UnitArr.__array_finalize__(self, obj)
        self._extent = getattr(obj, '_extent', [[0, 0]] * len(self.shape))

    def __array_wrap__(self, array, context=None):
        return UnitArr.__array_wrap__(self, array, context)

    def __getitem__(self, key):
        from warnings import simplefilter
        # ignore all future warnings
        simplefilter(action='ignore', category=FutureWarning)
        item = super(Map, self).__getitem__(key)
        # the warning: FutureWarning: Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of `arr[seq]`. In the future this will be interpreted as an array index, `arr[np.array(seq)]`, which will result either in an error or a different result.
        # is handles by pagad in the following code
        if item.ndim != self.ndim:
            if (isinstance(key, np.ndarray) and key.dtype == bool) or \
                    (not isinstance(key, slice) and
                     all(isinstance(k, bool) for k in key)):
                # TODO: handle: boolean masking, that turned out to mask out a dim'
                pass
            else:
                # TODO: how to handle np.newaxis/None?
                if not isinstance(key, (slice, tuple, list, np.ndarray)):
                    raise NotImplementedError("key/index is not of type 'tuple',"
                                              " but '%s'" % type(key))
                take = [isinstance(s, (slice, tuple, list, np.ndarray)) for s in key] + \
                       [True] * (self.ndim - len(key))
                item._extent = self._extent[take]
        return item

    def __repr__(self):
        r = self.view(UnitArr).__repr__(val='')
        r = r.replace('UnitArr(,', '<Map at 0x%x;' % id(self))
        r = r[:-1] + ', Npx=%s>' % (self.Npx,)
        # V = self.vol_tot()
        # r = r[:-1] + ', V=%g %s>' % (V, V.units)
        return r

    def __str__(self):
        s = self.view(UnitArr).__str__()
        s += ' (Npx=%s)' % (self.Npx,)
        return s

    @property
    def grid(self):
        """Just the underlying `UnitArr`."""
        return self.view(UnitArr)

    @property
    def Npx(self):
        """The side lengths in pixels in all dimensions."""
        return self.shape

    @property
    def extent(self):
        """The grid's extent in all dimensions."""
        return self._extent

    def res(self, axis=None, units=None, subs=None):
        """Resolution: pixels size in all the dimensions."""
        extent, Npx, res = grid_props(extent=self.extent, Npx=self.Npx)
        if units:
            res.convert_to(units, subs=subs)
        if axis is None:
            return res
        else:
            return UnitArr(res[axis], res.units)

    def vol_voxel(self):
        """The voxel volume."""
        return np.prod(self.res())

    def vol_tot(self):
        """The total grid volume."""
        return np.prod(self.extent.ptp(axis=1))

    def __copy__(self, *a):
        if a:
            duplicate = UnitArr.__copy__(self, *a).view(Map)
        else:
            duplicate = UnitArr.__copy__(self).view(Map)
        duplicate._extent = self._extent
        return duplicate

    def __deepcopy__(self, *a):
        duplicate = UnitArr.__deepcopy__(self).view(Map)
        duplicate._extent = self._extent
        return duplicate


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
    arr[arr > 1.0] = 1.0
    arr[arr < 0.0] = 0.0
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

    pxs = int(2 * np.ceil(sml) + 1)
    x = np.linspace(float(-(pxs - 1) / 2.), float((pxs - 1) / 2.), pxs) / sml
    D = len(grid.shape)
    x = np.meshgrid(*(x,) * D)
    x = np.array(x)
    dists = np.sqrt(np.sum(x ** 2, axis=0))
    conv_grid = kernel(dists.ravel())
    conv_grid = conv_grid.reshape((pxs, pxs))
    conv_grid /= np.sum(conv_grid)
    smooth = convolve(grid, conv_grid, mode=bndrymode)

    if isinstance(grid, UnitArr):
        smooth = UnitArr(smooth, grid.units)
    return smooth

