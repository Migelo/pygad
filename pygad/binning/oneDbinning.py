"""
Binning into some 1D grid (e.g. profiles).

Doctests:
    >>> from ..transformation import *
    >>> from ..snapshot import *
    >>> from mapping import *
    >>> s = Snap('snaps/snap_M1196_4x_470', physical=True)
    >>> Translation(UnitArr([-48087.1,-49337.1,-46084.3],'kpc')).apply(s)
    >>> s['vel'] -= UnitArr([-42.75,-15.60,-112.20],'km s**-1')
    load block vel... done.
    >>> ext = UnitArr( [[-200,200],[-200,200]], 'kpc' )
    >>> m = map_qty(s.gas, ext, True, 'rho', Npx=128) # doctest: +ELLIPSIS
    load block rho... done.
    ...
    done with SPH grid
    >>> prof, r_edges = profile_from_map(m, ext, reduction='mean', Nbins=40)
    >>> np.log10(prof) # doctest: +ELLIPSIS
    UnitArr([ 6.529...,  6.696...,  6.721...,  6.704...,  6.654...,
              6.646...,  6.421...,  5.934...,  5.857...,  5.885...,
              5.876...,  5.810...,  5.770...,  5.750...,  5.714...,
              5.689...,  5.668...,  5.652...,  5.638...,  5.628...,
              5.612...,  5.593...,  5.575...,  5.567...,  5.558...,
              5.549...,  5.539...,  5.529...,  5.519...,  5.516...,
              5.512...,  5.504...,  5.498...,  5.486...,  5.479...,
              5.465...,  5.455...,  5.448...,  5.442...,  5.430...])
"""
__all__ = ['profile_from_map']

import numpy as np
from ..units import UnitArr, UnitQty
from ..utils import dist, weighted_percentile

def profile_from_map(m, extent, av=None, Nbins=None, reduction='sum',
                     ref_center=None, surface_dens=False):
    """
    Create a profile from a 2D grid.

    Args:
        m (array-like):             The map to create the profile from.
        extent (array-like):        The extent of the map.
        av (array-like):            Another map used to weigh the maps values.
        Nbins (int):                The number of radial bins. Defaults to
                                    `min(*m.shape) / 3 + 1`.
        reduction (str):            The method to combine the pixel values into
                                    the bins. Available are:
                                    * 'sum':    Simply sum up the pixel values.
                                    * 'mean':   Take the mean/average of the
                                                pixels.
                                    * 'median': Take the median of the pixels.
                                    * 'std':    Calculate the standard deviation
                                                if the pixels in the radial bin.
        ref_center (array-like):    The reference center, i.e. the point from
                                    which to calculate the radii. Defaults to the
                                    center of the map.
        TODO

    Returns:
        profile (UnitQty):  The profile values for each bin.
        r_edges (UnitQty):  The bin edges.
    """
    # prepare arguments
    m = UnitQty(m)
    if len(m.shape) != 2:
        raise ValueError("The map is not 2D!")
    if av is not None:
        av = UnitQty(av)
        if av.shape != m.shape:
            raise ValueError("The 'average map' has different shape than the "
                             "main map!")
    if Nbins is None:
        Nbins = min(*m.shape) / 3 + 1
    Nbins = int(Nbins)
    extent = UnitQty(extent)
    if extent.shape != (2,2):
        raise ValueError("`extent` has to have shape (2,2)!")
    if ref_center is None:
        ref_center = np.sum(extent,axis=1) / 2.
    ref_center = UnitQty(ref_center, units=extent.units)
    if ref_center.shape != (2,):
        raise ValueError("`ref_center` has to have shape (2,)!")

    # pixel center positions and radii:
    X = np.linspace( extent[0,0], extent[0,1], m.shape[0]+1 )
    Y = np.linspace( extent[1,0], extent[1,1], m.shape[1]+1 )
    X = (X[:-1] + X[1:]) / 2.
    Y = (Y[:-1] + Y[1:]) / 2.
    X,Y = np.meshgrid(X,Y)
    pos = np.swapaxes( np.dstack((X,Y)), 0, 1 )
    del X, Y
    r_px = dist(pos.reshape((-1,2)), ref_center).reshape(pos.shape[:2])

    # radial bins
    #r_edges = np.linspace( r_px.min(), r_px.max(), Nbins+1 )
    r_edges = np.linspace( 0, np.min(extent[:,1]-extent[:,0])/2., Nbins+1 )
    r_edges = UnitArr(r_edges, extent.units)
    #dr = UnitScalar(r_edges[1] - r_edges[0], r_edges.units)
    idx = np.digitize(r_px.flatten(), r_edges).reshape(r_px.shape)

    # reduce map into bins:
    if av is None:
        if reduction in ['sum', 'mean', 'median', 'std']:
            red_func = getattr(np, reduction)
        else:
            raise ValueError("Unknown reduction method '%s'!" % reduction)
        profile = [ red_func(m[idx==i]) for i in xrange(1,Nbins+1) ]
    else:
        if reduction == 'mean':
            red_func = lambda q,a: np.average(q,weights=a)
        elif reduction == 'median':
            red_func = lambda q,a: weighted_percentile(q,[50],weights=a)[0]
        elif reduction in ['sum', 'mean', 'median', 'std']:
            raise ValueError("Cannot use weights with reduction method "
                             "'%s'!" % reduction)
        else:
            raise ValueError("Unknown reduction method '%s'!" % reduction)
        profile = [ red_func(m[idx==i],av[idx==i]) for i in xrange(1,Nbins+1) ]
    profile = UnitArr(profile, m.units)

    if surface_dens:
        ring_area = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)
        profile /= ring_area

    return profile, r_edges

