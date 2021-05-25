"""
Binning into some 1D grid (e.g. profiles).

Doctests:
    >>> from ..transformation import *
    >>> from ..snapshot import *
    >>> from .mapping import *
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_470', physical=True)
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
    UnitArr([6.52927349, 6.69610836, 6.72196057, 6.70476286, 6.6541627 ,
             6.64614539, 6.42186697, 5.93403278, 5.85748263, 5.88598258,
             5.87697597, 5.81081813, 5.77023788, 5.75011381, 5.71452994,
             5.68998774, 5.66865784, 5.6529112 , 5.63884698, 5.62835883,
             5.61282864, 5.59369405, 5.57579912, 5.56786731, 5.55873967,
             5.54992073, 5.53984417, 5.52930242, 5.51920267, 5.51656575,
             5.51265098, 5.50450026, 5.4986523 , 5.48687964, 5.47931559,
             5.46554878, 5.45591616, 5.44801523, 5.44204756, 5.43046586])
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
    X = np.linspace( float(extent[0,0]), float(extent[0,1]), m.shape[0]+1 )
    Y = np.linspace( float(extent[1,0]), float(extent[1,1]), m.shape[1]+1 )
    X = (X[:-1] + X[1:]) / 2.
    Y = (Y[:-1] + Y[1:]) / 2.
    X,Y = np.meshgrid(X,Y)
    pos = np.swapaxes( np.dstack((X,Y)), 0, 1 )
    del X, Y
    r_px = dist(pos.reshape((-1,2)), ref_center).reshape(pos.shape[:2])

    # radial bins
    #r_edges = np.linspace( float(r_px.min()), float(r_px.max()), Nbins+1 )
    r_edges = np.linspace( 0, float(np.min(extent[:,1]-extent[:,0])/2.), Nbins+1 )
    r_edges = UnitArr(r_edges, extent.units)
    #dr = UnitScalar(r_edges[1] - r_edges[0], r_edges.units)
    idx = np.digitize(r_px.flatten(), r_edges).reshape(r_px.shape)

    # reduce map into bins:
    if av is None:
        if reduction in ['sum', 'mean', 'median', 'std']:
            red_func = getattr(np, reduction)
        else:
            raise ValueError("Unknown reduction method '%s'!" % reduction)
        profile = [ red_func(m[idx==i]) for i in range(1,Nbins+1) ]
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
        profile = [ red_func(m[idx==i],av[idx==i]) for i in range(1,Nbins+1) ]
    profile = UnitArr(profile, m.units)

    if surface_dens:
        ring_area = np.pi * (r_edges[1:]**2 - r_edges[:-1]**2)
        profile /= ring_area

    return profile, r_edges

