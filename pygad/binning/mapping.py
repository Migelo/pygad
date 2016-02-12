'''
Mapping quantities onto a 2D-grid.

Example:
    >>> from ..analysis import *
    >>> from ..transformation import *
    >>> s = Snap('snaps/snap_M1196_4x_470', physical=True)
    >>> Translation(UnitArr([-48087.1,-49337.1,-46084.3],'kpc')).apply(s)    
    >>> s['vel'] -= UnitArr([-42.75,-15.60,-112.20],'km s**-1')
    load block vel... done.
    >>> orientate_at(s[s['r'] < '10 kpc'].baryons, 'L', total=True)
    load block pos... done.
    convert block pos to physical units... done.
    apply stored Translation to block pos... done.
    derive block r... done.
    load block mass... done.
    convert block mass to physical units... done.
    derive block momentum... done.
    derive block angmom... done.
    apply Rotation to "vel" of "snap_M1196_4x_470"... done.
    apply Rotation to "pos" of "snap_M1196_4x_470"... done.
    >>> sub = s[BallMask('60 kpc')]
    derive block r... done.
    load block hsml... done.
    convert block hsml to physical units... done.
    >>> m_b, px2 = map_qty(sub.baryons, '120 kpc', 'mass', Npx=256)
    create a 256 x 256 map (120 x 120 [kpc])...
    load block rho... done.
    convert block rho to physical units... done.
    derive block dV... done.
    create a 256 x 256 SPH-grid (120 x 120 [kpc])...
    done with SPH grid
    >>> m_s, px2 = map_qty(sub.stars, '120 kpc', 'mass', Npx=256)
    create a 256 x 256 map (120 x 120 [kpc])...
    >>> m_g, px2 = map_qty(sub.gas, '120 kpc', 'mass', Npx=256)
    create a 256 x 256 map (120 x 120 [kpc])...
    create a 256 x 256 SPH-grid (120 x 120 [kpc])...
    done with SPH grid
    >>> rel_err = np.abs(m_b - (m_s+m_g)) / m_b
    >>> assert rel_err.max() < 1e-10
'''
__all__ = ['map_qty']

from ..units import *
from ..snapshot import *
from core import *
from ..gadget import *
from ..kernels import *
from .. import environment

def map_qty(s, extent, qty, av=None, Npx=256, res=None, xaxis=0, yaxis=1, softening=None,
            sph=True, kernel=None):
    '''
    A fast pure-Python routine for binning SPH quantities onto a map.

    Args:
        s (Snap):           The (sub-)snapshot to bin from.
        extent (UnitQty):   The extent of the map. It can be a scalar and then is
                            taken to be the total side length of a square around
                            the origin or a sequence of the minima and maxima of
                            the directions: [[xmin,xmax],[ymin,ymax]]
        qty (UnitQty, str): The quantity to map. It can be a UnitArr of length
                            L=len(s) and dimension 1 (i.e. shape (L,)) or a string
                            that can be passed to s.get and returns such an array.
        av (UnitQty, str):  The quantity to average over. Otherwise as 'qty'.
        Npx (int, sequence):The number of pixel per side. Either an integer that
                            is taken for both sides or a pair of such, the first
                            for the x-direction, the second for the y-direction.
        res (UnitQty):      The resolution / pixel side length. If this is given,
                            Npx is ignored. It can also be either the same for
                            both, x and y, or seperate values for them.
        xaxis (int):        The coordinate for the x-axis. (0:'x', 1:'y', 2:'z')
        yaxis (int):        The coordinate for the y-axis. (0:'x', 1:'y', 2:'z')
        softening (UnitQty):A list of te softening lengthes that are taken for
                            smoothing the maps of the paticle types. Is
                            consequently has to have length 6. The entries for gas
                            particle types are ignored, if sph=True.
                            Default: None.
        sph (bool):         If set to True, do not use the softening length for
                            smoothing the SPH particles' quantity, but the actual
                            SPH smoothing lengthes, which differ from particle to
                            particle.
        kernel (str):       The kernel to use for smoothing. (Default: 'kernel'
                            from config file `gadget.cfg`)

    Returns:
        grid (UnitArr):     The quantity summed along the third axis over the area
                            of a pixel (column -- *not* column density).
        px_area (UnitArr):  The area of a pixel.
    '''
    if kernel is None:
        kernel = general['kernel']

    if isinstance(qty, str):
        qty = s.get(qty)

    if av is not None:
        if isinstance(av, str):
            av = s.get(av)
        grid, px2 = map_qty(s, extent, av*qty, av=None, Npx=Npx, res=res,
                            xaxis=xaxis, yaxis=yaxis, softening=softening,
                            sph=sph, kernel=kernel)
        norm, px2 = map_qty(s, extent, av, av=None, Npx=Npx, res=res,
                            xaxis=xaxis, yaxis=yaxis, softening=softening,
                            sph=sph, kernel=kernel)
        grid /= norm
        grid[np.isnan(grid)] = 0.0
        return grid, px2

    # prepare arguments
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, res=res, dim=2)
    if isinstance(extent, UnitArr):
        extent = extent.in_units_of(s['pos'].units, subs=s)
    if isinstance(res, UnitArr):
        res = res.in_units_of(s['pos'].units, subs=s)
    if xaxis not in range(3) or yaxis not in range(3) or xaxis==yaxis:
        raise ValueError('The x- and y-axis have to be 0, 1, or 2 and different!')
    if softening is not None:
        #softening = UnitArr([0.2, 0.45, 2.52, 20.0, 0.2, 0.2],'ckpc / h_0')
        softening = UnitQty(softening, s['pos'].units, subs=s)

    if environment.verbose:
        print 'create a %d x %d map (%.4g x %.4g %s)...' % (tuple(Npx) + \
                (extent[0,1]-extent[0,0],
                 extent[1,1]-extent[1,0], extent.units))

    grid = np.zeros(Npx)
    if isinstance(qty, UnitArr):
        grid = UnitArr(grid, qty.units)

    if len(s) == 0:
        return grid, np.prod(res)

    if sph:
        sph = s.gas
        sph_qty = qty[s.gas._mask]
        if len(sph) != 0:
            from cbinning import SPH_to_2Dgrid
            dV = sph['dV'].in_units_of(sph['pos'].units**3)
            sph_binned, px2 = SPH_to_2Dgrid(sph, qty=sph_qty/dV, extent=extent,
                                            Npx=Npx, xaxis=xaxis, yaxis=yaxis,
                                            kernel=kernel)
            px2 = np.prod(px2)
            sph_binned *= px2

            assert abs((px2 - np.prod(res)) / px2) < 1e-3
            grid += sph_binned

    if sph:
        ptypes = list(set(range(6))-set(families['gas']))
    else:
        ptypes = range(6)
    proj_kernel = project_kernel(kernel)
    for pt in ptypes:
        sub = SubSnap(s, [pt])
        if len(sub) == 0:
            continue

        if softening is not None:
            border = int( min(max(1, np.ceil(softening[pt] / res.min())),
                              0.3 * Npx.max()) )
            Npx_w = Npx + [2*border]*2
            extent_w = UnitArr( [extent[:,0] - border*res,
                                 extent[:,1] + border*res],
                               extent.units).T

            tmp = gridbin(sub['pos'][:,(xaxis,yaxis)], qty[sub._mask], extent=extent_w,
                          bins=Npx_w, nanval=0.0)
            tmp = smooth(tmp, softening[pt] / res.min(), kernel=proj_kernel)
            tmp = tmp[border:-border,border:-border]
        else:
            Npx_w = Npx
            extent_w = UnitArr( [extent[:,0], extent[:,1]],
                               extent.units).T
            tmp = gridbin(sub['pos'][:,(xaxis,yaxis)], qty[sub._mask], extent=extent_w,
                          bins=Npx_w, nanval=0.0)
        grid += tmp

    return grid, np.prod(res)

