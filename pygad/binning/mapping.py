'''
Mapping quantities onto a 2D-grid.

Example:
    >>> from ..analysis import *
    >>> from ..transformation import *
    >>> s = Snapshot(module_dir+'snaps/snap_M1196_4x_470', physical=True)
    >>> Translation(UnitArr([-48087.1,-49337.1,-46084.3],'kpc')).apply(s)
    >>> s['vel'] -= UnitArr([-42.75,-15.60,-112.20],'km s**-1')
    load block vel... done.
    >>> orientate_at(s[s['r'] < '10 kpc'].baryons, 'L', total=True)
    load block pos... done.
    apply stored Translation to block pos... done.
    derive block r... done.
    load block mass... done.
    derive block momentum... done.
    derive block angmom... done.
    apply Rotation to "pos" of "snap_M1196_4x_470"... done.
    apply Rotation to "vel" of "snap_M1196_4x_470"... done.
    >>> sub = s[BoxMask('120 kpc', sph_overlap=True)]
    load block hsml... done.
    >>> m_b = map_qty(sub.baryons, '120 kpc', False, 'mass', Npx=256)
    create a 256 x 256 map (120 x 120 [kpc])...
    load block rho... done.
    derive block dV... done.
    create a 256 x 256 SPH-grid (120 x 120 [kpc])...
    done with SPH grid
    >>> m_b # doctest: +ELLIPSIS
    <Map at 0x...; units="Msol", Npx=(256, 256)>
    >>> m_s = map_qty(sub.stars, '120 kpc', False, 'mass', Npx=256)
    create a 256 x 256 map (120 x 120 [kpc])...
    >>> m_g = map_qty(sub.gas, '120 kpc', False, 'mass', Npx=256)
    create a 256 x 256 map (120 x 120 [kpc])...
    create a 256 x 256 SPH-grid (120 x 120 [kpc])...
    done with SPH grid
    >>> rel_err = np.abs(m_b - (m_s+m_g)) / m_b
    >>> if rel_err.max() > 1e-10:
    ...     print(np.percentile(rel_err, [50,90,95,100]))
'''
__all__ = ['map_qty']

from ..units import *
from ..snapshot import *
from .core import *
from ..gadget import *
from ..kernels import *
from .. import environment
import numpy as np

def map_qty(s, extent, field, qty, av=None, reduction=None, Npx=256,
            xaxis=0, yaxis=1, softening=None, sph=True, kernel=None, dV='dV'):
    '''
    A fast routine for binning SPH quantities onto a map.

    Args:
        s (Snap):           The (sub-)snapshot to bin from.
        extent (UnitQty):   The extent of the map. It can be a scalar and then is
                            taken to be the total side length of a square around
                            the origin or a sequence of the minima and maxima of
                            the directions: [[xmin,xmax],[ymin,ymax]]
        field (bool):       If no `reduction` is given, this determines whether
                            the SPH-quantity is interpreted as a density-field or
                            its integral quantity.
                            For instance: rho would be the density-field of the
                            integral quantity mass.
        qty (UnitQty, str): The quantity to map. It can be a UnitArr of length
                            L=len(s) and dimension 1 (i.e. shape (L,)) or a string
                            that can be passed to s.get and returns such an array.
        av (UnitQty, str):  The quantity to average over. Otherwise as 'qty'.
        reduction (str):    If not None, interpret the SPH quantity not as a SPH
                            field, but as a particle property and reduce with this
                            given method along the third axis / line of sight.
                            See `SPH_to_2Dgrid_by_particle` for more information.
        Npx (int, sequence):The number of pixel per side. Either an integer that
                            is taken for both sides or a pair of such, the first
                            for the x-direction, the second for the y-direction.
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
        dV (UnitQty, str):  The volume elements of the particles.

    Returns:
        grid (Map):         The quantity summed along the third axis over the area
                            of a pixel (column -- *not* column density).
    '''
    zaxis = (set([0,1,2]) - set([xaxis, yaxis])).pop()
    if set([xaxis, yaxis, zaxis]) != set([0,1,2]):
        raise ValueError('Illdefined axes (x=%s, y=%s)!' % (xaxis, yaxis))

    if kernel is None:
        kernel = general['kernel']

    if isinstance(qty, str):
        qty = s.get(qty)

    if av is not None:
        if isinstance(av, str):
            av = s.get(av)
        if reduction is None:
            grid = map_qty(s, extent, field, av*qty, av=None, Npx=Npx,
                           xaxis=xaxis, yaxis=yaxis, softening=softening,
                           sph=sph, kernel=kernel, dV=dV)
            norm = map_qty(s, extent, field, av, av=None, Npx=Npx,
                           xaxis=xaxis, yaxis=yaxis, softening=softening,
                           sph=sph, kernel=kernel, dV=dV)
            grid /= norm
            grid[np.isnan(grid)] = 0.0
            return grid

    # prepare arguments
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, dim=2)
    extent = extent.in_units_of(s['pos'].units, subs=s)
    res = res.in_units_of(s['pos'].units, subs=s)
    if softening is not None:
        #softening = UnitArr([0.2, 0.45, 2.52, 20.0, 0.2, 0.2],'ckpc / h_0')
        softening = UnitQty(softening, s['pos'].units, subs=s)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('create a %d x %d map (%.4g x %.4g %s)...' % (tuple(Npx) + \
                (extent[0,1]-extent[0,0],
                 extent[1,1]-extent[1,0], extent.units)))

    grid = np.zeros(Npx)
    if isinstance(qty, UnitArr):
        grid = UnitArr(grid,
                (qty.units*s['pos'].units).gather() if field else qty.units)

    if len(s) == 0:
        return Map(grid, extent=extent)

    if sph:
        sph = s.gas
        sph_qty = qty[s.gas._mask]
        if len(sph) != 0:
            if isinstance(dV, str):
                dV = s.gas.get(dV)
            dV = UnitQty(dV, sph['pos'].units**3)
            if reduction is not None:
                from .cbinning import SPH_to_2Dgrid_by_particle
                sph_binned = SPH_to_2Dgrid_by_particle(sph, qty=sph_qty,
                                                       av=av, dV=dV,
                                                       extent=extent,
                                                       Npx=Npx,
                                                       reduction=reduction,
                                                       xaxis=xaxis,
                                                       yaxis=yaxis,
                                                       kernel=kernel)
            else:
                from .cbinning import SPH_to_2Dgrid
                sph_binned = SPH_to_2Dgrid(sph,
                                           qty = sph_qty if field
                                                       else sph_qty/dV,
                                           extent=extent,
                                           Npx=Npx, xaxis=xaxis, yaxis=yaxis,
                                           kernel=kernel)
                if not field:
                    sph_binned *= sph_binned.vol_voxel()

            grid += sph_binned

    if sph:
        ptypes = list(set(range(6))-set(families['gas']))
    else:
        ptypes = list(range(6))
    proj_kernel = project_kernel(kernel)
    for pt in ptypes:
        sub = s.SubSnap([pt])
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

    return Map(grid, extent=extent)

