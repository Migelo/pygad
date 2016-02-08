'''
A pure-Python way to map a SPH quantity onto a 2D-grid.

Example:
    >>> from ...snapshot import *
    >>> from ...analysis import *
    >>> from ...transformation import *
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
    >>> m, px = map_sph_qty(sub.gas, '120 kpc', 'mass', Npx=150)
    create a 150 x 150 SPH-map (120 x 120 [kpc])...
      do binning + smoothing for hsml < 20 px...
      done.
      calculate pixel center kernel-weighted quantity for
        925 particles with hsml > 16.0 [kpc] (px = 0.8 [kpc])...
      done.
    >>> m_min = s.gas[BallMask('60 kpc',fullsph=False)]['mass'].sum()
    >>> m_max = sub.gas['mass'].sum()
    >>> if not (1-1e-3)*m_min <= m.sum() <= (1+1e-3)*m_max:
    ...     print m_min, m.sum(), m_max

    >>> m, px = map_sph_qty(sub.gas, '120 kpc', 'mass', Npx=150, threshold=5)
    create a 150 x 150 SPH-map (120 x 120 [kpc])...
      do binning + smoothing for hsml < 5 px...
      done.
      calculate pixel center kernel-weighted quantity for
        6100 particles with hsml > 4.0 [kpc] (px = 0.8 [kpc])...
      done.
    >>> if not (1-1e-2)*m_min <= m.sum() <= (1+1e-2)*m_max:
    ...     print m_min, m.sum(), m_max
'''
__all__ = ['map_sph_qty']

import numpy as np
from ...kernels import *
from ...units import *
from ..core import *
from ...utils import *
from ... import environment
from ... import gadget

def map_sph_qty(s, extent, qty, Npx, res=None, xaxis=0, yaxis=1,
                kernel=None, threshold=20):
    '''
    A fast pure-Python routine for binning SPH quantities onto a map.

    Args:
        s (Snap):           The gas-only (sub-)snapshot to bin from.
        extent (UnitQty):   The extent of the map. It can be a scalar and then is
                            taken to be the total side length of a square around
                            the origin or a sequence of the minima and maxima of
                            the directions: [[xmin,xmax],[ymin,ymax]]
        qty (UnitQty, str): The quantity to map. It can be a UnitArr of length
                            L=len(s) and dimension 1 (i.e. shape (L,)) or a string
                            that can be passed to s.get and returns such an array.
        Npx (int, sequence):The number of pixel per side. Either an integer that
                            is taken for both sides or a pair of such, the first
                            for the x-direction, the second for the y-direction.
        res (UnitQty):      The resolution / pixel side length. If this is given,
                            Npx is ignored. It can also be either the same for
                            both, x and y, or seperate values for them.
        xaxis (int):        The coordinate for the x-axis. (0:'x', 1:'y', 2:'z')
        yaxis (int):        The coordinate for the y-axis. (0:'x', 1:'y', 2:'z')
        kernel (str):       The kernel to use for smoothing. (By default use the
                            kernel defined in `gadget.cfg`.)
        threshold (int):    Up to this smoothing length over pixel size (hsml/px),
                            the particles are first binned by hsml/px. For each
                            bin the quantity is simply binned onto the grid by the
                            particles' positions and then the resulting maps are
                            smoothed by the maximum hsml/px with the kernel.
                            For larger hsml/px, the SPH quantity is distributed to
                            the pixels weighted by the kernel value at the pixel
                            center. This is only a good approximation for hsml >>
                            px size and also faster than the kernel convolution in
                            this regime.

    Returns:
        grid (UnitArr):     The binned SPH quantity.
        px_area (UnitArr):  The area of a pixel.
    '''
    # prepare arguments
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, res=res, d=2)
    if isinstance(extent, UnitArr):
        extent = extent.in_units_of(s['pos'].units, subs=s)
    if isinstance(res, UnitArr):
        res = res.in_units_of(s['pos'].units, subs=s)
    if xaxis not in range(3) or yaxis not in range(3) or xaxis==yaxis:
        raise ValueError('The x- and y-axis has to be 0, 1, or 2 and different!')
    if kernel is None:
        kernel = gadget.general['kernel']
    proj_kernel = project_kernel(kernel)
    if threshold <= 0:
        raise ValueError('The smoothing threshold has to be a postive number of '
                         'pixels!')

    if environment.verbose:
        print 'create a %d x %d SPH-map (%.4g x %.4g %s)...' % (tuple(Npx) + \
                (extent[0,1]-extent[0,0],
                 extent[1,1]-extent[1,0], extent.units))

    Npx_w = Npx + [2*threshold]*2
    extent_w = UnitArr( [extent[:,0] - threshold*res,
                         extent[:,1] + threshold*res],
                       extent.units).T

    # prepare (sub-)snapshot
    if isinstance(qty, str):
        qty = s.get(qty)
    if len(s) == 0:
        return UnitArr(np.zeros(Npx), getattr(qty,'units',None)/s['pos'].units**2)

    if environment.verbose:
        print '  do binning + smoothing for hsml < %d px...' % threshold
    hsml_edges = range(threshold+1) * np.min(res)
    # enshure same units for smoothing lengthes (hsml_edges, res, and pos are
    # already consistent):
    hsml = s['hsml'].in_units_of(s['pos'].units)
    grid = np.zeros(Npx_w)
    if isinstance(qty, UnitArr):
        grid = UnitArr(grid, getattr(qty,'units',None))
    for i in xrange(len(hsml_edges)-1):
        mask = (hsml_edges[i]<hsml) & (hsml<=hsml_edges[i+1])
        sub = s[mask]
        if len(sub) == 0:
            continue
        tmp = gridbin(sub['pos'][:,(xaxis,yaxis)], qty[mask], extent=extent_w,
                      bins=Npx_w, nanval=0.0)
        # smooth with roughly sqrt(1/2) times the pixel size more than the maxium
        # smoothing length, since the particles could be in the corners as well
        tmp = smooth(tmp, hsml_edges[i+1]/np.min(res)+0.707, kernel=proj_kernel)
        grid += tmp
    # remove the threshold
    grid = grid[threshold:-threshold,threshold:-threshold].copy()
    if environment.verbose: print '  done.'

    # treat SPH particles with smoothing lengthes larger than hsml_edges[-1]
    mask = hsml_edges[-1]<hsml
    sub = s[mask]
    qty = qty[mask]
    if len(sub):
        if environment.verbose:
            import sys
            print '  calculate pixel center kernel-weighted quantity for'
            print '    %d particles with hsml > %s (px = %s)...' % (len(sub),
                    UnitQty(hsml_edges[-1],s['pos'].units), res.min())
            sys.stdout.flush()
        x = np.linspace(extent[0,0]+res[0]/2., extent[0,1]-res[0]/2., Npx[0])
        y = np.linspace(extent[1,0]+res[1]/2., extent[1,1]-res[1]/2., Npx[1])
        tmp = np.dstack( np.meshgrid(x,y) )
        pxcs = np.empty( (tmp.shape[1],tmp.shape[0],2) )
        pxcs[:,:,0] = tmp[:,:,0].T
        pxcs[:,:,1] = tmp[:,:,1].T
        # make copies for faster access (order of magnitude speed-up)
        pos = sub['pos'][:,(xaxis,yaxis)].view(np.ndarray).copy()
        sub_hsml = sub['hsml'].in_units_of(s['pos'].units).view(np.ndarray).copy()
        qty = qty.view(np.ndarray).copy()
        px_size = float(np.prod(res))
        grid_view = grid.view(np.ndarray)     # faster as well (hopefully)
        for ix in xrange(Npx[0]):
            for iy in xrange(Npx[1]):
                pxc = pxcs[ix,iy]
                d = dist(pos, pxc)
                mask = d < sub_hsml
                masked_hsml = sub_hsml[mask]
                u = d[mask] / masked_hsml
                w = proj_kernel(u) / masked_hsml**2
                grid_view[ix,iy] += px_size * np.sum(w * qty[mask])
        if environment.verbose: print '  done.'

    return grid, np.prod(res)

