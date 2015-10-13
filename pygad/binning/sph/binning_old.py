'''
Module for binning SPH properties onto grids.

Doctests:
    >>> from ..snapshot import Snap
    >>> from ..environment import module_dir
    >>> s = Snap(module_dir+'../test_sim/M1196/4x-Michi/out/snap_M1196_4x_320')
    >>> s.physical_units()
    >>> com = center_of_mass(s.gas)
    load block pos... done.
    load block mass... done.
    >>> gas_dists = utils.periodic_distance_to(s.gas.pos, com, s.boxsize)
    >>> gas_dists
    UnitArr([   42.52318051,    36.81107817,    36.46432429, ...,  2768.26857024,
              4452.89189233,  4193.30619816], units='kpc')
    >>> center = shrinking_sphere(s,
    ...                           R=np.percentile(gas_dists,98)/2.,
    ...                           center=com)
    >>> center
    UnitArr([ 33825.69629789,  34596.43666589,  32683.99256117], units='kpc')
    >>> R200, M200 = virial_info(s, center)
    >>> s.pos -= center
    >>> sub = s[np.abs(s.pos).sum(axis=1) < 50]
    >>> dens_map = density_map(sub.gas, [-50, 50], [-50, 50], 'mass',
    ...                        Q=1.0, N_sample_min=30, N_sample_max=300, N_px=100,
    ...                        verbose=False)
    load block 'HSML'...
    done.
    >>> assert (sub.gas.mass.sum() / dens_map.sum() -
    ...             Unit('1 kpc**2')) < 0.01
    >>> import doctest
    >>> doctest.ELLIPSIS_MARKER = '<IGNORE>'
    >>> dens_cube = density_cube(sub.gas, [-50, 50], [-50, 50], [-50, 50], 'mass',
    ...                          Q=1.0**(2./3.), N_sample_min=30, N_sample_max=300,
    ...                          N_px=100, verbose=True)  # doctest:+ELLIPSIS
    create density cube with Q=1
    particles are binned by hsml / pixel size into 6 bins:
    bin |   hsml/px.size  | particles | dots each |   dots total 
    ----+-----------------+-----------+-----------+--------------
      1 |  0.16 -     4.7 |    27,652 |        30 |      829,560
      2 |   4.7 -     6.2 |     2,779 |        37 |      102,823
      3 |   6.2 -     8.1 |     3,376 |        66 |      222,816
      4 |   8.1 -      11 |     3,550 |       115 |      408,250
      5 |    11 -      14 |     1,978 |       201 |      397,578
      6 |    14 -      20 |       187 |       300 |       56,100
    ----+-----------------+-----------+-----------+--------------
        |  0.16 -      20 |    39,522 |           |    2,017,127
    <BLANKLINE>
                      0% |                   50% |                   100% |
    particles in bin # 1 ..................................................
    particles in bin # 2 ..................................................
    particles in bin # 3 ..................................................
    particles in bin # 4 ..................................................
    particles in bin # 5 ..................................................
    particles in bin # 6 ..................................................
    creating the density cube took <IGNORE> sec

    For this line, I use that the sidelength of a pixel is 1 kpc.
    Check that the density maps from density_map and density_cube differ not
    too much in log-space.
    >>> dens_map_fromcube = dens_cube.sum(axis=2)*Unit('kpc')
    >>> dens_map          = np.log10(dens_map+1)
    >>> dens_map_fromcube = np.log10(dens_map_fromcube+1)
    >>> dens_diff = ( (dens_map - dens_map_fromcube) /
    ...               ((dens_map + dens_map_fromcube)/2.0) ).abs()
    >>> assert np.percentile(dens_diff.ravel(), 99) < 0.05
'''
__all__ = ['density_cube', 'density_map']

"""
from snapshot import Snap
from units import UnitArr, dist
import quantities
import numpy as np
import scipy.constants
import utils
import kernels
import units
import sys
import time
"""

def _hsml_binning(snap, Q_, px_size, N_sample_min, N_sample_max, verbose):
    '''
    Helper function for binning the particles according to hsml / pixel size.

    Args:
        snap (Snap):        The snapshot.
        Q_ (float):         The already to pixel size adapted quality parameter Q.
        px_size (float):    The smallest side of a pixel.
        N_sample_min, N_sample_max (int):
                            The limits for the number of pseudo-particles per
                            particle for the bins.
        verbose (bool):     Verbosity.

    Returns:
        hsml (np.ndarray):      The smoothing lengthes.
        hsml_px (np.ndarray):   hsml over pixel size.
        hsml_px_edges (list):   The edges (in hsml/px) of the bins.
        samples (list):         The number of pseudo-particles of the bins.
    '''
    hsml = snap.hsml.in_units_of(snap.pos.units).view(np.ndarray)
    hsml_px = hsml/px_size

    # create bins; equidistant in log-space between 0.5 and 100,
    # and optionally a bin above and below each, if there are smoothing
    # lengthes outside
    hsml_px_defmax = 100
    hsml_px_edges = list(np.logspace(np.log10(0.5), np.log10(hsml_px_defmax),
                                     num=20))
    hsml_px_lim = [hsml_px.min(), hsml_px.max()]
    if hsml_px_edges[0] > hsml_px_lim[0]:
        hsml_px_edges = [0.999*hsml_px_lim[0]] + hsml_px_edges
    if hsml_px_edges[-1] < hsml_px_lim[-1]:
        hsml_px_edges = hsml_px_edges + [1.001*hsml_px_lim[-1]]

    # create sampling number for each bin according to Q_ and
    # N_sample_[min,max]
    def h_to_points(h):
        # for particles with hsml/px.size around 1, increase the sampling a bit
        return (h + np.exp(-((h-1.)/2.)**2))**2
    samples = [Q_*h_to_points(h) for h in hsml_px_edges[1:]]
    samples = [int(max(N_sample_min,min(n,N_sample_max))) for n in samples]

    # count particles within the bins
    N_bin = []
    for i_bin in xrange(len(samples)):
        limits = [hsml_px_edges[i_bin], hsml_px_edges[i_bin+1]]
        N_bin.append( np.sum((hsml_px >= limits[0]) & (hsml_px < limits[1])) )

    # merge bins with same sample size
    i = 0
    while i < len(samples)-1:
        while i<len(samples)-1 and samples[i+1]==samples[i]:
            samples = samples[:i] + samples[i+1:]
            hsml_px_edges = hsml_px_edges[:i+1] + hsml_px_edges[i+2:]
            N_bin = N_bin[:i] + [N_bin[i]+N_bin[i+1]] + N_bin[i+2:]
        i += 1
    # remove empty bins
    i = 0
    while i < len(samples):
        while i<len(samples) and N_bin[i]==0:
            samples = samples[:i] + samples[i+1:]
            hsml_px_edges = hsml_px_edges[:i+1] + hsml_px_edges[i+2:]
            N_bin = N_bin[:i] + N_bin[i+1:]
        i += 1
    # for better output
    if hsml_px_edges[-1] == hsml_px_defmax:
        hsml_px_edges[-1] = 1.001*hsml_px_lim[-1]

    N_parts = len(hsml)

    # print some information about the binning
    if verbose:
        print 'particles are binned by hsml / pixel size into %d bins:' % len(samples)
        print 'bin |   hsml/px.size  | particles | dots each |   dots total '
        print '----+-----------------+-----------+-----------+--------------'
        bin_num = 0
        N_sample_tot = 0
        for i_bin in xrange(len(samples)):
            N_sample = samples[i_bin]
            limits = [hsml_px_edges[i_bin], hsml_px_edges[i_bin+1]]
            bin_num += 1
            mask = (hsml_px >= limits[0]) & (hsml_px < limits[1])
            N_parts_bh = np.sum(mask)
            N_sample_tot += N_parts_bh*N_sample
            print '%3d | %5.2g - %7.2g | %9s | %9d | %12s' % (
                    bin_num, limits[0], limits[1],
                    utils.nice_big_num_str(N_parts_bh), N_sample,
                    utils.nice_big_num_str(N_parts_bh*N_sample))
        print '----+-----------------+-----------+-----------+--------------'
        print '    | %5.2g - %7.2g | %9s |           | %12s' % (
                hsml_px_edges[0], hsml_px_edges[-1],
                utils.nice_big_num_str(N_parts),
                utils.nice_big_num_str(N_sample_tot))
        print
        print '                  0% |' + (' '*19) + '50% |' + (' '*19) + \
                '100% |'
        sys.stdout.flush()

    return hsml, hsml_px, hsml_px_edges, samples

def density_cube(snap, xlim, ylim, zlim, quantity='mass', average='none',
                 N_px=256, Q=0.3, N_sample_min=10, N_sample_max=1000,
                 return_average_binned=False,
                 kernel='Wendland C4', verbose=True):
    '''
    Create a 3-dim. UnitArr / cube of the specified quantity, accounting for SPH
    smoothing, if only gas is mapped.

    Args:
        snap (Snap):        The Snapshot to use.
        xlim (array-like):  The limits of the x-axis.
        ylim (array-like):  The limits of the y-axis.
        zlim (array-like):  The limits of the z-axis.
        quantity (str, UnitArr):
                            The quantity to calculate the density of. It can
                            either be the block itself (make shure it already has
                            the appropiate shape and size), the name of a block
                            to get from the snapshot or a expression that can be
                            passed to Snap.get.
        average ('none', 'particles', block):
                            'none':     a simple map of the surface density is
                                        created. Here it is also devided by pixel
                                        area, i.e. the result is a surface
                                        density.
                            'particle': it is averaged over the particles along
                                        the line of sight. Each particle has the
                                        same weighting.
                            block:      it has to be the name of a block, a
                                        string that is evaluable / can be passed
                                        to Snap.get() (e.g. 'Oxygen/mass'), or a
                                        np.ndarray/UnitArr. It is then averaged
                                        with the weighting of this block.
        N_px (int):         The number of pixels per side (the result will be a
                            (N_px x N_px x N_px)-array). Plotting time depends
                            quadratically on the number of pixels in case of
                            plotting gas, since the quality of the
                            Monte-Carlo-integration over the kernel depends on
                            pixel size. (See also parameter Q.) For other
                            particle species, however, there is almost no
                            dependence.
        Q (float):          A positive value that controls the quality of gas
                            quantity plotting.
                            If only gas particles are plotted, it is taken care
                            of the smoothing lengthes with a Monte-Carlo-like
                            integration. The quality of this integration is
                            controlled by Q. Be careful with large values
                            though: the plotting time goes quadratic with Q and
                            Q=1 is already good quality.
        N_sample_min (int): The minimum number of sampling per gas particle.
        N_sample_max (int): The maximum number of sampling per gas particle.
        return_average_binned (bool):
                            Whether to also return the quantity to average over
                            binned onto the cube.
        kernel (str):       The kernel to use to sample the gas particles.
        verbose (bool):     Whether or not to print which sample is mapped at the
                            moment.

    Returns:
        dcube (UnitArr):    The (N_px x N_px x N_px)-UnitArr cube of the densities.
       [normcube (UnitArr): The (N_px x N_px x N_px)-UnitArr cube of the quantity
                            over which it was averaged.]
    '''
    if verbose: print 'create density cube with Q=%g' % Q
    start_time = time.time()
    single_dcube = np.empty((N_px+2, N_px+2, N_px+2))   # add rows for off the grid!
    def single_density_cube(x, y, z, quantity, xbins, ybins, zbins, N_px):
        '''helper function'''
        x = np.digitize(x, xbins)
        y = np.digitize(y, ybins)
        z = np.digitize(z, zbins)
        N = len(quantity)
        single_dcube[:] = 0.0
        for i in xrange(N):
            single_dcube[x[i], y[i], z[i]] += quantity[i]
        return single_dcube[1:-1,1:-1,1:-1].copy()      # remove off-the-grid rows

    # prepare arguments
    gas_only = False
    if sum(snap.families)==1 and snap.families[0]:
        gas_only = True
    x = snap.x.view(np.ndarray)
    y = snap.y.view(np.ndarray)
    z = snap.z.view(np.ndarray)
    if isinstance(xlim, UnitArr):
        xlim = xlim.in_units_of(snap.pos.units).view(np.ndarray)
    if isinstance(ylim, UnitArr):
        ylim = ylim.in_units_of(snap.pos.units).view(np.ndarray)
    if isinstance(zlim, UnitArr):
        zlim = zlim.in_units_of(snap.pos.units).view(np.ndarray)
    xbins = np.linspace(xlim[0], xlim[1], N_px+1)   # actually edges, thus +1
    ybins = np.linspace(ylim[0], ylim[1], N_px+1)   # actually edges, thus +1
    zbins = np.linspace(zlim[0], zlim[1], N_px+1)   # actually edges, thus +1
    if isinstance(quantity, str):
        if quantity in snap._origin._blocks:
            quantity = snap[quantity]
        else:
            quantity = snap.get(quantity)
    if average not in ['none', 'particle']:
        if isinstance(average, str):
            if average.upper() in snap._origin._blocks:
                average = snap[average.upper()]
            else:
                average = snap.get(average)
        elif not isinstance(average, np.ndarray):   # includes UnitArr
            raise ValueError('average must be one of "none", "particle" or ' + \
                             'a block')
    if N_sample_min is None or N_sample_min < 1:
        N_sample_min = 1
    if kernel not in kernels._kernel:
        raise ValueError('Unknown kernel "%s"!\n' % kernel +
                         'Known kernels are:\n%s' % (kernels._kernel,))

    if len(quantity.shape) != 1 or len(quantity) != len(x):
        raise ValueError('the quantity array have to have the ' + \
                         'same length as positions.')

    if average not in ['none', 'particle']:
        # we have to sum over quantity*average...
        # *NOT* quantity *= average, that would possibly change a variable
        # outside this function!!!
        quantity = quantity * average

    average_units = getattr(average, 'units', None)
    if isinstance(average, UnitArr):
        average = average.view(np.ndarray)
    quantity_units = getattr(quantity, 'units', None)
    if isinstance(quantity, UnitArr):
        quantity = quantity.view(np.ndarray)

    # create cube
    if not gas_only:
        if average == 'none':
            dcube = single_density_cube(x, y, z, quantity, xbins, ybins, zbins, N_px)
            normcube = None
        elif average == 'particle':
            dcube = single_density_cube(x, y, z, quantity, xbins, ybins, zbins, N_px)
            normcube = np.histogramdd(x, y, z, bins=[xbins, ybins, zbins])[0]
            dcube /= normcube
            dcube[np.isnan(dcube)] = 0.0
        else:
            dcube = single_density_cube(x, y, z, quantity, xbins, ybins, zbins, N_px)
            normcube = single_density_cube(x, y, z, average, xbins, ybins, zbins, N_px)
            dcube /= normcube
            dcube[np.isnan(dcube)] = 0.0
    else:   # gas only
        # in preparation create bins according to quality parameter Q and
        # pixel size
        px_size = min(float(xlim[1]-xlim[0]) / N_px,
                      float(ylim[1]-ylim[0]) / N_px,
                      float(zlim[1]-zlim[0]) / N_px)
        hsml, hsml_px, hsml_px_edges, samples = \
                _hsml_binning(snap, (Q*N_px/100.)**3, px_size,
                              N_sample_min, N_sample_max, verbose)

        # actually create the cube
        bin_num = 0
        dcube = np.zeros((N_px, N_px, N_px))
        if average == 'none':
            normcube = None
        else:
            normcube = np.zeros((N_px, N_px, N_px))
        for i_bin in xrange(len(samples)):
            N_sample = samples[i_bin]
            limits = [hsml_px_edges[i_bin], hsml_px_edges[i_bin+1]]
            bin_num += 1
            if verbose:
                print 'particles in bin #%2d ' % (bin_num),
                sys.stdout.flush()
            mask = (hsml_px >= limits[0]) & (hsml_px < limits[1])
            N_parts_bh = np.sum(mask)
            if N_parts_bh == 0:
                continue
            hsml_hb = hsml[mask]
            quantity_hb = quantity[mask]
            if not isinstance(average, str):
                average_hb = average[mask]
            x_hb = x[mask]
            y_hb = y[mask]
            z_hb = z[mask]
            dcube_hb = np.zeros((N_px, N_px, N_px))
            normcube_hb = np.zeros((N_px, N_px, N_px))

            if verbose:
                n_outs = [i * N_sample / 50 for i in xrange(1,50+1)]
                n_outed = 0
            for n in xrange(N_sample):
                # get a random direction in 3D
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
                # get 2D offset that are in random directions with an amplitude
                # such that the points have the (3D) pdf of the (3D) kernel
                offset = np.tensordot(direction,
                                      hsml_hb * kernels.rand_r_kernel(kernel),
                                      axes=0)
                x_off = x_hb + offset[0]
                y_off = y_hb + offset[1]
                z_off = z_hb + offset[2]
                dcube_hb += single_density_cube(x_off, y_off, z_off, quantity_hb,
                                                xbins, ybins, zbins, N_px)
                if average == 'none':
                    pass
                elif average == 'particle':
                    normcube_hb += np.histogramdd(x_off, y_off, z_off,
                                                  bins=[xbins, ybins, zbins])[0]
                else:
                    normcube_hb += single_density_cube(x_off, y_off, z_off,
                                                       average_hb,
                                                       xbins, ybins, zbins, N_px)
                if verbose and n in n_outs:
                    n_to_out = np.ceil(50.*(n+1)/N_sample)
                    sys.stdout.write('.'*(n_to_out-n_outed))
                    n_outed = n_to_out
                    sys.stdout.flush()
            dcube += dcube_hb / N_sample  # norm to sample add to total
            if average != 'none':
                normcube += normcube_hb / N_sample
            if verbose:
                print
        if average != 'none':
            dcube /= normcube
            dcube[np.isnan(dcube)] = 0.0

    # promote to UnitArr
    dcube = dcube.view(UnitArr)
    dcube.units = (quantity_units if quantity_units else 1)
    if average == 'none':
        # divide by volume per pixel, if the quantity is not averaged
        dcube /= float(xlim[1]-xlim[0]) * (ylim[1]-ylim[0]) * (zlim[1]-zlim[0]) / N_px**3
        dcube.units /= snap.pos.units**3
    elif average == 'particle':
        pass
    else:
        if average_units:
            dcube.units /= average_units
    dcube.units = dcube.units.gather()

    if verbose:
        print 'creating the density cube took %.2f sec' % (time.time()-start_time)

    if return_average_binned:
        return dcube, normcube
    else:
        return dcube


def density_map(snap, xlim, ylim, quantity='mass', average='none', x_axis='x',
                y_axis='y', N_px=256, Q=0.3, N_sample_min=10, N_sample_max=1000,
                kernel='Wendland C4', verbose=True):
    '''
    Create a 2-dim. UnitArr / map of the specified quantity, accounting for SPH
    smoothing, if only gas is mapped.

    Args:
        snap (Snap):        The Snapshot to use.
        xlim (array-like):  The limits of the x-axis.
        ylim (array-like):  The limits of the y-axis.
        quantity (str, UnitArr):
                            The quantity to calculate the density of. It can
                            either be the block itself (make shure it already has
                            the appropiate shape and size), the name of a block
                            to get from the snapshot or a expression that can be
                            passed to Snap.get.
        average ('none', 'particles', block):
                            'none':     a simple map of the surface density is
                                        created. Here it is also devided by pixel
                                        area, i.e. the result is a surface
                                        density.
                            'particle': it is averaged over the particles along
                                        the line of sight. Each particle has the
                                        same weighting.
                            block:      it has to be the name of a block, a
                                        string that is evaluable / can be passed
                                        to Snap.get() (e.g. 'Oxygen/mass'), or a
                                        np.ndarray/UnitArr. It is then averaged
                                        with the weighting of this block.
        x_axis (str):       The axis to plot on the x-axis ('x', 'y', or 'z').
        y_axis (str):       The axis to plot on the y-axis ('x', 'y', or 'z').
        N_px (int):         The number of pixels per side (the result will be a
                            (N_px x N_px)-array). Plotting time depends
                            quadratically on the number of pixels in case of
                            plotting gas, since the quality of the
                            Monte-Carlo-integration over the kernel depends on
                            pixel size. (See also parameter Q.) For other
                            particle species, however, there is almost no
                            dependence.
        Q (float):          A positive value that controls the quality of gas
                            quantity plotting.
                            If only gas particles are plotted, it is taken care of
                            the smoothing lengthes with a Monte-Carlo-like
                            integration. The quality of this integration is
                            controlled by Q. Be careful with large values
                            though: the plotting time goes quadratic with Q and
                            Q=1 is already good quality.
        N_sample_min (int): The minimum number of sampling per gas particle.
        N_sample_max (int): The maximum number of sampling per gas particle.
        kernel (str):       The kernel to use to sample the gas particles.
        verbose (bool):     Whether or not to print which sample is mapped at the
                            moment.

    Returns:
        dmap (UnitArr):    The (N_px x N_px)-UnitArr map of the densities.
    '''
    if verbose: print 'create density map with Q=%g' % Q
    if x_axis not in ['x', 'y', 'z'] or y_axis not in ['x', 'y', 'z'] or \
            x_axis == y_axis:
        raise ValueError('x_axis and y_axis have to be "x", "y", or "z" and ' + \
                'not the same!')

    start_time = time.time()
    single_dmap = np.empty((N_px+2, N_px+2))    # add rows for off the grid!
    def single_density_map(x, y, quantity, xbins, ybins, N_px):
        '''helper function'''
        x = np.digitize(x, xbins)
        y = np.digitize(y, ybins)
        N = len(quantity)
        single_dmap[:] = 0
        for i in xrange(N):
            single_dmap[x[i], y[i]] += quantity[i]
        return single_dmap[1:-1,1:-1].copy()    # remove off-the-grid rows

    # prepare arguments
    gas_only = False
    if sum(snap.families)==1 and snap.families[0]:
        gas_only = True
    x = getattr(snap, x_axis).view(np.ndarray)
    y = getattr(snap, y_axis).view(np.ndarray)
    if isinstance(xlim, UnitArr):
        xlim = xlim.in_units_of(snap.pos.units).view(np.ndarray)
    if isinstance(ylim, UnitArr):
        ylim = ylim.in_units_of(snap.pos.units).view(np.ndarray)
    xbins = np.linspace(xlim[0], xlim[1], N_px+1)   # actually edges, thus +1
    ybins = np.linspace(ylim[0], ylim[1], N_px+1)   # actually edges, thus +1
    if isinstance(quantity, str):
        if quantity in snap._origin._blocks:
            quantity = snap[quantity]
        else:
            quantity = snap.get(quantity)
    if average not in ['none', 'particle']:
        if isinstance(average, str):
            if average.upper() in snap._origin._blocks:
                average = snap[average.upper()]
            else:
                average = snap.get(average)
        elif not isinstance(average, np.ndarray):   # includes UnitArr
            raise ValueError('average must be one of "none", "particle" or ' + \
                             'a block')
    if N_sample_min is None or N_sample_min < 1:
        N_sample_min = 1
    if kernel not in kernels._kernel:
        raise ValueError('Unknown kernel "%s"!\n' % kernel +
                         'Known kernels are:\n%s' % (kernels._kernel,))

    if len(quantity.shape) != 1 or len(quantity) != len(x):
        raise ValueError('the quantity array have to have the ' + \
                         'same length as positions.')

    if average not in ['none', 'particle']:
        # we have to sum over quantity*average...
        # *NOT* quantity *= average, that would possibly change a variable
        # outside this function!!!
        quantity = quantity * average

    average_units = getattr(average, 'units', None)
    if isinstance(average, UnitArr):
        average = average.view(np.ndarray)
    quantity_units = getattr(quantity, 'units', None)
    if isinstance(quantity, UnitArr):
        quantity = quantity.view(np.ndarray)

    # create map
    if not gas_only:
        dmap = single_density_map(x, y, quantity, xbins, ybins, N_px)
        if average != 'none':
            if average == 'particle':
                dmap /= np.histogram2d(x, y, bins=[xbins, ybins])[0]
            else:
                dmap /= single_density_map(x, y, average, xbins, ybins, N_px)
            dmap[np.isnan(dmap)] = 0.0
    else:   # gas only
        # in preparation create bins according to quality parameter Q and
        # pixel size
        px_size = min(float(xlim[1]-xlim[0]) / N_px,
                      float(ylim[1]-ylim[0]) / N_px)
        hsml, hsml_px, hsml_px_edges, samples = \
                _hsml_binning(snap, (Q*N_px/100.)**2, px_size,
                              N_sample_min, N_sample_max, verbose)

        # actually create the map
        bin_num = 0
        dmap = np.zeros((N_px, N_px))
        normmap = np.zeros((N_px, N_px))
        for i_bin in xrange(len(samples)):
            N_sample = samples[i_bin]
            limits = [hsml_px_edges[i_bin], hsml_px_edges[i_bin+1]]
            bin_num += 1
            if verbose:
                print 'particles in bin #%2d ' % (bin_num),
                sys.stdout.flush()
            mask = (hsml_px >= limits[0]) & (hsml_px < limits[1])
            N_parts_bh = np.sum(mask)
            if N_parts_bh == 0:
                continue
            hsml_hb = hsml[mask]
            quantity_hb = quantity[mask]
            if not isinstance(average, str):
                average_hb = average[mask]
            x_hb = x[mask]
            y_hb = y[mask]
            dmap_hb = np.zeros((N_px, N_px))
            normmap_hb = np.zeros((N_px, N_px))

            if verbose:
                n_outs = [i * N_sample / 50 for i in xrange(1,50+1)]
                n_outed = 0
            for n in xrange(N_sample):
                # get a random direction in 3D
                direction = np.random.normal(size=3)
                direction /= np.linalg.norm(direction)
                # get 2D offset that are in random directions with an amplitude
                # such that the points have the (3D) pdf of the (3D) kernel
                offset = np.tensordot(direction[:2],
                                      hsml_hb * kernels.rand_r_kernel(kernel),
                                      axes=0)
                x_off = x_hb + offset[0]
                y_off = y_hb + offset[1]
                dmap_hb += single_density_map(x_off, y_off, quantity_hb,
                                              xbins, ybins, N_px)
                if average != 'none':
                    if average == 'particle':
                        normmap_hb += np.histogram2d(x_off, y_off,
                                                     bins=[xbins, ybins])[0]
                    else:
                        normmap_hb += single_density_map(x_off, y_off, average_hb,
                                                         xbins, ybins, N_px)
                if verbose and n in n_outs:
                    n_to_out = np.ceil(50.*(n+1)/N_sample)
                    sys.stdout.write('.'*(n_to_out-n_outed))
                    n_outed = n_to_out
                    sys.stdout.flush()
            dmap += dmap_hb / N_sample  # norm to sample add to total
            if average != 'none':
                normmap += normmap_hb / N_sample
            if verbose:
                print
        if average != 'none':
            dmap /= normmap
            dmap[np.isnan(dmap)] = 0.0

    # promote to UnitArr
    dmap = dmap.view(UnitArr)
    dmap.units = (quantity_units if quantity_units else 1)
    if average == 'none':
        # divide by area per pixel, if the quantity is not averaged
        dmap /= float(xlim[1]-xlim[0]) * (ylim[1]-ylim[0]) / N_px**2
        dmap.units /= snap.pos.units**2
    elif average == 'particle':
        pass
    else:
        if average_units:
            dmap.units /= average_units
    dmap.units = dmap.units.gather()

    if verbose:
        print 'creating the density map took %.2f sec' % (time.time()-start_time)

    return dmap

