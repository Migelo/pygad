'''
Module for convenient plotting of snapshots.

This module is loaded automatically among all the other sub-modules of pygad, if
imported in interactive mode. Otherwise it has to be imported manually in order
to first reduce the time consumed by the import process significantly, and second
to allow to choose a different backend for the matplotlib other than the default
one.
'''
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.cm import get_cmap
from units import Unit, _UnitClass, UnitArr
import analysis
import collections
import luminosities
import environment
import sys

def plot_density(snap, quantity='MASS', average='none', xlim=None, ylim=None,
                 x_axis='x', y_axis='y', units=None, log_scale=True,
                 cbar_title='surface density', cmap=None, show_cbar=True,
                 vmin=None, vmax=None,
                 bg_color='black', N_px=256, Q=0.3, N_sample_min=10,
                 N_sample_max=1000, interpolation='bilinear',
                 kernel='Wendland C4', fontsize=18, axis=None,
                 verbose=environment.verbose):
    '''
    Plot the density of the specified quantity, accounting for SPH smooting, if
    only gas is plotted.

    Args:
        snap (Snap):            The Snapshot to use.
        quantity (str, UnitArr):
                                The quantity to calculate the density of. It can
                                either be the block itself (make shure it already
                                has the appropiate shape and size), the name of a
                                block to get from the snapshot or a expression
                                that can be passed to Snap.get.
        average ('none', 'particles', block):
                                'none':     a simple map of the surface density
                                            is created. Here it is also devided
                                            by pixel area, i.e. the result is a
                                            surface density.
                                'particle': it is averaged over the particles
                                            along the line of sight. Each
                                            particle has the same weighting.
                                block:      it has to be the name of a block, a
                                            string that is evaluable / can be
                                            passed to Snap.get() (e.g.
                                            'Oxygen/Mass'), or a np.ndarray /
                                            UnitArr. It is then averaged with
                                            the weighting of this block.
        xlim (array-like):      The limits of the x-axis.
        ylim (array-like):      The limits of the y-axis.
        x_axis (str):           The axis to plot on the x-axis ('x', 'y', or
                                'z').
        y_axis (str):           The axis to plot on the y-axis ('x', 'y', or
                                'z').
        units (str, Unit):      The units to for the map / colorbar.
        log_scale (bool):       Whether or not to plot the quantity
                                logarithmically.
        cbar_title (str):       The description of the plotted data biside the
                                colorbar.
        cmap (str, Colormap):   The colormap to use for the colorbar. If None is
                                passed, it is set to cm_k_b if only gas are
                                plotted, cm_k_y if only stars are plotted, and
                                'jet' otherwise.
        vmin, vmax (float):     The limits of the colorbar.
        N_px (int):             The number of pixels per side (the result will be
                                a (N_px x N_px)-array).
        Q (float):              A positive value that controls the quality of gas
                                quantity plotting.
                                If only gas particles are plotted, it is taken
                                care of the smoothing lengthes with a
                                Monte-Carlo-like integration. The quality of this
                                integration is controlled by Q. Be careful with
                                large values though: the plotting time goes
                                quadratic with Q and Q=1 is already good quality.
        N_sample_min (int):     The minimum number of sampling per gas particle.
        N_sample_max (int):     The maximum number of sampling per gas particle.
        interpolation (str):    The interpolation to use for plt.imshow().
        kernel (str):           The kernel to use to sample the gas particles
                                (default: 'Wendland C4').
        fontsize (int, float):  The fontsize to use for the axes labels (the
                                colorbar's font has 4/5 times this size).
        axis (AxesSubplot):     The axis to draw on. If None, a new figure is
                                created.
        verbose (bool):         Whether or not to print which sample is mapped at
                                the moment.

    Returns:
        fig (Figure):           The figure drawn on.
        ax (AxesSubplot):       The axis of the figure drawn on.
        cbar (Colorbar):        The color bar, if used, None otherwise.
    '''
    # prepaire arguments
    gas_only = False
    if sum(snap.families)==1 and snap.families[0]:
        gas_only = True

    if cmap is None:
        if sum(snap.families)==1 and (quantity.upper()=='MASS' or \
                getattr(quantity, 'name', '???').upper()=='MASS'):
            if snap.families[0]:
                cmap = cm_k_b
            elif snap.families[4]:
                cmap = cm_k_y
        if cmap is None:
            cmap = 'jet'
    if isinstance(cmap, str):
        cmap = get_cmap(cmap)
    cmap.set_bad(color=bg_color)
    x = getattr(snap, x_axis)
    y = getattr(snap, y_axis)
    if snap.N != 0:
        if xlim is None:
            xlim = [x.min().value, x.max().value]
        if ylim is None:
            ylim = [y.min().value, y.max().value]

        # create map
        dmap = analysis.density_map(snap, xlim, ylim, quantity=quantity,
                                    average=average, x_axis=x_axis,
                                    y_axis=y_axis, N_px=N_px, Q=Q,
                                    N_sample_min=N_sample_min,
                                    N_sample_max=N_sample_max,
                                    kernel=kernel, verbose=verbose)
        if units is not None:
            dmap.convert_to(units)

        # use reasonable limits if no were passed
        if vmin is None:
            vmin = np.percentile(dmap[np.where(dmap!=0)], 5)
        if vmax is None:
            vmax = dmap.max().value
    else:
        if xlim is None:
            xlim = [0, 1]
        if ylim is None:
            ylim = [0, 1]
        dmap = np.zeros((N_px,N_px)).view(UnitArr)
        if vmin is None:
            vmin = 1.0
        if vmax is None:
            vmax = 0.0

    # actually plot
    if axis is None:
        fig, ax = plt.subplots(1, 1, subplot_kw={'aspect':'equal'})
    else:
        fig = axis.get_figure()
        ax = axis

    im = ax.imshow(dmap.view(np.ndarray).T, origin='lower',
                   extent=(xlim[0],xlim[1],ylim[0],ylim[1]),
                   norm=LogNorm() if log_scale else None,
                   interpolation=interpolation, cmap=cmap, vmin=vmin, vmax=vmax)
    pos_units = snap._origin.pos.units.latex()
    ax.set_xlabel(r'$%s$ [$%s$]' % (x_axis, pos_units), fontsize=fontsize)
    ax.set_ylabel(r'$%s$ [$%s$]' % (y_axis, pos_units), fontsize=fontsize)
    
    if show_cbar:
        cbar = fig.colorbar(im)
        cbar_units = dmap.units.gather()
        if cbar_units != 1:
            cbar_title += r' [$%s$]' % cbar_units.latex()
        cbar.set_label(cbar_title,
                       fontsize=None if fontsize is None else fontsize*4/5)
        cbar.set_clim(vmin, vmax)
    else:
        cbar = None

    return fig, ax, cbar

def plot_density_4x4(snap, **kwargs):
    '''
    Do a (log) particle number density plot for the snapshot.

    Note:
        The related scatter_4x4 is much slower at the actual rendering! Also,
        this is only a proper density plot, if all particles have the same masses
        (and are point masses).

    Args:
        quantity (str, UnitArr):
                                The quantity to calculate the density of. It can
                                either be the block itself (make shure it already
                                has the appropiate shape and size), the name of a
                                block to get from the snapshot or a expression
                                that can be passed to Snap.get.
        average ('none', 'particles', block):
                                'none':     a simple map of the surface density
                                            is created. Here it is also devided
                                            by pixel area, i.e. the result is a
                                            surface density.
                                'particle': it is averaged over the particles
                                            along the line of sight. Each
                                            particle has the same weighting.
                                block:      it has to be the name of a block, a
                                            string that is evaluable / can be
                                            passed to Snap.get() (e.g.
                                            'Oxygen/Mass'), or a np.ndarray /
                                            UnitArr. It is then averaged with
                                            the weighting of this block.
                                (default: 'none')
        bg_color (str):         The background color of the axes (default:
                                'black').
        title (str):            The title of the figure (default: None).
        titlesize (float):      The font size of the title (default: 15).
        fontsize (float):       The font size of the axis labels (default: 11).
        lims (array-like):      The limits of all the axes.
        vmin, vmax (float):     The limits of the colorbar.
        log_scale (bool):       Whether or not to plot the quantity
                                logarithmically.
        cbar_title (str):       The title of the colorbar (default: 'surface
                                density').
        N_px (int):             The resolution of the images, pixel per side
                                (default: 256).
        Q (float):              A positive value that controls the quality of gas
                                quntity plotting.
                                If only gas particles are plotted, it is taken
                                care of the smoothing lengthes with a
                                Monte-Carlo-like integration. The quality of this
                                integration is controlled by Q. Be careful with
                                large values though: the plotting time goes
                                quadratic with Q and Q=1 is already good quality.
        N_sample_min (int):     The minimum number of sampling per gas particle.
        N_sample_max (int):     The maximum number of sampling per gas particle.
        kernel (str):           The kernel to use to sample the gas particles
                                (default: 'Wendland C4').
        fig_kw (dict):          Keyword arguments passed to figure().
        subplot_kw (dict):      Keyword arguments passed to figure.add_subplot().
                                'aspect' is by default set to 'equal'.
        gridspec_kw (dict):     Keyword arguments passed to the GridSpec
                                constructor.
        del_ax11 (bool):        Whether or not to delete the unused axis in the
                                bottom right corner (default: True).
        verbose (bool):         Whether or not to print which sample is mapped at
                                the moment.
        <remaining>:            Passed to scatter in the actual scatter plots
                                (eg. 's' and 'c'). If 'c' is not given and
                                bg_color == 'black', then it is set to 'white'.

    Returns:
        fig, axs:           The figure along with a 2x2 array with the axis with
                            axs[1][1]=None if del_ax11=True.
    '''
    # prepaire arguments
    quantity = kwargs.pop('quantity', 'MASS')
    average = kwargs.pop('average', 'none')
    title = kwargs.pop('title', None)
    titlesize = kwargs.pop('titlesize', 15)
    fontsize = kwargs.pop('fontsize', 11)
    lims = kwargs.pop('lims', [snap.pos.min().value, snap.pos.max().value] \
                              if snap.N else [-100,100])
    N_px = kwargs.pop('N_px', 256)
    Q = kwargs.pop('Q', 0.3)
    N_sample_min = kwargs.pop('N_sample_min', 10)
    N_sample_max = kwargs.pop('N_sample_max', 1000)
    kernel = kwargs.pop('kernel', 'Wendland C4')
    fig_kw = kwargs.pop('fig_kw', {})
    subplot_kw = kwargs.pop('subplot_kw', {})
    gridspec_kw = kwargs.pop('gridspec_kw', {})
    del_ax11 = kwargs.pop('del_ax11', True)
    verbose = kwargs.pop('verbose', False)
    units = kwargs.pop('units', None)
    log_scale = kwargs.pop('log_scale', True)
    draw_cbar = kwargs.pop('draw_cbar', True)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    cbar_title = kwargs.pop('cbar_title', 'surface density')

    subplot_kw['aspect'] = 'equal'
    gridspec_kw_defaults = {'left':0.1, 'right':0.87 if draw_cbar else 0.95,
                            'top':0.99 if title is None else 0.92,
                            'wspace':0.25, 'hspace':0.2}
    for key, value in gridspec_kw_defaults.iteritems():
        if key not in gridspec_kw:
            gridspec_kw[key] = value
    kwargs_defaults = {'norm':LogNorm() if log_scale else None,
                       'interpolation':'bilinear', 'cmap':'jet'}
    if sum(snap.families)==1 and (quantity.upper()=='MASS' or \
            getattr(quantity, 'name', '???').upper()=='MASS'):
        if snap.families[0]:
            kwargs_defaults['cmap'] = cm_k_b
        elif snap.families[4]:
            kwargs_defaults['cmap'] = cm_k_y
    for key, value in kwargs_defaults.iteritems():
        if key not in kwargs:
            kwargs[key] = value
    kwargs['aspect'] = 'equal'
    if isinstance(kwargs['cmap'], str):
        kwargs['cmap'] = get_cmap(kwargs['cmap'])
    kwargs['cmap'].set_bad(color=kwargs.pop('bg_color', 'black'))

    # create maps (and set vlimits)
    if snap.N > 0:
        if isinstance(quantity, str):
            # avoid multiple calculations and look-ups
            if quantity.upper() in snap._origin._blocks:
                quantity = snap[quantity.upper()]
            else:
                # especially that step could take some while
                quantity = snap.get(quantity)
        xy_dmap = analysis.density_map(snap, lims, lims, quantity=quantity,
                                       average=average, x_axis='x', y_axis='y',
                                       N_px=N_px, Q=Q, N_sample_min=N_sample_min,
                                       N_sample_max=N_sample_max,
                                       kernel=kernel, verbose=verbose)
        zy_dmap = analysis.density_map(snap, lims, lims, quantity=quantity,
                                       average=average, x_axis='z', y_axis='y',
                                       N_px=N_px, Q=Q, N_sample_min=N_sample_min,
                                       N_sample_max=N_sample_max,
                                       kernel=kernel, verbose=verbose)
        xz_dmap = analysis.density_map(snap, lims, lims, quantity=quantity,
                                       average=average, x_axis='x', y_axis='z',
                                       N_px=N_px, Q=Q, N_sample_min=N_sample_min,
                                       N_sample_max=N_sample_max,
                                       kernel=kernel, verbose=verbose)
        if units:
            xy_dmap.convert_to(units)
            zy_dmap.convert_to(units)
            xz_dmap.convert_to(units)

        if vmin is None:
            vmin = min( [np.percentile(dmap[np.where(dmap!=0)], 5) \
                            for dmap in [xy_dmap, zy_dmap, xz_dmap]] )
        if vmax is None:
            vmax = max( [dmap.max().value \
                            for dmap in [xy_dmap, zy_dmap, xz_dmap]] )
    else:
        xy_dmap = np.zeros((N_px,N_px)).view(UnitArr)
        zy_dmap = np.zeros((N_px,N_px)).view(UnitArr)
        xz_dmap = np.zeros((N_px,N_px)).view(UnitArr)
        if vmin is None:
            vmin = 1.0
        if vmax is None:
            vmax = 0.0

    # actually plot
    fig, axs = plt.subplots(2, 2, subplot_kw=subplot_kw, **fig_kw)
    if title is not None:
        fig.suptitle(title, fontsize=titlesize)
    fig.subplots_adjust(**gridspec_kw)
    if del_ax11:
        fig.delaxes(axs[1][1])

    pos_units = snap._origin.pos.units.latex()

    # x-y
    axs[0][0].imshow(xy_dmap.view(np.ndarray).T, origin='lower',
                     extent=(lims[0],lims[1],lims[0],lims[1]),
                     vmin=vmin, vmax=vmax, **kwargs)
    axs[0][0].set_xlabel(r'$x$ [$%s$]' % pos_units, fontsize=fontsize)
    axs[0][0].set_ylabel(r'$y$ [$%s$]' % pos_units, fontsize=fontsize)

    # z-y
    axs[0][1].imshow(zy_dmap.view(np.ndarray).T, origin='lower',
                     extent=(lims[0],lims[1],lims[0],lims[1]),
                     vmin=vmin, vmax=vmax, **kwargs)
    axs[0][1].set_xlabel(r'$z$ [$%s$]' % pos_units, fontsize=fontsize)
    axs[0][1].set_ylabel(r'$y$ [$%s$]' % pos_units, fontsize=fontsize)

    # x-z
    im = axs[1][0].imshow(xz_dmap.view(np.ndarray).T, origin='lower',
                          extent=(lims[0],lims[1],lims[0],lims[1]),
                          vmin=vmin, vmax=vmax, **kwargs)
    axs[1][0].set_xlabel(r'$x$ [$%s$]' % pos_units, fontsize=fontsize)
    axs[1][0].set_ylabel(r'$z$ [$%s$]' % pos_units, fontsize=fontsize)

    bottom = axs[1][0].get_position().get_points()[0][1]
    top = axs[0][0].get_position().get_points()[1][1]
    cbar_ax = fig.add_axes([0.9, bottom, 0.025, top-bottom])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar_units = xy_dmap.units.gather()
    if cbar_units != 1:
        cbar_title += r' [$%s$]' % cbar_units.latex()
    cbar.set_label(cbar_title,
                   fontsize=None if fontsize is None else fontsize*4/5)
    cbar.set_clim(vmin, vmax)

    return fig, axs

def plot_stars(snap, xlim=None, ylim=None, x_axis='x', y_axis='y', N_px=256,
               cmap=cm_age, show_cbar=True, min_rho=None, max_rho=None,
               fontsize=15, axis=None, fig_kw=None, verbose=environment.verbose):
    '''
    Plot the stars with age color coded (and brightness scaling with their mass).

    Args:
        snap (Snap):        The snapshot to use the gas of.
        xlim (array-like):  The limits of the x-axis.
        ylim (array-like):  The limits of the y-axis.
        title (str):        The title of the figure.
        N_px (int):         The number of pixels per side (the result will be a
                            (N_px x N_px)-image).
        cmap (Colormap):    The colormap to use for colorcode the (mean) stellar
                            ages.
        show_cbar (bool, int):
                            Whether to show the color bar. If it is an integer,
                            this sets the position.
        min_rho, max_rho (float, Unit, UnitArr):
                            The limits of the density for image brightness.
        fontsize (int):     The size of the labels.
        axis (AxesSubplot): The axis to draw on. If None, a new figure is created.
        fig_kw (dict):      Keyword argument to be passed to plot_density_4x4.
        verbose (bool):     Whether or not to print which sample is mapped at
                            the moment.

    Returns:
        fig, ax:            The figure along with the axes.
    '''
    if fig_kw is None:
        fig_kw = {}

    if xlim is None:
        xlim = [snap.stars.x.min().value, snap.stars.x.max().value]
    if ylim is None:
        ylim = [snap.stars.y.min().value, snap.stars.y.max().value]

    dens_units = None
    if min_rho is not None:
        if isinstance(min_rho, (_UnitClass, UnitArr)):
            dens_units = min_rho if isinstance(min_rho, _UnitClass) \
                    else min_rho.units
            min_rho = min_rho.in_units_of(dens_map.units)
        dens_units = np.log10(min_rho)
    if max_rho is not None:
        if isinstance(max_rho, (_UnitClass, UnitArr)):
            dens_units = max_rho if isinstance(max_rho, _UnitClass) \
                    else max_rho.units
            max_rho = max_rho.in_units_of(dens_map.units)
        dens_units = np.log10(max_rho)

    age_map = analysis.density_map(
                    snap.stars, xlim, ylim,
                    quantity=snap.stars.age.view(np.ndarray),
                    average=snap.stars.inim.view(np.ndarray),
                    x_axis=x_axis, y_axis=y_axis,
                    verbose=False).view(np.ndarray)
    age_map /= 12. * Unit('Gyr').in_units_of(snap.stars.age.units)
    dens_map = analysis.density_map(
                     snap.stars, xlim, ylim,
                     x_axis=x_axis, y_axis=y_axis,
                     verbose=False)
    if  dens_units is not None:
        dens_map = dens_map.in_units_of(dens_units)
    dens_map = np.log10(dens_map.view(np.ndarray))

    if min_rho is None:
        min_rho = np.percentile(dens_map[np.isfinite(dens_map)], 5)
    if max_rho is None:
        max_rho = np.percentile(dens_map[np.isfinite(dens_map)],99)

    rgb = cmap(age_map)
    v = (dens_map - min_rho) / (max_rho - min_rho)
    v[v<0.] = 0.
    v[v>1.] = 1.
    for i in xrange(3):
        rgb[:,:,i] = (rgb[:,:,i] * v).T
    
    if axis:
        fig, ax = axis.figure, axis
    else:
        fig, ax = plt.subplots(**fig_kw)
    im = ax.imshow(rgb, origin='lower', extent=(xlim[0],xlim[1],ylim[0],ylim[1]))
    pos_units = snap._origin.pos.units.latex()
    ax.set_xlabel(r'$%s$ [$%s$]' % (x_axis, pos_units), fontsize=fontsize)
    ax.set_ylabel(r'$%s$ [$%s$]' % (y_axis, pos_units), fontsize=fontsize)
    
    if show_cbar:
        if show_cbar is True:
            show_cbar = 1
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        axins = inset_axes(ax, loc=show_cbar, width='60%', height='3%')
        cbar = mpl.colorbar.ColorbarBase(axins, cmap=cmap,
                                         norm=mpl.colors.Normalize(vmin=0, vmax=12),
                                         ticks=range(0,12+1,2), format='%d',
                                         orientation='horizontal')
        cbar.ax.tick_params(labelsize=0.5*fontsize) 
        for i in cbar.ax.get_xticklabels():
            i.set_color('white')
        cbar.set_label('mean stellar age [Gyr]', fontsize=0.5*fontsize, color='w')

    return fig, ax

def plot_gas(snap, quantity='MASS', average='none', cut=False, title='gas',
             Q=0.3, N_sample_min=10, N_sample_max=1000, N_px=256, vmin=None,
             vmax=None, log_scale=True, units=None, titlesize=20, fontsize=15,
             cbar_title='surface density', fig_kw=None,
             verbose=environment.verbose):
    '''
    Plot the gas from three directions and the phase diagram.

    Args:
        snap (Snap):        The snapshot to use the gas of.
        quantity (str, UnitArr):
                            The quantity to calculate the density of. It can
                            either be the block itself (make shure it already
                            has the appropiate shape and size), the name of a
                            block to get from the snapshot or a expression
                            that can be passed to Snap.get.
        average ('none', 'particles', block):
                            'none':     a simple map of the surface density
                                        is created. Here it is also devided
                                        by pixel area, i.e. the result is a
                                        surface density.
                            'particle': it is averaged over the particles
                                        along the line of sight. Each
                                        particle has the same weighting.
                            block:      it has to be the name of a block, a
                                        string that is evaluable / can be
                                        passed to Snap.get() (e.g.
                                        'Oxygen/Mass'), or a np.ndarray /
                                        UnitArr. It is then averaged with
                                        the weighting of this block.
        cut (bool, float, Unit):
                            If False, simply plot all gas of the snapshot, if
                            True, cut out the subset of the snapshot such that
                            all coordinates have absolute values less than
                            0.2*R200. If a float / Unit is passed, use that as
                            the cut rather than 0.2*R200.
        title (str):        The title of the figure.
        Q (float):          A positive value that controls the quality of gas
                            quntity plotting.
                            If only gas particles are plotted, it is taken care of
                            the smoothing lengthes with a Monte-Carlo-like
                            integration. The quality of this integration is
                            controlled by Q. Be careful with large values
                            though: the plotting time goes quadratic with Q and
                            Q=1 is already good quality.
        N_sample_min (int): The minimum number of sampling per gas particle.
        N_sample_max (int): The maximum number of sampling per gas particle.
        N_px (int):         The number of pixels per side (the result will be a
                            (N_px x N_px)-image).
        vmin, vmax (float): The limits of the colorbar.
        log_scale (bool):   Whether or not to plot the quantity
                            logarithmically.
        units (str, Unit):  The units of the surface density.
        titlesize (int):    The size of the figure title.
        fontsize (int):     The size of the labels.
        cbar_title (str):   The title of the colorbar (default: 'surface
                            density').
        fig_kw (dict):      Keyword argument to be passed to plot_density_4x4.
        verbose (bool):     Whether or not to print which sample is mapped at
                            the moment.

    Returns:
        fig, axs:       The figure along with a 2x2 array with the axis.
    '''
    if cut:
        if verbose:
            print 'cut to desired region...'
            sys.stdout.flush()
        if cut is True:
            R200, M200 = analysis.virial_info(snap)
            if R200.scale == 0:
                RuntimeError('The virial radius is zero. Cannot cut from ' + \
                             'the halo for plotting.')
            cut = 0.2*R200
        if isinstance(cut, _UnitClass):
            cut = cut.in_units_of(snap.pos.units)
        sub_gas = snap.gas
        sub_gas = sub_gas[np.where( (sub_gas.x.abs() - sub_gas.hsml < cut)
                                  & (sub_gas.y.abs() - sub_gas.hsml < cut)
                                  & (sub_gas.z.abs() - sub_gas.hsml < cut) )]
        if verbose:
            print 'done.'
            sys.stdout.flush()
    else:
        cut = [snap.pos.min().value, snap.pos.max().value]
        sub_gas = snap.gas

    if fig_kw is None:
        fig_kw = {}

    fig, axs = plot_density_4x4(sub_gas,
                                quantity=quantity,
                                average=average,
                                lims=cut \
                                    if isinstance(cut, collections.Iterable) \
                                    else [-cut, cut],
                                title=title,
                                titlesize=titlesize, fontsize=fontsize,
                                Q=Q, N_sample_min=N_sample_min, N_px=N_px,
                                N_sample_max=N_sample_max,
                                units=units,
                                vmin=vmin, vmax=vmax,
                                log_scale=log_scale,
                                cbar_title=cbar_title,
                                fig_kw=fig_kw,
                                del_ax11=False,
                                verbose=verbose)
    axs[1][1].set_aspect(1.8)
    plot_phase_diagram(sub_gas,
                       T_unit='K', rho_unit='g/cm**3',
                       T_range=(1e1, 1e8), rho_range=(1e-34, 1e-20),
                       fontsize=fontsize,
                       axis=axs[1][1])

    return fig, axs

def plot_surface_density_profile(snap, r_edges=None, units=None, N_sample=50,
                                 grid=True, verbose=False, **kwargs):
    '''
    Plot the surface density profile using cylindrical bins.

    Args:
        snap (Snap):            The snapshot to use.
        r_edges (array-like):   The edges of the radial bins. If no units are
                                given, they are assumed to be thos eof snap.pos.
                                (default: 50 bins from 0 kpc through 100 kpc with
                                equal spacing.)
        units (str, Unit):      The units of the surface density. (default:
                                Msol/pc**2)
        N_sample (int):         If gas particles only, sample them with that many
                                points.
        grid (bool):            Whether or not to plot a grid.
        verbose (bool):         Whether or not to print which sample is mapped at
                                the moment.
        kwargs:                 Further kwywords will be passed to ax.plot().

    Returns:
        fig (Figure):           The figure drawn on.
        ax (AxesSubplot):       The axis of the figure drawn on.
    '''
    if r_edges is None:
        r_edges = UnitArr(np.arange(51), units='kpc', snap=snap)
    elif not isinstance(r_edges, UnitArr):
        r_edges = UnitArr(r_edges)
    if r_edges._units is None:
        r_edges.units = snap.pos.units
    if r_edges.snap is None:
        r_edges._snap = snap

    if units is None:
        units = Unit('Msol/pc**2')

    fig, ax = plt.subplots()

    r = (r_edges[1:]+r_edges[:-1])/2.
    if r_edges[0] == 0:
        r[0] = 0
    for s in [snap.stars, snap.gas]:
        if s.N == 0:
            continue
        Sigma = analysis.radial_surface_density_profile(s, r_edges=r_edges,
                                                        N_sample=N_sample,
                                                        units=units,
                                                        verbose=verbose)
        ax.plot(r, Sigma,
                c='orange' if s.family=='stars' else 'blue',
                label=s.family, **kwargs)

    ax.set_yscale('log')
    ax.set_xlabel(r'$r$ [$%s$]' % r_edges.units.latex())
    ax.set_ylabel(r'$\Sigma_\mathrm{*,gas}$ [$%s$]' % Sigma.units.latex())
    ax.set_xlim([r_edges[0], r_edges[-1]])
    ax.legend(loc='best')
    ax.grid(grid)

    return fig, ax

def plot_vcirc(snap, rrange=None, units=None, grid=True, axis=None, **kwargs):
    '''
    Plot the circular velocity curve.

    Here spherical symmetry is assumed. That implies that

        sqrt(G * M(r<R) / R)

    is plotted.

    Args:
        snap (Snap):            The snapshot to use (actually the original one).
        rrange (array-like):    The radial range to plot over. (default:
                                0 kpc - 100 kpc)
        units (str, Unit):      The units of the circular velocity.
                                (default: km/s)
        grid (bool):            Whether or not to plot a grid.
        axis (AxesSubplot):     The axis to draw on. If None, a new figure is
                                created.
        kwargs:                 Further keywords are passed to ax.plot().

    Returns:
        fig (Figure):           The figure drawn on.
        ax (AxesSubplot):       The axis of the figure drawn on.
    '''
    if rrange is None:
        rrange = UnitArr([0,100], units='kpc')
    if units is None:
        units = Unit('km/s')
    if snap is not snap._origin:
        print 'WARNING: using the original snapshot'
        snap = snap._origin

    # data preparation
    r = snap.r[snap.rind]
    v_circ = snap.vcir[snap.rind]

    v_circ.convert_to(units)

    i_min = max(np.abs(r-rrange[0]).argmin(), 1)
    i_max = min(np.abs(r-rrange[1]).argmin(), len(r)-2)

    # actual plotting
    if axis is None:
        fig, ax = plt.subplots()
    else:
        fig = axis.get_figure()
        ax = axis

    ax.plot(r[i_min:i_max], v_circ[i_min:i_max], **kwargs)

    ax.set_xlabel(r'$r$ [$%s$]' % r.units.latex())
    ax.set_ylabel(r'$v_\mathrm{circ}$ [$%s$]' % v_circ.units.latex())
    ax.set_xlim(rrange)
    ax.grid(grid)

    return fig, ax

def plot_stellar_light(snap, width, bands=None, weights=None, dynamic_range=3.,
                       faceon=False, enhance=0.2, axis=None, show=True):
    '''
    Create an image of the stars with luminosities of various bands mapped to the
    color channels.

    Args:
        snap (Snap):        The snapshot to use.
        width (Unit, UnitArr, str, float):
                            The extend of the image (which is centered at the
                            origion).
        bands (list):       The bands to use for the RGB channels.
                            default: ['r','v','b']
        weights (list):     Weights for the different channels / bands (applied
                            before the logarithm).
                            default: _band_weights for the chanels
        dynamic_range (float):
                            The dynamic range to plot over (decades in luminosity
                            to use for image).
                            default: 5.0
        faceon (bool):      Whether to plot face-on or not.
        enhance (float):    Enhance the colors by raising the color value (in
                            range 0-1) to this power (the smaller the power, the
                            bigger the enhancement).
                            default: 0.2
        axis (AxesSubplot): The axis to draw on. If None, a new figure is created.
        show (bool):        Whether to call show.

    Returns:
        rgb (np.ndarray):   The RGB image.
    '''
    if hasattr(width, 'in_units_of'):
        width = float( width.in_units_of(snap.pos.units) )
    lims = [-width/2., width/2.]
    if bands is None:
        bands = ['r', 'v', 'b']
    bands = map(str.lower, bands)
    if weights is None:
        weights = [luminosities._band_weights[b] for b in bands]
    print 'bands:', bands
    print 'band weights:', weights

    ms = []
    for w, band in zip(weights, bands):
        M = luminosities.calc_mags(snap, band)
        # sum up luminosities (!)
        lum_map = analysis.density_map(
                        snap.stars, lims, lims,
                        quantity=10.0**(-0.4*M),
                        x_axis=('y' if faceon else 'z'),
                        y_axis='x',
                        verbose=False).view(np.ndarray)
        # image brightness, however, should scale logarithmically
        ms.append( np.log10(w * lum_map) )

    # not quite the maximum; there can could be ouliers
    dynamic_range = float(dynamic_range)
    print 'dynamic range: %.2f mag' % dynamic_range
    maxM = max(map(lambda a: np.percentile(a[np.isfinite(a)], 99.5), ms))
    minM = maxM - dynamic_range

    # combine
    rgb = np.zeros((ms[0].shape[0], ms[0].shape[1], 3))
    for i, m in enumerate(ms):
        # don't forget the minus; smaller magnitude is more brightness
        v = (m - minM) / dynamic_range
        v[v<0.] = 0.
        v[v>1.] = 1.
        rgb[:,:,i] = v

    # enhance colors
    if enhance:
        hsv = mpl.colors.rgb_to_hsv(rgb)
        hsv[:,:,1] = hsv[:,:,1]**enhance
        rgb = mpl.colors.hsv_to_rgb(hsv)

    if axis:
        fig, ax = axis.figure, axis
    else:
        fig, ax = plt.subplots()
    ax.imshow(rgb[::-1,:], extent=(lims[0],lims[1],lims[0],lims[1]))

    if show:
        plt.show()

    return rgb

