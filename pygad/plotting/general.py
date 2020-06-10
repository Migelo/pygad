'''
Module for general convenience routines for plotting.

Only very little doctests are possible, since it is mostly visual inspection
required...
    # 'plasma' and 'viridis' are not always available (they are from mpl v2.0 on)
    >>> for cmap in ['jet', 'rainbow', #'plasma', 'viridis',
    ...              'age', 'isolum']:
    ...     if isinstance(cmap, str):
    ...         cmap = plt.cm.get_cmap(cmap)
    ...     normed_cmap = isolum_cmap(cmap)
    ...     colors = normed_cmap(np.arange(normed_cmap.N))
    ...     lum = luminance(colors)
    ...     if np.any( np.abs(lum - np.mean(lum)) > 1e-6 ):
    ...         print('%s:'%cmap.name, np.mean(lum), np.percentile(lum,[0,100]))
'''
__all__ = ['CM_DEF', 'luminance', 'isolum_cmap', 'color_code', 'show_image',
           'scatter_map', 'make_scale_indicators', 'add_cbar']

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from ..units import *
from ..binning import *
import sys

# 'jet' is bad -- the new default in matplotlib v2.0 ('viridis') is much better
CM_DEF = 'viridis'

# cf. http://alienryderflex.com/hsp.html
RGB_lum_weight = np.array([0.299, 0.587, 0.114])


def luminance(colors):
    '''
    Return the luminance of (a) color(s).

    Args:
        colors (array-like):    Either a single color in rgb space (shape (3,) or
                                (4,)) of an array of these (shape (N,3) of (N,4)).
                                The alpha channel is ignored.

    Returns:
        lum (np.ndarray):       The luminance of the color(s).
    '''
    colors = np.asarray(colors)
    if len(colors.shape) == 1:
        c = colors[:3]
    else:
        c = colors[:, :3]
    return np.sqrt(np.dot(c ** 2, RGB_lum_weight))


def isolum_cmap(cmap, isolum=1.0, desat=None):
    '''
    Return version of the colormap with constant luminance.

    Args:
        cmap (str, Colormap):   The colormap to norm in luminance.
        isolum (float):         The grade to which the colormap shall be converted
                                to a iso-luminosity one.
        desat (float):          If not None, a factor of how much to desatureate
                                the colormap first. This allows higher luminance.

    Returns:
        isolum (Colormap):      The iso-luminance colormap.
    '''
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    if desat is not None:
        if not (0.0 <= desat <= 1.0):
            raise ValueError('`desat` needs to be in [0,1], but is %s!' % desat)
        hsv = mpl.colors.rgb_to_hsv(colors[:, :3])
        hsv[:, 1] *= 1.0 - desat
        colors[:, :3] = mpl.colors.hsv_to_rgb(hsv)

    # convert RGBA to perceived greyscale luminance
    # cf. http://alienryderflex.com/hsp.html
    if not (0.0 <= isolum <= 1.0):
        raise ValueError('`isolum` needs to be in [0,1], but is %s!' % isolum)
    lum = (isolum * luminance(colors)) + (1. - isolum)
    for i in range(3):
        colors[:, i] /= lum
    # if rgb=(1,1,1), then lum=sqrt(3)>1
    colors[:, :3] /= np.max(colors[:, :3])
    lum = luminance(colors)

    return mpl.colors.LinearSegmentedColormap.from_list(cmap.name + '_lumnormed',
                                                        colors, cmap.N)


def color_code(im_lum, im_col, cmap=CM_DEF, vlim=None, clim=None,
               zero_is_white=False):
    '''
    Colorcode a mono-chrome image with onother one.

    Args:
        im_lum (np.ndarray):    The mono-chrome image to color code. In other
                                words: the map of the brightness values. Needs to
                                be of shape (w,h).
        im_col (np.ndarray):    The image/map values that are converted into
                                colors. Needs to be of the same shape as
                                im_lum: (w,h).
        cmap (str, Colormap):   The colormap to use to convert the values of
                                im_col into actual colors.
        vlim (sequence):        The limits for brightnesses.
        clim (sequence):        The limits for the colors.
        zero_is_white (bool):   Instead of scaling the image luminance by im_lum,
                                desaturate the colors to white for im_lum values
                                approaching the lower limit of vlim (or zero if
                                this is None).

    Returns:
        im (np.ndarray):        The rgb image (shape (w,h,3)).
    '''
    if vlim is None:
        vlim = np.percentile(im_lum[np.isfinite(im_lum)], [0, 100])
    if clim is None:
        clim = np.percentile(im_col[np.isfinite(im_col)], [0, 100])

    im_lum = scale01(im_lum, vlim)
    im_col = scale01(im_col, clim)

    if isinstance(cmap, str): cmap = mpl.cm.get_cmap(cmap)
    im = cmap(im_col)

    if zero_is_white:
        white = np.ones(im.shape[:-1])
        im[:, :, 0] = im_lum * im[:, :, 0] + (1. - im_lum) * white
        im[:, :, 1] = im_lum * im[:, :, 1] + (1. - im_lum) * white
        im[:, :, 2] = im_lum * im[:, :, 2] + (1. - im_lum) * white
    else:
        im[:, :, 0] *= im_lum
        im[:, :, 1] *= im_lum
        im[:, :, 2] *= im_lum

    return im


def show_image(m, extent=None, cmap=CM_DEF, vlim=None, aspect=None,
               interpolation='nearest', ax=None, **kwargs):
    '''
    Show a map or an image with the 'physicsit's orientation'.

    The first axis of `m` is the x-direction and the second one the y-direction,
    contrary to the default behaviour of plt.imshow.

    Args:
        m (Map, array-like):    The map/image to show. It can be a luminance image
                                (shape (w,h)), a rgb-image (shape (w,h,3)) or a
                                rgba-image (shape (w,h,4)).
        extent (array-like):    The ranges of the image. It can either be a sequence
                                of four values (xmin,xmax,ymin,ymax) or an
                                array-like, of the structure
                                [[xmin,xmax],[ymin,ymax]].
        cmap (str, Colormap):
                            The colormap to use, it a luminance image was passed.
                            Ignored otherwise.
        vlim (sequence):    The minimum and maximum value of the image to show. If
                            you want to specify just one, pass [vmin,None] for
                            instance. Ignored if the image is not a luminance only
                            one (shape==(w,h)).
        aspect (str, float):The aspect ratio of the image (see ax.imshow for more
                            information).
        interpolation (str):The interpolation to use between pixels. See imshow
                            for more details.
        ax (AxesSubplot):   The axis object to plot on. If None, a new one is
                            created by plt.subplots().
        **kwargs:           Further arguments are passed to ax.imshow.

    Returns:
        fig (Figure):       The figure of the axis plotted on.
        ax (AxesSubplot):   The axis plotted on.
        im (AxesImage):     The image instance created.
    '''
    if len(m.shape) != 2 and not (len(m.shape) == 3 and m.shape[-1] in [3, 4]):
        raise ValueError('Image has to have shape (w,h), (w,h,3) or (w,h,4)!')
    if extent is None:
        extent = getattr(m, 'extent', None)
        if extent is None:
            raise ValueError('No extent given and `m` is not a `Map`!')

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    vmin, vmax = vlim if vlim is not None else (None, None)
    if extent is not None:
        extent = np.asarray(extent).ravel()

    if len(m.shape) == 2:
        m = m.T
    else:
        m = np.dstack([m[:, :, i].T for i in range(m.shape[-1])])

    im = ax.imshow(m, origin='lower', extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect=aspect, interpolation=interpolation, **kwargs)

    return fig, ax, im


def scatter_map(x, y, s=None, qty=None, av=None, bins=150, extent=None,
                logscale=False, vlim=None, cmap=None, colors=None, colors_av=None,
                clim=None, clogscale=False, fontcolor=None, fontsize=14,
                outline=True, cbartitle=None, showcbar=True, aspect='auto',
                ax=None, zero_is_white=False, **kwargs):
    '''
    Do a binned scatter plot.

    Rather than doing a actual scatter plot, the data points are binned onto a
    grid (optionally color-coded by some quantity) and then the resulting map is
    displayed.

    Args:
        x (str, UnitArr):       The quantity to bin along the x-axis.
        y (str, UnitArr):       The quantity to bin along the y-axis.
        s (Snap):               The (sub-)snapshot from which to use the gas of.
                                Only needed, if one of x, y, qty, colors, or
                                colors_av is a string (block name).
        qty (str, array-like):  If given, not just scatter, but actually bin this
                                quantity onto a grid.
        av (str, array-like):   If given, plot the average the quantity `qty`,
                                weighted by this quantity.
        bins (int, sequence):   The number of bins per axis.
        extent (UnitArr):       The extent to plot over:
                                [[min(x),max(x)],[min(y),max(y)]]
        logscale (bool):        Whether to take the logarithm of the bins for
                                color-coding (or brightness, if colors are
                                provieded).
        vlim (sequence):        The limits for the bin values.
        cmap (str,Colormap):    The colormap to use.
        colors (str, UnitArr):  If passed, the bins are color-coded by this
                                quantity.
        colors_av (str, UnitArr):
                                If given, the colors quantity is averaged with
                                these weights.
        clim (sequence):        The limits for the color-coding.
        clogscale (bool):       Whether to do the color-coding logaritmically.
        fontcolor (str):        The color to use for the colorbar ticks and
                                labels.
        fontsize (int):         The size of the axis ticks and the cbar title.
                                Other font sizes are scaled accordingly.
        outline (array-like):   Draw an outline around text for better
                                readability. The first element is the thickness,
                                the second the color.
        cbartitle (str):        A specific colorbar title. If not given, a
                                automatic one is chosen.
        showcbar (bool):        Whether to show a colorbar for the color-coding.
        aspect (str, float):    The aspect ratio of the plot.
        ax (AxesSubplot):       The axis object to plot on. If None, a new one is
                                created by plt.subplots().
        zero_is_white (bool):   Instead of scaling the image luminance by `qty` if
                                if there is a color given, desaturate the colors
                                to white for `qty` values approaching the lower
                                limit of vlim (or zero if this is None).
        **kwargs:               Further arguments are passed to show_image.

    Returns:
        fig (Figure):       The figure of the axis plotted on.
        ax (AxesSubplot):   The axis plotted on.
        im (AxesImage):     The image instance created.
       [cbar (Colorbar):    The colorbar, if showcbar is True.]
    '''
    xname = ''
    yname = ''
    qtyname = ''
    cname = ''
    if isinstance(x, str):
        xname = x
        x = s.get(x)
    if isinstance(y, str):
        yname = y
        y = s.get(y)
    if isinstance(qty, str):
        qtyname = qty
        qty = s.get(qty)
    if isinstance(av, str):
        av = s.get(av)
    if cmap is None:
        cmap = CM_DEF if colors is None else 'isolum'
        cmap = mpl.cm.get_cmap(cmap)
        cmap.set_bad('w' if zero_is_white else 'k')
        if fontcolor is None:
            fontcolor = 'k' if zero_is_white else 'w'
    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)
    if isinstance(colors, str):
        cname = colors
        colors = s.get(colors)
    if isinstance(colors_av, str):
        colors_av = s.get(colors_av)
    if fontcolor is None:
        fontcolor = 'k' if zero_is_white else 'w'

    if extent is None:
        extent = np.asarray([[np.min(x), np.max(x)], [np.min(y), np.max(y)]])
    else:
        if getattr(extent, 'units', None):
            extent = [extent[0], extent[1]]
            if getattr(x, 'units', None) is not None:
                extent[0] = extent[0].in_units_of(x.units)
            if getattr(y, 'units', None) is not None:
                extent[1] = extent[1].in_units_of(y.units)
        extent = np.asarray(extent)

    if av is not None:
        grid = gridbin2d(x, y, av * qty, bins=bins, extent=extent,
                         nanval=0.0)
        grid /= gridbin2d(x, y, av, bins=bins, extent=extent,
                          nanval=0.0)
        # grid[np.isnan(grid)] = 0.0
    else:
        grid = gridbin2d(x, y, qty, bins=bins, extent=extent, nanval=0.0)
    if getattr(vlim, 'units', None) is not None \
            and getattr(grid, 'units', None) is not None:
        vlim = vlim.in_units_of(grid.units)
    if logscale:
        grid = np.log10(grid)
        if vlim is not None:
            vlim = np.log10(vlim)
    if vlim is None:
        vlim = np.percentile(grid[np.isfinite(grid)], [1, 99])

    if colors is None:
        clim = vlim
    else:
        if colors_av is not None:
            col = gridbin2d(x, y, colors_av * colors, bins=bins, extent=extent,
                            nanval=0.0)
            col /= gridbin2d(x, y, colors_av, bins=bins, extent=extent,
                             nanval=0.0)
            col[np.isnan(col)] = 0.0
        else:
            col = gridbin2d(x, y, colors, bins=bins, extent=extent, nanval=0.0)

        if clogscale:
            col = np.log10(col)
            if clim is not None:
                clim = np.log10(clim)

        if clim is None:
            finitegrid = col[np.isfinite(col)]
            clim = [finitegrid.min(), finitegrid.max()]
            del finitegrid
        grid = color_code(grid, col, cmap=cmap, vlim=vlim, clim=clim,
                          zero_is_white=zero_is_white)

    fig, ax, im = show_image(grid, extent=extent, cmap=cmap, aspect=aspect, ax=ax,
                             clim=clim, **kwargs)

    if showcbar:
        if cbartitle is None:
            name = cname if colors is not None else qtyname
            count_units = r'$\mathrm{count}$'
            if colors is not None:
                units = getattr(colors, 'units', None)
            elif qty is not None:
                units = getattr(qty, 'units', None)
            else:
                units = count_units
            if units == 1 or units is None:
                units = ''
            elif units is not count_units:
                units = r'[$%s$]' % units.latex()
            cbartitle = name + (' ' if (name != '' and units != '') else '') + units
            if (logscale and colors is None) or \
                    (clogscale and colors is not None):
                cbartitle = r'$\log_{10}$(' + cbartitle + ')'

        cbar = add_cbar(ax, cbartitle, clim=clim, cmap=cmap, fontcolor=fontcolor,
                        fontsize=fontsize, fontoutline=outline, nticks=7)

    xunits = r'[$%s$]' % x.units.latex() if getattr(x, 'units', None) else ''
    yunits = r'[$%s$]' % y.units.latex() if getattr(y, 'units', None) else ''
    ax.set_xlabel('%s%s' % (xname, xunits), fontsize=fontsize)
    ax.set_ylabel('%s%s' % (yname, yunits), fontsize=fontsize)
    for tl in ax.get_xticklabels():
        tl.set_fontsize(0.8 * fontsize)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(0.8 * fontsize)

    if showcbar:
        return fig, ax, im, cbar
    else:
        return fig, ax, im


def make_scale_indicators(ax, extent, scaleind='line', scaleunits=None,
                          xaxis=0, yaxis=1, fontsize=14, fontcolor='black',
                          outline=None):
    '''
    Set the scale indicators for a map.

    Args:
        ax (AxesSubplot):       The axis to set the indicators for.
        extent (UnitArr):       The extent plotted over. It needs to be of the
                                form: [[min(x),max(x)],[min(y),max(y)]]
        scaleind (str):         Can be:
                                    'labels':   ticks and labels are drawn on the
                                                axes
                                    'line':     a bar is drawn in the lower left
                                                corner indicating the scale
                                    'none':     neither nor is done -- make the
                                                axes unvisible
        scaleunits (str):       If the scale indication is 'labels', this is the
                                units of these. (Can still be None, though.)
        xaxis/yaxis (int):      The x- and y-axis in indices used to label the
                                axis in 'labels' scale mode.
        fontsize (int,float):   The font size to use.
        fontcolor (str):        The font color to use.
        outline (array-like):   Draw an outline around text for better
                                readability. The first element is the thickness,
                                the second the color.
    '''
    import matplotlib.patheffects

    if outline is True:
        outline = [
            3,
            np.array([1, 1, 1]) - mpl.colors.ColorConverter.to_rgb(fontcolor)
        ]
    if outline is None:
        path_effects = []
    else:
        path_effects = [mpl.patheffects.withStroke(linewidth=outline[0],
                                                   foreground=outline[1]),
                        mpl.patheffects.Normal()]
    if scaleind in [None, 'none']:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    elif scaleind == 'labels':
        if scaleunits is None:
            units_txt = ''
        else:
            units_txt = r' [$%s$]' % scaleunits.latex()
        x_label = r'$%s$%s' % ({0: 'x', 1: 'y', 2: 'z'}[xaxis], units_txt)
        ax.set_xlabel(x_label, fontsize=fontsize)
        y_label = r'$%s$%s' % ({0: 'x', 1: 'y', 2: 'z'}[yaxis], units_txt)
        ax.set_ylabel(y_label, fontsize=fontsize)
        for tl in ax.get_xticklabels():
            tl.set_fontsize(0.8 * fontsize)
        for tl in ax.get_yticklabels():
            tl.set_fontsize(0.8 * fontsize)
    elif scaleind == 'line':
        width = extent[:, 1] - extent[:, 0]
        scale = width.min() / 4.0
        order = 10.0 ** int(np.log10(scale))
        if scale / order < 2.0:
            scale = 1.0 * order
        elif scale / order < 5.0:
            scale = 2.0 * order
        else:
            scale = 5.0 * order
        scale_label = r'%g%s' % (scale,
                                 ' $%s$' % width.units.latex()
                                 if getattr(width, 'units', None) is not None
                                 else '')
        line = np.array([[extent[0, 0] + 0.05 * extent[0].ptp(),
                          extent[0, 0] + 0.05 * extent[0].ptp() + scale],
                         [extent[1, 0] + 0.12 * extent[1].ptp(),
                          extent[1, 0] + 0.12 * extent[1].ptp()]])
        if outline:
            ax.plot(line[0], line[1], color=outline[1], linewidth=3 + outline[0])
        ax.plot(line[0], line[1], color=fontcolor, linewidth=3)
        ax.text(np.mean(line[0]), extent[1, 0] + 0.10 * extent[1].ptp(),
                scale_label, color=fontcolor, size=0.9 * fontsize,
                horizontalalignment='center', verticalalignment='top',
                transform=ax.transData, path_effects=path_effects)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    else:
        import warnings
        warnings.warn('Unknown scaling indicator scaleind="%s"!' % scaleind)


def add_cbar(ax, cbartitle, clim, cmap=None, fontcolor='black', fontsize=14,
             fontoutline=None, nticks=None, tick_dist=None):
    '''
    Adding a pygad standard colorbar for a map.

    Args:
        ax (AxesSubplot):       The axis to set the indicators for.
        cbartitle (str):        The title for the color bar.
        clim (array-like):      The limits of the color bar.
        cmap (str, Colormap):   The colormap to use.
        fontcolor (str):        The font color for the ticks and the title.
        fontsize (int,float):   The font size of the title (ticks are 0.65 this
                                size).
        fontoutline (array-like):
                                Draw an outline around the text for better
                                readability. The first element is the thickness,
                                the second the color.
        nticks (int):           Set the number of ticks in the colorbar. If None,
                                it will be chosen automatically.
        tick_dist (float):      The distance between the tick labels, i.e. they
                                are placed at multiples of the given number.

    Returns:
        cbar (Colorbar):        The added colorbar instance.
    '''
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.patheffects

    if fontoutline is True:
        fontoutline = [
            3,
            np.array([1, 1, 1]) - mpl.colors.ColorConverter.to_rgb(fontcolor)
        ]
    if fontoutline is None:
        path_effects = []
    else:
        path_effects = [mpl.patheffects.withStroke(linewidth=fontoutline[0],
                                                   foreground=fontoutline[1]),
                        mpl.patheffects.Normal()]

    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)
    cax = inset_axes(ax, width="70%", height="3%", loc=1)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])

    cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                     orientation='horizontal')

    if tick_dist is not None:
        cbar.locator = mpl.ticker.MultipleLocator(tick_dist)
    if nticks is not None:
        cbar.locator = mpl.ticker.MaxNLocator(nbins=nticks)
    cbar.update_ticks()

    cbar.ax.tick_params(labelsize=8)
    for tl in cbar.ax.get_xticklabels():
        tl.set_path_effects(path_effects)
        tl.set_color(fontcolor)
        tl.set_fontsize(0.65 * fontsize)

    cbar.set_label(cbartitle, color=fontcolor, fontsize=fontsize, labelpad=12,
                   path_effects=path_effects)

    return cbar

