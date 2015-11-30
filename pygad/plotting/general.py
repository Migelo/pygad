'''
Module for general convenience routines for plotting.

Doctests impossible, since they would require visual inspection...
'''
__all__ = ['cm_k_b', 'cm_k_y', 'cm_age', 'cm_k_g', 'cm_k_p', 'color_code',
           'show_image', 'scatter_map']

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from ..units import *
from ..binning import *

# used 'http://colormap.org' for creation
cm_k_b = LinearSegmentedColormap('BlackBlue',
        {'red':   ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (1.0, 0.3, 0.3)),
         'green': ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (1.0, 0.75, 0.75)),
         'blue':  ((0.0, 0.0, 0.0),
                   (0.25, 0.5, 0.5),
                   (1.0, 1.0, 1.0))
        })
cm_k_b.set_bad('black')

cm_k_y = LinearSegmentedColormap('BlackYellow',
        {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 0.2, 0.2),
                   (1.0, 0.55, 0.55))
        })
cm_k_y.set_bad('black')

cm_k_g = LinearSegmentedColormap('BlackGreen',
        {'red':   ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (1.0, 0.3, 0.3)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),
         'blue':  ((0.0, 0.0, 0.0),
                   (0.25, 0.0, 0.0),
                   (1.0, 0.3, 0.3))
        })
cm_k_g.set_bad('black')

cm_k_p = LinearSegmentedColormap('BlackPurple',
        {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 0.4, 0.4)),
         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.1, 0.1)),
         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        })
cm_k_p.set_bad('black')

cm_age = LinearSegmentedColormap('Age',
        {'red':   ((0.0,  0.45, 0.45),
                   (0.1,  0.80, 0.80),
                   (0.15, 0.90, 0.90),
                   (0.25, 1.00, 1.00),
                   (1.0,  1.00, 1.00)),
         'green': ((0.0,  0.50, 0.50),
                   (0.1,  0.82, 0.82),
                   (0.15, 0.85, 0.85),
                   (0.30, 0.60, 0.60),
                   (1.0,  0.25, 0.25)),
         'blue':  ((0.0,  0.95, 0.95),
                   (0.1,  0.90, 0.90),
                   (0.15, 0.85, 0.85),
                   (0.30, 0.15, 0.15),
                   (1.0,  0.05, 0.05))
        })

def color_code(im_bright, im_col, cmap='jet', vlim=None, clim=None):
    '''
    Colorcode a mono-chrome image with onother one.

    Args:
        im_bright (np.ndarray): The mono-chrome image to color code. In other
                                words: the map of the brightness values. Needs to
                                be of shape (w,h).
        im_col (np.ndarray):    The image/map values that are converted into
                                colors. Needs to be of the same shape as
                                im_bright: (w,h).
        cmap (str, Colormap):   The colormap to use to convert the values of
                                im_col into actual colors.
        vlim (sequence):        The limits for brightnesses.
        clim (sequence):        The limits for the colors.

    Returns:
        im (np.ndarray):        The rgb image (shape (w,h,3)).
    '''
    if vlim is None:
        vlim = np.percentile(im_bright[np.isfinite(im_bright)], [0,100])
    if clim is None:
        clim = np.percentile(im_col[np.isfinite(im_col)], [0,100])

    im_bright = scale01(im_bright, vlim)
    im_col = scale01(im_col, clim)

    if isinstance(cmap,str): cmap = mpl.cm.get_cmap(cmap)
    im = cmap(im_col)

    im[:,:,0] *= im_bright
    im[:,:,1] *= im_bright
    im[:,:,2] *= im_bright

    return im

def show_image(im, extent=None, cmap='jet', vlim=None, aspect=None,
               interpolation='nearest', ax=None, **kwargs):
    '''
    Show an image with the 'physicsit's orientation'.

    The first axis of im is the x-direction and the second one the y-direction,
    contrary to the default behaviour of plt.imshow.

    Args:
        im (array-like):    The image to show. It can be a luminance image (shape
                            (w,h)), a rgb-image (shape (w,h,3)) or a rgba-image
                            (shape (w,h,4)).
        extent (array-like):The ranges of the image. It can either be a sequence
                            of four values (xmin,xmax,ymin,ymax) or an array-like,
                            of the structure [[xmin,xmax],[ymin,ymax]].
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
    if len(im.shape)!=2 and not (len(im.shape)==3 and im.shape[-1] in [3,4]):
        raise ValueError('Image has to have shape (w,h), (w,h,3) or (w,h,4)!')

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    vmin,vmax = vlim if vlim is not None else (None,None)
    if extent is not None:
        extent = np.asarray(extent).ravel()

    if len(im.shape) == 2:
        im = im.T
    else:
        im = np.dstack( [im[:,:,i].T for i in range(im.shape[-1])] )

    im = ax.imshow(im, origin='lower', extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect=aspect, interpolation=interpolation, **kwargs)

    return fig, ax, im

def scatter_map(x, y, s=None, qty=None, bins=150, extent=None, logscale=False,
                vlim=None, cmap=None, colors=None, colors_av=None, clim=None,
                clogscale=False, cbartxtcol=None, cbartitle=None, showcbar=True,
                aspect='auto', ax=None, **kwargs):
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
        cbartxtcol (str):       The color of the threshold lines.
        cbartitle (str):        A specific colorbar title. If not given, a
                                automatic one is chosen.
        showcbar (bool):        Whether to show a colorbar for the color-coding.
        aspect (str, float):    The aspect ratio of the plot.
        ax (AxesSubplot):       The axis object to plot on. If None, a new one is
                                created by plt.subplots().
        **kwargs:               Further arguments are passed to show_image.

    Returns:
        fig (Figure):       The figure of the axis plotted on.
        ax (AxesSubplot):   The axis plotted on.
        im (AxesImage):     The image instance created.
       [cbar (Colobar):     The colorbar, if showcbar is True.]
    '''
    xname = ''
    yname = ''
    qtyname = ''
    cname = ''
    if isinstance(x,str):
        xname = x
        x = s.get(x)
    if isinstance(y,str):
        yname = y
        y = s.get(y)
    if isinstance(qty,str):
        qtyname = qty
        qty = s.get(qty)
    if isinstance(cmap,str):
        cmap = mpl.cm.get_cmap(cmap)
    elif cmap is None:
        if colors is None:
            cmap = mpl.cm.get_cmap('Blues')
        else:
            cmap = mpl.cm.get_cmap('jet')
            if cbartxtcol is None:
                cbartxtcol = 'w'
    if isinstance(colors,str):
        cname = colors
        colors = s.get(colors)
    if isinstance(colors_av,str):
        colors_av = s.get(colors_av)
    if cbartxtcol is None:
        cbartxtcol = 'k'

    if extent is None:
        extent = np.asarray( [[np.min(x), np.max(x)], [np.min(y), np.max(y)]] )
    else:
        if getattr(extent,'units',None):
            extent = [extent[0], extent[1]]
            if getattr(x,'units',None) is not None:
                extent[0] = extent[0].in_units_of(x.units)
            if getattr(y,'units',None) is not None:
                extent[1] = extent[1].in_units_of(y.units)
        extent = np.asarray(extent)

    grid = gridbin2d(x, y, qty, bins=bins, extent=extent, nanval=0.0)
    if getattr(vlim,'units',None) is not None \
            and getattr(grid,'units',None) is not None:
        vlim = vlim.in_units_of(grid.units)
    if logscale:
        grid = np.log10( grid )
        if vlim is not None:
            vlim = np.log10(vlim)
    if vlim is None:
        finitegrid = grid[np.isfinite(grid)]
        vlim = [finitegrid.min(), finitegrid.max()]
        del finitegrid

    if colors is None:
        clim = vlim
    else:
        if colors_av is not None:
            col = gridbin2d(x, y, colors_av*colors, bins=bins, extent=extent,
                            nanval=0.0)
            col /= gridbin2d(x, y, colors_av, bins=bins, extent=extent,
                             nanval=0.0)
            col[np.isnan(col)] = 0.0
        else:
            col = gridbin2d(x, y, colors, bins=bins, extent=extent, nanval=0.0)

        if clogscale:
            col = np.log10( col )
            if clim is not None:
                clim = np.log10(clim)

        if clim is None:
            finitegrid = col[np.isfinite(col)]
            clim = [finitegrid.min(), finitegrid.max()]
            del finitegrid
        grid = color_code(grid, col, cmap=cmap, vlim=vlim, clim=clim)

    fig, ax, im = show_image(grid, extent=extent, cmap=cmap, aspect=aspect, ax=ax,
                             **kwargs)

    if showcbar:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(ax, width="70%", height="3%", loc=1)
        norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm,
                                         orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        for tl in cbar.ax.get_xticklabels():
            tl.set_color(cbartxtcol)
        if cbartitle is None:
            name = cname if colors is not None else qtyname
            count_units = r'$\mathrm{count}$'
            if colors is not None:
                units = getattr(colors,'units',None)
            elif qty is not None:
                units = getattr(qty,'units',None)
            else:
                units = count_units
            if units == 1 or units is None:
                units = ''
            elif units is not count_units:
                units = r'[$%s$]' % units.latex()
            cbartitle = name + (' ' if (name!='' and units!='') else '') + units
            if (logscale and colors is None) or \
                    (clogscale and colors is not None):
                cbartitle = r'$\log_{10}$(' + cbartitle + ')'
        cbar.set_label(cbartitle, color=cbartxtcol, fontsize=12, labelpad=12)

    xunits = r'[$%s$]' % x.units.latex() if getattr(x,'units',None) else ''
    yunits = r'[$%s$]' % y.units.latex() if getattr(y,'units',None) else ''
    ax.set_xlabel( '%s%s' % (xname, xunits), fontsize=16 )
    ax.set_ylabel( '%s%s' % (yname, yunits), fontsize=16 )

    if showcbar:
        return fig, ax, im, cbar
    else:
        return fig, ax, im

