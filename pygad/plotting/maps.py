'''
Module for convenience routines for plotting maps.

Doctests impossible, since they would require visual inspection...
'''
__all__ = ['image', 'phase_diagram']

import numpy as np
import matplotlib as mpl
from general import *
from ..units import *
from ..binning import *
from ..gadget import config
from ..snapshot import BoxMask
import warnings
from .. import environment

def image(s, qty=None, av=None, units=None, logscale=None, surface_dens=None,
          field=None, reduction=None, extent=None, Npx=256, xaxis=0, yaxis=1,
          vlim=None, cmap=None, normcmaplum=True, desat=0.1, colors=None,
          colors_av=None, cunits=None, clogscale=None, csurf_dens=None, clim=None,
          ax=None, showcbar=True, cbartitle=None, scaleind='line',
          scaleunits=None, fontcolor='white', fontsize=14,
          interpolation='nearest', maps=None, zero_is_white=False, **kwargs):
    '''
    Show an image of the snapshot.

    By default the mass is plotted with the colormap 'cubehelix' if `colors` is
    None or 'Bright' if `colors` is set. For stars only, however, the (V-band
    weighted) stellar ages are plotted with the colormap 'Age' and image luminance
    is the visual stellar luminosity (V-band); for gas only the mass-weighted
    log10(temperature) is colorcoded with 'Bright' and the image luminance is the
    surface density; and for dark matter only the colormap 'BlackGreen' is used.

    The defaults -- if `qty` and `av` are None -- are:
        for stars only:
            qty='lum_v', av=None, logscale=True, surface_dens=True
        otherwise:
            qty='mass', av=None, logscale=True, surface_dens=True
    if additionally `colors` and `colors_av` are None as well, the following
    defautls apply:
        for stars only:
            colors='age.in_units_of("Gyr")', colors_av='lum_v', cmap='Age',
            clim=[0,13], cbartitle = '(V-band weighted) age $[\mathrm{Gyr}]$'
        for gas only:
            colors='temp.in_units_of("K")', colors_av='rho', cmap='Bright',
            clogscale=True, cbartitle=r'$\log_{10}(T\,[\mathrm{K}])$'
        for otherwise:
            cmap=('BlackGreen' if len(s.baryons)==0 else 'BlackPurple'),
            cbartitle=(r'$\log_{10}(\Sigma\,[%s])$' % units.latex())

    Args:
        s (Snap):           The (sub-)snapshot to use.
        qty (UnitQty, str): The quantity to map. It can be a UnitArr of length
                            L=len(s) and dimension 1 (i.e. shape (L,)) or a string
                            that can be passed to s.get and returns such an array.
        av (UnitQty, str):  The quantity to average over. Otherwise as 'qty'.
        units (str, Unit):  The units to plot in (qty respectively qty/area).
        logscale (bool):    Whether to plot the image luminance in log-scale.
                            Default: True
        surface_dens (bool):Whether to plot the surface density of qty rather than
                            just sum it up along the line of sight per pixel.
        reduction (str):    If not None, interpret the SPH quantity not as a SPH
                            field, but as a particle property and reduce with this
                            given method along the third axis / line of sight.
        field (bool):       If no `reduction` is given, this determines whether
                            the SPH-quantity is interpreted as a density-field or
                            its integral quantity.
                            For instance: rho would be the density-field of the
                            integral quantity mass.
        extent (UnitQty):   The extent of the image. It can be a scalar and then
                            is taken to be the total side length of a square
                            around the origin or a sequence of the minima and
                            maxima of the directions: [[xmin,xmax],[ymin,ymax]]
        Npx (int, sequence):The number of pixel per side. Either an integer that
                            is taken for both sides or a pair of such, the first
                            for the x-direction, the second for the y-direction.
        xaxis (int):        The coordinate for the x-axis. (0:'x', 1:'y', 2:'z')
        yaxis (int):        The coordinate for the y-axis. (0:'x', 1:'y', 2:'z')
        vlim (sequence):    The limits of the quantity for the plot.
        cmap (str,Colormap):The colormap to use. If colors==None it is applied to
                            qty, otherwise to colors. Default: 'cubehelix' if
                            `colors` is not set, else 'Bright'.
        normcmaplum (bool): If there are colors and luminance information,
                            norm the colormap's luminance.
        desat (float):      If there are colors and luminance information,
                            desaturate the colormap by this factor before norming
                            its luminance. (See `plotting.general.isolum_cmap` for
                            more information!)
        colors (UnitQty, str):
                            The array for color coding (qty is just luminance
                            unless this is None). Otherwise same as qty.
        colors_av (UnitQty, str):
                            Same as av, but for colors.
        cunits (str, Unit): Same as units, but for colors.
        clogscale (bool):   Whether color-code in log-scale.
        csurf_dens (bool):  Same as surface_dens, but for colors.
        clim (sequence):    Same as vlim, but for colors.
        ax (AxesSubplot):   The axis object to plot on. If None, a new one is
                            created by plt.subplots().
        showcbar (cool):    Whether to add a colorbar (in the upper left corner
                            within the axis).
        cbartitle (str):    The title for the colorbar. In certain cases it can be
                            created automatically.
        scaleind (str):     Can be:
                                'labels':   ticks and labels are drawn on the axes
                                'line':     a bar is drawn in the lower left
                                            corner indicating the scale
                                'none':     neither nor is done -- make the
                                            axes unvisible
        scaleunits (str, Unit):
                            If scaleind=='line', these units are used for
                            indication.
        fontcolor (str):    The color to use for the colorbar ticks and labels as
                            well as for the scale bar.
        fontsize (int):     The size of the labels and other font sizes are scaled
                            accordingly.
        interpolation (str):The interpolation to use between pixels. See imshow
                            for more details.
        softening (UnitQty):A list of te softening lengthes that are taken for
                            smoothing the maps of the paticle types. Is
                            consequently has to have length 6. Default: None.
        maps (dict):        If this is a dictionary, the created 2D grids (qty,
                            and colors) are added for later use.
        zero_is_white (bool):
                            Instead of scaling the image luminance by `qty` if
                            there is a color given, desaturate the colors to white
                            for `qty` values approaching the lower limit of vlim
                            (or zero if this is None).
        **kwargs:           Further keyword arguments are to pass to map_qty. I
                            want to mention 'softening' here, which is a list of
                            the sotening lengthes used to smooth the maps of the
                            different particle types. For more details, however,
                            see the documentation of 'map_qty'.

    Returns:
        fig (Figure):       The figure of the axis plotted on.
        ax (AxesSubplot):   The axis plotted on.
        im (AxesImage):     The image instance created.
       [cbar (Colorbar):    The colorbar, if showcbar is True.]
    '''
    zaxis = (set([0,1,2]) - set([xaxis, yaxis])).pop()
    if set([xaxis, yaxis, zaxis]) != set([0,1,2]):
        raise ValueError('x- and y-axis must be different and in [0,1,2], ' + \
                         'but it is xaxis=%s, yaxis=%s!' % (xaxis, yaxis))

    # setting default values for arguments
    if units is not None:
        units = Unit(units)
    
    if cunits is not None:
        cunits = Unit(cunits)

    if extent is None:
        extent = UnitArr([np.percentile(s['pos'][:,xaxis], [1,99]),
                          np.percentile(s['pos'][:,yaxis], [1,99])],
                         s['pos'].units)
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, dim=2)
    extent = extent.in_units_of(s['pos'].units, subs=s)
    res = res.in_units_of(s['pos'].units, subs=s)

    # mask snapshot for faster plotting
    ext3D = UnitArr(np.empty((3,2)), extent.units)
    ext3D[xaxis] = extent[0]
    ext3D[yaxis] = extent[1]
    ext3D[zaxis] = [-np.inf, +np.inf]
    bmask = BoxMask(ext3D, sph_overlap=True)
    mask = bmask._get_mask_for(s)
    s = s[bmask]

    if isinstance(qty, (list,tuple)):   # enable masking for non-standard
        qty = np.asarray(qty)           # containers
    if isinstance(qty, np.ndarray):     # includes the derived UnitArr and SimArr
        qty = qty[mask]
    if isinstance(av, (list,tuple)):   # enable masking for non-standard
        av = np.asarray(av)            # containers
    if isinstance(av, np.ndarray):     # includes the derived UnitArr and SimArr
        av = av[mask]
    if 'dV' in kwargs:
        if isinstance(kwargs['dV'], (list,tuple)):  # enable masking for non-standard
            kwargs['dV'] = np.asarray(kwargs['dV']) # containers
        if isinstance(kwargs['dV'], np.ndarray):    # includes the derived UnitArr and SimArr
            kwargs['dV'] = kwargs['dV'][mask]

    if qty is None and av is None:
        """
        if (len(s)!=0 and len(s.gas)==len(s)) \
                or (s.descriptor.endswith('gas') and len(s)==0):
            qty, av = 'mass', None
            if logscale is None:        logscale = True
            if surface_dens is None:    surface_dens = True
        """
        if (len(s)!=0 and len(s.stars)==len(s)) \
                or (s.descriptor.endswith('stars') and len(s)==0):
            # stars only
            qty, av = 'lum_v', None
            if logscale is None:        logscale = True
            if surface_dens is None:    surface_dens = True
        else:
            qty, av = 'mass', None
            # needed for cbartitle:
            if units is None:           units = s['mass'].units / s['pos'].units**2
            if logscale is None:        logscale = True
            if surface_dens is None:    surface_dens = True
            if field is None:           field = False
        if colors is None and colors_av is None:
            if (len(s)!=0 and len(s.stars)==len(s)) \
                    or (s.descriptor.endswith('stars') and len(s)==0):
                # stars only
                colors, colors_av = 'age.in_units_of("Gyr")', 'lum_v'
                if cmap is None:        cmap = 'Age'
                if clim is None:        clim = [0,13]
                if cbartitle is None:   cbartitle = '(V-band weighted) age ' + \
                                                    '$[\mathrm{Gyr}]$'
            elif (len(s)!=0 and len(s.gas)==len(s)) \
                    or (s.descriptor.endswith('gas') and len(s)==0):
                # gas only
                colors, colors_av = 'temp.in_units_of("K")', 'rho'
                if clogscale is None:   clogscale = True
                if cmap is None:        cmap = 'Bright'
                if cbartitle is None:   cbartitle = r'$\log_{10}(T\,[\mathrm{K}])$'
                if field is None:       field = False
            else:
                if cmap is None:        cmap = 'BlackGreen' if len(s.baryons)==0 \
                                                    else 'BlackPurple'
                if cbartitle is None:   cbartitle = r'$\log_{10}(\Sigma\,[%s])$' % (
                                                        units.latex() )
    if logscale is None: logscale = True
    if cmap is None: cmap = 'cubehelix' if colors is None else 'Bright'
    if csurf_dens is None: csurf_dens = False
    if field is None:
        if reduction is None:
            field = (len(s.gas)==len(s) and len(s.gas)>0)
        else:
            field = False
    if surface_dens is None: surface_dens = (reduction is None and not field)
    if reduction is not None:
        field = False
        surface_dens = False

    if scaleunits is None:
        scaleunits = s['pos'].units
    else:
        scaleunits = Unit(scaleunits)
        extent.convert_to(scaleunits, subs=s)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'plot a map - paramters:'
        if isinstance(qty,(str,unicode)) or qty is None:
            print '  qty:         ', qty
        else:
            print '  qty:         ', type(qty)
        if isinstance(av,(str,unicode)) or av is None:
            print '  av:          ', av
        else:
            print '  av:          ', type(av)
        if isinstance(colors,(str,unicode)) or colors is None:
            print '  colors:      ', colors
        else:
            print '  colors:      ', type(colors)
        if isinstance(colors_av,(str,unicode)) or colors_av is None:
            print '  colors_av:   ', colors_av
        else:
            print '  colors_av:   ', type(colors_av)
        if reduction is None:
            print '  field:       ', 'ignored'
            print '  surface_dens:', 'ignored'
        else:
            print '  field:       ', field
            print '  surface_dens:', surface_dens
        print '  logscale:    ', logscale
        print '  clogscale:   ', clogscale
        print '  reduction:   ', reduction
        print '  [...]'

    # create luminance map
    if len(s) == 0:
        if isinstance(qty,(str,unicode)) and units is None:
            raise RuntimeError('Snapshot is empty and no units for the image are '
                               'given -- cannot create an empty map of correct '
                               'units!')
        if vlim is None: vlim = [1,2]
        im_lum = UnitArr(vlim[0]*np.zeros(Npx),
                         units if units is not None else
                             getattr(qty,'units',None))
        px2 = np.prod(res)
    else:
        im_lum, px2 = map_qty(s, extent=extent, field=field, qty=qty, av=av,
                              reduction=reduction, Npx=Npx,
                              xaxis=xaxis, yaxis=yaxis, **kwargs)
        if surface_dens:
            im_lum /= px2
        if units is not None:
            im_lum.convert_to(units, subs=s)
    if logscale:
        if isinstance(qty,(str,unicode)) and 'log' in qty:
            import sys
            print >> sys.stderr, 'WARNING: in log-scale, but "qty" already ' + \
                                 'contains "log"!'
        im_lum = np.log10(im_lum)
        if vlim is not None:
            vlim = map(np.log10, vlim)
    if maps is not None:
        maps['qty'] = im_lum

    if vlim is None:
        tmp = im_lum[np.isfinite(im_lum)]
        if len(tmp)==0:
            raise RuntimeError('The luminance array has no finite values!')
        if colors is None:
            vlim = np.percentile(tmp, [0.1, 99.9])
        else:
            vlim = np.percentile(tmp, [1, 98.5])
        del tmp

    # create color map
    if colors is None:
        im = im_lum
        clim = vlim
    elif len(s)==0:
        im = np.zeros(tuple(Npx)+(3,))  # at vlim[0] -> black with any color coding
    else:
        im_col, px2 = map_qty(s, extent=extent, field=field, qty=colors,
                              av=colors_av, reduction=reduction, Npx=Npx,
                              xaxis=xaxis, yaxis=yaxis,
                              **kwargs)
        if csurf_dens:
            im_col /= px2
        if cunits is not None:
            im_col.convert_to(cunits, subs=s)
        if clogscale:
            im_col = np.log10(im_col)
        if maps is not None:
            maps['colors'] = im_col
        if clim is None:
            clim = np.percentile(im_col[np.isfinite(im_col)], [0.1,99.99])
        if normcmaplum:
            cmap = isolum_cmap(cmap, desat=desat)
        im = color_code(im_lum, im_col, cmap=cmap, vlim=vlim, clim=clim,
                        zero_is_white=zero_is_white)

    fig, ax, im = show_image(im, extent=extent, ax=ax, cmap=cmap, vlim=vlim,
                             interpolation=interpolation)

    if showcbar:
        if cbartitle is None:
            cqty = colors if colors is not None else qty
            if isinstance(cqty,(str,unicode)):
                cname = cqty
                if reduction is not None:
                    cname = '%s(%s)' % (reduction, cname)
                if colors is not None:
                    if cunits is None and len(s)>0:
                        cunits = s.get(cqty).units
                        if field:
                            cunits = (cunits * s['pos'].units).gather()
                        if csurf_dens:
                            cunits = (cunits * px2.units).gather()
                    if csurf_dens:
                        cname = 'surface-density of ' + cname
                else:
                    if cunits is None:
                        if units is None:
                            if len(s) == 0:
                                cunits = None
                            else:
                                cunits = s.get(cqty).units
                                if field:
                                    cunits = (cunits * s['pos'].units).gather()
                        else:
                            cunits = units
            else:
                cname = ''
                if cunits is None:
                    cunits = units if units is not None else getattr(cqty,'units',None)
            if surface_dens and colors is None:
                if units is None and cunits is not None:
                    cunits = cunits/s['pos'].units**2
                cname = r'$\Sigma$ of ' + cname
            if cunits is None or cunits == 1:
                cunits = ''
            else:
                cunits = r'[$%s$]' % cunits.latex()
            cbartitle = cname + (' ' if (cname!='' and cunits!='') else '') + cunits
            if (logscale and colors is None) or \
                    (clogscale and colors is not None):
                cbartitle = r'$\log_{10}$(' + cbartitle + ')'

        cbar = add_cbar(ax, cbartitle, clim=clim, cmap=cmap,
                        fontcolor=fontcolor, fontsize=fontsize,
                        nticks=7)

    make_scale_indicators(ax, extent, scaleind=scaleind, scaleunits=scaleunits,
                          xaxis=xaxis, yaxis=yaxis, fontsize=fontsize,
                          fontcolor=fontcolor)

    if showcbar:
        return fig, ax, im, cbar
    else:
        return fig, ax, im

def phase_diagram(s, rho_units='Msol/pc**3', T_units='K',
                  T_threshold=None, rho_threshold=None,
                  threshold_col='black', **kwargs):
    '''
    Plot a phase diagram with the possiblity to color code it with some other
    quantity.

    Args:
        s (Snap):               The (sub-)snapshot from which to use the gas of.
        rho_units, T_units (Unit):
                                The units to plot in.
        T_threshold, rho_threshold (UnitScalar):
                                If not None, a line at this temperature and/or
                                density is drawn.
        threshold_col (str):    The color of the threshold lines.
        **kwargs:               Further arguments are passed to `scatter_map`.
                                Note the option to color by some other quantity
                                using `colors` (and `colors_av`)!

    Returns:
        fig (Figure):       The figure of the axis plotted on.
        ax (AxesSubplot):   The axis plotted on.
        im (AxesImage):     The image instance created.
       [cbar (Colorbar):    The colorbar, if showcbar is True.]
    '''
    if 'logscale' not in kwargs:    kwargs['logscale'] = True
    if 'showcbar' not in kwargs:    kwargs['showcbar'] = False
    res = scatter_map(np.log10(s['rho'].in_units_of(rho_units)),
                      np.log10(s['temp'].in_units_of(T_units)),
                      s, **kwargs)
    if kwargs['showcbar']:
        fig, ax, im, cbar = res
    else:
        fig, ax, im = res

    if rho_threshold:
        th = UnitScalar(rho_threshold)
        try:
            th = th.in_units_of(rho_units)
        except:
            # particle threshold, take the primorial mean particle mass
            # TODO: improve?
            from ..physics import m_u
            th = ((0.75*1+0.25*4)*m_u*th).in_units_of(rho_units)
        ax.vlines( np.log10(th), ax.get_ylim()[0], ax.get_ylim()[1],
                   color=threshold_col )
    if T_threshold:
        th = UnitScalar(T_threshold)
        th = th.in_units_of(T_units)
        ax.hlines( np.log10(th), ax.get_xlim()[0], ax.get_xlim()[1],
                   color=threshold_col )

    ax.set_xlabel( r'$\log_{10}(\rho\,[%s])$' % Unit(rho_units).latex(),
                   fontsize=16 )
    ax.set_ylabel( r'$\log_{10}(T\,[%s])$' % Unit(T_units).latex(),
                   fontsize=16 )

    if kwargs['showcbar']:
        return fig, ax, im, cbar
    else:
        return fig, ax, im

