'''
Module for convenience routines for plotting maps.

Doctests impossible, since they would require visual inspection...
'''
__all__ = ['image', 'phase_diagram', 'over_plot_species_phases', 'vec_field']

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from general import *
from ..units import *
from ..binning import *
from ..gadget import config
from ..snapshot import BoxMask
import warnings
from .. import environment

def image(s, qty=None, av=None, units=None, logscale=None, surface_dens=None,
          field=None, reduction=None, extent=None, Npx=256, xaxis=0, yaxis=1,
          vlim=None, cmap=None, normcmaplum=True, desat=None, colors=None,
          colors_av=None, cunits=None, clogscale=None, csurf_dens=None, clim=None,
          ax=None, showcbar=True, cbartitle=None, scaleind='line',
          scaleunits=None, fontcolor='white', fontsize=14,
          interpolation='nearest', maps=None, zero_is_white=False, **kwargs):
    '''
    Show an image of the snapshot.

    By default the mass is plotted with the colormap `CM_DEF` if `colors` is
    None or 'isolum' if `colors` is set. For stars only, however, the (V-band
    weighted) stellar ages are plotted with the colormap 'Age' and image luminance
    is the visual stellar luminosity (V-band); for gas only the mass-weighted
    log10(temperature) is colorcoded with 'isolum' and the image luminance is the
    surface density; and for dark matter only the colormap 'BlackGreen' is used.

    The defaults -- if `qty` and `av` are None -- are:
        for stars only:
            qty='lum_v', av=None, logscale=True, surface_dens=True
        otherwise:
            qty='mass', av=None, logscale=True, surface_dens=True
    if additionally `colors` and `colors_av` are None as well, the following
    defautls apply:
        for stars only:
            colors='age.in_units_of("Gyr")', colors_av='lum_v', cmap='age',
            clim=[0,13], cbartitle = '(V-band weighted) age $[\mathrm{Gyr}]$'
        for gas only:
            colors='temp.in_units_of("K")', colors_av='rho', cmap='isolum',
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
                            qty, otherwise to colors. Default: `CM_DEF` if
                            `colors` is not set, else 'isolum'.
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
                if cmap is None:        cmap = 'age'
                if clim is None:        clim = [0,13]
                if cbartitle is None:   cbartitle = '(V-band weighted) age ' + \
                                                    '$[\mathrm{Gyr}]$'
            elif (len(s)!=0 and len(s.gas)==len(s)) \
                    or (s.descriptor.endswith('gas') and len(s)==0):
                # gas only
                colors, colors_av = 'temp.in_units_of("K")', 'rho'
                if clogscale is None:   clogscale = True
                if cmap is None:        cmap = 'isolum'
                if cbartitle is None:   cbartitle = r'$\log_{10}(T\,[\mathrm{K}])$'
                if field is None:       field = False
            else:
                if cmap is None:        cmap = 'BlackGreen' if len(s.baryons)==0 \
                                                    else 'BlackPurple'
                if cbartitle is None:   cbartitle = r'$\log_{10}(\Sigma\,[%s])$' % (
                                                        units.latex() )
    if logscale is None: logscale = True
    if cmap is None: cmap = CM_DEF if colors is None else 'isolum'
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

def phase_diagram(s, rho_units='g/cm**3', T_units='K',
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

def over_plot_species_phases(s, species=None, extent=None, enclose=0.8,
                             frac_rel_to='phase', species_labels=None,
                             rho_units='g/cm**3', T_units='K', ax=None,
                             linewidths=2.5, colors=None, linestyles=None,
                             phase_kwargs=None):
    '''
    Plot the contours of specified species over the overall phase diagram.

    Args:
        s (Snap):                   The (sub-)snapshot from which to use the gas
                                    of.
        species (iterable):         The names (as in the block names) of species
                                    to plot. This needs to be a positive
                                    semi-definite quantity (like mass,
                                    luminosities, or volume).
                                    Default:
                                    ['HI', 'MgII', 'SiIII', 'CIV', 'OVI'].
        extent (array-like):        The extent of the phase diagram in the form
                                    [[rho_min,rho_max],[T_min,T_max]].
                                    Default: [[-31,-22],[2,7.5]].
        enclose (float, iterable):  Plot the contour lines such that they enclose
                                    this fraction of the total species quantity.
        frac_rel_to (str):          Whether to relate the enclosed fraction to the
                                    entire snapshot passed ('snap') or just the
                                    part that ends up in the phase diagram
                                    ('phase').
        species_labels (iterable):  The labels for the different species. Defaults
                                    to `species`.
        rho_units, T_units (Unit):  The units to plot in.
        ax (AxesSubplot):           The axis object containing a phase diagram to
                                    plot on. If None, a new one is created by
                                    `phase_diagram`.
        linewidths (float, iterable):
                                    The linewidths for the levels defined by
                                    `enclose`.
        colors (iterable, ...):     Color(s) for the contours of the different
                                    species distributions.
        linestyles (iterable, ...): The linestyle(s) for the contours of the
                                    different species distributions.
        phase_kwargs (dict):        Keyword arguments that are passed to
                                    `phase_diagram`, if `ax` is None. Default
                                    values are:
                                    qty='mass', showcbar=True, cmap='gray_r',
                                    fontcolor='k', and extent=extent.

    Returns:
        fig (Figure):               The figure of the axis plotted on.
        ax (AxesSubplot):           The axis plotted on.
    '''
    from numbers import Number
    from scipy.optimize import brentq
    if species is None:
        species = ['HI', 'MgII', 'SiIII', 'CIV', 'OVI']
    if species_labels is None:
        species_labels = species
    if len(species) != len(species_labels):
        raise ValueError('`species_labels` must be as long as `species`!')
    if extent is None:
        extent = [[-31,-22],[2,7.5]]
    extent = np.array(extent)
    if isinstance(enclose, Number):
        enclose = [enclose]
    if isinstance(linewidths, Number):
        linewidths = [linewidths] * len(enclose)
    if len(enclose) != len(linewidths):
        raise ValueError("Need as many linewidths as `enclose` levels!")
    argsort = np.argsort( enclose )
    linewidths = np.array(linewidths)[argsort[::-1]]
    enclose = np.array(enclose)[argsort[::-1]]
    if frac_rel_to not in ['phase', 'snap']:
        raise ValueError('Unkown mode frac_rel_to="%s"' % frac_rel_to)
    if colors is None:
        colors = [ str(c['color']) for c in mpl.rcParams['axes.prop_cycle'] ]
    if not hasattr(colors,'__getitem__'):
        colors = [colors]
    if linestyles is None:
        linestyles = [ ls for ls in mpl.lines.lineStyles.keys()
                        if ls not in ['',' ','None'] ]
    if isinstance(linestyles, str):
        linestyles = [linestyles]
    if phase_kwargs is None:
        phase_kwargs = dict()

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'plot temperature-density distribution of:'
        print '  ', species
        print '...'

    if ax is None:
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print 'plot underlying overall phase-diagram...'
        if 'qty' not in phase_kwargs:       phase_kwargs['qty'] = 'mass'
        if 'showcbar' not in phase_kwargs:  phase_kwargs['showcbar'] = True
        if 'cmap' not in phase_kwargs:      phase_kwargs['cmap'] = 'gray_r'
        if 'fontcolor' not in phase_kwargs: phase_kwargs['fontcolor'] = 'k'
        fig, ax, im, cbar = phase_diagram(s.gas, extent=extent,
                                          rho_units=rho_units, T_units=T_units,
                                          **phase_kwargs)
    else:
        fig = ax.get_figure()
    try:
        renderer = fig.canvas.get_renderer()
    except:
        renderer = None

    log_rho = np.log10( s.gas['rho'].in_units_of(rho_units) )
    log_T = np.log10( s.gas['temp'].in_units_of(T_units) )
    txt_pos = 0.04
    for n,spec in enumerate(species):
        label = species_labels[n]
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print 'plot distribution of %s...' % label
        c = colors[n%len(colors)]
        ls = linestyles[n%len(linestyles)]
        sub = s.gas.get(spec)
        if np.any(sub<0):
            raise ValueError('"%s" is not positive semi-definite!' % spec)
        sub_phase = gridbin2d(log_rho, log_T, qty=sub, extent=extent)
        M_sub = sub_phase.sum() if (frac_rel_to=='phase') else sub.sum()
        levels = [ brentq(lambda l: sub_phase[sub_phase>l].sum()-x*M_sub,
                          0, np.max(sub_phase), maxiter=10000 )
                   for x in enclose ]
        ax.contour(sub_phase.view(np.ndarray).T, extent=extent.flatten(),
                   levels=levels, colors=[c]*len(levels),
                   linewidths=linewidths, linestyles=[ls]*len(levels))
        txt = ax.text(txt_pos, 0.04, label,
                      transform=ax.transAxes, verticalalignment='bottom',
                      fontdict=dict(color=c, fontsize=14),
                      bbox=dict(edgecolor=c, facecolor='none', linewidth=2.5,
                                linestyle=ls))
        try:
            inv_trans = ax.transAxes.inverted()
            bb = inv_trans.transform_bbox(txt.get_window_extent(renderer=renderer))
            txt_pos += bb.width + 0.05
        except:
            txt_pos += 0.12

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print 'all done'
    return fig, ax

def vec_field(s, qty, extent, field=False, av=None, reduction=None,
              xaxis=0, yaxis=1, Npx=30, ax=None, color='k', pivot='mid',
              streamlines=False, **kwargs):
    """
    Plot a vector field (e.g. on an existing plot) using `plt.quiver`.

    Args:
        s (Snap):           The (sub-)snapshot to use.
        qty (UnitQty, str): The quantity to map. It can be a UnitArr of length
                            L=len(s) and dimension 1 (i.e. shape (L,)) or a string
                            that can be passed to s.get and returns such an array.
                            It has to be a vector quantity, of course.
        extent (UnitQty):   The extent of the plot. It can be a scalar and then
                            is taken to be the total side length of a square
                            around the origin or a sequence of the minima and
                            maxima of the directions: [[xmin,xmax],[ymin,ymax]]
        field (bool):       If no `reduction` is given, this determines whether
                            the SPH-quantity is interpreted as a density-field or
                            its integral quantity.
                            For instance: rho would be the density-field of the
                            integral quantity mass.
                            Here it default to False, as typically you would pass
                            'vel', which is not a density.
        av (UnitQty, str):  The quantity to average over. Otherwise as 'qty'.
        reduction (str):    If not None, interpret the SPH quantity not as a SPH
                            field, but as a particle property and reduce with this
                            given method along the third axis / line of sight.
        xaxis (int):        The coordinate for the x-axis. (0:'x', 1:'y', 2:'z')
        yaxis (int):        The coordinate for the y-axis. (0:'x', 1:'y', 2:'z')
        Npx (int, sequence):The number of pixel per side. Either an integer that
                            is taken for both sides or a pair of such, the first
                            for the x-direction, the second for the y-direction.
                            This determines the number of plotted arrows.
        ax (AxesSubplot):   The axis object to plot on. If None, a new one is
                            created by plt.subplots().
        color (str, color, color sequence):
                            The color of the arrows in the vector plot.
        pivot (str):        Where to align the arrows. See `plt.quiver` for more
                            information. Possible choices are: 'tail', 'mid', and
                            'tip'.
        streamlines (bool): Plot streamlines, not individual arrows (using
                            `streamplot`, not `quiver`). The `pivot` argument then
                            is ignored.
        **kwargs:           Further keyword arguments will be passed to `quiver`.

    Returns:
        fig (Figure):       The figure of the axis plotted on.
        ax (AxesSubplot):   The axis plotted on.
    """
    extent, Npx, res = grid_props(extent=extent, Npx=Npx, dim=2)
    extent = extent.in_units_of(s['pos'].units, subs=s)
    if reduction is not None:
        field = False

    if isinstance(qty,(str,unicode)):
        qty_x = s.get('%s[:,%d]' % (qty,xaxis))
        qty_y = s.get('%s[:,%d]' % (qty,yaxis))
    else:
        qty_x = qty[:,xaxis]
        qty_y = qty[:,yaxis]

    map_x, px2 = map_qty(s, extent=extent, field=field,
                         qty=qty_x, av=av, reduction=reduction, Npx=Npx,
                         xaxis=xaxis, yaxis=yaxis)
    map_y, px2 = map_qty(s, extent=extent, field=field,
                         qty=qty_y, av=av, reduction=reduction, Npx=Npx,
                         xaxis=xaxis, yaxis=yaxis)

    X, Y = np.meshgrid(np.linspace(extent[0,0],extent[0,1],Npx[0]),
                       np.linspace(extent[1,0],extent[1,1],Npx[1]))
    mx, my = map_x.view(np.ndarray).T, map_y.view(np.ndarray).T

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    if streamlines:
        ax.streamplot(X, Y, mx, my, color=color, **kwargs)
    else:
        ax.quiver(X, Y, mx, my, color=color, pivot=pivot, **kwargs)

    return fig, ax

