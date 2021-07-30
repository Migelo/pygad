'''
Module for convenience routines for plotting profiles.

Doctests impossible, since they would require visual inspection...
'''
__all__ = ['profile', 'history', 'SFR_history', 'flow_history']

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..units import *
from ..analysis import *
from .. import environment
from .. import utils
from .. import physics
import sys

def profile(s, Rmax, qty, av=None, units=None, dens=True, proj=None,
            N=50, logbin=False, minlog=None, logscale=True, ylabel=None,
            labelsize=14, ignoreZeros=False, ax=None, **kwargs):
    '''
    Plot a profile.

    Args:
        s (Snap):           The (sub-)snapshot to plot the quantity of.
        Rmax (UnitScalar):  The radius to plot out to.
        qty (str, UnitQty): The quantity to plot the profile for. It can be a
                            string that can be passed to s.get or a ready block
                            (of length len(s)).
        av (str, UnitQty):  The quantity to average 'qty' over. Otherwise as
                            'qty'.
        units (str, Unit):  The units in which the profile shall be plotted.
        dens (bool):        Whether to plot a density (surface density if proj is
                            not None, else volume density).
        proj (int):         Project along this axis and plot a cylindrical
                            profile. If None, however, plot a spherical profile.
                            (0:'x', 1:'y', 2:'z')
        N (int):            The number of bins.
        logbin (bool):      Whether to bin logarithmically in radius.
        minlog (UnitScalar):If logbin==True, this is the smallest bin edge (there
                            is one between 0 and this value, though).
                            Default: Rmax/100.
        logscale (bool):    Whether to plot y-values in log-scale.
        ylabel (str):       A custom y-axis label.
        labelsize (int):    The font size of the labels. The tick size will get
                            adjusted accordingly.
        ax (AxesSubplot):   The axis object to plot on. If None, a new one is
                            created by plt.subplots().
        ignoreZeros:        if set to True zero values are replaced by numpy.NaN
        **kwargs:           Further keyword arguments are passed to ax.plot (e.g.
                            'linewidth', 'color', or 'label').

    Returns:
        fig (Figure):       The figure of the axis plotted on.
        ax (AxesSubplot):   The axis plotted on.
    '''
    Rmax = UnitScalar(Rmax, s['pos'].units, subs=s)
    if logbin:
        if minlog is None:
            minlog = Rmax / 100.0
        else:
            minlog = UnitScalar(minlog, s['pos'].units)
        r_edges = np.logspace(np.log10(minlog), np.log10(float(Rmax)), N)
        r_edges = np.array( [0] + list(r_edges) )
        r = 10.0**((np.log10(r_edges[1:-1]) + np.log10(r_edges[2:])) / 2.0)
        r = np.array( [r_edges[1]/(r_edges[2]/r_edges[1])] + list(r) )
    else:
        r_edges = np.linspace(0, float(Rmax), N)
        r = (r_edges[:-1] + r_edges[1:]) / 2.0

    if dens:
        prof = profile_dens(s, qty, av, r_edges, proj=proj)
    else:
        prof = radially_binned(s, qty, av, r_edges, proj=proj)

    if units is not None:
        prof.convert_to(units, subs=s)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if ignoreZeros:
        prof[prof==0] = np.NaN
    ax.plot(r, prof, **kwargs)

    if logbin:
        ax.set_xscale('log')
    if logscale:
        ax.set_yscale('log')

    ax.set_xlabel(r'$%s$ [$%s$]' % ('r' if proj is None else 'R',
                                    s['pos'].units.latex()),
                  fontsize=labelsize)
    if ylabel is None:
        name = ''
        if isinstance(av,str):
            name += str(av) + '-weighted '
        if dens:
            name = r'$\Sigma' if proj else r'$\rho'
            if isinstance(qty,str) and qty!='mass':
                name += r'_\mathrm{%s}$' % str(qty)
            else:
                name += r'$'
        else:
            name += str(qty) if isinstance(qty,str) else ''
        ylabel = r'%s [$%s$]' % (name, prof.units.latex())
    ax.set_ylabel(ylabel, fontsize=labelsize)

    for tl in ax.get_xticklabels():
        tl.set_fontsize(0.8*labelsize)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(0.8*labelsize)

    return fig, ax

def history(s, qty, time=None, av=None, units=None, diff=False, N=50,
            t_edges=None, tlim=None, log=False, ylabel=None, labelsize=14,
            linewidth=2, add_z_ax=True, ax=None, **kwargs):
    '''
    Plot the archiological history of some stellar quantity.

    Args:
        s (Snap):               The (sub-)snapshot to plot the quantity of.
        qty (str, UnitQty):     The quantity to plot the profile for. It can be a
                                string that can be passed to s.get or a ready
                                block (of length len(s)).
        time (str, UnitQty):    The time to bin the quantity in. If None and the
                                (sub-)snapshot is stars only, it defaults to
                                'cosmic_time()-age'; otherwise it needs to be
                                given.
        av (str, UnitQty):      The quantity to average 'qty' over. Otherwise as
                                'qty'.
        units (str, Unit):      The units in which the profile shall be plotted.
        diff (boolean):         Whether to actually plot the time derivative.
        N (int):                The number of bins. It is ignored, if the edges
                                are given explicitly by `t_edges`.
        t_edges (UnitQty):      The time edges for the bins explicitly.
        tlim (list):            The limits in time in the plot. (Independent of
                                the bins used.)
        log (bool):             Plot in log-scale on the y-axis.
        ylabel (str):           A custom y-axis label.
        labelsize (int):        The font size of the labels. The tick size will
                                get adjusted accordingly.
        linewidth (int,float):  The linewidth.
        add_z_ax (boolean):     Whether the additional scale for redshifts is
                                added on top.
        ax (AxesSubplot):       The axis object to plot on. If None, a new one is
                                created by plt.subplots().
        **kwargs:               Further keyword arguments are passed to ax.plot
                                (e.g. 'linewidth', 'color', or 'label').

    Returns:
        fig (Figure):           The figure of the axis plotted on.
        ax (AxesSubplot):       The axis plotted on.
    '''
    now = s.cosmic_time()
    if isinstance(qty, str):
        Q = s.get(qty)
    else:
        Q = qty
    if len(s) != len(Q):
        raise ValueError('The length of the quantity array ' +
                         '(%s) ' % utils.nice_big_num_str(len(Q)) +
                         'does not match the length of the stellar ' +
                         '(sub-)snapshot (%s)!' % utils.nice_big_num_str(len(s)))
    if av is not None:
        if isinstance(av, str):
            AV = s.get(av)
        if len(s) != len(AV):
            raise ValueError('The length of the averaging quantity array ' +
                             '(%s) ' % utils.nice_big_num_str(len(AV)) +
                             'does not match the length of the stellar ' +
                             '(sub-)snapshot ' +
                             '(%s)!' % utils.nice_big_num_str(len(s)))
    if time is None:
        if len(s.stars) == len(s):
            time = 'cosmic_time()-age'
        else:
            raise ValueError('Time quantity is not defined!')
    if isinstance(time, str):
        time = s.get(time)
    if str(time.units).endswith('_form]'):
        from ..snapshot import age_from_form
        # only convert reasonable values & ensure not to overwrite blocks
        mask = (time!=-1) & np.isfinite(time)
        time = time.copy()
        new = now - age_from_form(time[mask],subs=s)
        time.units = new.units
        time[mask] = new

    if t_edges is None:
        t_edges = UnitArr(np.linspace(0,float(now),N+1), time.units)
        t_edges = UnitArr(np.linspace(0,float(now),N+1), time.units)
    else:
        t_edges = UnitQty(t_edges, time.units)

    Q_hist = []
    for t0, t1 in zip(t_edges[:-1], t_edges[1:]):
        mask = (t0<=time) & (time<t1)
        if av is None:
            Q_hist.append( Q[mask].sum() )
        else:
            av_hist.append( (Q[mask]*AV[mask]).sum() / AV[mask].sum() )
    Q_hist = UnitArr(Q_hist, Q.units)

    if diff:
        Q_hist /= t_edges[1:] - t_edges[:-1]

    if units is not None:
        Q_hist.convert_to(units, subs=s)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    t = (t_edges[:-1] + t_edges[1:]) / 2.0
    ax.plot(t, Q_hist, linewidth=linewidth, **kwargs)
    if tlim is None:
        tlim = UnitQty([t_edges.min(), t_edges.max()], units=t_edges.units)
    else:
        tlim = UnitQty(tlim, units=t_edges.units)
    ax.set_xlim(tlim)

    if add_z_ax:
        universe_age = s.cosmology.universe_age()
        z_minor = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                            1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        z_major = np.array([0, 0.5, 1, 2, 3, 4, 10])
        t_major = [universe_age-s.cosmology.lookback_time(z) for z in z_major]
        t_major = UnitArr(t_major, t_major[0].units).in_units_of(t.units)
        t_minor = [universe_age-s.cosmology.lookback_time(z) for z in z_minor]
        t_minor = UnitArr(t_minor, t_minor[0].units).in_units_of(t.units)
        ax_z = ax.twiny()
        ax_z.set_xlim(ax.get_xlim())
        ax_z.xaxis.set_tick_params(which='minor', size=4, width=1.5)
        ax_z.xaxis.set_tick_params(which='major', size=7, width=1.5)
        ax_z.set_xticks(t_minor, minor=True)
        ax_z.set_xticks(t_major)
        ax_z.set_xticklabels(['%g'%z for z in z_major])
        ax_z.set_xlabel(r'redshift $z$', fontsize=labelsize)
        for tl in ax_z.get_xticklabels():
            tl.set_fontsize(0.8*labelsize)
        for tl in ax_z.get_yticklabels():
            tl.set_fontsize(0.8*labelsize)
    ax.xaxis.set_tick_params(which='major', size=7, width=1.5)
    ax.xaxis.set_tick_params(which='minor', size=4, width=1.5)
    ax.yaxis.set_tick_params(which='major', size=7, width=1.5)
    ax.yaxis.set_tick_params(which='minor', size=3, width=1.5)

    if log:
        ax.set_yscale('log')

    ax.set_xlabel(r'cosmic time $t$ [$%s$]' % t.units.latex(),
                  fontsize=labelsize)
    if ylabel is None:
        name = ''
        if isinstance(av, str):
            name += str(av) + '-weighted '
        name += str(qty) if isinstance(qty,str) else ''
        ylabel = r'%s [$%s$]' % (name, Q_hist.units.latex())
    ax.set_ylabel(ylabel, fontsize=labelsize)

    for tl in ax.get_xticklabels():
        tl.set_fontsize(0.8*labelsize)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(0.8*labelsize)

    return fig, ax

def SFR_history(s, units='Msol/yr', **kwargs):
    """
    Plot the archeological star formation history of the stars of a snapshot.

    This function basically calls `history` with qty='inim'.

    Args:
        s (Snap):               The (sub-)snapshot to plot the SFR history of.
        units (str, Unit):      The units in which the profile shall be plotted.
        **kwargs:               Further keyword arguments are passed to `history`.
                                Se its help for more.

    Returns:
        fig (Figure):           The figure of the axis plotted on.
        ax (AxesSubplot):       The axis plotted on.
    """
    s = s.stars
    units = Unit(units)
    kwargs['diff'] = kwargs.get('diff', True)
    kwargs['ylabel'] = kwargs.get('ylabel', r'$\dot M_*$ [$%s$]'%units.latex())
    fig, ax = history(s, 'inim',
                      units=units, **kwargs)
    return fig, ax

def flow_history(s, qty='mass', inout='infall', recycles='first/re',
                 z_max_min='3 kpc', t_out_min='100 Myr', max_cycle=np.inf,
                 units='Msol/yr', **kwargs):
    """
    Plot the in-/outflow rates as from the gas trace files.

    This function uses `history` in the background.

    Args:
        s (Snap):               The (sub-)snapshot to plot the SFR history of.
        qty ('mass'/'metals'):  Mass or metal flow rates.
        inout ('infall'/'ejection'):
                                Inflow rates ('infall') or outflow rates
                                ('ejection').
        recycles (str,int):     The options are:
                                * 'first/re':   plot the first in-/outflow and the
                                                re-accretion seperately together
                                                with the total in-/outflow rate
                                * 'all':        Plot the in-/outflow rates for
                                                each number of recycling.
                                * <int>:        Only plot the given number of
                                                first in-/outflows / recycles.
        z_max_min (UnitScalar): Only include in-/outflow if the cycle got higher
                                in z-direction than this limit.
        t_out_min (UnitScalar): Limit the included cycles by a minimum time
                                outside of the region of interest.
        max_cycle (int):        Limit the number of cycles considered.
        units (str, Unit):      The units in which the profile shall be plotted.
        ylabel (str):           A custom y-axis label.
        **kwargs:               Further keyword arguments are passed to `history`.
                                Se its help for more.

    Returns:
        fig (Figure):           The figure of the axis plotted on.
        ax (AxesSubplot):       The axis plotted on.
    """
    g = s.gas
    units = Unit(units)
    z_max_min = UnitScalar(z_max_min, 'kpc', subs=s)
    t_out_min = UnitScalar(t_out_min, 'Gyr', subs=s)
    kwargs['diff'] = kwargs.get('diff', True)
    kwargs['N'] = kwargs.get('N', 50)
    kwargs['ylabel'] = kwargs.get('ylabel', r'$\dot M$ [$%s$]'%units.latex())
    kwargs['linewidth'] = kwargs.get('linewidth', 2)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('calculate the total %s rates...' % ('inflow' if inout=='infall'
                                                   else 'outflow'))
        sys.stdout.flush()
    max_N_cycle = min( np.max(g['num_recycled']), max_cycle )
    t_edges = np.linspace(0, float(g.cosmology.universe_age()), kwargs['N']+1)
    t_edges = UnitArr(t_edges, g.cosmology.universe_age().units)
    """
    mass = np.zeros(len(t_edges)-1, dtype=float)
    for i,t0,t1 in zip(np.arange(len(mass)),t_edges[:-1],t_edges[1:]):
        for n in range(max_N_cycle+1):
            tbin = (t0<=g[inout+'_time'][:,n]) & (g[inout+'_time'][:,n]<t1)
            mass[i] += g[qty+'_at_'+inout][tbin,n].sum()
    mass = UnitArr(mass, g[qty+'_at_'+inout].units)
    """

    t_lims = UnitArr([0,g.cosmology.universe_age()],
                     g.cosmology.universe_age().units)
    mass = UnitArr(np.zeros(kwargs['N'], dtype=float), g[qty+'_at_'+inout].units)
    from ..binning import gridbin
    for n in range(max_N_cycle+1):
        if n>1:
            sub = g[ (g['cycle_z_max'][:,n] > z_max_min) &
                     (g['out_time'][:,n-1] > t_out_min) ]
        else:
            sub = g[ (g['cycle_z_max'][:,n] > z_max_min) ]
        binned = gridbin(sub[inout+'_time'][:,n].reshape((len(sub),1)),
                         qty=sub[qty+'_at_'+inout][:,n],
                         bins=kwargs['N'],
                         extent=t_lims.reshape((1,2)))
        binned = binned.reshape((kwargs['N'],))
        if n==0:
            first_mass = binned
        mass += binned

    t = (t_edges[:-1]+t_edges[1:]) / 2.
    dt = t_edges[1:]-t_edges[:-1]

    if environment.verbose >= environment.VERBOSE_NORMAL:
        print('calculate the individual %s rates and plot...' % ('inflow'
                if inout=='infall' else 'outflow'))
        sys.stdout.flush()
    if 'ax' in kwargs:
        ax = kwargs['ax']
        fig = ax.get_figure()
    else:
        fig, ax = plt.subplots()
        kwargs['ax'] = ax

    pltargs = kwargs.copy()
    pltargs['linewidth'] = 1.4*kwargs['linewidth']
    for no in ['N', 'diff', 'ylabel', 'ax', 'add_z_ax', 'av']:
        if no in pltargs:
            del pltargs[no]
    ax.plot(t, (mass/dt).in_units_of(units,subs=s),
            label='total '+inout, **pltargs)

    if recycles == 'all':
        recycles = max_N_cycle
    if isinstance(recycles,int):
        for n in range(max_N_cycle+1):
            if n>1:
                sub = g[ (g['cycle_z_max'][:,n] > z_max_min) &
                         (g['out_time'][:,n-1] > t_out_min) ]
            else:
                sub = g[ (g['cycle_z_max'][:,n] > z_max_min) ]
            history(sub,
                    qty='%s_at_%s[:,%d]'%(qty,inout,n),
                    time='%s_time[:,%d]'%(inout,n),
                    units=units,
                    add_z_ax=(n==0),
                    label='%d. %s'%(n+1,inout), **kwargs)
    elif recycles == 'first/re':
        sub = g[ (g['cycle_z_max'][:,0] > z_max_min) ]
        history(sub,
                qty='%s_at_%s[:,0]'%(qty,inout),
                time='%s_time[:,0]'%(inout),
                units=units,
                add_z_ax=True,
                label='first %s'%inout, **kwargs)
        pltargs['linewidth'] = kwargs['linewidth']
        ax.plot(t, ((mass-first_mass)/dt).in_units_of(units,subs=s),
                label='re- '+inout, **pltargs)
    else:
        raise ValueError('`recycles` is %s -- not understood!' % recycles)

    ax.legend(loc='upper left')

    return fig, ax

