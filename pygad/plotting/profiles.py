'''
Module for convenience routines for plotting profiles.

Doctests impossible, since they would require visual inspection...
'''
__all__ = ['profile']

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..units import *
from ..analysis import *

def profile(s, Rmax, qty, av=None, units=None, dens=True, proj=None,
            N=50, logbin=False, minlog=None, ylabel=None, labelsize=14, ax=None,
            **kwargs):
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
        ylabel (str):       A custom y-axis label.
        labelsize (int):    The font size of the labels. The tick size will get
                            adjusted accordingly.
        ax (AxesSubplot):   The axis object to plot on. If None, a new one is
                            created by plt.subplots().
        **kwargs:           Further keyword arguments are passed to ax.plot (e.g.
                            'linewidth', 'color', or 'label').

    Returns:
        fig (Figure):       The figure of the axis plotted on.
        ax (AxesSubplot):   The axis plotted on.
    '''
    Rmax = UnitScalar(Rmax, s['pos'].units)
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

    ax.plot(r, prof, **kwargs)

    if logbin:
        ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel(r'$%s$ [$%s$]' % ('r' if proj is None else 'R',
                                    s['pos'].units.latex()),
                  fontsize=labelsize)
    if ylabel is None:
        name = ''
        if isinstance(av,str):
            name += av + '-weighted '
        if dens:
            if isinstance(qty,str) and qty!='mass':
                name += r'$\Sigma_\mathrm{%s}$' % qty
            else:
                name += r'$\Sigma$'
        else:
            name += qty if isinstance(qty,str) else ''
        ylabel = r'%s [$%s$]' % (name, prof.units.latex())
    ax.set_ylabel(ylabel, fontsize=labelsize)

    for tl in ax.get_xticklabels():
        tl.set_fontsize(0.8*labelsize)
    for tl in ax.get_yticklabels():
        tl.set_fontsize(0.8*labelsize)

    return fig, ax

