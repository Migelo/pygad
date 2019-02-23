'''
Module for plotting.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(general)
    TestResults(failed=0, attempted=1)
'''

from .colormaps import *
from .general import *
from .maps import *
from .profiles import *

def show_FoF_groups(s, groups, colors='rand', plot_center=True, plot_parts=False,
                    plot_hull=True, pointsize=20, alpha=0.02, linewidth=1,
                    linestyle='solid', **plotargs):
    '''
    Plot the image of the (sub-)snapshot `s` and overplot the convex hulls of the
    (projected) FoF-groups.

    Args:
        s (Snap):           The (sub-)snapshot from which the FoF-groups where
                            drawn.
        groups (list):      A list of the FoF groups as instances of ???.
        colors:             The line color. Can be:
                            * 'rand':       Choose a new random color for each
                                            FoF-group.
                            * str/3-tuple:  Define one color for all FoF-groups.
                            * iterable:     Individual colors for the FoF-groups.
        plot_center (str):  Whether to plot the centers of the groups and which
                            one (as attribute name of a group). If True, it will
                            be the center of mass.
        plot_parts (bool):  Whether to do a scatter plot of the particles of the
                            FoF-groups. The are transparent with an alpha as
                            specified.
        plot_hull (bool):   Whether to plot the convex hull.
        pointsize (float):  The side of the points for the scatter plot and the
                            centers markers ('plot_parts' and 'plot_center').
        alpha (float):      The alpha value for the scatter plot of the FoF-group.
        linewidth (float):  The linewidth for the convex hulls (constant for all
                            FoF-groups).
        linestyle (str):    The linestyle for the convex hulls (constant for all
                            FoF-groups).
        **plotargs:         Further arguments are passed to `image`.

    Returns:
        fig, ax, im, cbar:  As `image` does.
    '''
    import numpy as np
    import matplotlib as mpl
    from scipy.spatial import ConvexHull

    xaxis = plotargs.get('xaxis', 0)
    yaxis = plotargs.get('yaxis', 1)
    if not isinstance(colors, str):
        colors = np.asarray(colors)

    fig, ax, im, cbar = image(s, **plotargs)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for i, group in enumerate(groups):
        if colors == 'rand':
            color = mpl.colors.hsv_to_rgb( [np.random.rand(), 1, 1] )
        elif isinstance(colors,str) or colors.dtype in (str,str) \
                or colors.shape==(3,):
            color = colors
        else:
            color = colors[i]

        pos2D = s[group]['pos'][:,(xaxis,yaxis)]
        if plot_hull:
            hull = ConvexHull(pos2D)
            ax.fill(pos2D[hull.vertices,0], pos2D[hull.vertices,1], fill=False,
                    edgecolor=color, linestyle=linestyle, linewidth=linewidth)
        if plot_parts:
            ax.scatter(pos2D[:,0], pos2D[:,1], s=pointsize,
                       edgecolor='none', facecolor=color, alpha=alpha)
        if plot_center:
            try:
                center = getattr(group, plot_center)
            except:
                center = group.com
            c = mpl.colors.rgb_to_hsv(color)
            c = mpl.colors.hsv_to_rgb( [c[0], c[1]*0.6, c[2]*0.8] )
            ax.scatter(center[xaxis], center[yaxis],
                       marker='x', s=1.5*pointsize, color=c)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    return fig, ax, im, cbar

