'''
A collection of functions that calculate properties of snapshots (Snap) and
sub-snapshots (e.g. MaskedSnap).
'''
__all__ = ['radial_surface_density_profile']

from units import UnitArr
import physics
import numpy as np
import sph

def radial_surface_density_profile(snap, quantity='mass', r_edges=None,
                                   N_sample=50, units=None, verbose=True):
    '''
    Create a radial surface density profile.

    Args:
        snap (Snap):        The snapshot to use.
        quantity (str, UnitArr):
                            The quantity to do the radial surface density profile
                            of.
        r_edges (array-like):
                            The edges of the radial bins. If no units are set,
                            they are assumed to be those of snap.pos. (default:
                            UnitArr(np.arange(51), units='kpc'))
        units (str, Unit):  The units of the surface density.
        verbose (bool):     Whether or not to print which sample is mapped at the
                            moment.

    Returns:
        Sigma (UnitArr):   The profile, i.e. the quantity summed for the
                            intervals defined by r_edges and then divided by the
                            area.
    '''
    if r_edges is None:
        r_edges = UnitArr(np.arange(51), units='kpc')
    elif not isinstance(r_edges, UnitArr):
        r_edges = UnitArr(r_edges)
    if r_edges._units is None:
        r_edges.units = snap.pos.units

    Sigma = profile(snap, quantity, r_edges=r_edges, mode='cylindrical',
                    average='none', N_sample=N_sample, verbose=verbose)
    for i in range(len(r_edges)-1):
        Sigma[i] /= np.pi*(r_edges[i+1]**2-r_edges[i]**2)
    Sigma.units /= r_edges.units**2
    if units is not None:
        Sigma.convert_to(units)

    return Sigma
