'''
Utility functions regarding geometry.
'''
__all__ = ['angle', 'dist']

import numpy as np
from ..units import UnitArr, UnitQty, UnitQty

def angle(a,b):
    '''
    Return the angle (in radiants) between two 3D vectors.

    Examples:
        >>> angle( [1,2,3], [1,1.23,-1.4] )
        UnitArr(1.66444639785, units="rad")
        >>> angle( [0,1,0], [1,0,0] ).in_units_of('degree')
        UnitArr(90.0, units="degree")
        >>> angle( [1,0,0], [1,0,0] )
        UnitArr(0.000000e+00, units="rad")
        >>> a = UnitArr([1.23,-1.2,0.3], 'cm')
        >>> b = UnitArr([-0.79,0.43,-2.48], 'cm')
        >>> angle( a, np.cross(a,b) ).in_units_of('degree')
        UnitArr(90.0, units="degree")
    '''
    cos = np.dot(a,b) / ( np.linalg.norm(a) * np.linalg.norm(b) )
    return UnitArr(np.arccos(cos),'rad')

def dist(arr, pos=None, metric='euclidean', p=2, V=None, VI=None, w=None):
    '''
    Calculate the distances of the positions in arr to pos.

    This function uses scipy.spatial.distance.cdist and is, hence, faster than
    sqrt(sum((arr-pos)**2,axis=1)). Also see its documentation for more
    information. This is only a wrapper that handles the units. The overhead is
    marginal.

    Args:
        arr (array-like):   The array of positions (shape: (...,N)).
        pos (array-like):   The reference position (shape: (N,)).
                            Default: [0]*N
        [...]:              See scipy.spatial.distance.cdist.

    Returns:
        dists (UnitArr):    Basically cdist(arr, [pos], [...]).ravel() with units.

    Examples:
        >>> dist( [1,2,0], [-2,0,3] )
        UnitArr([ 4.69041576])
        >>> dist( UnitArr([1,2,0],'m'), UnitArr([-231,12,323],'cm') )
        UnitArr([ 4.99233412], units="m")
        >>> dist( UnitArr([1,2,0],'kpc') )
        UnitArr([ 2.23606798], units="kpc")
        >>> dist( [[1,2,0],[1,2,3],[-2,3,4]] )
        UnitArr([ 2.23606798,  3.74165739,  5.38516481])
    '''
    from scipy.spatial.distance import cdist
    arr = UnitQty(arr)
    if pos is None:
        if not isinstance(arr, np.ndarray): # includes UnitArr!
            arr = np.array(arr)
        pos = [0]*arr.shape[-1]
    pos = UnitQty(pos)

    if arr.units is not None:
        units = arr.units
        if pos.units is not None:
            pos = pos.astype(float).in_units_of(arr.units)
    elif pos.units is not None:
        units = pos.units
    else:
        units = None

    if arr.ndim == 1:
        arr = arr.reshape((1,-1))

    res = cdist(arr, [pos]).ravel().view(UnitArr)
    res.units = units
    return res

