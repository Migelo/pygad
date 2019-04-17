'''
Utility functions regarding geometry.
'''
__all__ = ['angle', 'dist', 'find_maxima_prominence_isolation']

import numpy as np
from ..units import UnitArr, UnitQty, UnitQty

def angle(a,b):
    '''
    Return the angle (in radiants) between two 3D vectors.

    Examples:
        >>> angle( [1,2,3], [1,1.23,-1.4] )
        UnitArr(1.664446397854663, units="rad")
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
        UnitArr([4.69041576])
        >>> dist( UnitArr([1,2,0],'m'), UnitArr([-231,12,323],'cm') )
        UnitArr([4.99233412], units="m")
        >>> dist( UnitArr([1,2,0],'kpc') )
        UnitArr([2.23606798], units="kpc")
        >>> dist( [[1,2,0],[1,2,3],[-2,3,4]] )
        UnitArr([2.23606798, 3.74165739, 5.38516481])
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

def find_maxima_prominence_isolation(arr, prominence=None, sortby='index',
                                     descending=False):
    '''
    Find all local maxima in an array (of a minimum prominence).

    This function does not only return the maxima, but also their position (as
    an index), their prominence, and their isolation (in pixels/indices). The
    latter two have the usual topographic definitions.

    Note:
        If two or more consequtive values are the same and together build a
        local extremum, they will all count as individual local extrema.

    Args:
        arr (array-like):       The 1-dimensional array in which to look for
                                the maxima.
        prominence (float):     A minimum prominence to filter for.
                                If None, do not filter the results.
        sortby (str):           Whether / how to sort the results. This must be
                                one of 'index', 'value', 'prominence', and
                                'isolation'.
                                They are naturally sorted by the indices. Hence,
                                this option is the fastest when chosen with
                                `descending=False`.
        descending (bool):      Sort in descending order.

    Returns:
        maxima (np.ndarray):    All the (filtered) maxima. It is an array with
                                named fields: 'index', 'value', 'prominence',
                                and 'isolation'.
    Examples:
        >>> m = find_maxima_prominence_isolation( np.array([1,2,3,2,0,2,1,3],dtype=float) )
        >>> m['index']
        array([2, 5, 7])
        >>> m['value']
        array([3., 2., 3.])
        >>> m['prominence']
        array([3., 1., 3.])
        >>> m['isolation']
        array([6, 2, 8])
        >>> find_maxima_prominence_isolation( [1,1,1] )['value']
        array([1, 1, 1])
        >>> find_maxima_prominence_isolation( [0,1,4,1,0] )['index']
        array([2])
        >>> find_maxima_prominence_isolation( [0,1,4,1,-3,-2,1,-1,0,3,5,6,4,1],
        ...                                   sortby='prominence',
        ...                                   descending=True )['value']
        array([6, 4, 1])
        >>> find_maxima_prominence_isolation( [0,5,-2,-2,1,0,8,0] )[0]
        (1, 5, 7, 5)
        >>> len(find_maxima_prominence_isolation( [0] ))
        0
    '''
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise ValueError('Only one dimensional array can be processed!')
    if len(arr) < 2:
        return np.array([],
                        dtype=[('index',int), ('value',arr.dtype), ('prominence',arr.dtype)])

    neg_arr = -arr
    def is_local_max(i, a=arr):
        next_other, j = a[i], i
        while next_other == a[i]:
            j += 1
            next_other = a[i]-1 if j==len(a) else a[j]
        prev_other, j = a[i], i
        while prev_other == a[i]:
            j -= 1
            prev_other = a[i]-1 if j==-1 else a[j]
        return prev_other < a[i] > next_other
    def is_local_min(i):
        return is_local_max(i, a=neg_arr)

    # find all the local extrema in the smoothed spectrum
    extrema = [ ]
    for i in range(len(arr)):
        if is_local_max(i):
            extrema.append( ['max', arr[i], i] )
        elif is_local_min(i):
            extrema.append( ['min', arr[i], i] )

    # get the prominence of the maxima
    for iex,ex in enumerate(extrema):
        if ex[0] == 'min':
            ex.append( -1.0 )
            continue

        # find the prominence
        def find_one_sided_prominence(iex, ex, left):
            prom_one = np.inf
            vmin = ex[1]
            nrange = range(iex-1,-1,-1) if left else range(iex+1,len(extrema))
            for n in nrange:
                if extrema[n][0] == 'min':
                    if extrema[n][1] < vmin: vmin = extrema[n][1]
                elif extrema[n][0] == 'max':
                    if extrema[n][1] >= ex[1]:
                        prom_one = ex[1] - vmin
                        break
            return prom_one
        # the total prominence is the minimum of the two
        prom = min(find_one_sided_prominence(iex,ex,left=True),
                   find_one_sided_prominence(iex,ex,left=False))
        if np.isinf(prom):
            prom = arr.ptp()
        ex.append( prom )
        # find the isolation
        iso = max( ex[2]+1, len(arr)-ex[2] )
        for i in range(1,max(ex[2],len(arr)-ex[2])):
            if ex[2]-i>= 0 and arr[ex[2]-i] > ex[1]:
                iso = i
                break
            if ex[2]+i<len(arr) and arr[ex[2]+i] > ex[1]:
                iso = i
                break
        ex.append( iso )

    # filter maxima with given prominence
    maxima = [ ex for ex in extrema
                if (ex[0]=='max' and (prominence is None or ex[3]>prominence)) ]
    if sortby == 'index':
        if descending:
            maxima = sorted(maxima, key=lambda ex: ex[2], reverse=descending)
        else:
            pass    # nothing to do...
    elif sortby == 'value':
        maxima = sorted(maxima, key=lambda ex: ex[1], reverse=descending)
    elif sortby == 'prominence':
        maxima = sorted(maxima, key=lambda ex: ex[3], reverse=descending)
    elif sortby == 'isolation':
        maxima = sorted(maxima, key=lambda ex: ex[4], reverse=descending)
    else:
        raise ValueError('Cannot sort by "%s" -  unknown property.' % sortby)
    return np.array( [(idx,val,prom,iso) for _,val,idx,prom,iso in maxima],
                      dtype=[('index',int), ('value',arr.dtype),
                             ('prominence',arr.dtype), ('isolation',int)] )

