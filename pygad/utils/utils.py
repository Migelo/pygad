'''
A collection of some general (low-level) functions.

Doctests are in the functions themselves.
'''
__all__ = ['static_vars', 'nice_big_num_str', 'float_to_nice_latex', 'perm_inv',
           'periodic_distance_to', 'positive_simple_slice', 'is_consecutive',
           'rand_dir']

import numpy as np
import re
import scipy.spatial.distance

def static_vars(**kwargs):
    '''
    Decorate a function with static variables.

    Example:
        >>> @static_vars(counter=0)
        ... def foo():
        ...     foo.counter += 1
        ...     print 'foo got called the %d. time' % foo.counter
        >>> foo()
        foo got called the 1. time
        >>> foo()
        foo got called the 2. time
        >>> foo()
        foo got called the 3. time
    '''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def nice_big_num_str(n, separator=','):
    '''
    Convert a number to a string with inserting separator all 3 digits.

    Note:
        With the default separator ',' one can also use the standard
        "{:,}".format(n).

    Example:
        >>> nice_big_num_str(12345678)
        '12,345,678'
        >>> nice_big_num_str(-12345)
        '-12,345'
        >>> nice_big_num_str(123456, separator=' ')
        '123 456'
        >>> nice_big_num_str(0)
        '0'
    '''
    if n < 0:
        return '-' + nice_big_num_str(-n)
    s = ''
    while n >= 1000:
        n, r = divmod(n, 1000)
        s = "%s%03d%s" % (separator, r, s)
    return "%d%s" % (n, s)

def float_to_nice_latex(x, dec=None):
    '''
    Convert a number to a nice latex representation.

    Args:
        x (float):  The number to convert.
        dec (int):  The number of digits for precision.

    Returns:
        repr (string):  The LaTeX representation.

    Example:
        >>> float_to_nice_latex(1.2345e67, 2)
        '1.23 \\\\times 10^{67}'
        >>> float_to_nice_latex(1e10)
        '10^{10}'
    '''
    s = ('%g' if dec is None else '%%.%dg' % (dec+1)) % x
    if 'e' in s:
        # two backslashes in raw-string, because it is a regex
        # replace 'e+'
        s = re.sub(r'e\+', r' \\times 10^{', s)
        # replace the 'e' in 'e-'
        s = re.sub(r'e(?=-)', r' \\times 10^{', s)
        s += '}'
        # remove potential '1 \times '
        if s.startswith(r'1 '):
            s = s[9:]
    return s

def perm_inv(perm):
    '''
    Invert a permutation.

    Args:
        perm (array-like):  The permutation in form of an array of the
                            integers 0, 1, 2, ..., N

    Returns:
        inverse (np.ndarray):   The inverse.

    Examples:
        >>> a = np.arange(5)
        >>> ind = a.copy()
        >>> np.random.shuffle(ind)
        >>> np.all( a == a[ind][perm_inv(ind)])
        True
    '''
    return perm.argsort()

def _ndarray_periodic_distance_to(pos, center, boxsize):
    '''periodic_distance_to assuming np.ndarray's as arguments. (for speed)'''
    min_dists = np.minimum((pos - center) % boxsize,
                           (center - pos) % boxsize)
    return np.sqrt((min_dists**2).sum(axis=1))

def periodic_distance_to(pos, center, boxsize):
    '''
    Calculate distances in a periodic box.

    Args:
        pos (array-like):       An array of points.
        center (array-like):    The reference point.
        boxsize (float, array-like):
                                The box size. Either a float, then it the box is
                                a cube, or an array-like object, defining the
                                sidelengths for each axis individually.

    Returns:
        dist (array-like):  The distance(s) between the points.

    Examples:
        >>> from .. import units
        >>> from ..units import UnitArr
        >>> from ..environment import module_dir
        >>> units.undefine_all()
        >>> units.define_from_cfg([module_dir+'units/units.cfg'])
        reading units definitions from "pygad/units/units.cfg"
        >>> pos = UnitArr([[1.1,2.1,3.7], [2.8,-1.4,5.4], [7.0,3.4,-5.6]],
        ...               units='m')
        >>> ref = UnitArr([10.,120.,280.], units='cm')
        >>> periodic_distance_to(pos, ref, '5 m')
        UnitArr([ 1.61864141,  4.1       ,  3.318132  ], units="m")
    '''
    from ..units import UnitArr

    if isinstance(boxsize, str):
        from ..units import Unit
        boxsize = Unit(boxsize)

    # go through all the unit mess...
    unit = None
    if isinstance(pos, UnitArr):
        unit = pos.units
        if isinstance(center, UnitArr):
            if not hasattr(boxsize, 'in_units_of') and pos.units != center.units:
                raise ValueError('boxsize is unitless and pos and center have '
                                 'different units. Ambiguous to interpret.')
            center = center.in_units_of(unit).view(np.ndarray)
        else:
            center = np.array(center)
        if hasattr(boxsize, 'in_units_of'):
            boxsize = boxsize.in_units_of(unit)
            if isinstance(boxsize, UnitArr):
                boxsize = boxsize.view(np.ndarray)
        pos = pos.view(np.ndarray)
    elif isinstance(center, UnitArr):
        unit = center.units
        if hasattr(boxsize, 'in_units_of'):
            boxsize = boxsize.in_units_of(unit)
            if isinstance(boxsize, UnitArr):
                boxsize = boxsize.view(np.ndarray)
        center = center.view(np.ndarray)
        pos = np.array(pos)
    elif hasattr(boxsize, 'in_units_of'):
        from ..units import _UnitClass
        unit = boxsize.units
        if isinstance(boxsize, _UnitClass):
            boxsize = float(boxsize)
        elif isinstance(boxsize, UnitArr):
            boxsize = boxsize.view(np.ndarray)
        center = np.array(center)
        pos = np.array(pos)

    r = _ndarray_periodic_distance_to(pos, center, boxsize).view(UnitArr)
    r.units = unit
    return r

def positive_simple_slice(s, N):
    '''
    Convert a slice into an equivalent one with step>0 and 0<=start<=stop<=N and
    start, stop, and step are all not None.

    The new slice is equivalent in the sense, that it yields the same values when
    applied. These are, however, reverted with respect to the original slice, if
    it had a negative step, of course.

    Args:
        s (slice):  The slice to make sane.
        N (int):    The length of the sequence to apply the slice to. (Needed for
                    negative indices and None's in the slice).

    Returns:
        simple (slice): The sane start and stop position and the (positive) step.

    Raises:
        ValueError:     If a slice with step==0 was passed.

    Examples:
        >>> positive_simple_slice(slice(None,2,-3), 10)
        slice(3, 10, 3)
        >>> positive_simple_slice(slice(-3,None,None), 10)
        slice(7, 10, 1)

        Test some random parameters for the requirements:
        >>> from random import randint
        >>> for i in xrange(10000):
        ...     N = randint(0,10)
        ...     start, stop = randint(-3*N, 3*N), randint(-3*N, 3*N)
        ...     step = randint(-3*N, 3*N)
        ...     if step==0: step = None     # would raise exception
        ...     a = range(N)
        ...     s = slice(start,stop,step)
        ...     ss = positive_simple_slice(s,N)
        ...     if not ( sorted(a[s]) == a[ss] and 0 <= ss.start <= ss.stop <= N
        ...                 and ss.step > 0 ):
        ...         print 'ERROR:'
        ...         print N, s, ss
        ...         print a[s], a[ss]
    '''
    step = s.step if s.step is not None else 1
    if step == 0:
        ValueError('Slice step cannot be 0!')

    # get first index
    first = s.start
    if first is None:
        first = 0 if step > 0 else N-1
    elif first < 0:
        first = N+first
        if first < 0:
            if step < 0:
                return slice(0,0,1)
            first = 0
    elif first >= N:
        if step > 0:
            return slice(0,0,1)
        first = N-1
    # get stop index
    stop = s.stop
    if stop is None:
        stop = N if step > 0 else -1
    elif stop < 0:
        stop = max(-1, N+stop)

    # check for etmpy slices
    if (stop-first)*step <= 0:
        return slice(0,0,1)

    if step > 0:
        return slice(first, min(stop,N), step)
    elif step == -1:
        return slice(stop+1, first+1, -step)
    else:
        # calculate the new start -- does not have to be next to old stop, since
        # the stepping might lead elsewhere
        step = -step
        dist = first - max(stop,-1)
        last = first - dist / step * step
        if dist % step == 0:
            last += step
        return slice(last, first+1, step)

def is_consecutive(l):
    '''
    Test whether an iterable that supports indexing has consecutive elements.

    Args:
        l (iterable):   Some iterable, that supports indexing for which
                        list(l) == [l[i] for i in range(len(l))]
    
    Examples:
        >>> is_consecutive([1,2,3])
        True
        >>> is_consecutive([-2,-1,0,1])
        True
        >>> is_consecutive([1,3,2])
        False
        >>> is_consecutive({1:'some', 2:'dict'})
        Traceback (most recent call last):
        ...
        TypeError: Cannot check whether a dict has consecutive values!
    '''
    if isinstance(l, dict):
        raise TypeError('Cannot check whether a dict has consecutive values!')
    return np.all( np.arange(len(l)) + l[0] == list(l) )

def rand_dir(dim=3):
    '''
    Create a vectors with uniform spherical distribution.

    Args:
        dim (int):      The number of dimensions the vector shall have.

    Returns:
        r (np.ndarray): A vector of shape (dim,) with length of one pointing into
                        a random direction.

    Examples:
        >>> N = 1000
        >>> for dim in [2,3,4]:
        ...     for n in xrange(10):
        ...         assert abs( np.linalg.norm(rand_dir(dim=dim)) - 1 ) < 1e-4
        ...     v = np.empty([N,dim])
        ...     for n in xrange(N):
        ...         v[n] = rand_dir(dim=dim)
        ...     assert np.linalg.norm(v.sum(axis=0))/N < 2.0/np.sqrt(N)
    '''
    if dim < 1:
        raise ValueError('Can only create vectors with at least one dimension!')
    length = 2.
    while length > 1.0 or length < 1e-4:
        r = np.random.uniform(-1.,1., size=dim)
        length = np.linalg.norm(r)
    return r / length

