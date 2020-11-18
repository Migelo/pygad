'''
In this module there is class defined that decorates a numpy array with units.

Note:
    Since the unit is global to the array, the speed penality is not too big.
    However, if one accesses elements, slices, etc. of a UnitArr, it might help to
    work with 'a.view(np.ndarray)' and add the units again at the end, rather than
    directly working with the UnitArr 'a'.

Examples:
    >>> v = UnitArr([[1,-2], [-3,1], [2,3]], dtype=np.float64, units='m/s')
    >>> v
    UnitArr([[ 1., -2.],
             [-3.,  1.],
             [ 2.,  3.]], units="m s**-1")
    >>> v.units
    Unit("m s**-1")
    >>> v[0]
    UnitArr([ 1., -2.], units="m s**-1")
    >>> v[0,0]
    1.0

    >>> v = UnitArr([1.5,-2.,3.], units='m/s')
    >>> w = UnitArr(v, 'km/h')
    >>> w
    UnitArr([  5.4,  -7.2,  10.8], units="km h**-1")
    >>> v.in_units_of('km/h')   # does not change v
    UnitArr([  5.4,  -7.2,  10.8], units="km h**-1")
    >>> v
    UnitArr([ 1.5, -2. ,  3. ], units="m s**-1")
    >>> v.convert_to('km/h')    # convertion in-place
    >>> v
    UnitArr([  5.4,  -7.2,  10.8], units="km h**-1")
    >>> UnitArr(UnitArr(v, 'm/s'), 'm/s')
    UnitArr([ 1.5, -2. ,  3. ], units="m s**-1")
    >>> v
    UnitArr([  5.4,  -7.2,  10.8], units="km h**-1")
    >>> v[:2].convert_to('m/s') # for consistency: convert entire array!
    >>> v
    UnitArr([ 1.5, -2. ,  3. ], units="m s**-1")
    >>> v.units = 'kg'  # just setting the units, no conversion
    >>> v
    UnitArr([ 1.5, -2. ,  3. ], units="kg")
    >>> sub = v[1:]
    >>> v[:2].units = 's'   # for consistency: convert entire array!
    >>> v
    UnitArr([ 1.5, -2. ,  3. ], units="s")
    >>> sub     # units are changed for the view as well!
    UnitArr([-2.,  3.], units="s")

    >>> v + v
    UnitArr([ 3., -4.,  6.], units="s")
    >>> v + UnitArr('1 min')
    UnitArr([ 61.5,  58. ,  63. ], units="s")
    >>> v *= 2.3
    >>> v
    UnitArr([ 3.45, -4.6 ,  6.9 ], units="s")
    >>> v -= [1.05, -5.1, 2.9]     # understood as in current units
    >>> v
    UnitArr([ 2.4,  0.5,  4. ], units="s")
    >>> v * v
    UnitArr([  5.76,   0.25,  16.  ], units="s**2")
    >>> v
    UnitArr([ 2.4,  0.5,  4. ], units="s")
    >>> v * np.array([1,2,3])
    UnitArr([  2.4,   1. ,  12. ], units="s")
    >>> np.array([1,2,3]) * v
    UnitArr([  2.4,   1. ,  12. ], units="s")
    >>> np.sqrt(v)
    UnitArr([ 1.54919334,  0.70710678,  2.        ], units="s**1/2")
    >>> np.round(v**Fraction(1,3), 3)   # no floats allowed, however, Fraction are!
    array([1.339,  0.794,  1.587])
    >>> np.mean(v)
    UnitArr(2.3, units="s")
    >>> v.sum()
    UnitArr(6.9, units="s")
    >>> v.prod()
    UnitArr(4.8, units="s**3")
    >>> for prop in [np.sum, np.cumsum, np.mean, np.std, np.median, np.ptp,
    ...              np.transpose, np.min, np.max]:
    ...     res = prop(v)
    ...     assert res.units == v.units
    ...     assert res._unit_carrier is not v._unit_carrier
    ...     b = res.base
    ...     while hasattr(b, '_unit_carrier'):
    ...         assert b._unit_carrier is res._unit_carrier
    ...         b = b.base
    ...     while b is not None:
    ...         assert not hasattr(b, '_unit_carrier')
    ...         b = b.base
    >>> M_earth = UnitArr('5.972e24 kg')
    >>> R_earth = UnitArr('12740 km', dtype=float) / 2.0
    >>> v_earth = UnitArr('29.78 km/s')
    >>> M_earth / (4/3.*np.pi * R_earth**3)
    UnitArr(5.515856e+12, units="kg km**-3")
    >>> (0.5 * M_earth * v_earth**2).in_units_of('J')
    UnitArr(2.648129e+33, units="J")
    >>> UnitArr('1.2 N*m').in_base_units()
    UnitArr(1.200000e+03, units="g m**2 s**-2")

    Unit conversion at construction:
    >>> UnitArr('10**8 km', units='AU')
    UnitArr(0.668458712227, units="AU")
    >>> UnitArr(UnitArr([1e8, 1e9], 'km'), units='AU')
    UnitArr([ 0.66845871,  6.68458712], units="AU")

    conversion to native types
    >>> s = UnitScalar(1.2, 'm')
    >>> float(s)
    1.2
    >>> int(s)
    1
    >>> l = UnitArr([2.4], 'km')
    >>> float(l)
    2.4
    >>> s / l
    UnitArr([ 0.5], units="km**-1 m")
    >>> float(s/l)
    0.0005
    >>> float( UnitArr([1.2], 'km/cm') )
    120000.0

    Pickling:
    >>> import pickle
    >>> v_pickled = pickle.dumps(UnitArr([1,2,3], 'inch'))
    >>> v_unpickled = pickle.loads(v_pickled)
    >>> v_unpickled.units
    Unit("inch")

    Some tests for underlying implementation
    >>> v = UnitArr([1.5,-2.,3.], units='m/s')
    >>> assert v.copy()._unit_carrier is not v._unit_carrier
    >>> assert v.astype(float)._unit_carrier is not v._unit_carrier
    >>> assert v.T._unit_carrier is not v._unit_carrier
    >>> assert (v+v)._unit_carrier is not v._unit_carrier
'''
__all__ = ['UnitArr', 'UnitQty', 'UnitScalar']

import numpy as np
import numpy.core.umath_tests
from . import units
from .units import *
from .units import _UnitClass
from fractions import Fraction
import warnings
import functools
import numbers


def UnitQty(obj, units=None, subs=None, dtype=None, copy=False):
    '''
    Convert to a UnitArr with enshured units.

    Args:
        obj (UnitArr, array-like, Unit, str, float):
                                The object to convert.
        units (Units, str):     If obj had units, convert to these, otherwise set
                                these units.
        subs (dict, Snap):      Substitutions as passe to UnitArr.convert_to.
        dtype (np.dtype, str):  The requested dtype.

    Returns:
        obj (UnitArr):          A unit array with the desired properties.

    Examples:
        >>> a = UnitArr([1,2,3], 'm')
        >>> UnitQty(a, units='m') is a
        True
        >>> UnitQty(a, copy=True) is a
        False
        >>> UnitQty(a, units='cm', dtype=float)
        UnitArr([ 100.,  200.,  300.], units="cm")
        >>> a
        UnitArr([1, 2, 3], units="m")
        >>> a = UnitArr([1.,2.,3.], 'm')
        >>> UnitQty(a, units='cm')
        UnitArr([ 100.,  200.,  300.], units="cm")
        >>> a
        UnitArr([ 1.,  2.,  3.], units="m")
        >>> UnitQty([1,2,3.5], units='s')
        UnitArr([ 1. ,  2. ,  3.5], units="s")
        >>> UnitQty('2.5 AU')
        UnitArr(2.5, units="AU")
        >>> UnitQty('0.5 min', units='s')
        UnitArr(30.0, units="s")
        >>> UnitQty('2.5 ckpc/h_0', 'kpc', subs=dict(a=0.35, h_0=0.7))
        UnitArr(1.25, units="kpc")
        >>> UnitQty(Unit('2.5 AU'))
        UnitArr(2.5, units="AU")
        >>> UnitQty([2.5, -0.3])
        UnitArr([ 2.5, -0.3])
        >>> UnitQty(2.5, 'AU')
        UnitArr(2.5, units="AU")
        >>> UnitQty('2.5')
        UnitArr(2.5)
        >>> UnitQty('2.5', 'min')
        UnitArr(2.5, units="min")
        >>> UnitQty({}) # doctest:+ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: Only native dtype allowed for UnitArr, not object!
    '''
    if isinstance(obj, UnitArr) and not copy:
        if (units is None or obj.units == units) and \
                (dtype is None or obj.dtype == dtype):
            return obj
    return UnitArr(obj, units=units, subs=subs, dtype=dtype, copy=True)


def UnitScalar(obj, units=None, subs=None, dtype=None):
    '''
    Convert to a UnitArr with enshured units of a single number.

    Calls UnitQty and tests for the shape of the object.

    Args:
        obj (UnitArr, Unit, str, float):
                                The object to convert.
        units (Units, str):     If obj had units, convert to these, otherwise set
                                these units.
        subs (dict, Snap):      Substitutions as passe to UnitArr.convert_to.
        dtype (np.dtype, str):  The requested dtype.

    Raises:
        ValueError:             If the obj is actually an array.

    Examples:
        >>> UnitScalar(1.2, 'm')
        UnitArr(1.2, units="m")
        >>> UnitScalar('2.5 ckpc/h_0', 'kpc', subs=dict(a=0.35, h_0=0.7))
        UnitArr(1.25, units="kpc")
        >>> UnitScalar([1.2], 'm')
        Traceback (most recent call last):
        ...
        ValueError: object is not a scalar!
        >>> UnitScalar([1,2,3])
        Traceback (most recent call last):
        ...
        ValueError: object is not a scalar!
    '''
    obj = UnitQty(obj, units=units, subs=subs, dtype=dtype)
    if obj.shape:
        raise ValueError('object is not a scalar!')
    return obj


def with_own_units(func):
    '''Decorate a function return value with the units of the first argument.'''

    def u_func(self, *args, **kwargs):
        a = UnitQty(func(self.view(np.ndarray), *args, **kwargs))
        a._set_units_and_carrier_on_base(self.units)
        return a

    u_func.__name__ = func.__name__
    u_func.__doc__ = func.__doc__
    return u_func


class UnitArr(np.ndarray):
    '''
    A numpy array decorated with units.

    Args:
        data (array-like):  Either 'raw data' passed to the underlying numpy
                            array or a UnitArr. In the latter case, units are
                            taken from it.
        units (Unit, str):  The units of the array. If not None, they overwrite
                            those from data.
        keyword arguments:  Passed on to the array factory function of numpy
                            (np.array).

    New attributes:
        units (Unit):       The units of the data (can be None!).
    '''

    _ufunc_registry = {}

    def __new__(subtype, data, units=None, subs=None, **kwargs):
        # handle cases where `obj` is a string of a unit:
        if isinstance(data, str):
            data = Unit(data)
        if isinstance(data, _UnitClass):
            if units is None:
                data, units = data.scale, data.free_of_factors()
            else:
                if len(data._composition) == 0:
                    data = float(data)
                else:
                    data = data.in_units_of(units, subs=subs)

        # actually create the new object
        if isinstance(data, UnitArr) and 'copy' not in kwargs:
            # avoid interference of different `_unit_carrier`s and units
            kwargs['copy'] = True
        obj = np.array(data, **kwargs).view(subtype)
        # bool needed in some situations (such as np.median(UnitArr))
        if obj.dtype.kind not in 'uifb':
            raise ValueError('Only native dtype allowed for UnitArr, not %s!' %
                             obj.dtype)

        # all UnitArr's need to have a _unit_carrier
        obj._unit_carrier = obj
        obj._units = getattr(data, 'units', None)

        # bring data into desired units
        if units is not None:
            if obj.units is not None:
                obj.convert_to(units, subs=subs)
            else:
                obj.units = units

        return obj

    def __array_finalize__(self, obj):
        self._unit_carrier = getattr(obj, '_unit_carrier', self)
        self._units = getattr(obj, 'units', None)

    def __array_wrap__(self, array, context=None):
        if context is None:
            return array

        try:
            ufunc = context[0]
            units = UnitArr._ufunc_registry[ufunc](*context[1])
            if units is not None:
                units = units.gather()
                units._composition.sort()
            array = array.view(UnitArr)
            array._set_units_and_carrier_on_base(units)
            return array
        except KeyError:
            if not ufunc.__name__.endswith('(vectorized)'):
                warnings.warn('Operation \'%s\' on units is ' % ufunc.__name__ + \
                              '*not* defined! Return normal numpy array.')
            return np.asarray(array)
        except:
            raise

    @property
    def units(self):
        '''The units of the array.'''
        return self._units

    @units.setter
    def units(self, value):
        if value is not None:
            value = Unit(value)
        self._units = value

    def _set_units_and_carrier_on_base(self, units=None):
        uc = self
        if isinstance(self.base, UnitArr):
            uc = self.base._set_units_and_carrier_on_base(units)
            self._units = uc._units
        else:
            self._units = None if units is None else Unit(units)
        self._unit_carrier = uc

        return uc

    def __float__(self):
        try:
            a = self.astype(float).in_units_of(1)
        except:
            a = self
        return float(a.view(np.ndarray))

    def __int__(self):
        try:
            a = self.astype(float).in_units_of(1)
        except:
            a = self
        return int(a.view(np.ndarray))

    def __copy__(self, *a):
        if a:
            dupl = np.ndarray.copy(self, *a)
        else:
            dupl = np.ndarray.copy(self)
        dupl._set_units_and_carrier_on_base(self.units)
        return dupl

    def __deepcopy__(self, *a):
        dupl = np.ndarray.__deepcopy__(self).view(UnitArr)
        dupl._set_units_and_carrier_on_base(self.units)
        return dupl

    def copy(self, order=None):
        '''Create a real copy.'''
        if order:
            return self.__copy__(order)
        else:
            return self.__copy__()

    def __repr__(self, val=None):
        if not self.shape and self.dtype.kind == 'f':
            r = 'UnitArr('
            if val is None:
                f = float(self.view(np.ndarray))  # avoid conversion of units!
                r += str(f) if (1e-3 <= f <= 1e3) else ('%e' % f)
            else:
                r += str(val)
            if self.dtype not in ['int', 'float']:
                r += ', dtype=' + str(self.dtype)
            r += ')'
        else:
            if val is None:
                r = repr(self.view(np.ndarray))
                r = r[r.find('('):].replace('\n', '\n  ')
            else:
                r = '(' + str(val) + ')'
            r = 'UnitArr' + r
        if hasattr(self, 'units'):
            if self.units is not None and self.units != 1:
                r = r[:-1] + (', units="%s")' % str(self.units)[1:-1])
                if len(r) - r.rfind('\n') > 82:
                    right = r.find('dtype=')
                    if right == -1:
                        right = r.find('units=')
                    arr_end = r.rfind('],') + 2
                    if arr_end == 1: arr_end = r.find(',') + 1
                    r = r[:arr_end] + '\n' + ' ' * 8 + r[right:]
        else:
            r = r[:-1] + ', no units!)'
        return r

    def __str__(self):
        if not self.shape and self.dtype.kind == 'f':
            f = float(self.view(np.ndarray))  # avoid conversion of units!
            s = str(f) if (1e-3 <= f <= 1e3) else ('%e' % f)
        else:
            s = str(self.view(np.ndarray))
        if self.units is not None and self.units != 1:
            s += ' %s' % self.units
        return s

    # needed for pickling
    def __reduce__(self):
        pickled_state = np.ndarray.__reduce__(self)
        return (pickled_state[0], pickled_state[1],
                pickled_state[2] + (self.units,))

    # needed for unpickling
    def __setstate__(self, state):
        self.units = state[-1]
        np.ndarray.__setstate__(self, state[:-1])

    def astype(self, dtype, *args, **kwargs):
        '''
        Convert the type of the array.

        Internally np.ndarray.astype is called. See the documentation for more
        details.
        '''
        new = np.ndarray.astype(self, dtype, *args, **kwargs)
        if new is not self:
            new._set_units_and_carrier_on_base(self.units)
        return new

    def convert_to(self, units, subs=None):
        '''
        Convert the array into other units in place.

        If this is just a view of a bigger underlying UnitArr, this entire
        underlying is converted in order to avoid different units within the same
        array.

        Args:
            units (Unit, str):  The units to convert to.
            subs (Snap, dict):  Substitutions to be made during conversion. (For
                                more information see _UnitClass.substitute). If
                                this is a Snap, the values for the redshift 'z',
                                the scale factor 'a', and the Hubble parameter
                                'h_0' from it are used as well as the cosmology,
                                if not set explicitly by the other argument.

        Raises:
            UnitError:          In case the current units and the target units are
                                not convertable.
        '''
        units = Unit(units)
        if self.units == units:
            if str(self.units) != str(units):
                self.units = units
            return
        elif self.units is None:
            self.units = units
            return

        from ..snapshot.snapshot import Snapshot
        if isinstance(subs, Snapshot):
            snap, subs = subs, {}
            subs['a'] = snap.scale_factor
            subs['z'] = snap.redshift
            subs['h_0'] = snap.cosmology.h_0

        uc = self._unit_carrier
        if uc.dtype.kind != 'f':
            raise RuntimeError('Cannot convert UnitArr inplace, that do not ' + \
                               'have floating point data type!')
        fac = self._units.in_units_of(units, subs=subs)
        view = uc.view(np.ndarray)
        view *= fac
        self._units = units

    def in_units_of(self, units, subs=None, copy=False):
        '''
        Return the array in other units.

        Args:
            units (Unit, str):  The target units.
            subs (Snap, dict):  See 'convert_to'.
            copy (bool):        If set to False, return the array itself, if the
                                target units are the same as the current ones.

        Returns:
            converted (UnitArr):    The new, converted array. If copy is False and
                                    the target units are the same as the current
                                    ones, however, simply self is returned.

        Raises:
            UnitError:          In case the current units and the target units are
                                not convertable.
        '''
        if not isinstance(units, (_UnitClass, str, numbers.Number)):
            raise TypeError('Cannot convert type %s to units!' % \
                            type(units).__name__)
        units = Unit(units)
        if not copy and units == self.units:
            self.units = units
            return self
        c = self.copy()
        c.convert_to(units=units, subs=subs)
        return c

    def convert_to_base_units(self, subs=None):
        '''
        Convert to the base units, i.e. the units all other units are defined in.

        For more information see `convert_to`.
        '''
        units = self.units.standardize().free_of_factors()
        return self.convert_to(units, subs=subs)

    def in_base_units(self, subs=None, copy=False):
        '''
        Return the array in the base units, i.e. the units all other units are
        defined in.

        For more information see `in_units_of`.
        '''
        units = self.units.standardize().free_of_factors()
        return self.in_units_of(units, subs=subs, copy=copy)

    def __setitem__(self, i, value):
        # if both value and self have units and they are different,
        # take care of them
        if isinstance(value, UnitArr) and value.units is not None \
                and self.units is not None and value.units != self.units:
            np.ndarray.__setitem__(self, i, value.in_units_of(self.units))
        elif isinstance(value, _UnitClass) and self.units is not None and \
                value != self.units:
            np.ndarray.__setitem__(self, i, value.in_units_of(self.units))
        else:
            np.ndarray.__setitem__(self, i, value)

    def __setslice__(self, a, b, value):
        self.__setitem__(slice(a, b), value)

    @staticmethod
    def ufunc_rule(for_ufunc):
        def x(fn):
            UnitArr._ufunc_registry[for_ufunc] = fn
            return fn

        return x

    def _generic_add(self, x, op):
        return op(self, UnitQty(x, self.units))

    def __add__(self, x):
        return self._generic_add(x, op=np.add)

    def __sub__(self, x):
        return self._generic_add(x, op=np.subtract)

    def __iadd__(self, x):
        return self._generic_add(x, op=np.ndarray.__iadd__)

    def __isub__(self, x):
        return self._generic_add(x, op=np.ndarray.__isub__)

    def __mul__(self, x):
        return np.multiply(self, UnitQty(x))

    def __div__(self, x):
        return np.divide(self, UnitQty(x))

    def __truediv__(self, x):
        return np.true_divide(self, UnitQty(x))

    def __floordiv__(self, x):
        return np.floor_divide(self, UnitQty(x))

    def __mod__(self, x):
        return np.remainder(self, UnitQty(x, self.units))

    def __imul__(self, x):
        x = UnitQty(x)
        if self.units is None:
            self.units = x.units
        elif x.units is not None:
            self.units *= x.units
        np.ndarray.__imul__(self, x)
        return self

    def __idiv__(self, x):
        x = UnitQty(x)
        if x.units is not None:
            if self.units is None:
                self.units = 1 / x.units
            else:
                self.units /= x.units
        np.ndarray.__idiv__(self, x)
        return self

    def __itruediv__(self, x):
        x = UnitQty(x)
        if x.units is not None:
            if self.units is None:
                self.units = 1 / x.units
            else:
                self.units /= x.units
        np.ndarray.__itruediv__(self, x)
        return self

    def __ifloordiv__(self, x):
        x = UnitQty(x)
        if x.units is not None:
            if self.units is None:
                self.units = 1 / x.units
            else:
                self.units /= x.units
        np.ndarray.__ifloordiv__(self, x)
        return self

    def __imod__(self, x):
        x = UnitQty(x, self.units)
        np.ndarray.__imod__(self, x)
        return self

    def __pow__(self, x):
        if isinstance(x, Fraction) and self.dtype.kind == 'f':
            # way faster to use floats:
            res = np.ndarray.__pow__(self.view(np.ndarray), float(x))
            res = UnitQty(res, units=self.units ** x)
        else:
            res = np.ndarray.__pow__(self, x)
        return res

    def __ipow__(self, x):
        if isinstance(x, Fraction) and self.dtype.kind == 'f':
            # way faster to use floats:
            np.ndarray.__ipow__(self.view(np.ndarray), float(x))
        else:
            np.ndarray.__ipow__(self, x)
        self.units **= x
        return self

    def prod(self, axis=None, *args, **kwargs):
        x = np.ndarray.prod(self, axis, *args, **kwargs).view(UnitArr)
        if self.units is not None:
            if axis is None:
                x.units = self.units ** np.prod(self.shape)
            else:
                x.units = self.units ** self.shape[axis]
        return x

    def cumprod(self, axis=None, *args, **kwargs):
        raise NotImplementedError('Units of a cumulative product would not ' + \
                                  'be constant!')

    def var(self, *args, **kwargs):
        x = np.ndarray.var(self, *args, **kwargs).view(UnitArr)
        if self.units is not None:
            x.units = self.units ** 2
        return x

    @with_own_units
    def sum(self, *args, **kwargs):
        return np.ndarray.sum(self, *args, **kwargs)

    @with_own_units
    def cumsum(self, *args, **kwargs):
        return np.ndarray.cumsum(self, *args, **kwargs)

    @property
    def T(self):
        return self.transpose()

    @with_own_units
    def transpose(self, *args, **kwargs):
        return np.ndarray.transpose(self, *args, **kwargs)

    @with_own_units
    def byteswap(self, *args, **kwargs):
        return np.ndarray.byteswap(self, *args, **kwargs)

    @with_own_units
    def conj(self, *args, **kwargs):
        return np.conj(self, *args, **kwargs)

    @with_own_units
    def conjugate(self, *args, **kwargs):
        return np.conjugate(self, *args, **kwargs)

    @with_own_units
    def min(self, *args, **kwargs):
        return np.ndarray.min(self, *args, **kwargs)

    @with_own_units
    def max(self, *args, **kwargs):
        return np.ndarray.max(self, *args, **kwargs)

    @with_own_units
    def mean(self, *args, **kwargs):
        return np.mean(self, *args, **kwargs)

    @with_own_units
    def std(self, *args, **kwargs):
        return np.std(self, *args, **kwargs)

    @with_own_units
    def ptp(self, *args, **kwargs):
        return np.ptp(self, *args, **kwargs)

    def argpartition(self, *args, **kwargs):
        return np.ndarray.argpartition(self.view(np.ndarray), *args, **kwargs)

    def argsort(self, *args, **kwargs):
        return np.ndarray.argsort(self.view(np.ndarray), *args, **kwargs)


for f in (np.ndarray.__lt__, np.ndarray.__le__, np.ndarray.__eq__,
          np.ndarray.__ne__, np.ndarray.__gt__, np.ndarray.__ge__):
    # N.B. cannot use functools.partial because it doesn't implement the
    # descriptor protocol
    @functools.wraps(f, assigned=("__name__", "__doc__"))
    def wrapper_function(self, other, comparison_op=f):
        try:
            if hasattr(self, 'snap'):  # e.g. for SimArr
                other = UnitQty(other, self.units, subs=self.snap)
            else:
                other = UnitQty(other, self.units)
        except units.UnitError as e:
            if not e.msg.endswith('convertable!'):
                return NotImplemented
            other = UnitQty(other)
            if isinstance(other, UnitArr):
                raise units.UnitError('%r and %r ' % (self.units, other.units) +
                                      'are not comparible.')
        except:
            raise
        res = comparison_op(self, other)
        return res


    setattr(UnitArr, f.__name__, wrapper_function)


@UnitArr.ufunc_rule(np.add)
@UnitArr.ufunc_rule(np.subtract)
@UnitArr.ufunc_rule(np.minimum)
@UnitArr.ufunc_rule(np.maximum)
def _same_units_binary(a, b):
    a_units = getattr(a, 'units', None)
    b_units = getattr(b, 'units', None)
    if a_units is not None and b_units is not None:
        if a_units == b_units:
            return a_units
        else:
            a_units.in_units_of(b_units)
            raise units.UnitError('Units are not the same but they are ' +
                                  'convertable. Use operator instead of ' +
                                  'function or convert manually.')
    else:
        return a_units if a_units is not None else b_units


@UnitArr.ufunc_rule(np.negative)
@UnitArr.ufunc_rule(np.abs)
@UnitArr.ufunc_rule(np.floor)
@UnitArr.ufunc_rule(np.ceil)
@UnitArr.ufunc_rule(np.trunc)
@UnitArr.ufunc_rule(np.round)  # TODO: does not work, since the
@UnitArr.ufunc_rule(np.around)  # function does something more inbetween
@UnitArr.ufunc_rule(np.round_)  # than others (and calls np.rint)
@UnitArr.ufunc_rule(np.rint)
@UnitArr.ufunc_rule(np.fix)
@UnitArr.ufunc_rule(np.transpose)
@UnitArr.ufunc_rule(np.conjugate)
def _same_units_unary(a):
    return a.units


@UnitArr.ufunc_rule(np.multiply)
@UnitArr.ufunc_rule(np.cross)
@UnitArr.ufunc_rule(np.dot)
@UnitArr.ufunc_rule(np.core.umath_tests.inner1d)
def _mul_units(a, b):
    a_units = getattr(a, 'units', None)
    b_units = getattr(b, 'units', None)
    if a_units is not None and b_units is not None:
        ab_units = a_units * b_units
        return ab_units
    elif a_units is not None:
        return a_units
    else:
        return b_units


@UnitArr.ufunc_rule(np.divide)
@UnitArr.ufunc_rule(np.true_divide)
@UnitArr.ufunc_rule(np.floor_divide)
def _div_units(a, b):
    a_units = getattr(a, 'units', None)
    b_units = getattr(b, 'units', None)
    if a_units is not None and b_units is not None:
        ab_units = a_units / b_units
        return ab_units
    elif a_units is not None:
        return a_units
    else:
        return 1 / b_units


@UnitArr.ufunc_rule(np.reciprocal)
def _inv_units(a):
    if a.units is not None:
        return a.units ** -1
    else:
        return None


@UnitArr.ufunc_rule(np.remainder)
def _remainder_units(a, b):
    a_units = getattr(a, 'units', None)
    b_units = getattr(b, 'units', None)
    if a_units is not None and b_units is not None and a_units != b_units:
        a_units.in_units_of(b_units)
        raise units.UnitError('Units are not the same but they are ' +
                              'convertable. Use operator instead of ' +
                              'function or convert manually.')
    return a_units


@UnitArr.ufunc_rule(np.sqrt)
def _sqrt_units(a):
    if a.units is not None:
        return a.units ** Fraction(1, 2)
    else:
        return None


@UnitArr.ufunc_rule(np.square)
def _square_units(a):
    if a.units is not None:
        return a.units ** 2
    else:
        return None


@UnitArr.ufunc_rule(np.greater)
@UnitArr.ufunc_rule(np.greater_equal)
@UnitArr.ufunc_rule(np.less)
@UnitArr.ufunc_rule(np.less_equal)
@UnitArr.ufunc_rule(np.equal)
@UnitArr.ufunc_rule(np.not_equal)
@UnitArr.ufunc_rule(np.logical_and)
@UnitArr.ufunc_rule(np.logical_or)
@UnitArr.ufunc_rule(np.logical_xor)
@UnitArr.ufunc_rule(np.logical_not)
def _comparison_units(a, b):
    return None


@UnitArr.ufunc_rule(np.bitwise_and)
@UnitArr.ufunc_rule(np.bitwise_or)
@UnitArr.ufunc_rule(np.bitwise_xor)
@UnitArr.ufunc_rule(np.invert)
def _bitwise_op_units(a, b):
    return None


@UnitArr.ufunc_rule(np.power)
def _pow_units(a, b):
    if getattr(a, 'units', None) is not None:
        return a.units ** b
    else:
        return None


@UnitArr.ufunc_rule(np.arctan)
@UnitArr.ufunc_rule(np.arctan2)
@UnitArr.ufunc_rule(np.arcsin)
@UnitArr.ufunc_rule(np.arccos)
@UnitArr.ufunc_rule(np.arcsinh)
@UnitArr.ufunc_rule(np.arccosh)
@UnitArr.ufunc_rule(np.arctanh)
@UnitArr.ufunc_rule(np.sin)
@UnitArr.ufunc_rule(np.tan)
@UnitArr.ufunc_rule(np.cos)
@UnitArr.ufunc_rule(np.sinh)
@UnitArr.ufunc_rule(np.tanh)
@UnitArr.ufunc_rule(np.cosh)
def _trig_units(*a):
    return None


@UnitArr.ufunc_rule(np.exp)
@UnitArr.ufunc_rule(np.log)
@UnitArr.ufunc_rule(np.log2)
@UnitArr.ufunc_rule(np.log10)
def _exp_log_units(*a):
    return None


@UnitArr.ufunc_rule(np.sign)
def _no_units(*a):
    return None


@UnitArr.ufunc_rule(np.isinf)
@UnitArr.ufunc_rule(np.isneginf)
@UnitArr.ufunc_rule(np.isposinf)
@UnitArr.ufunc_rule(np.isnan)
@UnitArr.ufunc_rule(np.isfinite)
def _testfunc_units(*a):
    return None

