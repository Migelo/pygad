'''
In this module there is class defined that decorates a numpy array with units.

Note:
    Since the unit is global to the array, the speed penality is not too big.
    However, if one accesses elements, slices, etc. of a UnitArr, it might help to
    work with 'a.view(np.ndarray)' and add the units again at the end, rather than
    directly working with the UnitArr 'a'.

Example:
    >>> from ..environment import module_dir
    >>> s = UnitArr([[1,-2], [-3,1], [2,3]], dtype=np.float64, units='km/h')
    >>> s.in_units_of('cm/s')   # in different units (without touching s)
    UnitArr([[ 27.77777778, -55.55555556],
             [-83.33333333,  27.77777778],
             [ 55.55555556,  83.33333333]], units="cm s**-1")
    >>> s -= [3, 4]                     # one can calculate with the arrays
    >>> s /= 10                         # as one is used to
    >>> s
    UnitArr([[-0.2, -0.6],
             [-0.6, -0.3],
             [-0.1, -0.1]], units="km h**-1")
    >>> s += UnitArr(1, units='cm/s')   # different units are converted implicitly
    >>> s = 2.5 * s
    >>> s
    UnitArr([[-0.5 , -1.5 ],
             [-1.5 , -0.75],
             [-0.25, -0.25]], units="h**-1 km")
    >>> s[2,0]
    -0.25
    >>> s[2,0] = 5
    >>> s
    UnitArr([[-0.5 , -1.5 ],
             [-1.5 , -0.75],
             [ 5.  , -0.25]], units="h**-1 km")
    >>> np.mean(s,axis=0)
    UnitArr([ 1.        , -0.83333333], units="h**-1 km")
    >>> np.std(s,axis=0)**2
    UnitArr([ 8.16666667,  0.26388889], units="h**-2 km**2")
    >>> np.all(np.abs(np.var(s,axis=0) - np.std(s,axis=0)**2) < 1e-6)
    UnitArr(True, dtype=bool)
    >>> s[::2].convert_to('m/s')    # converts all, not just every 2nd row!
    >>> s
    UnitArr([[-0.13888889, -0.41666667],
             [-0.41666667, -0.20833333],
             [ 1.38888889, -0.06944444]], units="h**-1 km")
    >>> s.units = 'ckpc / h_0'
    >>> s.in_units_of('kpc', subs={'a':0.8, 'h_0':0.7})
    UnitArr([[-0.15873016, -0.47619048],
             [-0.47619048, -0.23809524],
             [ 1.58730159, -0.07936508]], units="kpc")
    >>> UnitArr('10 kpc')
    UnitArr(10.0, units="kpc")
    >>> UnitArr('10', 'kpc')
    UnitArr(10.0, units="kpc")
    >>> UnitArr('10 m', units='kpc')
    Traceback (most recent call last):
    ...
    ValueError: conflicting units in UnitArr instantiation!
    >>> dist(s)
    UnitArr([ 0.43920523,  0.4658475 ,  1.39062392], units="ckpc h_0**-1")

    >>> UnitScalar(3.)
    UnitArr(3.0)
    >>> UnitScalar(42, units='m')
    UnitArr(42, units="m")
    >>> UnitScalar('5 kpc')
    UnitArr(5.0, units="kpc")
    >>> UnitScalar(UnitArr(0.1,'kpc'), units='pc')
    UnitArr(100.0, units="pc")
    >>> from units import Unit
    >>> UnitScalar(Unit('Msol'))
    UnitArr(1.0, units="Msol")
    >>> UnitScalar([1,2,3])
    Traceback (most recent call last):
    ...
    ValueError: object is an array!
    >>> UnitQty([1,2,3])
    UnitArr([1, 2, 3])
    >>> import numpy as np
    >>> UnitQty(np.arange(4), 'km')
    UnitArr([0, 1, 2, 3], units="km")
    >>> UnitQty('3 km')
    UnitArr(3.0, units="km")
    >>> UnitQty('3 a kpc / h_0', 'kpc', subs={'a':0.5, 'h_0':0.5})
    UnitArr(3.0, units="kpc")
'''
__all__ = ['UnitArr', 'dist', 'UnitQty', 'UnitScalar']

import numpy as np
import copy
import numpy.core.umath_tests
import units
from units import *
from units import _UnitClass
from fractions import Fraction
from multiprocessing import Pool, cpu_count
import warnings
import functools
import numbers
from .. import environment

def _Gyr2z_vec(arr, cosmo):
    '''Needed to pickle cosmo.lookback_time_2z for Pool().apply_async.'''
    return np.vectorize(lambda t: cosmo.lookback_time_2z(t))(arr)

def _z2Gyr_vec(arr, cosmo):
    '''Needed to pickle cosmo.lookback_time_in_Gyr for Pool().apply_async.'''
    return np.vectorize(cosmo.lookback_time_in_Gyr)(arr)

def UnitQty(obj, units=None, subs=None, dtype=None):
    '''
    Convert to a UnitArr with enshured units.

    Args:
        obj (UnitArr, array-like, Unit, str, float):
                                The object to convert.
        units (Units, str):     If obj had units, convert to these, otherwise set
                                these units.
        subs (dict, Snap):      Substitutions as passe to UnitArr.convert_to.
        dtype (np.dtype, str):  The requested dtype.
    '''
    if not isinstance(obj, UnitArr):
        if obj is None:
            return None
        obj = UnitArr(obj)

    if dtype is not None:
        obj = obj.astype(dtype)

    if units is not None:
        if obj.units is not None:
            obj.convert_to(units, subs=subs)
        else:
            obj.units = units

    return obj

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
    '''
    obj = UnitQty(obj, units=units, subs=subs, dtype=dtype)
    if getattr(obj, 'shape', None):
        raise ValueError('object is an array!')
    return obj


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

    def __new__(subtype, data, units=None, **kwargs):
        if isinstance(data, str):
            data = Unit(data)
        if isinstance(data, _UnitClass):
            if data.composition:
                if units is not None:
                    raise ValueError('conflicting units in UnitArr instantiation!')
                data, units = data.scale, data.free_of_factors()
            else:
                data = data.scale

        new = np.array(data, **kwargs).view(subtype)

        if units is None:
            #units = getattr(data, '_units', None)
            new._units = getattr(data, '_units', None)#Unit(1) if units is None else units
        elif isinstance(units, _UnitClass):
            new._units = units
        else:
            new._units = Unit(units)

        return new

    def __array_finalize__(self, obj):
        if obj is None:
            return
        else:
            self._units = getattr(obj, '_units', None)
            if isinstance(self.base, UnitArr):
                self.base._units = self._units

    def __array_wrap__(self, array, context=None):
        if context is None:
            return array.view(UnitArr)

        try:
            n_arr = array.view(UnitArr)
            ufunc = context[0]
            # prevent warnings in parallel automatic conversions
            # TODO: find better way!
            if ufunc.__name__ not in ['lookback_time_in_Gyr (vectorized)',
                    'lookback_time_2z (vectorized)']:
                n_arr.units = UnitArr._ufunc_registry[ufunc](*context[1])
            if n_arr._units:
                n_arr._units = n_arr._units.gather()
                n_arr._units._composition.sort()
            return n_arr
        except KeyError:
            #if ufunc.__name__.split()[-1] != '(vectorized)':
            warnings.warn('Operation \'%s\' on units is ' % ufunc.__name__ + \
                          '*not* defined! Return normal numpy array.')
            return array.view(np.ndarray)
        except:
            raise

    @property
    def units(self):
        '''The units of the array.'''
        return self._units

    @units.setter
    def units(self, value):
        if isinstance(value, _UnitClass):
            value = value
        elif value is not None:
            value = Unit(value)

        self._units = value
        if isinstance(self.base, UnitArr):
            self.base._units = value

    @property
    def value(self):
        '''Return the value, if this is just a single value.'''
        if self.shape:
            raise RuntimeError('This is not a single value.')
        return self.dtype.type(self)

    def __copy__(self, *a):
        if a:
            duplicate = np.ndarray.__copy__(self, *a).view(UnitArr)
        else:
            duplicate = np.ndarray.__copy__(self).view(UnitArr)
        if self._units is None:
            duplicate._units = None
        else:
            duplicate._units = _UnitClass(self._units._scale,
                                          copy.deepcopy(self._units._composition))
        return duplicate

    def __deepcopy__(self, *a):
        if a:
            duplicate = np.ndarray.__deepcopy__(self, *a).view(UnitArr)
        else:
            duplicate = np.ndarray.__deepcopy__(self).view(UnitArr)
        if self._units is None:
            duplicate._units = None
        else:
            duplicate._units = _UnitClass(self._units._scale,
                                          copy.deepcopy(self._units._composition))
        return duplicate

    def copy(self, order=None):
        '''Create a real copy.'''
        return self.__copy__()
        """
        if order:
            return self.__copy__(order) # does not work
        else:
            return self.__copy__()
        """

    def __repr__(self):
        if not self.shape and self.dtype.kind == 'f':
            r = 'UnitArr('
            f = float(self)
            r += str(f) if (1e-3<=f<=1e3) else ('%e' % f)
            if self.dtype not in ['int', 'float']:
                r += ', dtype=' + str(self.dtype)
            r += ')'
        else:
            r = np.ndarray.__repr__(self)
            r = 'UnitArr' + r[r.find('('):].replace('\n', '\n  ')
        if self._units is not None and self._units != 1:
            r = r[:-1] + (', units="%s")' % str(self._units)[1:-1])
            if len(r)-r.rfind('\n')>82:
                right = r.find('dtype=')
                if right == -1:
                    right = r.find('units=')
                arr_end = r.rfind('],')+2
                if arr_end == 1: arr_end = r.find(',')+1
                r = r[:arr_end]+'\n'+' '*8+r[right:]
        return r

    def __str__(self):
        if not self.shape and self.dtype.kind == 'f':
            f = float(self)
            s = str(f) if (1e-3<=f<=1e3) else ('%e' % f)
        else:
            s = np.ndarray.__str__(self)
        if self._units is not None and self._units != 1:
            s += ' %s' % self._units
        return s

    def __reduce__(self):
        T = np.ndarray.__reduce__(self)
        T = (T[0], T[1],
             (self._units, T[2][0], T[2][1], T[2][2], T[2][3], T[2][4]))
        return T

    def __setstate__(self, args):
        self._units = args[0] if isinstance(args[0], _UnitClass) else None
        np.ndarray.__setstate__(self, args[1:])

    def astype(self, dtype, order='K', casting='unsafe', subok=True, copy=True):
        '''
        Convert the type of the array.

        Internally np.ndarray.astype is called. See the documentation for more
        details.

        Args:
            dtype (str, dtype):     Typecode or data-type to convert to.
            casting {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}:
                                    Controls the kind of casting that may occur.
                                    For more details see np.ndarray.astype.
            subok (bool):           If True, the sub-class will be
                                    passed-through, otherwise the returned array
                                    will be forced to be a base-class array.
            copy (bool):            If set to false (and dtype requirement is
                                    fulfilled, see np.array.astype), the input
                                    UnitArr with the original data (casted) is
                                    returned.
        '''
        new = np.ndarray.astype(self, dtype, order, casting, subok, copy)
        new._units = self._units
        return new

    def in_units_of(self, units, subs=None, cosmo=None, parallel=None, copy=False):
        '''
        Return the array in other units.

        Args:
            units (Unit, str):  The target units.
            subs (Snap, dict):  See 'convert_to'.
            cosmo (FLRWCosmo):  See 'convert_to'.
            parallel (bool):    See 'convert_to'.
            copy (bool):        If set to false, return the array itself, if the
                                target units are the same as the current ones.

        Returns:
            converted (UnitArr):    The new, converted array. If copy is False and
                                    the target units are the same as the current
                                    ones, however, simply self is returned.

        Raises:
            UnitError:          In case the current units and the target units are
                                not convertable.
        '''
        if not isinstance(units, (_UnitClass,str,numbers.Number)):
            raise TypeError('Cannot convert type %s to units!' % \
                            type(units).__name__)
        if not copy and self.units == units:
            return self
        c = self.copy()
        c.convert_to(units=units, subs=subs, cosmo=cosmo, parallel=parallel)
        return c

    def convert_to(self, units, subs=None, cosmo=None, parallel=None):
        '''
        Convert the array into other units in place.

        Args:
            units (Unit, str):  The units to convert to.
            subs (Snap, dict):  Substitutions to be made during conversion. (For
                                more information see _UnitClass.substitute). If
                                this is a Snap, the values for the redshift 'z',
                                the scale factor 'a', and the Hubble parameter
                                'h_0' from it are used as well as the cosmology,
                                if not set explicitly by the other argument.
            cosmo (FLRWCosmo):  A FLRW cosmology to use, when it is needed to
                                convert lookbacktime to z_form (or a_form) or
                                vice versa.
            parallel (bool):    If units are converted from Gyr (or some other
                                time unit) to z_form / a_form, one can choose to
                                use multiple threads. By default, the function
                                chooses automatically whether to perform in
                                parallel or not.

        Raises:
            UnitError:          In case the current units and the target units are
                                not convertable.
        '''
        if not isinstance(units, _UnitClass):
            units = Unit(units)
        if self._units == units:
            return
        elif self._units is None:
            self.units = units
            return

        # if this is not the entire array, pass down to base
        if isinstance(self.base, UnitArr):
            self.base.convert_to(units, subs=subs, cosmo=cosmo, parallel=parallel)
            self._units = units
            return

        from ..snapshot.snapshot import _Snap
        if subs is None:
            subs = {}
        elif isinstance(subs, _Snap):
            snap, subs = subs, {}
            subs['a'] = snap.scale_factor
            subs['z'] = snap.redshift
            subs['h_0'] = snap.cosmology.h_0
            if cosmo is None:
                cosmo = snap.cosmology

        # for these kind of conversion ignore subs(titutions)
        if self._units in ['a_form', 'z_form'] or units in ['a_form', 'z_form']:
            from ..physics import a2z, z2a
            # We have functions to convert: a <-> z <-> Gyr or other time units
            # first bring own units to z_form
            if self._units == 'a_form':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.setfield(np.vectorize(a2z)(self), dtype=self.dtype)
            elif self._units != 'z_form':
                # then the unit is (or should be) some time unit
                # to present day (z=0) ages
                self += cosmo.lookback_time(subs['z'])
                # from present day lookback times (ages) to z
                if environment.allow_parallel_conversion and (
                        parallel or (parallel is None and len(self) > 10)):
                    N_threads = cpu_count()
                    chunk = [[i*len(self)/N_threads, (i+1)*len(self)/N_threads] 
                                for i in xrange(N_threads)]
                    p = Pool(N_threads)
                    res = [None] * N_threads
                    for i in xrange(N_threads):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            res[i] = p.apply_async(_Gyr2z_vec,
                                                   (self[chunk[i][0]:chunk[i][1]],
                                                       cosmo))
                    for i in xrange(N_threads):
                        self[chunk[i][0]:chunk[i][1]] = res[i].get()
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.setfield(np.vectorize(lambda t:
                                        cosmo.lookback_time_2z(t))(self),
                                      dtype=self.dtype)
            self.units = 'z_form'
            # if target unit is z_form, we are done
            if units == 'z_form':
                return
            # convert own units (now in z_form) to target unit
            if units == 'a_form':
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.setfield(np.vectorize(z2a)(self), dtype=self.dtype)
            else:   # target unit is (or should be) some time unit
                if environment.allow_parallel_conversion and (
                        parallel or (parallel is None and len(self) > 1000)):
                    N_threads = cpu_count()
                    chunk = [[i*len(self)/N_threads, (i+1)*len(self)/N_threads]
                                for i in xrange(N_threads)]
                    p = Pool(N_threads)
                    res = [None] * N_threads
                    with warnings.catch_warnings():
                        # warnings.catch_warnings doesn't work in parallel
                        # environment...
                        warnings.simplefilter("ignore") # for _z2Gyr_vec
                        for i in xrange(N_threads):
                            res[i] = p.apply_async(_z2Gyr_vec,
                                        (self[chunk[i][0]:chunk[i][1]], cosmo))
                    for i in xrange(N_threads):
                        self[chunk[i][0]:chunk[i][1]] = res[i].get()
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore") # for _z2Gyr_vec
                        self.setfield(_z2Gyr_vec(self,cosmo), dtype=self.dtype)
                self.units = 'Gyr'
                if units != 'Gyr':
                    view = self.view(np.ndarray)
                    view *= self._units.in_units_of(units, subs=subs)
                # from present day ages (lookback time) to actual current ages
                self -= cosmo.lookback_time(subs['z'])
            self.units = units
        else:
            # not a_form of z_form
            view = self.view(np.ndarray)
            view *= self._units.in_units_of(units, subs=subs)
            self.units = units

    def __getitem__(self, i):
        item = np.ndarray.__getitem__(self, i)
        if isinstance(i, slice) or \
                (isinstance(i, tuple) and len(i)==1 \
                    and isinstance(i[0], np.ndarray)):
            item._units = self._units
        return item

    def __getslice__(self, a, b):
        return self.__getitem__(slice(a,b))

    def __setitem__(self, i, value):
        # if both value and self have units and they are different,
        # take care of them
        if isinstance(value, UnitArr) and value._units is not None \
                and self._units is not None and value._units != self._units:
            np.ndarray.__setitem__(self, i, value.in_units_of(self._units))
        elif isinstance(value, _UnitClass) and self._units is not None and \
                value != self._units:
            np.ndarray.__setitem__(self, i, value.in_units_of(self._units))
        else:
            np.ndarray.__setitem__(self, i, value)

    def __setslice__(self, a, b, value):
        self.__setitem__(slice(a,b), value)

    @staticmethod
    def ufunc_rule(for_ufunc):
        def x(fn):
            UnitArr._ufunc_registry[for_ufunc] = fn
            return fn

        return x

    def _generic_add(self, x, op):
        return op(self, UnitQty(x,self._units))

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
        return np.remainder(self, UnitQty(x))

    def __imul__(self, x):
        x = UnitQty(x)
        if self._units is None:
            self._units = x._units
        elif x._units is not None:
            self.units *= x._units
        np.ndarray.__imul__(self, x)
        return self

    def __idiv__(self, x):
        x = UnitQty(x)
        if x._units is not None:
            if self._units is None:
                self.units = 1 / x._units
            else:
                self.units /= x._units
        np.ndarray.__idiv__(self, x)
        return self

    def __itruediv__(self, x):
        x = UnitQty(x)
        if x._units is not None:
            if self._units is None:
                self.units = 1 / x._units
            else:
                self.units /= x._units
        np.ndarray.__itruediv__(self, x)
        return self

    def __ifloordiv__(self, x):
        x = UnitQty(x)
        if x._units is not None:
            if self._units is None:
                self.units = 1 / x._units
            else:
                self.units /= x._units
        np.ndarray.__ifloordiv__(self, x)
        return self

    def __imod__(self, x):
        x = UnitQty(x)
        if self._units is not None and x._units is not None:
            x = x.in_units_of(self._units)
        np.ndarray.__imod__(self, x)
        return self

    def __pow__(self, x):
        if isinstance(x,Fraction) and self.dtype.kind=='f':
            res = np.ndarray.__pow__(self,float(x)) # way faster to use floats
        else:
            res = np.ndarray.__pow__(self,x)
        return res

    """
    def append(self, array):
        '''
        Append a numpy array or a UnitArr to this one.

        Args:
            array (np.ndarray, UnitArr):
                    The array to append. It has to have a compatible data-type as
                    this array. If it is a UnitArr, the units are converted
                    automatically before appending.
        '''
        if not hasattr(array, 'shape'):
            array = np.array(array, dtype=self._array.dtype)
        if self._array.shape[1:] != array.shape[1:]:
            raise ValueError('The array to append has to have ' + \
                             'compatible shapes!')
        if isinstance(array, UnitArr) and self._units != array._units:
            array = array.copy()    # do not destroy the input
            array.convert_to(self._units)
        if self._array.flags.owndata:
            self._array.resize((self._array.shape[0]+array.shape[0],) + \
                               self._array.shape[1:])
            self._array[-array.shape[0]:] = array
        else:
            self._array = np.concatenate( (self._array, array), axis=0)
    """

    def cumsum(self, axis=None, dtype=None, out=None):
        x = np.ndarray.cumsum(self, axis, dtype, out)
        x.units = self._units
        return x

    def prod(self, axis=None, dtype=None, out=None):
        x = np.ndarray.prod(self, axis, dtype, out)
        if axis is None:
            x.units = self._units ** np.prod(self.shape)
        else:
            x.units = self._units ** self.shape[axis]
        return x

    def sum(self, *args, **kwargs):
        x = np.ndarray.sum(self, *args, **kwargs)
        x.units = self._units
        return x

    def mean(self, *args, **kwargs):
        x = np.ndarray.mean(self, *args, **kwargs)
        x.units = self._units
        return x

    def max(self, *args, **kwargs):
        x = np.ndarray.max(self, *args, **kwargs)
        x.units = self._units
        return x

    def min(self, *args, **kwargs):
        x = np.ndarray.min(self, *args, **kwargs)
        x.units = self._units
        return x

    def ptp(self, *args, **kwargs):
        x = np.ndarray.ptp(self, *args, **kwargs)
        x.units = self._units
        return x

    def std(self, *args, **kwargs):
        x = np.ndarray.std(self, *args, **kwargs)
        x.units = self._units
        return x

    def var(self, *args, **kwargs):
        x = np.ndarray.var(self, *args, **kwargs)
        x.units = self._units**2
        return x

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
    '''
    from scipy.spatial.distance import cdist
    arr_units = getattr(arr, '_units', None)
    pos_units = getattr(pos, '_units', None)
    if pos is None:
        if not isinstance(arr, np.ndarray): # includes UnitArr
            arr = np.array(arr)
        pos = [0]*arr.shape[-1]
    if arr_units is not None:
        units = arr_units
        if pos_units is not None:
            pos = pos.in_units_of(arr_units)
    elif pos_units is not None:
        units = pos_units
    else:
        units = None
    res = cdist(arr, [pos]).ravel().view(UnitArr)
    res._units = units
    return res

for f in (np.ndarray.__lt__, np.ndarray.__le__, np.ndarray.__eq__,
        np.ndarray.__ne__, np.ndarray.__gt__, np.ndarray.__ge__):
    # N.B. cannot use functools.partial because it doesn't implement the
    # descriptor protocol
    @functools.wraps(f, assigned=("__name__", "__doc__"))
    def wrapper_function(self, other, comparison_op=f):
        try:
            if hasattr(self,'snap'):
                other = UnitQty(other, self._units, subs=self.snap)
            else:
                other = UnitQty(other, self._units)
        except units.UnitError as e:
            if not e.msg.endswith('convertable!'):
                return NotImplemented
            other = UnitQty(other)
            if isinstance(other, UnitArr):
                raise units.UnitError('%r and %r ' % (self._units, other._units) +
                                      'are not comparible.')
        except:
            raise
        res = comparison_op(self, other)
        return res

    setattr(UnitArr, f.__name__, wrapper_function)

@UnitArr.ufunc_rule(np.add)
@UnitArr.ufunc_rule(np.subtract)
def _same_units_binary(a, b):
    a_units = getattr(a, '_units', None)
    b_units = getattr(b, '_units', None)
    if a_units is not None and b_units is not None:
        if a_units == b_units:
            return a_units
        else:
            a_units.in_units_of(b_units)
            raise units.UnitError('Units are not the same, however, they are ' +
                                  'convertable. Use operator instead of ' +
                                  'function or convert manually.')
    elif a_units is not None:
        return a_units
    else:
        return b_units

@UnitArr.ufunc_rule(np.negative)
@UnitArr.ufunc_rule(np.abs)
@UnitArr.ufunc_rule(np.floor)
@UnitArr.ufunc_rule(np.ceil)
#@UnitArr.ufunc_rule(np.minimum)    not unary
#@UnitArr.ufunc_rule(np.maximum)    not unary
def _same_units_unary(a):
    return a._units

@UnitArr.ufunc_rule(np.multiply)
@UnitArr.ufunc_rule(np.core.umath_tests.inner1d)
def _mul_units(a, b):
    a_units = getattr(a, '_units', None)
    b_units = getattr(b, '_units', None)
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
    a_units = getattr(a, '_units', None)
    b_units = getattr(b, '_units', None)
    if a_units is not None and b_units is not None:
        ab_units = a_units / b_units
        return ab_units
    elif a_units is not None:
        return a_units
    else:
        return 1/b_units

@UnitArr.ufunc_rule(np.remainder)
def _remainder_units(a, b):
    return getattr(a, '_units', None)

@UnitArr.ufunc_rule(np.sqrt)
def _sqrt_units(a):
    if a._units is not None:
        return a._units**Fraction(1,2)
    else:
        return None

@UnitArr.ufunc_rule(np.square)
def _square_units(a):
    if a._units is not None:
        return a._units**2
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
    if getattr(a,'_units',None) is not None:
        return a._units**b
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

@UnitArr.ufunc_rule(np.isinf)
@UnitArr.ufunc_rule(np.isneginf)
@UnitArr.ufunc_rule(np.isposinf)
@UnitArr.ufunc_rule(np.isnan)
@UnitArr.ufunc_rule(np.isfinite)
def _testfunc_units(*a):
    return None

