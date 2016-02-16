'''
A UnitArr that belongs to a snapshot and can have derived arrays, which are
deleted from the snapshot, if the array changes.

Examples:
    >>> from snapshot import Snap
    >>> from ..environment import module_dir
    >>> s = Snap(module_dir+'../snaps/snap_M1196_4x_470')
    >>> sa = SimArr([1,2,3], units='a kpc / h_0', dtype=float, snap=s)
    >>> sa[:2]
    SimArr([ 1.,  2.], units="a kpc h_0**-1", snap="snap_M1196_4x_470")
    >>> sa.in_units_of('kpc')
    UnitArr([ 1.38888889,  2.77777778,  4.16666667], units="kpc")
    >>> sa.convert_to('kpc')
    >>> sa
    SimArr([ 1.38888889,  2.77777778,  4.16666667],
           units="kpc", snap="snap_M1196_4x_470")
    >>> sa+sa
    UnitArr([ 2.77777778,  5.55555556,  8.33333333], units="kpc")
    >>> 2*sa
    UnitArr([ 2.77777778,  5.55555556,  8.33333333], units="kpc")
'''

import numpy as np
from ..units import *
from snapshot import _Snap
import functools

class SimArr(UnitArr):
    '''
    A UnitArr that belongs to a snapshot and can have derived arrays, which are
    deleted from the snapshot, if the array changes.

    Operations on this array return UnitArr's, e.g. type(a*a) = units.UnitArr,
    even if a is a SimArr.
    '''

    def __new__(subtype, data, units=None, snap=None, subs=None, **kwargs):
        ua = UnitArr(data, units=units, subs=subs, **kwargs)
        
        new = ua.view(subtype)
        copy = kwargs.get('copy',True)
        new._unit_carrier = new if copy else getattr(ua, '_unit_carrier', new)
        new._unit_carrier._units = getattr(ua, 'units', None)
        new._snap = snap if snap else getattr(data, 'snap', None)
        new._dependencies = getattr(data, '_dependencies', set())

        return new

    # operations of unit array should always return simple UnitArr
    # otherwise one would have 'memeory leaks' where the hidden references to
    # snapshots do not get deleted
    def __array__(self, dtype=None):
        ua = self.view(UnitArr)
        del ua._snap
        del ua._dependencies
        return ua

    def __array_finalize__(self, obj):
        UnitArr.__array_finalize__(self, obj)
        self._snap = getattr(obj, '_snap', None)
        self._dependencies = getattr(obj, '_dependencies', set())

    def __array_wrap__(self, array, context=None):
        return UnitArr.__array_wrap__(self, array, context)

    @property
    def snap(self):
        '''The associated snapshot.'''
        if isinstance(self.base,SimArr):
            return self.base.snap
        else:
            return self._snap

    @property
    def dependencies(self):
        '''A set of the names of the derived blocks that depend on this one.'''
        if isinstance(self.base,SimArr):
            return self.base.dependencies
        else:
            return self._dependencies

    def __repr__(self):
        r = super(SimArr, self).__repr__()
        r = r.replace('UnitArr', 'SimArr').replace('\n ', '\n')
        if self.snap is not None:
            r = r[:-1] + (', snap=%s)' % self.snap.descriptor)
            if len(r)-r.rfind('\n')>80:
                right = r.find('dtype=')
                if right == -1:
                    right = r.find('units=')
                    if right == -1:
                        right = r.find('snap=')
                arr_end = r.rfind('],')+2
                if arr_end == 1: arr_end = r.find(',')+1
                r = r[:arr_end]+'\n'+' '*7+r[right:]
        return r

    def __copy__(self, *a):
        if a:
            duplicate = UnitArr.__copy__(self, *a).view(SimArr)
        else:
            duplicate = UnitArr.__copy__(self).view(SimArr)
        duplicate._snap = self.snap
        duplicate._dependencies = self.dependencies.copy()
        return duplicate

    def __deepcopy__(self, *a):
        if a:
            duplicate = UnitArr.__deepcopy__(self, *a).view(SimArr)
        else:
            duplicate = UnitArr.__deepcopy__(self).view(SimArr)
        duplicate._snap = self.snap
        duplicate._dependencies = self.dependencies.copy()
        return duplicate

    def convert_to(self, units, subs=None):
        '''See UnitArr for documentation.'''
        if subs is None:
            subs = self.snap
        #super(SimArr, self).convert_to(units, subs=subs)
        UnitArr.convert_to(self, units, subs=subs)

    def in_units_of(self, units, subs=None, copy=False):
        '''See UnitArr for documentation. This, however, returns a UnitArr view.'''
        if subs is None:
            subs = self.snap
        conv = UnitArr.in_units_of(self, units, subs=subs, copy=copy)
        if conv is not self:
            del conv._snap
            del conv._dependencies
        return conv.view(UnitArr)

    def invalidate_dependencies(self):
        '''
        Invalidate all dependencies by simply deleting them from the snapshot.
        (They get rederived automatically, if needed, anyway.)

        There is no need to call this function directly. It gets called once the
        array changes.
        '''
        for dep in self.dependencies:
            host = self.snap.get_host_subsnap(dep)
            if dep in host._blocks:
                del host[dep]

# __i*__ functions shall not erase the _snap and _dependencies attributes and
# shall invalidate all dependent SimArr's
def _ichange_wrapper(f):
    def ichange__(self, other):
        self.invalidate_dependencies()
        return f(self, other)
    ichange__.__name__ = f.__name__
    return ichange__
for fn in ('__iadd__', '__isub__', '__imul__', '__idiv__',
           '__itruediv__', '__ifloordiv__', '__imod__',
           '__ipow__', '__ilshift__', '__irshift__',
           '__iand__', '__ior__', '__ixor__'):
    setattr(SimArr, fn, _ichange_wrapper(getattr(UnitArr,fn)))

# __set*__ functions shall invalidate all dependant SimArr's
def _set_wrapper(f):
    def set__(self, *y, **kw):
        self.invalidate_dependencies()
        if kw:
            f(self, *y, **kw)
        else:
            f(self, *y)
    set__.__name__ = f.__name__
    return set__
for fn in ('__setitem__', '__setslice__'):
    setattr(SimArr, fn, _set_wrapper(getattr(UnitArr,fn)))

"""
# arithmetic functions that yield new arrays shall return UnitArr's
def _arith_wrapper(f):
    def arith__(self, other):
        res = f(self, other)
        if res is not NotImplemented:
            res = res.view(UnitArr)
        return res
    arith__.__name__ = f.__name__
    return arith__
for fn in ('__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
           '__div__', '__rdiv__', '__truediv__', '__floordiv__', '__mod__',
           '__pow__', '__eq__', '__gt__', '__lt__', '__ge__', '__le__',
           '__rshift__', '__lshift__'):
    setattr(SimArr, fn, _arith_wrapper(getattr(UnitArr,fn)))

# "properties" should be UnitArr's
def _prop_wrapper(f):
    def prop__(self, *a, **kw):
        res = f(self, *a, **kw)
        if res is not NotImplemented:
            res = res.view(UnitArr)
        return res
    prop__.__name__ = f.__name__
    return prop__
for fn in ('cumsum', 'prod', 'sum', 'max', 'min', 'ptp', 'mean', 'std', 'var'):
    setattr(SimArr, fn, _prop_wrapper(getattr(UnitArr,fn)))
"""

