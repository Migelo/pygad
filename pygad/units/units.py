'''
An implementation of units.

For a system of units one defined base units (globally registered) and can then
build derived units from them. This can be done by direct arithmetic of the units
or by defining them via string. These can again be registered under a name for
later use:
>>> undefine_all()
>>> define('m'), define('s')
(Unit("m"), Unit("s"))
>>> m = Unit('m')
>>> print(m**3)
[m**3]
>>> cm = define('cm', '1e-2 m')
>>> define('in', '2.54 cm')
Traceback (most recent call last):
...
pygad.units.units.UnitError: '"in" is a keyword or a function or constant from math!'
>>> define('inch', '2.54 cm')
Unit("inch")
>>> define('km', 1e3*m)
Unit("km")
>>> m, km, s, h = Units('m,km,s,3600*s')
>>> h # not named/registered, however
Unit("3600 s")
>>> Unit('s**1/2')**2   # even fractional powers are allowed (no brackets needed)
Unit("s")
>>> Unit('s**1 / 2')**2 # this, however, is half a second (squared)
Unit("0.25 s**2")

Conversion factors can conveniently be calculated as long as they are convertable:
>>> (km/h).in_units_of('inch/s')
10.936132983377076
>>> Unit('m').in_units_of(s)
Traceback (most recent call last):
...
pygad.units.units.UnitError: 'Units [m] and [s] are not convertable!'
>>> convertable('m', 'km')
True
>>> convertable('m', 'km/s')
False

Expanding, comparing, and undefining units is also straight forward:
>>> kg = define('kg')
>>> N = define('N', m*kg/s**2)
>>> J = define('J', 'N m')
>>> print(J.expand(full=False))
[N m]
>>> print(J.expand().standardize())
[kg m**2 s**-2]
>>> N == m*kg/s**2
True
>>> J == 'N m', J == N
(True, False)
>>> undefine('N')
>>> defined_units()
['m', 's', 'cm', 'inch', 'km', 'kg', 'J']
>>> Unit('N')
Traceback (most recent call last):
...
pygad.units.units.UnitError: "Undefined symbol: 'N'"
>>> pc = define('pc', 3.08568e+16*m)
>>> Msol = define('Msol', 1.989e30*kg, latex=r'M_\odot')
>>> (Msol/pc**2).latex()
'M_\\\\odot\\\\,\\\\mathrm{pc}^{-2}'

You can also substitute units with numbers:
>>> define('a'), define('h_0')
(Unit("a"), Unit("h_0"))
>>> define('kpc', '1e3 pc')
Unit("kpc")
>>> ckpc = Unit('a kpc')
>>> ckpc.substitute({'a':0.5})
Unit("0.5 kpc")
>>> (ckpc / Unit('h_0')).in_units_of(Unit('kpc/h_0'), {'a':0.5})
0.5

One can also read config files, that define units. See define_from_cfg for more
dertails.
>>> from ..environment import module_dir
>>> define_from_cfg([module_dir+'config/units.cfg'], allow_redef=True,
...                 warn=False)
reading units definitions from "units.cfg"
>>> Unit('erg').in_units_of('J')
1e-07
>>> Unit('Msol / h_0').latex()
'M_\\\\odot\\\\,h_0^{-1}'
'''
__all__ = ['UnitError', 'define', 'set_latex_repr', 'undefine', 'undefine_all',
           'defined_units', 'define_from_cfg', 'Unit', 'Units', 'Fraction',
           'convertable']

from numbers import Number
from fractions import Fraction
import numpy
import math
from ..utils import *
from keyword import iskeyword
import re
from configparser import ConfigParser
import sys
import ast
import operator as op
from os import path
from .. import environment

class UnitError(Exception):
    '''
    Exception class for any errors that directly arise from units.

    Args:
        msg (object):   The message to be displayed.
    '''
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)

class _UnitClass(object):
    '''
    Implementation of a unit class. Should not be instantiated directly, but via
    the factory function 'Unit'.

    Note:
        The scale *NEEDS* to be a float and the composition *NEEDS* to be a list
        of length-2-lists with the first being a string and the second an integer
        or a Fraction! Otherwise the class might not work properly. Checks or
        conversions of the arguments are not performed for reasons of speed.

    Args:
        scale (float):      The scale.
        composition (list): A list of length-2-lists, where the first is a name of
                            a unit and the second its power.
    '''

    def __init__(self, scale, composition):
        self._scale = scale
        self._composition = composition

    @property
    def scale(self):
        '''The prefactor of the unit.'''
        return self._scale

    @property
    def composition(self):
        '''The definition of the unit by other units.'''
        return self._composition

    def _as_string(self):
        if self._scale != 1 or not self._composition:
            s = '%g ' % (self._scale)
        else:
            s = ''
        comp_sort = sorted(self._composition, key=lambda x: x[0])
        s += ' '.join('%s**%s'%(u,p) if p!=1 else u for u,p in comp_sort)
        return s.strip()

    def __str__(self):
        if self == 0:
            return 'base unit'
        return '[%s]' % self._as_string()

    def __repr__(self):
        return 'Unit("%s")' % self._as_string()

    def latex(self):
        '''
        Return a representation of the unit formatted for LaTeX.

        By default units will get is usual name in a mathrm environment, unless
        there is a individual LaTeX representation registered (either in define
        itself or later by set_latex_repr).

        Returns:
            latex (str):    The LaTeX representation. It's without the '$' for a
                            math environment, though!
        '''
        if self._scale != 1 or not self._composition:
            s = float_to_nice_latex(self._scale) + r'\,'
        else:
            s = ''
        for u, p in self._composition:
            if u in _unit_latex:
                s += _unit_latex[u]
            else:
                s += r'\mathrm{%s}' % u
            if p != 1:
                s += r'^{%d}' % p
            s += r'\,'
        return s.rstrip(r'\,')

    def expand(self, full=True):
        '''
        Replace all non-base units with their definitions.

        Args:
            full (bool):        Whether to to the replacements recursively.

        Returns:
            expanded (Unit):    A unit with all replacements.
        '''
        expanded = _UnitClass(self._scale, [])
        for unit,power in self._composition:
            u_exp = _unit_definitions.get(unit,None)
            if u_exp and u_exp._composition:
                u_exp = u_exp.expand() if full else u_exp
                expanded._scale *= u_exp._scale**power
                expanded._composition += [[u,p*power] for u,p in u_exp._composition]
            else:
                # a undefined unit
                expanded._composition.append([unit,power])
        return expanded

    def gather(self):
        '''Return a unit where all equal units are gathered under one exponent.'''
        composition = []
        for unit in set(u for u,p in self._composition):
            power = sum(p for u,p in self._composition if u==unit)
            if power != 0:
                composition.append( [unit, power] )
        return _UnitClass(self._scale, composition)

    def standardize(self):
        '''
        Return a standardized representation of the unit.

        This function is equivalent with self.expand().gather() followed by
        sorting the composition lexicographically.
        '''
        std = self.expand().gather()
        std._composition.sort()
        return std

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, _UnitClass):
            other = _UnitClass(0,[]) if other==0 else Unit(other)
        a = self.standardize()
        b = other.standardize()
        return a._scale == b._scale and a._composition == b._composition

    def __ne__(self, other):
        return not self.__eq__(other)

    def __mul__(self, rhs):
        if isinstance(rhs, Number):
            return _UnitClass(self._scale * rhs, self._composition)
        return _UnitClass(self._scale * rhs._scale,
                          self._composition + rhs._composition)

    def __rmul__(self, lhs):
        if isinstance(lhs, Number):
            return _UnitClass(lhs * self._scale, self._composition)
        return _UnitClass(lhs._scale * self._scale,
                          lhs._composition + self._composition)

    def __div__(self, rhs):
        if isinstance(rhs, Number):
            return _UnitClass(self._scale / rhs, self._composition)
        return _UnitClass(self._scale / rhs._scale,
                          self._composition + [[u,-p] for u,p in rhs._composition])

    def __rdiv__(self, lhs):
        inv_comp = [[u,-p] for u,p in self._composition]
        if isinstance(lhs, Number):
            return _UnitClass(lhs / self._scale, inv_comp)
        return _UnitClass(lhs._scale / self._scale,
                          lhs._composition + inv_comp)

    def __truediv__(self, rhs):
        return self.__div__(rhs)

    def __rtruediv__(self, rhs):
        return self.__rdiv__(rhs)

    def sqrt(self):
        return self.__pow__(Fraction(1,2))

    def __pow__(self, power):
        if not isinstance(power, (int, Fraction, numpy.int32, numpy.int64)):
            raise UnitError('Units can only be raised to integer of fractional '
                            'powers!')
        '''
        if power % 1:
            raise UnitError('Units can only be raised to integer powers!')
        '''
        return _UnitClass(self._scale**power,
                          [[u,p*power] for u,p in self._composition])

    def __float__(self):
        std = self.standardize()
        if std._composition:
            raise UnitError('Unit is not dimensionless!')
        return std._scale

    def _substitute(self, subs):
        '''Helper function for substitute.'''
        scale = self._scale
        subst_comp = []
        did = False
        for u,p in self._composition:
            if u in subs:
                scale *= subs[u]**p
                did = True
            else:
                e = _UnitClass(1.0, [[u,p]]).expand(full=False)
                if len(e._composition) == 1:
                    subst_comp.append([u,p])
                else:
                    e, did_e = e._substitute(subs)
                    if did_e:
                        scale *= e._scale
                        subst_comp += e._composition
                        did = True
                    else:
                        subst_comp.append([u,p])
        return _UnitClass(scale, subst_comp), did
    def substitute(self, subs):
        '''
        Substitute all occurances of the given units with the corresponding
        numbers, even if hidden in unit definitions used for this unit.

        Args:
            subs (dict, Snap):  The units to replace with the corresponding
                                numbers as values. If a snapshot is passed, the
                                redshift ('z'), the scale factor ('a') and the
                                Hubble parameter ('h_0') are read from it.

        Returns:
            substituted (Unit): The new unit.
        '''
        from ..snapshot.snapshot import Snapshot
        if isinstance(subs, Snapshot):
            subs = {'a': subs.scale_factor,
                    'z': subs.redshift,
                    'h_0': subs.cosmology.h_0}
        return self._substitute(subs)[0]

    def free_of_factors(self, unit_names=None):
        '''
        Remove all occurances of the given units, even if hidden in unit
        definitions used for this unit, and set scale to 1.0.

        If no unit names are given, this gives the pure units with scale = 1.0.

        Args:
            unit_names (list):  The unit name to remove.

        Returns:
            clean (Unit):       The new unit.
        '''
        if unit_names is None:
            unit_names = {}
        subs = {name:1 for name in unit_names}
        return _UnitClass(1.0, self._substitute(subs)[0]._composition)

    def in_units_of(self, other, subs=None):
        '''
        Return the convertion factor from this unit to the other one.

        Equivalent to float(self/Unit(other)).

        Args:
            other (Unit, str):      The reference unit.
            subs (dict, Snap):      Unit to substitute with numbers in the
                                    convertion. For more information see
                                    'substitute'.

        Returns:
            conv_factor (float):    The conversion factor from the current unit
                                    to the other one.

        Raises:
            UnitError:              If the units are not convertable.
        '''
        if not isinstance(other, _UnitClass):
            other = Unit(other)
        try:
            if subs is None:
                return float(self/other)
            else:
                return float(self.substitute(subs)/other.substitute(subs))
        except UnitError:
            raise UnitError('Units %s and %s are not convertable!' % (self, other))

_unit_definitions = {}
_unit_latex = {}

def define(name, u=None, latex=None, allow_redef=True, warn=True):
    '''
    Define a new unit.

    Args:
        name (str):         Name of the new unit. Keywords or names of elements of
                            the math module are not allowed.
        u (Unit, str):      The definition of the new unit. If None, the new unit
                            is a base unit.
        latex (str):        A individual LaTeX representation for that unit. By
                            default its the unit name in a mathrm environment
                            (r'\mathrm{%s}' % name).
        allow_redef (bool): If True, only warn about the redefinition of units and
                            totally ignore it, if the definition actually does not
                            change.
        warn (bool):        Turn off warning, if there was a unit redefined
                            differently.

    Returns:
        unit (Unit):    The newly defined unit.
    '''
    if iskeyword(name) or name in math.__dict__:
        raise UnitError('"%s" is a keyword or a function or constant from math!'
                % name)
    elif not re.match(r"^[_a-zA-Z][_a-zA-Z0-9]*$", name):
        raise UnitError('"%s" is not a valid name!' % name)

    if u is None:
        u = _UnitClass(0, [])
    elif not isinstance(u, _UnitClass):
        u = Unit(u)
    if name in _unit_definitions:
        if allow_redef:
            if warn and _unit_definitions[name] != u:
                print('WARNING: Changing definition of ' + \
                        '"%s": %s -> %s)!' % (name, _unit_definitions[name], u), file=sys.stderr)
        else:
            raise UnitError('There is already a unit defined as "%s"!' % name)
    _unit_definitions[name] = u
    if latex:
        set_latex_repr(name, latex)
    return Unit(name)

def set_latex_repr(name, latex):
    '''Set a special latex representation for a unit.'''
    if name not in _unit_definitions:
        raise KeyError('There is no unit named "%s" registered. Cannot ' % name +
                       'set its LaTeX representation!')
    if name in _unit_latex:
        if _unit_latex[name] != latex:
            print('WARNING: Changing LaTeX representation of ' + \
                    '"%s": %s -> %s)!' % (name, _unit_latex[name], latex), file=sys.stderr)
    _unit_latex[name] = latex

def undefine(name):
    '''Delete the definition of a unit.'''
    if name not in _unit_definitions:
        print('WARNING: Tried to undefine not defined unit "%s"!' % (name), file=sys.stderr)
        return
    del _unit_definitions[name]
    if name in _unit_latex:
        del _unit_latex[name]

def undefine_all():
    '''Delete all unit definitions.'''
    global _unit_definitions, _unit_latex
    _unit_definitions.clear()
    _unit_latex.clear()

def defined_units():
    '''Return a list of the names of all defined units.'''
    return list(_unit_definitions.keys())

def define_from_cfg(config, allow_redef=False, warn=True, undefine_old=True):
    '''
    Define unit from a config-file.

    The config file has to be readable by ConfigParser.SafeConfigParser, i.e. it
    basically has to have the format of a Microsoft Windows INI file.

    The section 'base' is required which lists all base units. All other sections
    are optional.
    The section 'prefixes' one can define some prefixes (e.g. 'k' for 1e3), which
    are applied automatically to all units unless there is a section called
    'prefix'.
    The section 'prefix' (which, of course, requires section 'prefixes') would
    define explicitly which prefixes shall be applied to which units.
    The section 'derived' defines more units that can be derived from any other
    unit, even other (maybe prefixed) derived units. Te author of the config file,
    however, has to care that all units are well defined, i.e. there are no
    circularities of units defined by undefined units, for instance.
    Finally, in the optional section 'latex' one can define some deviant LaTeX
    representations of (already otherwise defined) units.

    Args:
        config (list):      list of possible filenames for the config file.
        allow_redef (bool): If True, only warn about the redefinition of units.
                            This includes previously defined units. (Also see
                            function 'define'!)
        warn (bool):        Turn off warning, if there was a unit redefined
                            differently.
        undefine_old (bool):Undefine all previously defined units.

    Raises:
        RuntimeError:       In case not all units are well defined (e.g. ther are
                            circular definitions) or in case the config does does
                            not exist.

    A sample config file:
    [base]
    m
    g
    s
    [prefixes]
    m   = 1e-3
    c   = 1e-2
    k   = 1e3
    [prefix]
    m:  m,c,k
    g:  m,k
    N:  k
    [derived]
    min     = 60 s
    h       = 60 min
    N       = kg m / s**2
    Msol    = 1.989e30 kg
    [latex]
    Msol    = M_\odot

    It defines:
    mm, cm, m, km, mg, g, kg, s, min, h, N, kN, Msol
    '''
    def apply_prefixes(units, prefixes):
        if cfg.has_section('prefix'):
            for name, pres in cfg.items('prefix'):
                if name in units:
                    for pre in map(str.strip, pres.split(',')):
                        define(pre+name, prefixes[pre]*Unit(name),
                               allow_redef=allow_redef, warn=warn)
        else:
            for name in units:
                for pre, factor in prefix:
                    define(pre+name, factor*Unit(name), allow_redef=allow_redef,
                           warn=warn)

    for filename in config:
        if path.exists(path.expanduser(filename)):
            break
    else:
        raise IOError('Config file "%s" does not exist!' % config)

    if environment.verbose >= environment.VERBOSE_NORMAL:
        pfilename = path.split(filename)[1]
        print('reading units definitions from "%s"' % pfilename)

    cfg = ConfigParser(allow_no_value=True, inline_comment_prefixes=('#', ';'))
    # new to python3. ignores comments at end of values
    cfg.optionxform = str
    cfg.read(filename)

    if not cfg.has_section('base'):
        raise KeyError('Section "base" is required in unit config file.')

    if undefine_all:
        undefine_all()

    prefixes = dict()
    if cfg.has_section('prefixes'):
        for pre, factor in cfg.items('prefixes'):
            prefixes[pre] = float(factor)

    for name in cfg.options('base'):
        define(name, allow_redef=allow_redef, warn=warn)
    if prefixes:
        apply_prefixes(cfg.options('base'), prefixes)

    if cfg.has_section('derived'):
        # Since some may depend on others (that are maybe even prefixed), do
        # definitions only of those that can be defined and then repeat until no
        # units can be defined anymore.
        to_define = dict(cfg.items('derived'))
        defined = True  # just to enter the loop
        while defined:
            defined = []
            for name, definition in list(to_define.items()):
                try:
                    define(name, Unit(definition), allow_redef=allow_redef,
                           warn=warn)
                    del to_define[name]
                    defined.append(name)
                except UnitError as e:
                    if not str(e).startswith('"Undefined symbol'):
                        raise
                except:
                    raise
            if prefixes:
                apply_prefixes(defined, prefixes)
        if to_define:
            print('Undefinable units:', file=sys.stderr)
            for name, definition in to_define.items():
                print('  %-10s = %s' % (name, definition), file=sys.stderr)
            raise RuntimeError('There are %d not well-defined ' % len(to_define) +
                               'units in the unit config file "%s"!' % config)

    if cfg.has_section('latex'):
        for name, latex in cfg.items('latex'):
            set_latex_repr(name, latex)

_re_mul_space = re.compile(r'(?<=\w|\)|\.)\s+(?=[A-Za-z_]|\()')
_re_ident = re.compile(r'[A-Za-z_][\w]*')
_re_frac_power = list(map(re.compile,
                     [r'\*\*(?P<nom>\d+)/(?P<den>\d+)(?=[^(e|\.)]|$)',
                      r'\*\*\(\s*(?P<nom>\d+)\s*/\s*(?P<den>\d+)\s*\)']))
_unit_evaluator = Evaluator({'Fraction':Fraction}, my_math=math)
def Unit(x, allow_undefined=False):
    '''
    Construct a unit.

    Strings are parsed with Python syntax with the following restrictions:
        * The parsing is done with pygad.utils.eval (restricted and save, esp. no
          underscore identifiers).
        * Divisions are true divisions (operator.truediv), i.e. '3/4' evaluates to
          0.75 rather than 0. If it is a power of a unit, however, it is
          interpreted as a Fraction (even if not placed in backets: 'm**1/2' is
          'm**Fraction(1,2)').
        * Spaces between two identifiers or between a leading number and an
          identifier are replaced with '*' to make expressions like 'kg m / s**2'
          or '1e3 m' work. Note, however, that 'm 3' (should it be 'm*3' or
          'm**3') is not valid.

    Args:
        x (str, Number, Unit):  The definition of the unit to construct. It can
                                simply be a number or already a unit, however,
                                usually one would pass a string (e.g. "km/h") that
                                defines the new unit.
        allow_undefined (bool): If True, allow to use undefined unit names. These
                                are later treated as base units.

    Returns:
        unit (Unit):            The constructed unit.
    '''
    if isinstance(x, Number):
        if x == 0:
            raise ValueError('Zero-unit is not allowed!')
        return _UnitClass(float(x), [])
    elif isinstance(x, _UnitClass):
        return x
    elif not isinstance(x, str):
        raise TypeError(x.__class__.__name__ + ' cannot be converted into ' +
                        'a unit.')
    # x is a string...!
    x = x.strip()

    if x in _unit_definitions:
        return _UnitClass(1., [[x,1]])
    else:
        # place missing multiplications
        x = re.sub(_re_mul_space, '*', x)
        for pattern in _re_frac_power:
            x = re.sub(pattern, '**Fraction(\g<nom>,\g<den>)', x)
        variables = { n:_UnitClass(1.,[[n,1]]) for n in _unit_definitions }
        if allow_undefined:
            if allow_undefined == 'debug':
                print('Parsing the string: "%s"' % x)
                if len(variables) != len(_unit_definitions):
                    undef = set(variables)-set(_unit_definitions)
                    print('WARNING: there are %d undefined units:' % len(undef), file=sys.stderr)
                    print(' ', list(undef), file=sys.stderr)
            variables.update( { n:_UnitClass(1.,[[n,1]])
                    for n in re.findall(_re_ident, x)
                    if not n in _unit_evaluator.namespace } )
        try:
            exp_val = _unit_evaluator.eval(x, variables)
        except EvalError as e:
            raise UnitError('%s' % e.msg)
        except:
            raise

        if isinstance(exp_val, _UnitClass):
            if exp_val._scale == 0 and len(exp_val._composition) == 0:
                raise ValueError('Zero-unit is not allowed!')
            return exp_val
        else:
            return Unit(exp_val)

def Units(l, allow_undefined=False):
    '''
    Conveniently construct multiple units at once.

    In the backend it calls 'Unit' for the individual creations. See its
    documention for more details.

    Args:
        l (list, str):          A list of unit definitions or a comma- or
                                semicolon-separated string of such definitions.
        allow_undefined (bool): If True, allow to use undefined unit names. These
                                are later treated as base units.

    Returns:
        units (list):           A list of the construct units.
    '''
    if isinstance(l, str):
        l = str(l).replace(';',',')
        if ',' in l:
            return Units(list(map(str.strip,l.split(','))), allow_undefined)
        return [Unit(l, allow_undefined)]
    return [Unit(n, allow_undefined) for n in l]

def convertable(u1, u2, subs=None):
    '''
    Tests whether to units are convertable (provided substitutions).

    Args:
        u1, u2 (Unit, str):     The units to test.
        subs (dict, Snap):      Unit to substitute with numbers in the convertion.
                                For more information see '_UnitClass.substitute'.

    Returns:
        convertable (bool):     The result of the test.
    '''
    u1, u2 = Unit(u1), Unit(u2)
    try:
        u1.in_units_of(u2, subs=subs)
    except:
        return False
    else:
        return True

