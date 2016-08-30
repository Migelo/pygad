'''
Module for units and arrays with units.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(units)
    TestResults(failed=0, attempted=38)
    >>> doctest.testmod(unit_arr)
    TestResults(failed=0, attempted=85)
'''
from units import *
from unit_arr import *
import os

from ..environment import module_dir
define_from_cfg(['./units.cfg',
                 os.getenv("HOME")+'/.config/pygad/units.cfg',
                 module_dir+'units/units.cfg'])

