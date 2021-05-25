"""
Module for units and arrays with units.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(units)
    TestResults(failed=0, attempted=38)
    >>> doctest.testmod(unit_arr)
    TestResults(failed=0, attempted=85)
"""
import os

from ..environment import module_dir
from .unit_arr import *
from .units import *

define_from_cfg(
    [
        "./units.cfg",
        os.getenv("HOME") + "/.config/pygad/units.cfg",
        module_dir + "config/units.cfg",
    ]
)
