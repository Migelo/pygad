'''
Module for the pygad snapshots.

These are structures for Gadget snapshots. The reading, though, is done by the
gadget module.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(snapshot)
    TestResults(failed=0, attempted=120)
    >>> doctest.testmod(derived)
    TestResults(failed=0, attempted=12)
    >>> doctest.testmod(sim_arr)
    TestResults(failed=0, attempted=18)
    >>> doctest.testmod(masks)
    TestResults(failed=0, attempted=33)
'''
from snapshot import *
from derived import *
from sim_arr import *
from masks import *
import os

from ..environment import module_dir
read_derived_rules(['./derived.cfg',
                   os.getenv("HOME")+'/.config/pygad/derived.cfg',
                   module_dir+'snapshot/derived.cfg'])

