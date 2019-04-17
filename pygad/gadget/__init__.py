'''
Module for handling Gadget files.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(config)
    TestResults(failed=0, attempted=10)
    >>> doctest.testmod(lowlevel_file)
    TestResults(failed=0, attempted=12)

    handler module is implicitly tested in the snapshot module...
'''
from .config import *
from . import lowlevel_file
from .handler import *
import os

from ..environment import module_dir
read_config(['./gadget.cfg',
             os.getenv("HOME")+'/.config/pygad/gadget.cfg',
             module_dir+'config/gadget.cfg'])

