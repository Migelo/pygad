'''
Module for physics that are not directly connected to snapshots.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(quantities)
    TestResults(failed=0, attempted=33)
    >>> doctest.testmod(cosmology)
    TestResults(failed=0, attempted=42)
    >>> doctest.testmod(cooling)
    TestResults(failed=0, attempted=14)
'''

from .quantities import *
from .cosmology import *
from .cooling import *

