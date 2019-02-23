'''
Some (convenience) functions for binning.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(core)
    TestResults(failed=0, attempted=32)
    >>> doctest.testmod(cbinning)
    TestResults(failed=0, attempted=34)
    >>> doctest.testmod(mapping)
    TestResults(failed=0, attempted=13)
    >>> doctest.testmod(oneDbinning)
    TestResults(failed=0, attempted=10)
'''

from .core import *
from .cbinning import *
from .mapping import *
from .oneDbinning import *

