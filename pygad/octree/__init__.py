'''
A module for octrees on 3-dim. points in form of np.ndarrays.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(coctree)
    TestResults(failed=0, attempted=48)
    >>> doctest.testmod(octree)
    TestResults(failed=0, attempted=29)
'''

from .coctree import *
from .octree import *

