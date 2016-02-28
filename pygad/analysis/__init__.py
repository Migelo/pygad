'''
Module for snapshot analysis.

Also doctest other parts of this sub-module:
    >>> import doctest
    >>> doctest.testmod(sph_eval)
    TestResults(failed=0, attempted=15)
    >>> doctest.testmod(properties)
    TestResults(failed=0, attempted=32)
    >>> doctest.testmod(halo)
    TestResults(failed=0, attempted=26)
    >>> doctest.testmod(profiles)
    TestResults(failed=0, attempted=19)

    #>>> doctest.testmod(analysis)
    #TestResults(failed=0, attempted=20)
'''
from sph_eval import *
from properties import *
from halo import *
from profiles import *
#from analysis import *

