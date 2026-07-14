'''
Some general (low-level) functions.

Also doctest other parts of this sub-module:
    >>> import pygad.doctest as doctest
    >>> doctest.testmod(utils)
    TestResults(failed=0, attempted=62)
    >>> doctest.testmod(safe_eval)
    TestResults(failed=0, attempted=11)
    >>> doctest.testmod(term)
    TestResults(failed=0, attempted=0)
    >>> doctest.testmod(geo)
    TestResults(failed=0, attempted=20)
'''
# isort: skip_file
from .safe_eval import *
from .term import *
from .utils import *
from .geo import *
