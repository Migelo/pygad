import pytest
import pygad

def test_import():
    """Importing the package should succeed without triggering heavy bootstrapping.
    The custom doctest harness is exercised via pytest's doctest plugin.
    """
    assert pygad.__version__ is not None
