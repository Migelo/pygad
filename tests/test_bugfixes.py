"""Regression checks for the three pygad bug fixes.

Run with the project venv:

    .venv/bin/python tests/test_bugfixes.py
"""
import inspect
import os

import pygad
from pygad.snapshot.snapshot import Snapshot
from pygad.units.units import (
    define, define_from_cfg, defined_units, Unit, UnitError,
)
from pygad.units.unit_arr import UnitArr, _div_units


def check_fix1():
    # FIX 1: _host_derive_block gained a `cache` parameter so the non-cached
    # derive path (called with a second positional arg) no longer crashes.
    params = inspect.signature(Snapshot._host_derive_block).parameters
    assert 'cache' in params, "missing 'cache' parameter"
    assert params['cache'].default is True, "cache default should be True"


def check_fix2():
    # FIX 2: define_from_cfg's `undefine_old` flag is now honoured; passing
    # undefine_old=False preserves previously defined units.
    # NOTE: allow_redef=True is required here because `import pygad` already
    # auto-loads units.cfg (152 units incl. 'm'); without it the re-definition
    # of 'm' would raise UnitError independently of this fix. allow_redef lets
    # the call run so we can assert zzzcustom survives (it would be wiped if
    # the bug were still present).
    define('zzzcustom', 'm')
    assert 'zzzcustom' in defined_units()
    cfg = os.path.join(os.path.dirname(pygad.__file__), 'config', 'units.cfg')
    define_from_cfg([cfg], undefine_old=False, allow_redef=True)
    assert 'zzzcustom' in defined_units()   # preserved now


def check_fix3a():
    # FIX 3a: _div_units no longer evaluates `1 / None` when both operands are
    # unitless; it returns None instead.
    assert _div_units(UnitArr([1., 2.]), UnitArr([2., 4.])) is None


def check_fix3b():
    # FIX 3b: dividing two unitful UnitArrs yields a UnitArr with combined units.
    r = UnitArr([1., 2.], 'm') / UnitArr([2., 4.], 's')
    assert type(r).__name__ == 'UnitArr'
    assert str(r.units) == '[m s**-1]'


def check_fix3c():
    # FIX 3c: dividing two unitless UnitArrs keeps the UnitArr subclass with
    # units=None.
    r2 = UnitArr([1., 2.]) / UnitArr([2., 4.])
    assert type(r2).__name__ == 'UnitArr'
    assert r2.units is None


def check_sanity():
    # Sanity: a unitful multiplication still yields a UnitArr with units.
    r = UnitArr([2., 4.], 'm') * UnitArr([2., 4.], 's')
    assert type(r).__name__ == 'UnitArr'
    assert str(r.units) == '[m s]'


if __name__ == '__main__':
    check_fix1()
    check_fix2()
    check_fix3a()
    check_fix3b()
    check_fix3c()
    check_sanity()
    print("all regression checks passed")
