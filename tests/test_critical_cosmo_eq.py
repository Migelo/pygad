"""Regression tests for CRITICAL #10: FLRWCosmo.__eq__ compared 4 of its 6
fields (Omega_m, Omega_b, sigma_8, n_s) against `self` instead of `other`,
so any two cosmologies equal in h_0 and Omega_Lambda compared equal.

Run with: .venv/bin/python -m pytest tests/test_critical_cosmo_eq.py
"""
from pygad.physics import FLRWCosmo

_BASE = dict(h_0=0.7, Omega_Lambda=0.7, Omega_m=0.3, Omega_b=0.045,
             sigma_8=0.8, n_s=0.96)


def _cosmo(**overrides):
    params = dict(_BASE)
    params.update(overrides)
    return FLRWCosmo(**params)


def test_identical_cosmologies_compare_equal():
    # Regression guard: passed pre- and post-fix.
    a = _cosmo()
    b = _cosmo()
    assert a == b
    assert not (a != b)


def test_different_Omega_m_compares_unequal():
    a = _cosmo()
    b = _cosmo(Omega_m=0.25)  # Omega_b also overridden: default is 0.16*Omega_m
    assert a != b
    assert not (a == b)


def test_different_Omega_b_compares_unequal():
    a = _cosmo()
    b = _cosmo(Omega_b=0.05)
    assert a != b
    assert not (a == b)


def test_different_sigma_8_compares_unequal():
    a = _cosmo()
    b = _cosmo(sigma_8=0.9)
    assert a != b
    assert not (a == b)


def test_different_n_s_compares_unequal():
    a = _cosmo()
    b = _cosmo(n_s=1.0)
    assert a != b
    assert not (a == b)


def test_different_h_0_and_Omega_Lambda_compare_unequal():
    # These two fields were compared correctly even before the fix.
    a = _cosmo()
    assert a != _cosmo(h_0=0.72)
    assert a != _cosmo(Omega_Lambda=0.72)


def test_ne_is_negation_of_eq():
    a = _cosmo()
    b = _cosmo(Omega_m=0.25)
    c = _cosmo()
    assert (a != b) == (not (a == b))
    assert (a != c) == (not (a == c))
