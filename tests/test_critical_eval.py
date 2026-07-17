"""Regression test for CRITICAL #2 (review/REVIEW.md).

The ``SnapshotCache.family`` setter (pygad/snapshot/snapshotcache.py)
resolved the user-supplied family name via
``eval('gx.' + str(value), globals(), locals())``.  Any string that forms a
valid Python expression continuation (e.g. ``"gas, __import__(...)"``) was
executed with the module's globals -> arbitrary code execution; a simple
typo raised a confusing eval-induced ``AttributeError``.

The fix validates the name against ``gadget.families`` and resolves it with
``getattr``; invalid names raise a ``ValueError`` naming the valid choices.
"""
import os

import pytest

import pygad as pg
from pygad.snapshot.snapshotcache import SnapshotCache

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAP = os.path.join(REPO, "pygad", "snaps", "snap_M1196_4x_470")

snap = pg.Snapshot(SNAP, load_double_prec=False)
cache = SnapshotCache(SNAP)
cache.snapshot = snap
# Mimic what load_snapshot/_center_gx set up: the cached all-families galaxy.
cache._SnapshotCache__galaxy_all = snap
cache._SnapshotCache__galaxy = snap


def test_family_setter_does_not_execute_code(tmp_path):
    # SECURITY EXHIBIT: with the old eval(), the string below evaluates as
    # ``gx.gas, __import__('pathlib').Path(<canary>).touch()`` -- a tuple
    # expression whose second element runs the injected payload.
    canary = tmp_path / ("pygad_pwned_%d" % os.getpid())
    payload = "gas, __import__('pathlib').Path(r'%s').touch()" % canary
    try:
        cache.family = payload
    except ValueError:
        pass  # fixed behavior: rejected by validation before any evaluation
    assert not canary.exists(), \
        "family setter eval()ed the payload -> arbitrary code execution"
    # the galaxy must not have been replaced by some eval() result
    assert cache.galaxy is snap


def test_family_setter_accepts_valid_family():
    cache.family = 'gas'
    assert cache.family == 'gas'
    assert cache.galaxy is snap.gas
    assert len(cache.galaxy) == len(snap.gas)
    # None resets to the whole (all-families) galaxy
    cache.family = None
    assert cache.family is None
    assert cache.galaxy is snap


def test_family_setter_rejects_unknown_family():
    with pytest.raises(ValueError) as excinfo:
        cache.family = 'no_such_family'
    msg = str(excinfo.value)
    assert 'no_such_family' in msg
    # the error message names the valid choices
    for fam in ('gas', 'stars', 'dm'):
        assert fam in msg
    # state untouched by the failed assignment
    assert cache.galaxy is snap
