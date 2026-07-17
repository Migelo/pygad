"""Regression test for CRITICAL #13 (review/REVIEW.md).

pygad/plotting/profiles.py history() (lines ~168-176)::

    if av is not None:
        if isinstance(av, str):
            AV = s.get(av)
        if len(s) != len(AV):
            ...

``AV`` is only assigned inside ``if isinstance(av, str):`` but referenced
unconditionally on the next line.  Passing ``av`` as a non-string
array-like (e.g. a ``UnitArr`` of per-particle weights) therefore raised
``NameError: name 'AV' is not defined``.  The fix adds an
``else: AV = av`` branch so ready blocks are used directly -- exactly as
already done for ``qty``/``Q`` a few lines above.

The tests plot the history of the stars of a test snapshot with ``av``
given as an explicit array:

* pre-fix the call fails with the NameError (exhibiting the bug);
* post-fix the plotted values match a manual per-bin weighted mean when
  ``av`` is the mass block, and the plain per-bin mean when ``av`` is an
  array of ones;
* the string-``av`` path (CRITICAL #12) keeps working.

Run with the project venv:

    .venv/bin/python -m pytest tests/test_critical_av_nonstr.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pygad
from pygad import UnitArr
from pygad.plotting.profiles import history

SNAP = "pygad/snaps/snap_M1196_4x_470"
N = 10


def _manual_binned(stars, Q, W, n=N):
    """Manual per-bin weighted mean of Q with weights W over time bins."""
    time = stars.cosmic_time() - stars["age"]
    now = stars.cosmic_time()
    t_edges = np.linspace(0, float(now), n + 1)
    expected = []
    for t0, t1 in zip(t_edges[:-1], t_edges[1:]):
        mask = (t0 <= time) & (time < t1)
        expected.append((Q[mask] * W[mask]).sum() / W[mask].sum())
    return np.asarray(expected, dtype=float)


def test_history_av_array_weights():
    """Non-string `av` (UnitArr of masses) must work and be correct."""
    s = pygad.Snapshot(SNAP)
    stars = s.stars
    assert len(stars) > 0

    masses = stars["mass"]  # a UnitArr, not a string
    fig, ax = history(stars, "metallicity", av=masses, N=N, add_z_ax=False)
    try:
        assert len(ax.lines) == 1
        y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert len(y) == N
        assert np.all(np.isfinite(y))
        expected = _manual_binned(stars, stars["metallicity"], masses)
        np.testing.assert_allclose(y, expected, rtol=1e-10)
    finally:
        plt.close(fig)


def test_history_av_array_ones_equals_plain_mean():
    """Non-string `av` of ones must reproduce the plain per-bin mean."""
    s = pygad.Snapshot(SNAP)
    stars = s.stars

    ones = UnitArr(np.ones(len(stars)))
    fig, ax = history(stars, "metallicity", av=ones, N=N, add_z_ax=False)
    try:
        y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        # with unit weights the weighted mean equals the plain mean
        expected = _manual_binned(stars, stars["metallicity"], ones)
        np.testing.assert_allclose(y, expected, rtol=1e-10)

        # cross-check against an explicit plain mean per bin (the block is
        # stored in float32, so allow for float32 rounding here)
        time = stars.cosmic_time() - stars["age"]
        Z = stars["metallicity"]
        now = stars.cosmic_time()
        t_edges = np.linspace(0, float(now), N + 1)
        plain = []
        for t0, t1 in zip(t_edges[:-1], t_edges[1:]):
            mask = (t0 <= time) & (time < t1)
            plain.append(Z[mask].mean())
        np.testing.assert_allclose(y, np.asarray(plain, dtype=float),
                                   rtol=1e-6)
    finally:
        plt.close(fig)


def test_history_av_string_still_works():
    """Regression guard: the string-`av` path (CRITICAL #12) still works."""
    s = pygad.Snapshot(SNAP)
    stars = s.stars

    fig, ax = history(stars, "metallicity", av="mass", N=N, add_z_ax=False)
    try:
        y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        assert len(y) == N
        expected = _manual_binned(stars, stars["metallicity"], stars["mass"])
        np.testing.assert_allclose(y, expected, rtol=1e-10)
    finally:
        plt.close(fig)
