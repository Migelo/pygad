"""Regression test for CRITICAL #12 (review/REVIEW.md).

pygad/plotting/profiles.py history(): in the weighted branch of the
binning loop the code appended to ``av_hist`` (line ~206)::

    Q_hist = []
    for t0, t1 in zip(t_edges[:-1], t_edges[1:]):
        mask = (t0<=time) & (time<t1)
        if av is None:
            Q_hist.append( Q[mask].sum() )
        else:
            av_hist.append( (Q[mask]*AV[mask]).sum() / AV[mask].sum() )
    Q_hist = UnitArr(Q_hist, Q.units)

``av_hist`` is never defined, so any call to ``history`` with a
weighted-history quantity (``av is not None``) raised
``NameError: name 'av_hist' is not defined``.  The weighted average
``sum(Q*AV)/sum(AV)`` has the units of ``Q``, so the existing
``UnitArr(Q_hist, Q.units)`` conversion is correct; the fix simply
appends to ``Q_hist``.

The test plots the mass-weighted metallicity history of the stars of a
test snapshot (the ``av`` path).  Pre-fix it fails with the NameError;
post-fix it asserts the call returns a figure/axis whose plotted values
match a manual mass-weighted mean per time bin.

Run with the project venv:

    .venv/bin/python -m pytest tests/test_critical_avhist.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pygad
from pygad.plotting.profiles import history

SNAP = "pygad/snaps/snap_M1196_4x_470"
N = 10


def test_weighted_history():
    s = pygad.Snapshot(SNAP)
    stars = s.stars
    assert len(stars) > 0

    fig, ax = history(stars, "metallicity", av="mass", N=N, add_z_ax=False)
    try:
        # the quantity histogram was plotted as a single line
        assert len(ax.lines) == 1
        y = np.asarray(ax.lines[0].get_ydata(), dtype=float)
        x = np.asarray(ax.lines[0].get_xdata(), dtype=float)
        assert len(y) == N

        # manual mass-weighted mean of metallicity per time bin
        time = stars.cosmic_time() - stars["age"]
        Z = stars["metallicity"]
        m = stars["mass"]
        now = stars.cosmic_time()
        t_edges = np.linspace(0, float(now), N + 1)
        expected = []
        for t0, t1 in zip(t_edges[:-1], t_edges[1:]):
            mask = (t0 <= time) & (time < t1)
            expected.append((Z[mask] * m[mask]).sum() / m[mask].sum())
        expected = np.asarray(expected, dtype=float)

        # all bins populated -> finite, and equal to the manual weighted mean
        assert np.all(np.isfinite(y))
        np.testing.assert_allclose(y, expected, rtol=1e-10)
        # bin centers as x values
        np.testing.assert_allclose(x, (t_edges[:-1] + t_edges[1:]) / 2.0)
    finally:
        plt.close(fig)
