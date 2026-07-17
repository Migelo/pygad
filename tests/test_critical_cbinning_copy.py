"""Regression tests for CRITICAL #6 (review/REVIEW.md, slice 4 finding 1):

pygad/binning/cbinning.py passed non-contiguous arrays to the C extension:
the guards

    if qty.base is not None:
        qty.copy()

*discard* the contiguous copy that `ndarray.copy()` returns, so the raw
pointer of the original (potentially non-contiguous) buffer was still handed
to C (`qty.ctypes.data`), which reads it with assumed unit stride -> silently
wrong binned results.  Fixed by keeping the copy: `qty = qty.copy()` (same for
pos/hsml/dV/av at all four call sites).

How the bug is exhibited here
-----------------------------
In the current production pipeline the defect is *masked*: every array passed
to C is already a fresh C-contiguous copy, because (a) `sub._mask` is a
boolean array whose fancy indexing copies and (b) `.astype(np.float64)`
copies unconditionally (numpy `copy=True` default).  Hence the
`if X.base is not None` guards never even fire with stock numpy and stock
pygad masks -- verified empirically: contiguous and strided inputs binned
identically (to ~1e-15 threading noise) already before the fix.

To exercise the guard lines -- the exact code this fix changes -- these tests
feed the wrappers inputs through a *view-preserving* path, simulating array
-likes/future refactors (e.g. `np.asarray`-style copy elision) the guards were
written for:

* `NonCopyingArr`: an `np.ndarray` subclass whose `view()`/`astype()` elide
  copies when the dtype already matches (like `np.asarray`).  The numerically
  identical but non-contiguous variant is a stride-2 view into an interleaved
  buffer whose odd slots hold poison; a wrong-stride C read therefore lands on
  poison and produces an O(1) relative error, deterministically.
* a shimmed `BoxMask.get_mask_for` returning `slice(None)`: pygad natively
  supports slice masks (`s[::2]` etc.), and slice indexing preserves both the
  subclass and the strides, letting the non-contiguous array reach the guard.
  (For `SPH_3D_to_line`, whose mask is built from `periodic_distance_to`, the
  same full-selection slice is injected there instead.)

Both shims are applied identically to the contiguous control and the strided
variant, so the comparison stays apples-to-apples.  The contiguous controls
pass before and after the fix; the non-contiguous variants fail before the
fix (relative error ~1, from stale pointers) and pass after.

The C binning uses multi-threading, so identical inputs reproduce only to
~1e-15 relative; the agreement tolerance is 1e-9, six orders above the noise
and nine orders below the bug signal.
"""

import contextlib

import numpy as np
import pytest

import pygad  # noqa: F401  (ensures units/config are loaded)
from pygad import environment
from pygad.binning import cbinning
from pygad.snapshot import Snapshot, masks
from pygad.transformation import Translation
from pygad.units import UnitArr

RTOL = 1e-9          # agreement tolerance (threading noise is ~1e-15)
POISON = -12345.678  # sentinel in the odd slots of the interleaved buffer

EXTENT3 = UnitArr([[-0.2, 0.2], [-0.2, 0.2], [-0.2, 0.2]], 'Mpc')
EXTENT2 = EXTENT3[:2]
EXTENT_LINE = UnitArr([-0.2, 0.2], 'Mpc')
NPX3 = [16, 16, 16]
NPX2 = [16, 16]
NPX_LINE = 64


class NonCopyingArr(np.ndarray):
    """ndarray with np.asarray-like copy elision in view()/astype().

    Stands in for array-likes (or future pygad/numpy code paths) that do not
    re-copy when the requested dtype/class already matches -- exactly the
    situation the `.base is not None` guards in cbinning were written for.
    """

    def view(self, dtype=None, *args, **kwargs):
        if dtype is None or dtype is np.ndarray:
            return self
        return super().view(dtype, *args, **kwargs)

    def astype(self, dtype, *args, **kwargs):
        if np.dtype(dtype) == self.dtype:
            return self
        return super().astype(dtype, *args, **kwargs)


def contiguous_input(x):
    """Contiguous float64 input (copy-eliding array-like, base set via view)."""
    return np.ascontiguousarray(x, dtype=np.float64).view(NonCopyingArr)


def strided_input(x):
    """Numerically identical but non-contiguous input (stride-2 view).

    The odd slots of the underlying buffer hold POISON, so a C read that
    wrongly assumes unit stride picks up poison instead of the data.
    """
    buf = np.empty(2 * len(x), dtype=np.float64)
    buf[::2] = x
    buf[1::2] = POISON
    v = buf[::2].view(NonCopyingArr)
    assert v.base is not None and not v.flags['C_CONTIGUOUS']
    return v


@contextlib.contextmanager
def boxmask_slice_shim():
    """Let BoxMask select everything via a view-preserving slice mask."""
    real = masks.BoxMask.get_mask_for
    masks.BoxMask.get_mask_for = lambda self, s: slice(None)
    try:
        yield
    finally:
        masks.BoxMask.get_mask_for = real


@contextlib.contextmanager
def line_all_shim():
    """Same full-selection slice for SPH_3D_to_line's distance mask."""
    class _All:
        def __lt__(self, other):
            return slice(None)

    real = cbinning.periodic_distance_to
    cbinning.periodic_distance_to = lambda *a, **k: _All()
    try:
        yield
    finally:
        cbinning.periodic_distance_to = real


def rel_diff(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.max(np.abs(a - b)) / np.abs(a).max()


@pytest.fixture(scope='module')
def gas():
    environment.verbose = environment.VERBOSE_QUIET
    s = Snapshot(environment.module_dir + 'snaps/snap_M1196_4x_320',
                 physical=True)
    Translation(-UnitArr([34.7828, 35.5898, 33.6147], 'cMpc/h_0')).apply(s)
    return s.gas


@pytest.fixture(scope='module')
def qty(gas):
    return np.ascontiguousarray(
        gas.get('rho').in_units_of('Msol/kpc**3').view(np.ndarray),
        dtype=np.float64)


@pytest.fixture(scope='module')
def av(gas):
    return np.ascontiguousarray(
        gas['mass'].in_units_of('Msol').view(np.ndarray), dtype=np.float64)


def call_3dgrid(gas, qty):
    with boxmask_slice_shim():
        return cbinning.SPH_to_3Dgrid(gas, qty, extent=EXTENT3, Npx=NPX3,
                                      hsml='hsml', dV=None)


def call_2dgrid(gas, qty):
    with boxmask_slice_shim():
        return cbinning.SPH_to_2Dgrid(gas, qty, extent=EXTENT2, Npx=NPX2,
                                      hsml='hsml', dV=None)


def call_line(gas, qty):
    with line_all_shim():
        return cbinning.SPH_3D_to_line(gas, qty, los=[0, 0],
                                       extent=EXTENT_LINE, Npx=NPX_LINE,
                                       hsml='hsml', dV=None)


def call_by_particle(gas, qty, av):
    with boxmask_slice_shim():
        return cbinning.SPH_to_2Dgrid_by_particle(
            gas, qty, extent=EXTENT2, Npx=NPX2, reduction='mean', av=av,
            hsml='hsml', dV=None)


# --- SPH_to_3Dgrid (cbinning.py:180-187) ------------------------------------

def test_sph_to_3dgrid_contiguous_consistent(gas, qty):
    truth = call_3dgrid(gas, contiguous_input(qty))
    control = call_3dgrid(gas, contiguous_input(qty))
    assert rel_diff(control, truth) < RTOL


def test_sph_to_3dgrid_noncontiguous_matches_contiguous(gas, qty):
    truth = call_3dgrid(gas, contiguous_input(qty))
    result = call_3dgrid(gas, strided_input(qty))
    assert rel_diff(result, truth) < RTOL


# --- SPH_to_2Dgrid (cbinning.py:317-322) -------------------------------------

def test_sph_to_2dgrid_contiguous_consistent(gas, qty):
    truth = call_2dgrid(gas, contiguous_input(qty))
    control = call_2dgrid(gas, contiguous_input(qty))
    assert rel_diff(control, truth) < RTOL


def test_sph_to_2dgrid_noncontiguous_matches_contiguous(gas, qty):
    truth = call_2dgrid(gas, contiguous_input(qty))
    result = call_2dgrid(gas, strided_input(qty))
    assert rel_diff(result, truth) < RTOL


# --- SPH_3D_to_line (cbinning.py:444-449) ------------------------------------

def test_sph_3d_to_line_contiguous_consistent(gas, qty):
    truth = call_line(gas, contiguous_input(qty))
    control = call_line(gas, contiguous_input(qty))
    assert rel_diff(control, truth) < RTOL


def test_sph_3d_to_line_noncontiguous_matches_contiguous(gas, qty):
    truth = call_line(gas, contiguous_input(qty))
    result = call_line(gas, strided_input(qty))
    assert rel_diff(result, truth) < RTOL


# --- SPH_to_2Dgrid_by_particle (cbinning.py:577-585) -------------------------

def test_sph_to_2dgrid_by_particle_contiguous_consistent(gas, qty, av):
    truth = call_by_particle(gas, contiguous_input(qty), contiguous_input(av))
    control = call_by_particle(gas, contiguous_input(qty), contiguous_input(av))
    assert rel_diff(control, truth) < RTOL


def test_sph_to_2dgrid_by_particle_noncontiguous_matches_contiguous(gas, qty,
                                                                    av):
    truth = call_by_particle(gas, contiguous_input(qty), contiguous_input(av))
    result = call_by_particle(gas, strided_input(qty), strided_input(av))
    assert rel_diff(result, truth) < RTOL
