# Review: Binning, Octree, Kernels (Slice 4)

## Slice summary

**Severity counts:** 2 CRITICAL, 3 HIGH, 5 MEDIUM, 3 LOW

This slice covers the SPH binning pipeline (Python + C backend), the pure-Python and C-accelerated octree, SPH kernel definitions, and kernel integral utilities. The most serious findings are a **contiguity bug** in `cbinning.py` where non-contiguous arrays are passed to C without actually making them contiguous (`.copy()` return value is discarded), and an **undefined variable** crash in `cOctree.__init__` (`s` instead of `pos`). Additionally, `Map.vol_tot()` uses the removed `.ptp()` ndarray method (NumPy 2.x compat), `smooth()` hardcodes a 2D convolution kernel that crashes on 1D grids, and several ctypes `restype`/`argtypes` declarations are incorrect.

---

## Finding 1

**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/binning/cbinning.py:180-187`
**Description:** Non-contiguous arrays are passed to C without being made contiguous. In `SPH_to_3Dgrid`, lines 180-187 check `if pos.base is not None: pos.copy()` — but `.copy()` returns a **new** contiguous array without mutating `pos` in place. The original non-contiguous `pos` (with `.base is not None`) is still used at line 197 (`pos.ctypes.data`). The same bug exists for `hsml`, `dV`, and `qty`. This applies to all four C-binning functions:
- `SPH_to_3Dgrid`: lines 180-187
- `SPH_to_2Dgrid`: lines 317-322
- `SPH_3D_to_line`: lines 444-449
- `SPH_to_2Dgrid_by_particle`: lines 577-585

Note that `pos` in `SPH_to_2Dgrid` (line 299-300) and `SPH_to_2Dgrid_by_particle` (line 557-558) already call `.copy()` correctly (chained assignment). But `hsml`, `dV`, and `qty` in those functions still have the bug.

**Suggested fix:** Replace `pos.copy()` with `pos = pos.copy()` (and same for hsml, dV, qty) in all four functions. Example:
```python
if pos.base is not None:
    pos = pos.copy()
```

---

## Finding 2

**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/octree/coctree.py:188`
**Description:** `cOctree.__init__` references the undefined variable `s` instead of `pos` in the verbose print statement: `utils.nice_big_num_str(len(s))`. The parameter is named `pos`, not `s`. This will raise a `NameError` whenever the tree is built with `environment.verbose >= environment.VERBOSE_TALKY`. Confirmed by ruff as F821.

**Suggested fix:** Change `len(s)` to `len(pos)` on line 188.

---

## Finding 3

**Severity:** HIGH
**Category:** compat
**Location:** `pygad/binning/core.py:371`
**Description:** `Map.vol_tot()` uses `self.extent.ptp(axis=1)` — the `.ptp()` method on ndarray was **removed in NumPy 2.0**. The function `np.ptp(arr, axis=1)` still works, but `arr.ptp(axis=1)` raises `AttributeError`. The doctest for `vol_tot()` (line 84) would fail on NumPy >= 2.0 if executed.

Note: The `DOCTEST_FAILURES.md` FIX 10 addresses `np.ptp()` vs `.ptp()` but this specific instance at `core.py:371` uses the **method form** `.ptp()` and does not appear to be covered by that fix.

**Suggested fix:** Replace `self.extent.ptp(axis=1)` with `np.ptp(self.extent, axis=1)`.

---

## Finding 4

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/binning/core.py:444`
**Description:** `smooth()` always reshapes the convolution kernel to 2D: `conv_grid.reshape((pxs, pxs))` (line 444). However, the kernel is built from `D = len(grid.shape)`, which can be 1 for 1D grids (line projections). When `D=1`, `dists` has shape `(pxs,)`, and reshaping to `(pxs, pxs)` fails with a `ValueError` because `pxs != pxs*pxs`. The function will crash on any 1D grid input.

**Suggested fix:** Generalize the reshape to handle arbitrary dimensions, e.g.:
```python
conv_grid = conv_grid.reshape((pxs,) * D)
```

---

## Finding 5

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/octree/coctree.py:135`
**Description:** `cpygad.get_octree_in_region.restypes = c_int` — `restypes` (with an 's') is not a valid ctypes attribute. The correct attribute is `restype`. This line silently does nothing (Python sets an arbitrary attribute on the module), leaving the return type of `get_octree_in_region` at the default `c_int`. By coincidence, the C function returns `int`, so the default is correct. But if the C signature ever changed (e.g., to return `unsigned`), this would silently break.

**Suggested fix:** Change `restypes` to `restype` on line 135.

---

## Finding 6

**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/octree/coctree.py:131`
**Description:** `cpygad.get_octree_max_depth.restype = c_int` — the C function `get_octree_max_depth` returns `size_t` (8 bytes on 64-bit), but Python declares `restype = c_int` (4 bytes). This truncates the upper 32 bits of the return value. For tree depths (max 25), this is currently safe, but it is inconsistent with `get_octree_node_count` which correctly uses `c_size_t`. Similarly, `get_octree_next_ngb` (line 153-154) has no `restype` set at all, defaulting to `c_int`, while the C function returns `size_t`. The sentinel `-1` check on line 474 works by accident (lower 32 bits of `SIZE_MAX` equal `c_int(-1)`), but valid particle indices > 2^31 would be silently truncated.

**Suggested fix:**
```python
cpygad.get_octree_max_depth.restype = c_size_t
cpygad.get_octree_next_ngb.restype = c_size_t
```

---

## Finding 7

**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/octree/coctree.py:127-130`
**Description:** Duplicate declaration: `get_octree_max_H.restype` and `get_octree_max_H.argtypes` are set twice (lines 127-128 and 129-130). This is harmless (idempotent) but indicates a copy-paste error or merge artifact.

**Suggested fix:** Remove the duplicate lines 129-130.

---

## Finding 8

**Severity:** MEDIUM
**Category:** compat
**Location:** `pygad/binning/core.py:92`
**Description:** `from scipy.ndimage.filters import convolve` — the `scipy.ndimage.filters` namespace is deprecated since SciPy 1.8 and will be removed in SciPy 2.0.0. Currently produces a `DeprecationWarning`. The `smooth()` docstring (line 423) also references the deprecated path.

**Suggested fix:** Change import to `from scipy.ndimage import convolve`. Update docstring references.

---

## Finding 9

**Severity:** MEDIUM
**Category:** qol
**Location:** `pygad/kernels/integral.py:60,92`
**Description:** Both `project_kernel` and `integrate_kernel` use `scipy.interpolate.interp1d`, which is marked as **legacy** in SciPy 1.14+ with a deprecation path toward removal. Additionally, `integrate_kernel` (line 121) calls `np.linspace(0,1)` which defaults to `num=50` points — the parameter `N=100` is declared but never passed to `np.linspace`. This means the kernel is integrated at only 50 sample points regardless of `N`, producing a coarser interpolation table than intended.

**Suggested fix:** Pass `N` to linspace: `rs = np.linspace(0, 1, N)`. Consider migrating from `interp1d` to a modern alternative (e.g., `CubicSpline`) before SciPy removes it.

---

## Finding 10

**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/binning/oneDbinning.py:100-108`
**Description:** `profile_from_map` uses `np.digitize` to assign pixels to radial bins. Pixels outside the bin range get index 0 or `Nbins+1`. The loop `range(1, Nbins+1)` (line 108) only processes bins 1 through Nbins, correctly excluding out-of-range pixels. However, if a radial bin is **empty** (no pixels assigned), `m[idx==i]` returns an empty array, and `np.std([])` returns `nan` (with a RuntimeWarning), `np.mean([])` also returns `nan`. These NaN values propagate into the profile without any warning to the user.

**Suggested fix:** Handle empty bins explicitly, either by substituting `np.nan`, `0.0`, or interpolating from neighbors, and suppress or warn about RuntimeWarnings.

---

## Finding 11

**Severity:** LOW
**Category:** qol
**Location:** `pygad/C/__init__.py:13,17`
**Description:** The `print(f"cpygad found at {cpygad_path}")` statements execute unconditionally (outside the `try/except`) and always print on every import. In the `try` branch (line 13), the print is inside the `try` block but before any `except`, so it only prints on success — but the `except` fallback (line 15-16) also prints. This means every `import pygad` prints the library path to stdout, which is noisy for library users.

**Suggested fix:** Remove or gate behind verbose/debug flag:
```python
if environment.verbose >= environment.VERBOSE_TALKY:
    print(f"cpygad found at {cpygad_path}")
```

---

## Finding 12

**Severity:** LOW
**Category:** correctness
**Location:** `pygad/octree/coctree.py:384`
**Description:** `find_ngbs_within` and `find_ngbs_SPH` (line 430) use `np.resize(ngbs, N_ngbs.value)` to truncate the neighbor array. `np.resize` repeats the array if the target size exceeds the source size, but here `N_ngbs <= max_ngbs` is guaranteed by the C code, so this works correctly. However, using `ngbs[:N_ngbs.value]` (a simple slice) would be clearer and avoid any confusion about `np.resize`'s repeat semantics.

**Suggested fix:** Replace `np.resize(ngbs, N_ngbs.value)` with `ngbs[:N_ngbs.value]` in both `find_ngbs_within` (line 384) and `find_ngbs_SPH` (line 430).

---

## Finding 13

**Severity:** LOW
**Category:** qol
**Location:** `pygad/octree/octree.py:267-268`
**Description:** Bare `except:` clause in `Octree()` catches all exceptions (including `KeyboardInterrupt`, `SystemExit`) silently. The fallback on line 268 is nearly identical to the try-block on line 266 (both call `np.ptp`), just with an extra `np.array()` wrapper. The bare except is likely intended to handle cases where `pos_data` is a `UnitArr` (which may not support `np.ptp` directly), but it swallows real errors.

**Suggested fix:** Use `except (TypeError, AttributeError):` to catch only the expected failure modes.
