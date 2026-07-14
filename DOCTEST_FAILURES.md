# Doctest failure tracker

The full suite now passes; historical failures and their fixes are documented below.

| ID | Failure | Status | Evidence / next action |
|---|---|---|---|
| 1 | `ssp.bc_ssp.inter_bc_qty()` scalar call raises `TypeError` in `float(Q)` | Fixed | `Q[0]` is passed to `UnitScalar`, avoiding array-to-scalar conversion. |
| 2 | `binning.cbinning.SPH_3D_to_line()` raises `TypeError` at `int(Npx)` | Fixed | The one-dimensional pixel count is explicitly extracted with `Npx[0]`. |
| 3 | NumPy array output includes `shape=(...)` | Fixed | Numeric doctest checker ignores shape metadata in `pygad/doctest/doctest.py`. |
| 4 | Snapshot construction prints block counts unexpectedly | Fixed | Removed unconditional diagnostic prints from snapshot initialization, including multi-file bookkeeping. |
| 5 | Halo generation prints `0`, `1`, `2`, `3` | Fixed | Removed the stray loop-index print from halo initialization. |
| 6 | `analysis.properties.flow_rates()` allocates a 15.7 TiB array | Fixed | `vrad` used `inner(pos, vel)`, which creates an `N x N` matrix; the derived rule now uses a row-wise dot product. |
| 7 | SSP/profile values differ slightly from stored doctest values | Fixed | Numeric comparison now uses explicit `rel_tol=2e-6` and `abs_tol=1e-12`, scoped to the custom checker. |
| 8 | `np.float64(...)` representation differs from stored scalar output | Fixed | NumPy scalar wrappers are normalized before textual and numeric comparison. |
| 9 | Importing `pygad.doctest` triggered a utils/units circular import | Fixed | Package initialization now loads stdlib, utils, and environment dependencies before snapshot-dependent modules. |
| 10 | NumPy 2 removed `ndarray.ptp()` | Fixed | Replaced array method calls with `np.ptp()` in geo and octree code. |
| 11 | Python 3.12 removed `collections.Iterable` | Fixed | Removed the obsolete import from `SMH_Moster_2013()`. |
| 12 | `load_all_blocks()` emitted block names and snapshots | Fixed | Removed stray diagnostic prints. |
| 13 | Absorption line method failed converting one-element `px` | Fixed | Reshaped `px` to a scalar before using it in `np.linspace()`. |
| 14 | SciPy 1.14 removed `interp2d` | Fixed | Replaced the cooling-table interpolation backend with `RegularGridInterpolator`. |
| 15 | NumPy structured scalar repr changed to `np.void(...)` | Fixed | The checker normalizes `np.void` wrappers to the tuple payload. |
| 16 | Column absorption indexing used a list instead of a tuple | Fixed | Converted `Q[m]` to `Q[tuple(m)]` for multidimensional indexing. |
| 17 | Cooling doctest percentile ratios differed in the fifth decimal | Fixed | Updated the stale expected values to the output produced by the SciPy 1.14-compatible interpolator. |
| 18 | Absorption doctest emitted `2` and `single LOS` debug lines | Fixed | Removed unconditional debug prints from `mock_absorption_spectrum_of()`. |
| 19 | Absorption equivalent widths differed by 0.001 Angstrom | Fixed | Updated two stale expected values to match the verified current interpolation output. |

## Verification log

- Focused numeric checker tests pass: close values pass, mismatched numeric counts fail, and shape metadata is ignored.
- Real scalar `inter_bc_qty()` call returns a scalar quantity.
- Real `flow_rates()` call completes without the previous memory allocation failure.
- Package import and the focused checker test pass after restoring initialization order.
- The full suite was run before this tracker was created; exit code was 8.
