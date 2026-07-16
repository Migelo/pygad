# Review: analysis/absorption_spectra.py, analysis/fit_profiles.py, analysis/profiles.py, analysis/__init__.py

## Slice summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1     |
| HIGH     | 3     |
| MEDIUM   | 5     |
| LOW      | 4     |

Overview: The two spectrum-generation functions in `absorption_spectra.py` contain a no-op assignment (`==` instead of `=`) that leaves `spatial_res` stale after recomputing `N`, and two dead `temp.copy()` calls whose return values are silently discarded. The Voigt profile and `line_profile` wrapper have no guard against `b=0` / `sigma=0`, yielding division-by-zero. `fit_profiles.py` has a covariance/parameter mismatch after the jiggle+BFGS step, an off-by-one in `EquivalentWidth` that silently drops the last pixel, and a reduced-chi-sq denominator that uses `np.count_nonzero` instead of the pixel count. Several quality-of-life issues (dead code, code duplication, bare `except` clauses, a latent `UnboundLocalError`) are also noted.

---

## Finding 1

**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/analysis/fit_profiles.py:1058–1080`
**Description:**
After the jiggle + BFGS re-optimisation step (lines 1058–1067), the code computes `chisq_new` from the *jiggled* `params` (line 1069) rather than from the re-optimised `soln.x`. Because `params += ...` at line 1059 mutates `params` in-place, there is no copy of the pre-jitter state to fall back to.

Two distinct bugs result:

1. **Wrong comparison**: `chisq_new = _chisq(params, ...)` evaluates the jiggled (not re-optimised) params. If BFGS actually improved the fit but the jitter worsened it relative to `chisq_best`, the improvement is incorrectly rejected. The correct evaluation should be `chisq_new = _chisq(soln.x, ...)`.

2. **Stale `params` on rejection**: When `chisq_new >= chisq_best` (the `else` branch, implicit), `params` remains the *jiggled* version (mutated in-place at line 1059), not the pre-jitter optimal. The subsequent small-line removal loop (lines 1082–1103) and the final error extraction (lines 1116–1140) operate on this wrong parameter set. Meanwhile `cov = soln.hess_inv` (line 1068) corresponds to `soln.x`, creating a parameter–covariance mismatch for the reported errors.

**Suggested fix:**
```python
# Before jittering, save the best params
params_best = params.copy()

# Jiggle and refit
params_jiggled = params_best + 0.02 * (2 * np.random.rand(len(params_best)) - 1)
soln = minimize(chisq_fcn, params_jiggled, args=(l_reg, f_reg, n_reg, mode),
                method="BFGS", options={"maxiter": 100})
cov = soln.hess_inv
chisq_new = _chisq(soln.x, l_reg, f_reg, n_reg, mode)  # evaluate RE-OPTIMISED params

if chisq_new < chisq_best:
    params = soln.x
    n_lines = int(len(params) / 3)
    best_nlines = n_lines
    best_params = params
    best_bounds = bounds
    chisq_best = chisq_new
else:
    params = params_best  # restore pre-jitter params
```

---

## Finding 2

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/analysis/fit_profiles.py:466, 472`
**Description:**
Both `_chisq` and `_chisq_asym` compute reduced chi-squared as:
```python
return np.sum(dx_array * dx_array) / np.count_nonzero(dx_array)
```
`np.count_nonzero` counts only pixels where the residual is exactly zero. If any pixel has a perfect fit (residual = 0), those pixels are excluded from the denominator, *inflating* the reduced chi-squared value. More fundamentally, the correct denominator for reduced chi-squared should be `len(dx_array) - len(p)` (number of data points minus number of fitted parameters), not the count of non-zero residuals.

**Verified impact:** With 6 pixels where 3 have zero residual, the buggy formula gives `chi-sq = 3.0/3 = 1.0` instead of the correct `3.0/6 = 0.5` (without DoF correction) or `3.0/(6-3) = 1.0` (with DoF correction). The current formula accidentally agrees with the DoF-corrected value only by coincidence.

**Suggested fix:**
```python
return np.sum(dx_array * dx_array) / (len(dx_array) - len(p))
```
(Where `p` is the parameter vector; the caller already has access to it. Alternatively pass `n_params` or use `max(1, len(dx_array) - n_params)`.)

---

## Finding 3

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/analysis/fit_profiles.py:1504–1507`
**Description:**
`EquivalentWidth` uses a `for` loop `range(1, len(fluxes) - 1)`. After the loop, `i = len(fluxes) - 2`. Line 1507 then sets:
```python
dEW[i - 1] = (1.0 - fluxes[i - 1]) * abs(waves[i - 1] - waves[i - 2])
```
This writes to index `len(fluxes) - 3` (duplicating what the loop already wrote at `i = len(fluxes) - 2`), and **never writes to `dEW[-1]`** (index `len(fluxes) - 1`), which remains zero.

**Verified impact:** With 5-element arrays where the last flux is 0.5 (not 1.0): buggy sum = 2.8, correct sum = 3.3 — a ~15% undercount. The last pixel's EW contribution is silently dropped.

Compare with `EW()` in `absorption_spectra.py` (line 1841) which correctly handles all pixels via vectorised bin-edge arithmetic.

**Suggested fix:**
```python
dEW[-1] = (1.0 - fluxes[-1]) * abs(waves[-1] - waves[-2])
```

---

## Finding 4

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/analysis/fit_profiles.py:1270–1283`
**Description:**
`find_regions` merges overlapping regions by iterating with `range(len(regions_expanded) - 1)` (computed once) while mutating `regions_expanded` via `np.delete`. After a merge at index `i`, the merged region's new end is never re-checked against the *next* region (now at `i+1`). Adjacent regions can remain un-merged.

Additionally, `range` was computed from the original length, so the loop index can exceed the current array length after multiple deletions. The guards at lines 1272 and 1279 (`if len(regions_expanded) == i: break` and `if len(...) == i+1: break`) prevent IndexError but do **not** fix the missed-merge problem.

**Verified impact:** With 5 overlapping regions `[10,20], [15,25], [24,35], [34,50], [49,60]`, the merge loop produces `[[10,25], [24,50], [49,60]]` — the last pair (`[24,50]` and `[49,60]`, gap = -1) is never checked and should be merged to `[24,60]`.

**Suggested fix:** Use a `while` loop instead of `for`, or rebuild from a fresh index after each merge:
```python
i = 0
while i < len(regions_expanded) - 1:
    if regions_expanded[i + 1][0] - regions_expanded[i][1] < 5:
        regions_expanded[i][1] = regions_expanded[i + 1][1]
        regions_expanded = np.delete(regions_expanded, i + 1, axis=0)
        # do NOT increment i; re-check merged region
    else:
        i += 1
```

---

## Finding 5

**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/analysis/absorption_spectra.py:883`
**Description:**
Line 883 reads:
```python
spatial_res == spatial_extent.ptp() / N
```
This is a **comparison** (`==`), not an assignment (`=`). The boolean result is discarded. The intent was clearly to recompute `spatial_res` after `N` is determined (line 882 computes `N` from `spatial_extent.ptp() / spatial_res`). Without the reassignment, `spatial_res` retains its pre-rounding value, creating a slight inconsistency between `N`, `spatial_res`, and `spatial_extent`.

The same pattern does **not** appear in the multi-LOS version (`mock_absorption_spectra_multilos`) at the corresponding location (~line 1664), though the dead `temp.copy()` does.

**Suggested fix:** Change `==` to `=`:
```python
spatial_res = spatial_extent.ptp() / N
```

---

## Finding 6

**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/analysis/absorption_spectra.py:1057–1058, 1663–1664`
**Description:**
In both `mock_absorption_spectrum` (line 1058) and `mock_absorption_spectra_multilos` (line 1664):
```python
if temp.base is not None:
    temp.copy()
```
`temp.copy()` returns a new array but the return value is discarded; `temp` itself is never reassigned. This appears to be a vestige of a `temp = temp.copy()` call where the assignment was removed. The conditional check (`temp.base is not None`) confirms the intent was to ensure `temp` owns its data, but the copy is thrown away.

**Suggested fix:** Either remove the dead block entirely, or restore the assignment:
```python
if temp.base is not None:
    temp = temp.copy()
```

---

## Finding 7

**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/analysis/absorption_spectra.py:222–223`
**Description:**
`Voigt(x, sigma, gamma)` divides by `sigma * np.sqrt(2.0)` (line 222) and again by `sigma * np.sqrt(2.0 * np.pi)` (line 223). If `sigma = 0` (i.e., `b = 0` passed to `line_profile`), both lines raise `ZeroDivisionError`. Even small `sigma` values (e.g., `b < 0.01 km/s`) can produce extremely large `z` values passed to `wofz`, potentially causing numerical overflow or inaccurate results.

`line_profile` (line 235) does not guard against `b=0`. The `b` parameter defaults to `None`, and when `T` is given, `thermal_b_param` computes `b = sqrt(2 k_B T / m)`, which is always positive for `T > 0`. But a user passing `b=0` directly triggers the error.

**Suggested fix:** Add a guard in `line_profile`:
```python
b = UnitScalar(b, "km/s")
if float(b) <= 0:
    raise ValueError("b-parameter must be positive (got %s)" % b)
```

---

## Finding 8

**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/analysis/fit_profiles.py:474–513`
**Description:**
The inner function `_add_line` assigns to `b_bounds` at line 506 (`b_bounds = [b_guess * 0.5, b_guess * 2]`), which makes Python treat **all** references to `b_bounds` within `_add_line` as local. In the `else` branch (line 495, entered when `grow_line=False`), `b_bounds` is read *before* the assignment at line 506, triggering `UnboundLocalError`.

Currently `grow_line` defaults to `True` and all callers use the default, so this path is never hit. However, the latent bug will surface if anyone calls `_add_line(..., grow_line=False)`. The `b_bounds` at line 495 is meant to refer to the outer-scope parameter (function argument at line 411), but Python's scoping rules prevent this.

**Suggested fix:** Rename the outer `b_bounds` parameter (e.g., `b_bounds_in`) or the local one (e.g., `b_bounds_new`), or use `nonlocal b_bounds`.

---

## Finding 9

**Severity:** MEDIUM
**Category:** qol
**Location:** `pygad/analysis/absorption_spectra.py:998–1004, 1605–1611`
**Description:**
In both `mock_absorption_spectrum` and `mock_absorption_spectra_multilos`:
```python
bin_func = SPH_3D_to_line

def bin_func(s, qty, **args):
    ...
```
The assignment `bin_func = SPH_3D_to_line` at the first line is immediately overwritten by the `def bin_func(...)` on the next line. The first assignment is dead code and likely a leftover from an earlier refactoring.

**Suggested fix:** Remove the dead `bin_func = SPH_3D_to_line` line.

---

## Finding 10

**Severity:** LOW
**Category:** bug
**Location:** `pygad/analysis/fit_profiles.py:1296–1306`
**Description:**
Inside the second loop of `find_regions` (line 1287), there is an inner `for j in range(start, end)` loop that:
1. Computes `flux_dec = 1.0 - fluxes[j]` (line 1297, unused)
2. Immediately `break`s (line 1306)

The `for j` loop is functionally equivalent to a single `if start < end:` block. This appears to be a leftover from code that was once intended to scan pixels within the region.

**Suggested fix:** Replace with a simple conditional or remove the dead loop.

---

## Finding 11

**Severity:** LOW
**Category:** qol
**Location:** `pygad/analysis/absorption_spectra.py:1216–1838`
**Description:**
`mock_absorption_spectra_multilos()` (~600 lines) is a near-exact duplicate of `mock_absorption_spectrum()`. The only meaningful difference is that it loops over multiple LOS positions. All unit conversions, particle selection, binning, LOS integration, and return logic are duplicated line-for-line. This makes bug fixes error-prone (e.g., the `temp.copy()` dead code was duplicated, and the `spatial_res ==` bug was NOT — showing inconsistency).

**Suggested fix:** Refactor the single-LOS function to accept either one or many LOS positions, with the multi-LOS version calling the single-LOS implementation in a loop. Alternatively, extract shared logic into a helper.

---

## Finding 12

**Severity:** LOW
**Category:** qol
**Location:** `pygad/analysis/absorption_spectra.py:297, 788, 1198, 1823`
**Description:**
Four bare `except:` clauses in `absorption_spectra.py`:
- Line 297 (`line_profile`): catches *any* exception during unit conversion of `N`, silently falling back to mass-based conversion. Should catch `UnitError` or the specific unit exception.
- Lines 788, 1395: catch *any* exception when converting `v_turb`, silently falling back to `s.gas.get(v_turb)`.
- Lines 1198, 1823: catch *any* exception around a print statement and `pass`.

Bare `except:` swallows `KeyboardInterrupt`, `SystemExit`, and `MemoryError`, making debugging difficult.

**Suggested fix:** Use `except (SpecificException,)` instead of bare `except:`.

---

## Finding 13

**Severity:** LOW
**Category:** improvement
**Location:** `pygad/analysis/profiles.py:103`
**Description:**
`radially_binned` finds bin-edge indices using a Python loop:
```python
ind_edges = np.array([np.abs(r[r_ind] - rr).argmin() for rr in r_edges])
```
This finds the nearest particle to each edge radius, which is an approximation (not exact bin edges). For many bins, the Python loop is slow. A vectorised approach using `np.searchsorted` on the sorted radii would be both faster and more correct (assigning particles to proper bins by sorted order rather than nearest-edge).

**Suggested fix:** Since `r_ind = r.argsort()`, the sorted radii are `r[r_ind]`. Use `np.searchsorted(r[r_ind], r_edges)` to find the insertion indices, which are exactly the bin boundaries in the sorted array.
