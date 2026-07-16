# Review: analysis/halo, properties, sph_eval, vpfit, analysis

## Slice summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH     | 2 |
| MEDIUM   | 7 |
| LOW      | 5 |

The analysis/halo slice contains one confirmed data-corruption bug in `periodic_wrap` (duplicates a pixel and drops another), a NameError crash in `analysis.py`, a logic bug in `find_most_massive_progenitor` where the duplicate-suppression inner loop is dead code, and a closure-capture bug in `fit_profiles`. Several quality-of-life issues around mutable default arguments, deprecated exception APIs, dead code, and an inefficient O(N*M) temperature lookup.

---

## Finding 1

**Severity:** CRITICAL  
**Category:** bug  
**Location:** `pygad/analysis/vpfit.py:476-477`

**Description:** `periodic_wrap` uses `flux[starting_pixel:-1]` which **excludes the last element**. When concatenated with `flux[0:starting_pixel+1]`, the element at `starting_pixel` appears twice and the last element (`flux[-1]`) is silently dropped. For example, with `flux = [0.5, 0.9, 0.7, 0.3, 0.8]` and `starting_pixel=1` (argmax), the result is `[0.9, 0.7, 0.3, 0.5, 0.9]` — the last pixel (0.8) is lost and 0.9 is duplicated. The same bug applies to the `noise` array on the next line. This corrupts every spectrum that passes through `periodic_wrap`.

Verified by running the exact code path with small arrays:
```python
flux = np.array([0.5, 0.9, 0.7, 0.3, 0.8])
sp = 1  # argmax
buggy = np.concatenate((flux[sp:-1], flux[0:sp+1]))
# Result: [0.9, 0.7, 0.3, 0.5, 0.9]  -- lost 0.8, duped 0.9
```

**Suggested fix:** Change line 476 to use `flux[starting_pixel:]` (include last element) and line 477 similarly, and adjust the second slice to `flux[0:starting_pixel]` (exclude the starting pixel from the second part):
```python
flux = np.concatenate((flux[starting_pixel:], flux[0:starting_pixel]))
noise = np.concatenate((noise[starting_pixel:], noise[0:starting_pixel]))
```

---

## Finding 2

**Severity:** HIGH  
**Category:** bug  
**Location:** `pygad/analysis/analysis.py:8,41`

**Description:** The file has two fatal defects that prevent it from ever being imported:
1. **Line 8**: `from units import UnitArr` is a bare (non-relative) import that only works if `analysis.py` is run as a standalone script with the `pygad/units` directory on `sys.path`. When imported as part of the `pygad.analysis` package, it raises `ModuleNotFoundError: No module named 'units'`.
2. **Line 41**: `profile(snap, quantity, ...)` calls an undefined name `profile`. No `profile` function is imported or defined anywhere in this file. Even if the import were fixed, this would raise `NameError`.

The `__init__.py` of `pygad.analysis` does **not** import this module (it is commented out in the doctest block), so the broken file is never loaded during normal operation. However, any user who tries `from pygad.analysis.analysis import radial_surface_density_profile` will get an immediate crash. The ruff F821 diagnostic on this line is correct — `profile` is genuinely undefined.

**Suggested fix:** 
- Line 8: Change to `from ..units import UnitArr` (relative import consistent with the package structure).
- Line 41: Replace `profile(...)` with the actual function from `pygad.analysis.profiles` (e.g. `from .profiles import profile` or inline the logic). Currently the entire function `radial_surface_density_profile` is dead code.

---

## Finding 3

**Severity:** HIGH  
**Category:** bug  
**Location:** `pygad/analysis/vpfit.py:176,220`

**Description:** Inside the `fit_profiles` function, a lambda is defined inside a `while` loop:
```python
chisq_fcn = lambda *args: _chisq(*args)
```
This is a closure that captures the loop variable environment. While in this specific case the lambda only calls `_chisq` (which doesn't depend on loop variables), this pattern is fragile and the lambda is completely unnecessary — it is just `chisq_fcn = _chisq`. The lambda wrapping provides no benefit but makes the intent unclear.

More importantly, the lambda shadows any previous `chisq_fcn` without purpose, and on line 220 the **same lambda is redefined in the retry branch**, which creates a second unnecessary closure. If someone later modifies `_chisq` to close over loop variables, this pattern will silently break due to late binding.

**Suggested fix:** Replace `chisq_fcn = lambda *args: _chisq(*args)` with `chisq_fcn = _chisq` at both lines 176 and 220. Or simply use `_chisq` directly in the `minimize` calls.

---

## Finding 4

**Severity:** HIGH  
**Category:** bug  
**Location:** `pygad/analysis/halo.py:1133`

**Description:** `find_most_massive_progenitor` has an unconditional `print(h0, h0.mass)` statement on line 1133 that fires every time the function is called, regardless of verbosity settings. This is debug output that was likely left in during development. For a function that may be called in batch processing of halo catalogs, this produces unwanted stdout noise.

Additionally, `h0.mass` triggers lazy computation of the mass property on the Halo object, which could be expensive if the halo has not had its mass computed yet.

**Suggested fix:** Remove or guard behind a verbosity check:
```python
if verbose >= environment.VERBOSE_TACITURN:
    print(h0, h0.mass)
```

---

## Finding 5

**Severity:** MEDIUM  
**Category:** bug  
**Location:** `pygad/analysis/halo.py:1141-1151`

**Description:** In `find_most_massive_progenitor`, the inner loop intended to skip halos already in `closest` is dead code:
```python
for h in halos:
    d = np.linalg.norm(h.com - h0_com)
    if d < close_d and h.mass > min_mass:
        for nh in closest:      # <-- inner loop
            if h is nh:
                continue         # <-- only continues INNER loop!
        close = h               # <-- always reached
        close_d = d
```
The `continue` statement on line 1149 only skips the rest of the **inner** `for nh` loop, not the outer assignment. The code after the inner loop (`close = h; close_d = d`) executes regardless. This means the same halo can be added to `closest` multiple times across the 3 outer iterations, defeating the deduplication intent.

**Suggested fix:** Replace the inner loop with a simple check:
```python
for h in halos:
    d = np.linalg.norm(h.com - h0_com)
    if d < close_d and h.mass > min_mass and h not in closest:
        close = h
        close_d = d
```

---

## Finding 6

**Severity:** MEDIUM  
**Category:** bug  
**Location:** `pygad/analysis/halo.py:560`

**Description:** In `generate_Rockstar_halos`, the error handler accesses `e.message`:
```python
except RuntimeError as e:
    if ignore_inconsistency:
        print('WARNING:', e.message)
```
Python 3 exceptions do not have a `.message` attribute. This will raise `AttributeError` whenever `ignore_inconsistency=True` and a consistency check fails. Verified: `hasattr(RuntimeError('test'), 'message')` returns `False` in Python 3.12.

**Suggested fix:** Use `str(e)` instead of `e.message`:
```python
print('WARNING:', e)
# or: print('WARNING: ' + str(e))
```

---

## Finding 7

**Severity:** MEDIUM  
**Category:** correctness  
**Location:** `pygad/analysis/halo.py:1092,1097`

**Description:** `generate_FoF_catalogue` uses `max_halos` in `min(N_FoF, max_halos)` on line 1092 and in `len(halos) == max_halos` on line 1097, but `max_halos` defaults to `None`. When called without `max_halos`, line 1092 raises `TypeError: '<' not supported between instances of 'NoneType' and 'int'`. This makes the function crash with default arguments whenever `exclude is None`.

Verified:
```python
>>> min(5, None)
TypeError: '<' not supported between instances of 'NoneType' and 'int'
```

**Suggested fix:** Guard the `max_halos` comparisons:
```python
for i in range(min(N_FoF, max_halos) if (exclude is None and max_halos is not None) else N_FoF):
    ...
    if max_halos is not None and len(halos) == max_halos:
        break
```

---

## Finding 8

**Severity:** MEDIUM  
**Category:** correctness  
**Location:** `pygad/analysis/vpfit.py:323-327`

**Description:** `EquivalentWidth` computes the endpoint contribution using a loop-variable leak:
```python
for i in range(1, len(fluxes) - 1):
    dEW[i] = (1.0 - fluxes[i]) * abs(waves[i + 1] - waves[i - 1]) * 0.5
dEW[0] = (1.0 - fluxes[0]) * abs(waves[1] - waves[0])
dEW[i - 1] = (1.0 - fluxes[i - 1]) * abs(waves[i - 1] - waves[i - 2])
```
After the loop, `i = len(fluxes) - 2`, so `dEW[i-1] = dEW[len(fluxes)-3]` overwrites the **third-to-last** element instead of the last element `dEW[-1]`. The intended line should be:
```python
dEW[-1] = (1.0 - fluxes[-1]) * abs(waves[-1] - waves[-2])
```
This causes the EW contribution of the last pixel to be silently dropped and the third-to-last pixel's contribution to be double-counted.

**Suggested fix:** Replace `dEW[i - 1]` with `dEW[-1]`:
```python
dEW[-1] = (1.0 - fluxes[-1]) * abs(waves[-1] - waves[-2])
```

---

## Finding 9

**Severity:** MEDIUM  
**Category:** compat  
**Location:** `pygad/analysis/vpfit.py:64-65`

**Description:** `fit_profiles` uses mutable default arguments `logN_bounds=[12, 19]` and `b_bounds=[5, 200]`. While these lists are currently only read (never mutated) within the function, Python's mutable-default-argument pitfall means that if any caller or future modification accidentally mutates them (e.g. via `b_bounds.append(300)`), the default will be permanently changed for all future calls. This is a latent hazard.

The same pattern appears in `fit_profiles.py` at line 180 (`b_bounds=[2, 100]`).

**Suggested fix:** Use `None` defaults and create the list inside the function:
```python
def fit_profiles(line, l, flux, noise, chisq_lim=2.0, max_lines=10,
                  mode="Voigt", logN_bounds=None, b_bounds=None):
    if logN_bounds is None:
        logN_bounds = [12, 19]
    if b_bounds is None:
        b_bounds = [5, 200]
```

---

## Finding 10

**Severity:** MEDIUM  
**Category:** bug  
**Location:** `pygad/analysis/halo.py:805-806`

**Description:** `Halo.__getattr__` uses `self._props.get(name, None)` and then checks `if attr is not None`. This means that if a property is legitimately set to `None` (e.g. via `Halo(properties={'some_prop': None})`), accessing it as `halo.some_prop` will raise `AttributeError` instead of returning `None`. The `_props` dictionary correctly stores the value, but the attribute accessor cannot distinguish between "missing" and "explicitly None".

**Suggested fix:** Use a sentinel value:
```python
_MISSING = object()
def __getattr__(self, name):
    if name.startswith('__'):
        return super(Halo, self).__getattr__(name)
    if name in self._props:
        return self._props[name]
    raise AttributeError('%s has no attribute "%s"!' % (self, name))
```

---

## Finding 11

**Severity:** MEDIUM  
**Category:** qol  
**Location:** `pygad/analysis/properties.py:532-535`

**Description:** `x_ray_luminosity` uses a manual O(N_bins × N_particles) loop with double `np.where` calls per bin to find the nearest temperature bin. This is both slow and hard to read. A `np.searchsorted` or `np.digitize` approach would be O(N_particles × log(N_bins)) and much cleaner. For typical emission tables with ~30-50 bins and millions of gas particles, this loop is a significant bottleneck.

The second loop (lines 538-540) also uses `np.where` twice per bin, scanning the entire particle array for each bin.

**Suggested fix:** Replace both loops with vectorized binning:
```python
indices = np.searchsorted(tempbin, kB_T) - 1
indices = np.clip(indices, 0, len(tempbin) - 1)
# Handle ties (pick closer bin)
mask_high = (np.abs(kB_T - tempbin[np.clip(indices+1, 0, len(tempbin)-1)]) <
             np.abs(kB_T - tempbin[indices]))
indices[mask_high] = np.clip(indices[mask_high] + 1, 0, len(tempbin) - 1)
# Then vectorized lookup:
lx = lx0bin[indices] + (Z - Zref) * dlxbin[indices]
lx[kB_T < tlow] = 0
```

---

## Finding 12

**Severity:** MEDIUM  
**Category:** correctness  
**Location:** `pygad/analysis/vpfit.py:495`

**Description:** In `fit_profiles.py:_add_line` (line 495), `b_bounds` is a **function parameter** that is used on line 495, but then **reassigned as a local variable** on line 506 (`b_bounds = [b_guess * 0.5, b_guess * 2]`). This shadows the parameter for the remainder of the function. Ruff correctly flags this as F823 "referenced before assignment" because the reassignment at line 506 creates ambiguity about which `b_bounds` is being used at line 575 (`b_range = np.linspace(start=np.log10(b_bounds[0]), ...)`). After line 506, all uses of `b_bounds` refer to the local copy, not the original parameter — this appears intentional but is confusing and error-prone.

**Suggested fix:** Rename the local variable:
```python
b_bounds_local = [b_guess * 0.5, b_guess * 2]
```

---

## Finding 13

**Severity:** LOW  
**Category:** qol  
**Location:** `pygad/analysis/vpfit.py:687,626`

**Description:** In both `write_lines` and `plot_fit`, the `else` branch uses a confusing double-assignment:
```python
l = dl = line_list["l"]
```
followed immediately by:
```python
dl = line_list["dl"]
```
The `dl = line_list["l"]` assignment is immediately overwritten, making the combined assignment misleading. This pattern occurs at `vpfit.py:626` and `vpfit.py:687`.

**Suggested fix:** Simplify to:
```python
l = line_list["l"]
dl = line_list["dl"]
```

---

## Finding 14

**Severity:** LOW  
**Category:** improvement  
**Location:** `pygad/analysis/vpfit.py:249`

**Description:** `fit_profiles` uses `soln.hess_inv` as the covariance matrix when the optimizer is L-BFGS-B (default method for `minimize` with bounds). However, `hess_inv` from L-BFGS-B is an **approximation** of the inverse Hessian of the objective, not the inverse Hessian of the chi-squared surface at the minimum. For proper error estimates, the Hessian should be evaluated numerically (e.g. via `scipy.optimize.minimize` with `method='Nelder-Mead'` followed by numerical Hessian, or using `scipy.optimize.least_squares` which provides proper covariance). The BFGS refit at line 242 also does not use bounds, potentially allowing parameters to drift outside physical ranges during the Hessian computation.

**Suggested fix:** Compute the Hessian numerically at the best-fit solution, or use `scipy.optimize.least_squares` which returns the Jacobian from which a proper covariance can be derived.

---

## Finding 15

**Severity:** LOW  
**Category:** qol  
**Location:** `pygad/analysis/halo.py:814,818`

**Description:** `Halo.__repr__` uses bare `except:` clauses (lines 814 and 818) that silently swallow all exceptions including `KeyboardInterrupt` and `SystemExit`. While this is only used for display purposes, bare excepts are an anti-pattern.

**Suggested fix:** Replace `except:` with `except Exception:`.

---

## Finding 16

**Severity:** LOW  
**Category:** improvement  
**Location:** `pygad/analysis/halo.py:348`

**Description:** `find_FoF_groups` counts unique groups with `N_FoF = len(set(FoF)) - 1`, which creates a Python set from a potentially large array of uintp values. For simulations with millions of particles, this allocates a large temporary set. A faster approach would be `N_FoF = FoF.max() + 1` (since FoF IDs are contiguous 0..N-1 when sorted).

**Suggested fix:** Replace with `N_FoF = int(FoF.max()) + 1` (also fixing the same issue at `halo.py:1075`).

---
