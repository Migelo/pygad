# Review: Physics, SSP, Transformation, and Cloudy Modules

## Slice summary

| Severity | Count |
|----------|-------|
| CRITICAL | 2     |
| HIGH     | 4     |
| MEDIUM   | 6     |
| LOW      | 4     |

Two critical bugs found: `FLRWCosmo.__eq__` compares `self` fields to `self` instead of `other` (making equality tests meaningless for 4 of 6 parameters), and the `_apply_to_block` shape-check in both `Rotation` and `Translation` has a Python operator-precedence error that silently skips validation for wrong-column-count 2D arrays. Additional HIGH findings include a broken `sigma_8` setter (`NameError`) and two instances of dead infinite-recursion `copy()` methods shadowed by correct redefinitions. Several medium-severity issues around interpolation edge behavior, documentation errors, and code quality.

---

## Finding 1

**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/physics/cosmology.py:195-201`
**Description:**
`FLRWCosmo.__eq__` compares 4 of 6 fields to `self` instead of `other`. Lines 198–201 read `self._Omega_m == self._Omega_m` (always `True`), `self._Omega_b == self._Omega_b` (always `True`), `self._sigma_8 == self._sigma_8` (always `True`), and `self._n_s == self._n_s` (always `True`). Only `h_0` and `Omega_Lambda` are compared against `other`. This means any two `FLRWCosmo` instances with the same `h_0` and `Omega_Lambda` compare equal regardless of their `Omega_m`, `Omega_b`, `sigma_8`, or `n_s`.

```python
def __eq__(self, other):
    return (self._h_0 == other._h_0 and
            self._Omega_Lambda == other._Omega_Lambda and
            self._Omega_m == self._Omega_m and      # BUG: should be other._Omega_m
            self._Omega_b == self._Omega_b and      # BUG: should be other._Omega_b
            self._sigma_8 == self._sigma_8 and       # BUG: should be other._sigma_8
            self._n_s == self._n_s)                  # BUG: should be other._n_s
```

**Suggested fix:**
Replace `self._` with `other._` on lines 198–201. Also consider adding a `NotImplemented` return for non-`FLRWCosmo` `other`.

---

## Finding 2

**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/transformation/transformation.py:312` and `pygad/transformation/transformation.py:417`
**Description:**
Both `Translation._apply_to_block` (line 312) and `Rotation._apply_to_block` (line 417) have the same operator-precedence bug in their shape-validation guard:

```python
if not len(block.shape)==2 and block.shape[1]==self._R.shape[0]:
    raise ValueError(...)
```

Due to Python precedence (`not` < `==` < `and`), this parses as:
`(not (len(block.shape)==2)) and (block.shape[1]==self._R.shape[0])`

The intended logic is `not (len(block.shape)==2 and block.shape[1]==...)`.

Consequences:
- A 2D block with the **wrong** column count silently passes validation (no error raised).
- A 1D block crashes with `IndexError` on `block.shape[1]` instead of a helpful `ValueError`.

For `Translation`, the same pattern at line 312 uses `self._trans.shape[0]` instead of `self._R.shape[0]`.

**Suggested fix:**
Add parentheses: `if not (len(block.shape)==2 and block.shape[1]==self._R.shape[0]):` in both locations (lines 312 and 417).

---

## Finding 3

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/physics/cosmology.py:278-281`
**Description:**
The `sigma_8` property setter references the bare name `sigma_8` in its assert (line 280):

```python
@sigma_8.setter
def sigma_8(self, value):
    assert sigma_8 >= 0      # NameError: 'sigma_8' is not defined
    self._sigma_8 = value
```

The bare name `sigma_8` is neither a local variable nor a global/module-level name. This raises `NameError` whenever the setter is called. Note that `sigma_8` is a property descriptor — accessing it via `self.sigma_8` would invoke the getter, but the bare name does not go through the descriptor protocol. The setter is currently never exercised in tests or doctests, so this bug is latent.

**Suggested fix:**
Change to `assert value >= 0`.

---

## Finding 4

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/transformation/transformation.py:134`
**Description:**
`Transformation.__init__` line 134 contains a typo: `prost.copy()` instead of `post.copy()`:

```python
self._post = [] if post is None else prost.copy()
```

The bare name `prost` does not exist anywhere in scope, so passing a non-`None` `post` argument raises `NameError`. Currently, all subclass constructors (`Translation.__init__`, `Rotation.__init__`) pass `post=None` (the default), so this code path is never reached in practice. However, this prevents `Transformation` from being instantiated with a custom `post` list.

**Suggested fix:**
Change `prost.copy()` to `post.copy()`.

---

## Finding 5

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/transformation/transformation.py:140`
**Description:**
`Transformation.__copy__` at line 140 references `self._prost` (which is never set — the attribute is `self._post`):

```python
def __copy__(self):
    cp = Transformation.__new__(Transformation)
    cp._change = self._change.copy()
    cp._pre  = self._pre.copy()
    cp._post = self._prost.copy()    # AttributeError: _prost does not exist
    return cp
```

This always raises `AttributeError` if called. Both `Translation` and `Rotation` override `__copy__`, so this base-class method is not currently called, but it is dead-broken code.

**Suggested fix:**
Change `self._prost.copy()` to `self._post.copy()`.

---

## Finding 6

**Severity:** HIGH
**Category:** correctness
**Location:** `pygad/physics/cosmology.py:108`
**Description:**
The `WMAP7()` docstring states `sigma_8=810`, but the actual function call on line 118 passes `sigma_8=0.810`. The docstring value is off by a factor of 1000, which would mislead anyone reading the docstring to understand the cosmological parameters.

```python
# Docstring (line 108):
#   sigma_8=810
# Actual (line 118):
    sigma_8=0.810
```

**Suggested fix:**
Change the docstring to `sigma_8=0.810`.

---

## Finding 7

**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/transformation/transformation.py:284-285` and `pygad/transformation/transformation.py:358-359`
**Description:**
Both `Translation` and `Rotation` define `copy()` twice. The first definition is an infinite recursion (`return self.copy()`), immediately shadowed by a correct second definition later in the class body:

- `Translation`: line 284–285 (`return self.copy()`) shadowed by line 297–299 (`return Translation(self._trans)`)
- `Rotation`: line 358–359 (`return self.copy()`) shadowed by line 399–401 (`return Rotation(self._R)`)

Python class bodies execute top-to-bottom, so the second definition replaces the first. The first is dead code. If someone ever reorders the methods or removes the second definition, the first would cause `RecursionError`.

**Suggested fix:**
Remove the first (broken) `copy()` definition from both classes.

---

## Finding 8

**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/physics/cosmology.py:46`
**Description:**
The `z2a` function docstring (line 46) says "Convert a scalefactor to redshift" but the function actually converts a **redshift to a scalefactor** (the opposite direction). The function name `z2a` (redshift-to-scalefactor) and its implementation (`1.0 / (1.0 + z)`) confirm the docstring is wrong.

**Suggested fix:**
Change the docstring to "Convert a redshift to a scalefactor."

---

## Finding 9

**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/physics/cooling.py:291-294`
**Description:**
In `get_cooling_for_species`, when a metal species is queried with `len(T) > 1`, the code falls back to a Python loop calling the `RegularGridInterpolator` one particle at a time:

```python
if len(T) == len(nH) and len(T) > 1:
    Lambda = np.array([
        self._metals_interp[species](t, n) for t, n in zip(T, nH)
    ]).reshape(len(T))
else:
    Lambda = self._metals_interp[species](T, nH)
```

`RegularGridInterpolator` supports vectorized evaluation — the `else` branch already does this. The `if` branch unnecessarily loops over particles, which is extremely slow for large snapshots. The condition `len(T) == len(nH) and len(T) > 1` triggers the loop; the `else` branch only fires for scalar inputs or mismatched lengths.

**Suggested fix:**
Remove the per-particle loop and always use `self._metals_interp[species](T, nH)`, adapting the call to pass 2D point array if needed: `self._metals_interp[species](np.column_stack([T, nH]))`.

---

## Finding 10

**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/ssp/bc_ssp.py:307-322`
**Description:**
The metallicity interpolation in `inter_bc_qty` uses strict `<` comparisons for bin boundary assignment:

- Line 310: `Z_mask = Z < z` (particles below the lowest metallicity)
- Line 314: `Z_mask = z <= Z` (particles at or above the highest metallicity)
- Line 319: `Z_mask = (z[0] <= Z) & (Z < z[1])` (interpolation between two bins)

For a particle whose metallicity exactly equals one of the table metallicity bin edges (except the highest), the strict `<` on the upper bound excludes it from the interpolation bin, while the `<=` on the lower bound of the next bin captures it. This means particles at exact bin edges get assigned to the *higher* bin rather than being properly interpolated between adjacent bins. This creates a small discontinuity in the interpolated quantity at each bin boundary.

**Suggested fix:**
Consider using `<=` on the upper bound and `<` on the lower bound consistently, or handle exact-boundary particles via explicit nearest-neighbor assignment.

---

## Finding 11

**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/cloudy/treecol.py:39-40`
**Description:**
The `sigHI` function assigns a local variable `sigHI` that shadows the function name:

```python
def sigHI(z, UVB=gadget.general['UVB']):
    ...
    sigHI = -4.89946e-19 * (1. + z) ** (0.46765) + 3.072648e-18
    ...
    return sigHI
```

While this works at runtime (the `return` statement accesses the local variable), it prevents any recursive or re-entrant call and makes the code confusing. It also shadows the function in any nested scope. This pattern appears in multiple UVB branches (FG11/FG19 at line 39, HM01 at line 46, HM12 at line 50).

**Suggested fix:**
Rename the local variable to e.g. `cross_section` or `_sig`.

---

## Finding 12

**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/physics/cosmology.py:195-201` (same method as Finding 1)
**Description:**
`FLRWCosmo.__eq__` does not guard against comparing to non-`FLRWCosmo` objects. If `other` is not an `FLRWCosmo` instance, accessing `other._h_0` will raise `AttributeError` instead of returning `NotImplemented` (which would allow Python's comparison protocol to try the reflected operation).

**Suggested fix:**
Add at the top of `__eq__`: `if not isinstance(other, FLRWCosmo): return NotImplemented`

---

## Finding 13

**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/physics/cosmology.py:172-174`
**Description:**
In `FLRWCosmo.__init__`, the `sigma_8` validation uses a truthiness check:

```python
if sigma_8:
    assert sigma_8 > 0
```

If `sigma_8=0.0` is passed, the `if` branch is skipped (0.0 is falsy), so the assertion never fires. While `sigma_8=0.0` is physically meaningless, the intent was clearly to validate that `sigma_8` is positive. A negative value like `sigma_8=-0.5` would correctly be caught, but `sigma_8=0.0` silently passes.

**Suggested fix:**
Use `if sigma_8 is not None: assert sigma_8 > 0` to properly handle the `None` (not provided) case.

---

## Finding 14

**Severity:** LOW
**Category:** qol
**Location:** `pygad/physics/quantities.py:40-42`
**Description:**
`Iterable` is imported from `collections.abc` inside a `try/except ImportError: pass` block but is never used anywhere in `quantities.py`. The import was presumably added for Python 3.12 compatibility, but it is dead code. The `except` clause silently swallows the `ImportError` if it ever occurred.

**Suggested fix:**
Remove the unused import block entirely.

---

## Finding 15

**Severity:** LOW
**Category:** qol
**Location:** `pygad/physics/quantities.py:612`
**Description:**
The `Jeans_mass` docstring (line 612) says "You can also calculate **Jeans lengthes** for arrays of parameters" — this is a copy-paste from `Jeans_length` and should say "Jeans masses."

**Suggested fix:**
Change "Jeans lengthes" to "Jeans masses" in the `Jeans_mass` docstring.

---

## Finding 16

**Severity:** LOW
**Category:** improvement
**Location:** `pygad/physics/quantities.py:418`
**Description:**
`SMH_Kravtsov_2014` emits `warnings.warn('Scatter of 0.2 dex in Kravtsov et al. (2014) not sure!')` when `return_scatter=True`. The message "not sure!" is vague and unprofessional. The scatter value of 0.2 dex is stated in the paper (Kravtsov et al. 2014, Appendix A), so the uncertainty note should be more specific.

**Suggested fix:**
Rewrite to something like `'Scatter of 0.2 dex assumed from Kravtsov et al. (2014); value is approximate.'`

---

## Finding 17

**Severity:** LOW
**Category:** qol
**Location:** `pygad/physics/quantities.py:425`
**Description:**
`Reff_van_der_Wel_2014` uses `type` as a parameter name (`def Reff_van_der_Wel_2014(M_stars, z, type, ...)`), which shadows the Python builtin `type`. This is flagged by ruff (E741/F841) and is generally considered bad practice. The same pattern appears in `SMH_Kravtsov_2014` (line 376, `type='200c'`).

**Suggested fix:**
Rename the parameter to `galaxy_type` or `morph_type` in both functions.
