# Review Slice 02: Snapshot Framework

## Slice summary

**Severity counts:** 2 CRITICAL, 3 HIGH, 6 MEDIUM, 5 LOW

The snapshot framework (snapshot.py, snapshotcache.py, derived.py, derive_rules.py, masks.py, sim_arr.py, __init__.py) is largely correct but carries several latent hazards. The most dangerous findings are a debug `exit(1)` left inside production code that will hard-kill the process, and an `eval()` call on untrusted user input in the snapshot cache family setter. Memory handling in the snapshot cache and derived block caching logic also have edge cases that can silently swallow errors.

---

## Finding 1
**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/snapshot/snapshotcache.py:1549-1550`
**Description:** Inside `read_traced_gas`, a bare `except Exception:` handler at line 1549 prints the error and then calls `exit(1)` at line 1550, hard-killing the entire Python process. The `exit(1)` is followed by an unreachable `return` on line 1551. This means any failure during traced gas reading (e.g. a malformed file, a missing column, an I/O error) will silently terminate the user's Python session with no chance for the caller to catch or recover. This is extremely dangerous in interactive or pipeline usage.
**Suggested fix:** Replace `exit(1)` with `raise` (re-raise the caught exception), or raise a custom exception. The current `except Exception:` pattern is already problematic (silently swallowing), but `exit(1)` makes it catastrophic. At minimum:
```python
except Exception:
    import traceback
    traceback.print_exc()
    raise
```

---

## Finding 2
**Severity:** CRITICAL
**Category:** security / bug
**Location:** `pygad/snapshot/snapshotcache.py:315`
**Description:** The `galaxy` property setter calls `eval('gx.' + str(value), globals(), locals())` to resolve a family name string to an attribute access. The `value` parameter comes from the user (it is the property's public setter). A maliciously crafted `value` such as `"__class__.__base__"` or any arbitrary Python expression after the dot would execute arbitrary code within the current module's globals and locals. While this is an astrophysics library and not a web service, this is still an unsafe pattern that can lead to unexpected behavior or crashes from accidental malformed input (e.g. `snapshot.galaxy = "foo"` would eval `gx.foo` and raise a confusing `AttributeError` instead of a clear `ValueError`).
**Suggested fix:** Validate `value` against the known families in `gadget.families` before eval, or better yet, replace `eval` with `getattr(gx, str(value))`:
```python
if value not in gadget.families:
    raise ValueError(f"Unknown galaxy family '{value}'")
self.__galaxy = getattr(self.__galaxy_all, str(value))
```

---

## Finding 3
**Severity:** HIGH
**Category:** bug
**Location:** `pygad/snapshot/snapshotcache.py:532`
**Description:** A dead/always-true guard at line 532: `if True or rs == 0.0:  # load only for z=0`. The `True or` short-circuits Python's boolean evaluation, so `rs == 0.0` is never evaluated. The comment says "load only for z=0", suggesting the original intent was to conditionally load star-forming info only at redshift zero. As written, this block always executes regardless of redshift, which may cause unnecessary file loads or incorrect behavior when loading data meant only for z=0 snapshots. The `exlude` typo on line 882 in the FoF lambda is a related minor spelling error (though it is a keyword argument name passed to `analysis.generate_FoF_catalogue`, so if that function uses `exclude`, this is a runtime `TypeError`).
**Suggested fix:** Change to `if rs == 0.0:` to match the stated intent, or if the guard was intentionally disabled, add a clear comment and remove the dead operand.

---

## Finding 4
**Severity:** HIGH
**Category:** bug
**Location:** `pygad/snapshot/snapshotcache.py:814`
**Description:** A debug print `print("$$$Korrekturfaktor ", mass_total, mass_dm, factor)` remains in `_correct_virial_info`. This function is itself commented out at line 907 (`# if dm_only and dm_halo is not None: self._correct_virial_info(halo_properties)`), so this is not currently user-facing. However, if someone re-enables the function (which is plausible since the virial correction is an important physical calculation), the debug print with the `$$$` prefix will appear in user output. This is a latent cleanup issue — more importantly, the entire `_correct_virial_info` function appears to be disabled, meaning virial radii/masses from the profile reader may be uncorrected when they should be.
**Suggested fix:** Remove the debug print. Evaluate whether `_correct_virial_info` should be re-enabled; if not, remove the entire function to avoid confusion.

---

## Finding 5
**Severity:** HIGH
**Category:** correctness
**Location:** `pygad/snapshot/snapshotcache.py:450-451, 498-499, 788-789, 2108-2109`
**Description:** Four separate `except Exception:` / `except Exception as e:` handlers that silently swallow errors:
- Line 450-451: In `has_profile` method — catches all exceptions, returns `False`. This means any error (e.g. permission denied, disk corruption) is silently reported as "no profile", making debugging impossible.
- Line 498-499: In galaxy property loading — catches all exceptions, sets `halo = None` and `gx = None`. If the halo catalog file exists but is corrupted, the user gets no error message, just `None` properties.
- Line 788-789: In profile loading — catches all exceptions, silently sets default profile properties. The user may work with incorrect/empty profile data without knowing.
- Line 2108-2109: In `SnapshotProperty` data retrieval — catches all exceptions with bare `pass`, returns `None`.

These broad exception handlers mask real problems. If a file is corrupted or a code bug occurs, the user sees silent wrong behavior instead of a meaningful error.
**Suggested fix:** Narrow each `except Exception` to catch only the specific expected exceptions (e.g. `FileNotFoundError`, `KeyError`, `OSError`). At minimum, add `logging.warning()` or `warnings.warn()` calls so the user is informed that something went wrong.

---

## Finding 6
**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/snapshot/derive_rules.py:286`
**Description:** In `calc_ion_mass`, the return value is `f_ion * s.gas.get(el)`. The `f_ion` comes from `10. ** iontbl.interp_snap(...)` which returns a dimensionless numpy array. The `s.gas.get(el)` returns the element abundance block (typically derived from `'elements'` column, with units of mass or dimensionless depending on how the block was defined). If the element block `el` happens to have units (e.g. Msol), the multiplication would attach those units to the ion mass, but the function's docstring says the result should be "in units of the block given by `el`". However, `f_ion` is always unitless (`10**x`), so the units propagate only from `s.gas.get(el)`. This is correct if `el` is defined as a mass block (which it typically is via `_rules[el] = 'elements[:,%d]' % i` at `derived.py:210`, and elements are mass fractions times mass, so they carry mass units). However, if someone defines `el` differently (e.g. as a pure fraction), the result would be dimensionless mass, losing units silently.
**Suggested fix:** This is architecturally correct for the standard use case. Add an assertion or explicit check: `assert s.gas.get(el).units is not None, "Element block must have units for calc_ion_mass"`.

---

## Finding 7
**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/snapshot/masks.py:486-488`
**Description:** `ExprMask._get_mask_for` calls `s.get(self._expr)` with no validation or sandboxing of the expression string. The expression is passed directly to `Snapshot.get()`, which uses `utils.Evaluator` with numpy and all snapshot blocks in the namespace. While `utils.Evaluator` may provide some restriction, the expression can access any attribute of the snapshot, any imported module in the namespace (including `os`, `sys` via attribute chains), and execute arbitrary numpy operations. This is a design note rather than an exploitable bug (the expression comes from the Python user, not external input), but it means a typo in an expression string can trigger unexpected block loads (expensive I/O) rather than failing fast with a clear syntax error.
**Suggested fix:** This is acceptable as a design choice for a research tool. Consider adding `__builtins__` stripping from the evaluator namespace if not already done, and document that expressions can trigger block loads.

---

## Finding 8
**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/snapshot/masks.py:226`
**Description:** `BallMask._get_mask_for` uses strict inequality `r < R` (line 226), while the sph_overlap branch uses `r - sub['hsml'] < R` (line 237). The docstring says "within a given radius", which is ambiguous about boundary inclusion. The strict `<` means particles exactly at radius R are excluded. For the sph_overlap branch, particles at R + hsml are also included, which may be intentional (smoothed particles that overlap the ball). However, this inconsistency between the strict and overlap modes is worth noting: a user might expect `BallMask(R)` to be equivalent to `np.linalg.norm(pos-center, axis=1) <= R`, but it is `<`, not `<=`.
**Suggested fix:** This is a deliberate design choice documented in the sph_overlap flag. Add a note to the docstring clarifying the strict inequality (`<`, not `<=`) for both the strict mode and the overlap mode.

---

## Finding 9
**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/snapshot/derived.py:176`
**Description:** When `delete_old=True` is passed to `read_derived_rules`, the function calls `general.clear()` at line 176, then immediately re-initializes `general` with defaults on lines 177-178. However, at line 169, `general['cache_derived']` was already updated from the config. The `clear()` at line 176 discards this freshly-read value, and line 177 resets it to `True` regardless of what the config file says. This means that with `delete_old=True`, the config file's `cache_derived` setting is ignored and always reset to `True`. The subsequent code at line 169 may have also already modified `general['always_cache']` from the config, which is also lost by the `clear()`.
**Suggested fix:** Reorder the code: apply `delete_old` reset *before* reading from the config file, or move the config reading after the reset:
```python
if delete_old:
    _rules.clear()
    general.clear()
    general['cache_derived'] = True
    general['always_cache'] = set()
    # ... reset iontable

# now read from config (which overwrites the defaults)
general['cache_derived'] = cfg.getboolean('general', 'cache_derived')
```

---

## Finding 10
**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/snapshot/sim_arr.py:68`
**Description:** In `SimArr.__new__`, when `snap is None`, the fallback is `self._snap = getattr(data, '_snap', lambda: None)`. This means if `data` is a `SimArr` that has a dead weakref (the snapshot has been garbage collected), `self._snap` will be set to the *dead weakref callable*, not a new `lambda: None`. The `snap` property at line 104 calls `self._snap()`, which would return `None` (since the weakref is dead), so it technically works — but the old dead weakref object is retained unnecessarily. More subtly, if `data._snap` happens to be the `lambda: None` from a previous such fallback, `getattr` will return it, and both the old and new SimArr share the same lambda object. This is harmless but indicates the pattern is fragile.
**Suggested fix:** Use a more explicit check:
```python
if snap is not None:
    new._snap = weakref.ref(snap)
elif hasattr(data, '_snap'):
    existing = data._snap
    if callable(existing) and not isinstance(existing, weakref.ref):
        new._snap = existing  # already lambda: None
    else:
        new._snap = lambda: None
else:
    new._snap = lambda: None
```
Or more simply, always use `lambda: None` when `snap is None`.

---

## Finding 11
**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/snapshot/snapshot.py:421-425`
**Description:** In `_initsnap`, block name normalization uses a fragile format-specific check at line 421: `if s._file_handlers[0]._format == 3 and '%-4s' % name not in gadget.std_name_to_HDF5`. The `%-4s` formatting left-pads the block name to 4 characters. This is used to check whether a block name exists in the HDF5 name mapping. However, `name` may already be longer than 4 characters, in which case `%-4s` is a no-op. The real issue is that this check accesses `s._file_handlers[0]._format` before the block name loop, meaning if the first file handler has a different format than the block's source, blocks could be renamed incorrectly. Additionally, the `'%-4s' % name` format would raise a `TypeError` if `name` is not a string (unlikely but not guarded).
**Suggested fix:** Guard against non-string names and document the assumption that all file handlers share the same format. Consider using the block's own source format rather than always checking the first handler.

---

## Finding 12
**Severity:** LOW
**Category:** qol
**Location:** `pygad/snapshot/snapshotcache.py:779`
**Description:** Typo in debug print: `'create profle folder: "%s"'` should be `'create profile folder: "%s"'`. This is cosmetically incorrect but has no functional impact.
**Suggested fix:** Fix the typo to `'profile'`.

---

## Finding 13
**Severity:** LOW
**Category:** qol
**Location:** `pygad/snapshot/snapshotcache.py:1355`
**Description:** Typo in comment: `'wihtin 10 kpc'` should be `'within 10 kpc'`. No functional impact.
**Suggested fix:** Fix the typo.

---

## Finding 14
**Severity:** LOW
**Category:** qol
**Location:** `pygad/snapshot/snapshotcache.py:1370`
**Description:** Typo in print statement: `'angular momentum of the galaxtic baryons'` should be `'galactic'`.
**Suggested fix:** Fix the typo.

---

## Finding 15
**Severity:** LOW
**Category:** qol
**Location:** `pygad/snapshot/snapshotcache.py:882`
**Description:** Keyword argument typo: `exlude=lambda h, s: ...` — if the receiving function `generate_FoF_catalogue` expects `exclude`, this will raise a `TypeError` at runtime. If the function accepts `**kwargs` and ignores unknown keywords, this silently does nothing, meaning the FoF mass filter intended to exclude low-resolution contaminated halos is never applied.
**Suggested fix:** Change `exlude` to `exclude` and verify the parameter name matches the function signature.

---

## Finding 16
**Severity:** LOW
**Category:** improvement
**Location:** `pygad/snapshot/snapshot.py:891-912`
**Description:** The `hsml3` property (lines 891-912) contains dead commented-out code from an older `get_arepo_blocks` method (lines 914-951, ~40 lines of comments). It also hardcodes `nthreads = 32` at line 904, which is not portable and may fail or perform poorly on machines with fewer cores. The property loads `Volume` if not present, computes hsml from it, then also calls `pysph` tree-based hsml, takes the maximum, and stores it as a custom block via `_add_custom_block`. This has side effects every time `hsml3` is accessed if the block is not cached.
**Suggested fix:** Remove the ~40 lines of dead commented-out code. Replace `nthreads = 32` with a configurable value or `os.cpu_count()`. Document the side effects of accessing this property.

---
