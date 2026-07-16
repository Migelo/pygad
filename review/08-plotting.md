# Review: pygad/plotting/

## Slice summary

**Severity counts:** 2 CRITICAL, 3 HIGH, 6 MEDIUM, 6 LOW

The plotting module contains two critical bugs: an undefined variable (`av_hist`) in `profiles.py:history()` that makes weighted history plots crash at runtime whenever `av` is passed, and an undefined variable (`AV`) when `av` is a non-string array. The `correlation_chart` function silently fails to share x-axes due to accessing uninitialized array slots. Colormap registration at module import time crashes on reload. Several deprecated matplotlib APIs (`ColorConverter`) and stale documentation round out the findings.

---

## Finding 1
**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/plotting/profiles.py:206`
**Description:** In `history()`, when `av` is not `None`, line 206 appends to `av_hist` which is never defined. Only `Q_hist` (line 200) is initialized as a list. This means any call to `history(s, qty=..., av=<non-None>)` raises `NameError: name 'av_hist' is not defined`. Since `flow_history` does not use `av`, this bug only triggers when users call `history` directly with an averaging quantity.

**Suggested fix:** Change `av_hist.append(...)` to `Q_hist.append(...)` on line 206, and on line 207 wrap the list in `UnitArr(Q_hist, Q.units / AV.units)` (or the appropriate combined units) since the weighted average has different units than the unweighted sum.

---

## Finding 2
**Severity:** CRITICAL
**Category:** bug
**Location:** `pygad/plotting/profiles.py:168-176`
**Description:** In `history()`, when `av` is not a string (e.g. it's already a `UnitArr`), the assignment `AV = s.get(av)` on line 170 is skipped (it's inside `if isinstance(av, str):`). Then line 171 references `AV` unconditionally, raising `NameError`. The guard on line 168 checks `if av is not None:` but the inner assignment only covers the string case.

**Suggested fix:** Add an `else` clause:
```python
if isinstance(av, str):
    AV = s.get(av)
else:
    AV = av
```

---

## Finding 3
**Severity:** HIGH
**Category:** bug
**Location:** `pygad/plotting/maps.py:866`
**Description:** In `correlation_chart()`, the rcParam restore at line 866 reads:
```python
mpl.rcParams["axes.formatter.useoffset"] = useoffset
```
This sets it to the *parameter* `useoffset` (which is `False` by default), NOT to the saved value `useoffset_sys` from line 865. The intent is to restore the original setting, but the code restores `False` unconditionally, silently clobbering the user's rcParam setting.

**Suggested fix:** Change line 866 to:
```python
mpl.rcParams["axes.formatter.useoffset"] = useoffset_sys
```

---

## Finding 4
**Severity:** HIGH
**Category:** bug
**Location:** `pygad/plotting/maps.py:892-893`
**Description:** In `correlation_chart()`, the inner loop at line 893 uses `sharex=axs[N-1,y]`, but `axs[N-1,y]` has not been created yet when `x < N-1`. The array `axs` is initialized with `np.empty((N,N), dtype=object)` which fills with `None`. Passing `sharex=None` to `add_subplot` silently does nothing, so axes that should share x-ranges actually do not. The chart still renders but column alignment is broken for all columns except the last.

**Suggested fix:** Either reverse the loop order (iterate x from N-1 downward), or create the bottom-row axes first, or defer the `sharex` via a two-pass approach.

---

## Finding 5
**Severity:** HIGH
**Category:** bug
**Location:** `pygad/plotting/colormaps.py:40,54,66,324`
**Description:** All colormap registrations (`age`, `BlackGreen`, `BlackPurple`, `isolum`) call `mpl.colormaps.register(name=..., cmap=...)` at module import time **without** `force=True`. If the module is imported twice (e.g. during testing, reload, or multiprocessing), `register` raises `ValueError: A colormap named "<name>" is already registered.` This crashes the import.

**Suggested fix:** Add `force=True` to every `mpl.colormaps.register()` call, or guard registration with a try/except, or use `mpl.colormaps.register(cmap, name=..., force=True)`.

---

## Finding 6
**Severity:** MEDIUM
**Category:** compat
**Location:** `pygad/plotting/general.py:423,509`
**Description:** `mpl.colors.ColorConverter.to_rgb(fontcolor)` is used in `make_scale_indicators` and `add_cbar`. While `ColorConverter` is not yet removed in matplotlib 3.11, it is a legacy API. The modern equivalent is `mpl.colors.to_rgb(fontcolor)` (a simple module-level function). Using the old class-based access may break in future matplotlib releases.

**Suggested fix:** Replace `mpl.colors.ColorConverter.to_rgb(fontcolor)` with `mpl.colors.to_rgb(fontcolor)` at both sites.

---

## Finding 7
**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/plotting/general.py:479-541` (`add_cbar`)
**Description:** `add_cbar` uses `mpl.colorbar.ColorbarBase` (line 523), which in matplotlib 3.11 is already an alias for `mpl.colorbar.Colorbar` and may be removed in future versions. The `ColorbarBase` name has been deprecated since matplotlib 3.6.

**Suggested fix:** Use `mpl.colorbar.Colorbar` instead of `mpl.colorbar.ColorbarBase`.

---

## Finding 8
**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/plotting/maps.py:159`
**Description:** In `plot_map()`, when `logscale=True`, `m = np.log10(m)` is called without checking for non-positive values. If `m` contains zeros or negative values (e.g. an empty region of the map), `np.log10` produces `-inf` and `NaN` respectively, which propagate through the rest of the pipeline. While `scale01` clips to `[0,1]`, NaN and `-inf` pass through unclamped and produce garbled pixels.

**Suggested fix:** Guard with `m = np.where(m > 0, np.log10(m), np.nan)` or similar, so non-positive values become NaN (rendered as the colormap's "bad" color) rather than `-inf`.

---

## Finding 9
**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/plotting/__init__.py:67-68`
**Description:** In `show_FoF_groups()`, the `colors` dtype check at line 68 reads:
```python
elif isinstance(colors, str) or colors.dtype in (str, str) \
        or colors.shape == (3,):
```
The tuple `(str, str)` checks against the Python `str` type twice — this is clearly a copy-paste error. The intent was likely to check for a numeric dtype (e.g. `float` or `np.floating`) or some other meaningful condition. As written, this branch is never entered via the `dtype` check, so a numpy array of strings is silently handled as the else branch (per-element indexing).

**Suggested fix:** Determine the intended type check (e.g. `np.floating` or just remove the duplicate and rely on `shape==(3,)`).

---

## Finding 10
**Severity:** MEDIUM
**Category:** qol
**Location:** `pygad/plotting/__init__.py:6-8` vs `pygad/__init__.py:85`
**Description:** The `pygad/plotting/__init__.py` docstring (and the `pygad/__init__.py` docstring at line 5-8) states that the plotting sub-module is "only imported automatically when in interactive mode." However, `pygad/__init__.py:85` does `from . import plotting` unconditionally — there is no interactive-mode guard. This stale documentation misleads users about the import behavior and the matplotlib dependency.

**Suggested fix:** Either update the docstrings to reflect unconditional import, or actually gate the import on interactive mode.

---

## Finding 11
**Severity:** MEDIUM
**Category:** correctness
**Location:** `pygad/plotting/profiles.py:195-196`
**Description:** Line 195 and 196 are identical duplicate statements:
```python
t_edges = UnitArr(np.linspace(0,float(now),N+1), time.units)
t_edges = UnitArr(np.linspace(0,float(now),N+1), time.units)
```
While harmless (the second overwrites the first with the same value), this is dead code that suggests an incomplete edit.

**Suggested fix:** Remove line 196.

---

## Finding 12
**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/plotting/general.py:283-290`
**Description:** In `scatter_map()`, when `cmap` is `None` (lines 283-286), the code resolves it, calls `cmap.set_bad(...)`, and sets `fontcolor`. Then at lines 289-290, if the user passed a string `cmap`, it's resolved again via `mpl.colormaps[cmap]` — but `set_bad` is **not** called for this path. This means NaN values in the scatter data render as transparent when the user passes a string colormap, but render as black/white when using the default. The behavior is inconsistent.

**Suggested fix:** After line 290, add `cmap.set_bad('w' if zero_is_white else 'k')` (or move the `set_bad` call to after all resolution paths).

---

## Finding 13
**Severity:** LOW
**Category:** qol
**Location:** `pygad/plotting/maps.py:389-395`
**Description:** Lines 389-395 inside `image()` contain a triple-quoted string that serves as dead/commented-out code:
```python
if qty is None and av is None:
    """
    if (len(s)!=0 and len(s.gas)==len(s)) ...
    """
```
The string is assigned to nothing and acts as a no-op. While it doesn't cause a runtime error, it's confusing dead code that suggests incomplete refactoring (the gas-only default case was removed).

**Suggested fix:** Remove the dead triple-quoted string block.

---

## Finding 14
**Severity:** LOW
**Category:** qol
**Location:** `pygad/plotting/__init__.py:84`, `pygad/plotting/maps.py:595,721,753`
**Description:** Four bare `except:` clauses swallow all exceptions silently (including `KeyboardInterrupt`, `SystemExit`). Specifically:
- `__init__.py:84`: catching `getattr` failure for `plot_center` — should catch `AttributeError`.
- `maps.py:595`: catching unit conversion failure for `rho_threshold` — should catch the specific conversion error.
- `maps.py:721`: catching `get_renderer()` failure — should catch `AttributeError` or the specific backend error.
- `maps.py:753`: catching text bounding box failure — should catch the specific exception.

**Suggested fix:** Replace each `except:` with a specific exception type (e.g. `except AttributeError:` or `except Exception:`).

---

## Finding 15
**Severity:** LOW
**Category:** qol
**Location:** `pygad/plotting/maps.py:124-126`
**Description:** `plot_map()` has debug prints at lines 124-125:
```python
print(colors)
print(colors.shape, m.shape)
```
These execute before the `ValueError` on line 126, printing raw array data and shapes to stdout whenever a shape mismatch occurs. This is diagnostic print left in production code.

**Suggested fix:** Remove the two `print` statements or convert to `logging.debug`.

---

## Finding 16
**Severity:** LOW
**Category:** qol
**Location:** `pygad/plotting/profiles.py:85`
**Description:** `prof[prof==0] = np.NaN` uses `==` on what could be a `UnitArr`. If `prof` is a `UnitArr` with units, the comparison `prof==0` creates a boolean mask (which is fine), but the right-hand side `np.NaN` is unitless. This may trigger a unit mismatch warning or error depending on the `UnitArr` implementation. Should use `np.nan` (standard) and ensure the assignment is compatible.

**Suggested fix:** Use `np.nan` (lowercase) and verify `UnitArr.__setitem__` accepts unitless NaN assignment.

---

## Finding 17
**Severity:** LOW
**Category:** improvement
**Location:** `pygad/plotting/maps.py:556-585` (`phase_diagram`)
**Description:** `phase_diagram()` uses `s['rho']` and `s['temp']` (line 583) which implicitly assumes these blocks exist on the snapshot's gas particles. The docstring says "from which to use the gas of" but no `s.gas` filter is applied. If `s` is the full snapshot (all types), `s['rho']` returns gas-only data (blocks are particle-type specific), but this relies on an implicit convention rather than being explicit.

**Suggested fix:** Use `s.gas['rho']` and `s.gas['temp']` explicitly, matching the docstring's intent.

---
