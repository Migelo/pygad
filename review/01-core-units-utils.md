# Slice 1 — Core, Units, Utils — Review

## Slice summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH | 2 |
| MEDIUM | 6 |
| LOW | 5 |

This slice covers 11 files in the pygad package bootstrap, unit system, and shared utilities. The most significant finding is a confirmed data-structure bug in `tools.py:693` where a missing colon turns a slice into a single-element access. A HIGH-severity tarfile deprecation warning will become a breaking change in Python 3.14. The unit system and safe eval are generally well-designed, with only minor robustness issues. Several QOL and compatibility improvements are noted.

---

## Finding 1

**Severity:** CRITICAL  
**Category:** bug  
**Location:** `pygad/tools.py:693`  
**Description:** In `read_traced_gas`, the `t == 19` branch (particles that left the region and turned into a star) appends `e[-4]` instead of `e[-4:]`. This is a missing colon in a slice expression. All other type branches use proper slices (`e[-4:]`, `e[-10:-4]`, `e[-14:-8]`, etc.). With `e[-4]`, a **single scalar element** (the second-to-last element of `e`) is appended to `new` instead of the expected 4-element array. Every subsequent consumer of this data (e.g. `fill_gas_from_traced`, `fill_derived_gas_trace_qty`) that assumes the last entry is a 4-element array will get incorrect data or crash with a shape mismatch.

**Suggested fix:** Change `e[-4]` to `e[-4:]`:
```python
# line 693
new += [e[-14:-8], e[-8:-4], e[-4:]]
```

---

## Finding 2

**Severity:** HIGH  
**Category:** compat  
**Location:** `pygad/__init__.py:168`  
**Description:** `archive.extractall(module_dir)` is called without the `filter` parameter. As of Python 3.12 this emits a `DeprecationWarning`, and Python 3.14 will default to `'data'` filtering which may reject members or modify metadata (e.g. setting permissions on extracted files). Since `_download_and_extract` runs at import time for every first-time user, this warning will be visible and in Python 3.14 the extraction may silently change behavior or fail.

**Suggested fix:** Add `filter='data'` (or `'fully_trusted'` if the tarballs contain expected metadata):
```python
archive.extractall(module_dir, filter='data')
```

---

## Finding 3

**Severity:** HIGH  
**Category:** bug  
**Location:** `pygad/utils/utils.py:42`  
**Description:** `DevNull.read(size=0)` is missing `self` as its first parameter. It is defined as a plain function inside the class body rather than an instance method. Any call to `DevNull().read()` raises `TypeError: read() takes 0 positional arguments but 1 was given`. Although `DevNull` is documented as "write-only" and `read()` is likely never called in practice, the method signature is broken.

**Suggested fix:** Add `self`:
```python
def read(self, size=0):
    raise IOError("Invalid operation")
```

---

## Finding 4

**Severity:** MEDIUM  
**Category:** bug  
**Location:** `pygad/tools.py:680`  
**Description:** `len(sub) / 15` uses true division (`/`) which returns a float (e.g. `1.0`) in Python 3. This float is placed into a list `[tt, len(sub) / 15]` that is used as metadata by downstream consumers. While downstream code may tolerate a float, it is unexpected for a "number of cycles" count and is almost certainly an integer-division oversight from the Python 2 era.

**Suggested fix:** Use floor division:
```python
new = [[tt, len(sub) // 15]]
```

---

## Finding 5

**Severity:** MEDIUM  
**Category:** correctness  
**Location:** `pygad/utils/geo.py:27`  
**Description:** The `angle()` function computes `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))` without clamping the cosine to `[-1, 1]`. When either vector is zero-length, `arccos` receives `NaN`. When vectors are nearly parallel (cosine slightly outside `[-1, 1]` due to floating-point rounding), `arccos` returns `NaN`. For the zero-vector case, NumPy emits a `RuntimeWarning: invalid value encountered in scalar divide` and returns `NaN`. The function should either clamp or handle zero-length inputs explicitly.

**Suggested fix:**
```python
cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
cos = np.clip(cos, -1.0, 1.0)
return UnitArr(np.arccos(cos), 'rad')
```

---

## Finding 6

**Severity:** MEDIUM  
**Category:** correctness  
**Location:** `pygad/__init__.py:101-106` / `pygad/__init__.py:175-195`  
**Description:** `_ensure_auxiliary_data` has a classic TOCTOU race condition: it checks `os.path.exists(path)`, then downloads and extracts if missing. Two concurrent imports (e.g. in a multiprocessing or multi-threaded context) could both see the file missing and start downloading simultaneously, potentially corrupting the extracted data or failing. No file locking is used during the download/extract cycle. This is a theoretical concern for interactive use but could become a practical problem in HPC batch environments.

**Suggested fix:** Use a lock file (e.g. `fcntl.flock` on Unix, or a directory-based lock) around the check-download-extract sequence.

---

## Finding 7

**Severity:** MEDIUM  
**Category:** qol  
**Location:** `pygad/utils/term.py`  
**Description:** `get_terminal_size` uses `os.popen('stty size')` which is Unix-only. On Windows or in environments where `stty` is not available (e.g. certain CI runners, Jupyter notebooks, or redirected stdin), this call fails silently and returns `(0, 0)`. While `os.get_terminal_size()` has been available since Python 3.3 and handles fallbacks natively, this code predates it.

**Suggested fix:** Replace with `os.get_terminal_size()` which provides a cross-platform `os.terminal_size` named tuple with fallback logic:
```python
def get_terminal_size():
    try:
        size = os.get_terminal_size()
        return (size.lines, size.columns)
    except OSError:
        return (0, 0)
```

---

## Finding 8

**Severity:** MEDIUM  
**Category:** qol  
**Location:** `pygad/units/units.py:586-587` and `pygad/units/units.py:668-669`  
**Description:** Two bare `except:` / `except: raise` clauses that exist only to re-raise the original exception. In `define_from_cfg` (line 586-587):
```python
except:
    raise
```
This catches and re-raises every exception including `KeyboardInterrupt` and `SystemExit`, which is an anti-pattern. The clause in `Unit()` (line 668-669) is identical. Both are functionally equivalent to removing the `try/except` entirely.

**Suggested fix:** Remove both bare `except: raise` clauses. If the intent is to catch and re-raise only specific exceptions, use `except Exception: raise` (or better, just remove the clause).

---

## Finding 9

**Severity:** MEDIUM  
**Category:** correctness  
**Location:** `pygad/__init__.py:101-106`  
**Description:** `_human_size` produces nonsensical output for negative byte counts: `_human_size(-1)` returns `"-1.0 B"`. While unlikely in practice, negative bytes indicate a bug upstream and the function should at least handle `0` bytes cleanly. Currently `_human_size(0)` returns `"0.0 B"` which is technically correct but arguably `"0 B"` would be more natural.

**Suggested fix:** Add a guard at the top:
```python
def _human_size(nbytes):
    if nbytes <= 0:
        return '0 B'
    ...
```

---

## Finding 10

**Severity:** LOW  
**Category:** compat  
**Location:** `pygad/utils/utils.py:31`  
**Description:** `DevNull.__init__` sets `self.softspace = 0`. The `softspace` attribute was removed in Python 3 (`PEP 3138`). While setting it does not cause an error, it is a vestige of Python 2 file objects and may confuse readers.

**Suggested fix:** Remove the `self.softspace = 0` line.

---

## Finding 11

**Severity:** LOW  
**Category:** qol  
**Location:** `pygad/tools.py:641`  
**Description:** `import pickle as pickle` is a redundant aliased import. The `as pickle` is unnecessary since `pickle` is already the module name. This is a very minor style issue.

**Suggested fix:** Simplify to `import pickle`.

---

## Finding 12

**Severity:** LOW  
**Category:** qol  
**Location:** `pygad/tools.py:23-27`  
**Description:** `from .analysis import *`, `from .snapshot import *`, `from .transformation import *`, `from .units import *`, and `from .utils import *` are wildcard imports in `tools.py`. This pollutes the namespace of the `tools` module with potentially hundreds of names, making it harder to trace where a given name comes from and increasing the risk of name collisions.

**Suggested fix:** Import only the specific names needed, or at minimum document which names are expected to come from each star-import.

---

## Finding 13

**Severity:** LOW  
**Category:** qol  
**Location:** `pygad/utils/geo.py:51`  
**Description:** In `dist()`, when `pos is None` and `arr` is not already an ndarray, the code does `arr = np.array(arr)` but only after checking `not isinstance(arr, np.ndarray)`. This check is correct but the comment says "includes UnitArr!" which is misleading — the check succeeds for UnitArr (which IS an ndarray subclass), so `pos = [0]*arr.shape[-1]` is always reached when `pos is None`.

**Suggested fix:** Minor — the comment is slightly confusing but the logic is correct. No change needed, or clarify the comment.

---

## Finding 14

**Severity:** LOW  
**Category:** compat  
**Location:** `pygad/utils/safe_eval.py:136-139`  
**Description:** The `Evaluator.__init__` `bin_op` dictionary maps `ast.Div` to `op.truediv` by default (when `truediv=True`). However, if someone instantiates `Evaluator(truediv=False)`, the code attempts to use `op.div`, which does not exist in Python 3. This would raise an `AttributeError` at construction time. While no existing code passes `truediv=False`, the API contract is broken for that parameter value.

**Suggested fix:** Remove support for `truediv=False` entirely, or replace `op.div` with a fallback:
```python
if truediv:
    self.bin_op = {ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ...}
else:
    # Python 3 doesn't have op.div; closest approximation
    div_func = lambda a, b: int(a / b)
    self.bin_op = {ast.Div: div_func, ...}
```
Alternatively, raise `NotImplementedError` for `truediv=False`.

---

## Finding 15

**Severity:** LOW  
**Category:** qol  
**Location:** `pygad/tools.py:1087-1144`  
**Description:** A large block of code inside `fill_derived_gas_trace_qty` is wrapped in a triple-quoted string (lines 1087–1144), effectively commented out. This dead code computes "metal_gain_in" and "metal_gain_out" blocks. It references undefined variables (`gididx`, `trididx`, `trtype`) suggesting it was never finished. This should be removed or moved to a proper comment/docstring.

**Suggested fix:** Delete the dead code block or convert it to a `# TODO` comment describing the intended functionality.
