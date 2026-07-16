# Review Slice 09: Scripts, Build, CI, Doctest

## Slice summary

| Severity | Count |
|----------|-------|
| CRITICAL | 0     |
| HIGH     | 5     |
| MEDIUM   | 8     |
| LOW      | 4     |

Overview: The CLI and scaffolding layer has several correctness bugs: a type-mismatch in `gconv`'s argparse default, a reversed string-slice bug in `gCache.py`'s `.py` extension check, `exit(1)` used instead of `sys.exit(1)` in three bin/ scripts (bypassing cleanup), and non-existent GitHub Action tag versions. The Dockerfile uses a Debian release past EOL. The custom doctest checker's output-normalization design choice could silently mask regressions, but is intentional per `DOCTEST_FAILURES.md` and not a new bug.

---

## Finding 1

**Severity:** HIGH
**Category:** bug
**Location:** `bin/gconv:16`
**Description:** The `--format` argument has `default=470` (an integer) but `choices=['1','2','3','hdf5','HDF5']` (all strings). argparse does **not** validate the default against `choices`, so `args.format` silently becomes `470` (int) when no `--format` is given. Downstream at line 63, `int(args.format)` works on `470` (identity), but the integer `470` is not a valid Gadget format — it passes silently to `pygad.snapshot.write` which presumably expects `1`, `2`, `3`, or something meaningful. The user gets no warning and a garbage conversion happens silently.

Verified: argparse does **not** reject `default=470` even though `470` is not in `choices`. An explicit `--format 470` *does* get rejected.

**Suggested fix:** Change `default=470` to `default='2'` (or whichever is the sensible default Gadget format), and remove the downstream `int()` conversion since the value is already a string that gets converted.

---

## Finding 2

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/cmdtool/gCache.py:257`
**Description:** The `.py` extension check uses `script_file[:-3].lower() != '.py'`. The slice `[:-3]` returns everything *except* the last 3 characters. For `"myscript.py"` this yields `"myscript"`, which is compared to `'.py'` — they are never equal for any filename of length > 3. The condition is therefore **always True**, meaning `'.py'` is always appended to any filename longer than 3 characters, producing paths like `"myscript.py.py"`.

The intended logic is `script_file[-3:].lower() == '.py'` (check that the *last 3 characters* are `.py`, and only append if they are not).

Verified: for `"script.py"`, `script_file[:-3]` → `"script"` (compared to `'.py'` → not equal → appends `.py`).

**Suggested fix:** Change line 257 to:
```python
if len(script_file) > 3 and script_file[-3:].lower() != '.py':
```

---

## Finding 3

**Severity:** HIGH
**Category:** bug
**Location:** `bin/gCache3:14,19,27`; `bin/gCatalog3:34,39,52`; `bin/gStarform3:22,27,43`
**Description:** These three scripts use bare `exit(1)` instead of `sys.exit(1)`. The builtin `exit()` is intended for the interactive interpreter and may behave differently in some environments (e.g., it can be shadowed by site-packages, or behave unexpectedly in embedded Python). In contrast, `ginsp` and `gconv` correctly use `sys.exit()`. This inconsistency means these scripts may fail to terminate properly in non-standard Python environments.

**Suggested fix:** Add `import sys` at the top and replace all `exit(1)` with `sys.exit(1)`. (gCache3 already imports sys; gCatalog3 and gStarform3 do not.)

---

## Finding 4

**Severity:** HIGH
**Category:** bug
**Location:** `pygad/cmdtool/gCache.py:380,419`
**Description:** `gCache.py` uses bare `exit(1)` at lines 380 and 419 in addition to `sys.exit(1)` elsewhere in the same file. Same issue as Finding 3 — inconsistent termination method within a single file.

**Suggested fix:** Replace `exit(1)` with `sys.exit(1)` at lines 380 and 419.

---

## Finding 5

**Severity:** HIGH
**Category:** compat
**Location:** `.github/workflows/workflow.yml:14,16`
**Description:** The workflow references `actions/checkout@v7` and `actions/setup-python@v6`. As of July 2026, the latest stable tags for these actions are `@v4` and `@v5` respectively. Tags `@v7` and `@v6` do not exist, which will cause the CI pipeline to fail on every push.

**Suggested fix:** Update to `actions/checkout@v4` and `actions/setup-python@v5`.

---

## Finding 6

**Severity:** MEDIUM
**Category:** bug
**Location:** `runTestsAll.py:27`
**Description:** `sys.exit(res.failed)` exits with the *count* of failed doctests rather than a fixed exit code. With 0 failures → exit 0 (correct). With 3 failures → exit 3. CI systems treat any non-zero as failure, so this works functionally, but it's unconventional and could confuse log parsers or downstream scripts that expect a boolean-like exit code (0 or 1). More importantly, if a caller checks `exit_code == 1` specifically, it would miss failures.

**Suggested fix:** Use `sys.exit(1 if res.failed else 0)` or `sys.exit(min(res.failed, 1))`.

---

## Finding 7

**Severity:** MEDIUM
**Category:** compat
**Location:** `Dockerfile:1`
**Description:** The base image `python:3.11-slim-buster` uses Debian Buster, which reached EOL in August 2022. Security updates are no longer provided, making the Docker image insecure and potentially causing package installation failures as mirrors drop Buster support.

Additionally, line 45 uses `--NotebookApp.token=` and `--NotebookApp.password=`, which are deprecated in Jupyter 7+. The correct flags are `--ServerApp.token=` and `--ServerApp.password=`.

**Suggested fix:** Update to `python:3.11-slim-bookworm` (or newer). Replace `--NotebookApp.` with `--ServerApp.`.

---

## Finding 8

**Severity:** MEDIUM
**Category:** correctness
**Location:** `commands/CmdCatalog.py:25`; `commands/CmdStarform.py` (similar pattern)
**Description:** `write_line()` references `snap_fname` and `snap_num` as bare names, but they are not function parameters — they rely on globals set in the gcache3-init section. While this works because gCache.py uses `exec(command_str, globals(), locals())`, it makes the code fragile and non-reusable. Any refactor that changes the global scope will break these functions silently.

**Suggested fix:** Pass `snap_fname` and `snap_num` as explicit parameters to `write_line`.

---

## Finding 9

**Severity:** MEDIUM
**Category:** correctness
**Location:** `commands/CmdCatalog.py:51-52,55`; `commands/CmdStarform.py:74-75`; `commands/CmdTraceGas.py:261-262`
**Description:** All three command scripts contain hardcoded absolute paths to `/mnt/hgfs/Astro/...` and `/mnt/hgfs/AstroDaten/...`. These are VMware shared-folder paths and will not exist on any other system. If these scripts are ever used as documentation examples or accidentally committed as-is, they will fail immediately. These are clearly personal development artifacts.

**Suggested fix:** Replace hardcoded paths with environment variables or command-line arguments. If these are purely personal scripts, they should not be in the repository, or at minimum should have a clear comment marking them as templates.

---

## Finding 10

**Severity:** MEDIUM
**Category:** correctness
**Location:** `setup.py:28`
**Description:** `subprocess.run(["make", "clean"], cwd=setup_dir + "/pygad/C", check=False)` runs `make clean` at module-import time (top-level scope in `setup.py`). If `make` is not installed or the Makefile has issues, this silently fails (due to `check=False`), but it still runs an external command during Python's initial setup scan, which can slow down `pip install` and cause confusing output.

**Suggested fix:** Move the `make clean` into the `BuildCtypesLibrary.run()` method so it only executes during the actual build_ext step.

---

## Finding 11

**Severity:** MEDIUM
**Category:** bug
**Location:** `pygad/C/Makefile:1,8`
**Description:** Two Makefile portability issues:
1. **Line 8:** `CC = g++-13` is hardcoded on Darwin. Many macOS systems have `g++-14`, `g++-15`, or only `clang++` (via Xcode Command Line Tools). The build will fail if `g++-13` is not installed.
2. **Line 1:** `GSL_HOME := ${GSL_HOME} /opt/local/ ...` — if the environment variable `GSL_HOME` is unset, it expands to an empty string, so the first path searched is literally the empty string (or the current directory), producing a spurious `-I/include` and `-L/lib` flag that may cause warnings.

**Suggested fix:** For (1), use `CC ?= g++-13` or detect available compiler. For (2), guard the variable: `GSL_HOME := $(if $(GSL_HOME),$(GSL_HOME) /opt/local/ /opt/homebrew/opt/gsl/,/opt/local/ /opt/homebrew/opt/gsl/)`.

---

## Finding 12

**Severity:** MEDIUM
**Category:** correctness
**Location:** `.github/workflows/workflow.yml:45`
**Description:** The "Extract auxiliary data" step runs `find data/*.tar.gz | xargs -I% tar xzf % -C pygad` **without** `set -euo pipefail` (unlike the download step above it). If `find` returns no results (glob doesn't match), `xargs` receives no input and does nothing — silently. If a tar extraction fails, the pipeline continues. Also, the glob `data/*.tar.gz` is shell-expanded, so if no `.tar.gz` files exist, the unexpanded glob is passed to `find`, which will error.

**Suggested fix:** Add `set -euo pipefail` at the top of the run block. Use `shopt -s nullglob` before the find, or add an explicit check like `[ -n "$(ls data/*.tar.gz 2>/dev/null)" ]`.

---

## Finding 13

**Severity:** MEDIUM
**Category:** qol
**Location:** `renovate.json:11,13,24`
**Description:** `platformAutomerge: true` appears three times in the config. In Renovate, `platformAutomerge` is a boolean that must be configured at the top level (not nested inside `lockFileMaintenance` or `packageRules`). The `automerge` + `automergeType: "pr"` properties within those blocks already control automerging. The `platformAutomerge` key inside nested blocks may be silently ignored, or could cause validation warnings depending on Renovate version.

Additionally, the `devDependencies` rule (lines 27-38) has `matchPackageNames: ["/lint/", "/prettier/"]` which only matches packages whose names literally contain `/lint/` or `/prettier/`. This is extremely narrow and unlikely to match any real Python dev dependencies (like ruff, pytest, mypy, etc.).

**Suggested fix:** Remove `platformAutomerge` from nested blocks (keep it only at top level if desired). Widen `matchPackageNames` to match actual dev tool names, or use a different match strategy.

---

## Finding 14

**Severity:** MEDIUM
**Category:** correctness
**Location:** `setup.py:88-89`
**Description:** `BuildCtypesLibrary.get_ext_filename` forces `.so` extension on all platforms: `ext_name.replace(".", os.sep) + ".so"`. On macOS, the convention is `.dylib`, though Python's `ctypes.cdll.LoadLibrary` can load `.so` files on macOS too (via `dlopen`), so this works. However, it means `pygad.C.cpygad.so` will appear as a `.so` file on macOS, which is unusual and may confuse users or tools that expect platform-appropriate extensions.

**Suggested fix:** This is intentional for ctypes compatibility. Consider adding a comment explaining why `.so` is forced on all platforms.

---

## Finding 15

**Severity:** LOW
**Category:** improvement
**Location:** `pygad/doctest/doctest.py:40-90`
**Description:** The custom `NumericOutputChecker` normalizes output by stripping array shapes (`_without_array_shape`) and numpy scalar wrappers (`_without_numpy_scalar_wrapper`) before comparison. These are acknowledged fixes (per `DOCTEST_FAILURES.md` items 3, 8, 15) that prevent spurious failures from repr format changes across NumPy versions. However, this normalization means doctests will **never** catch shape regressions or type changes (e.g., a function returning `np.float32` instead of `np.float64` would pass). This is a known trade-off documented in the fixes, not a new bug.

**Suggested fix:** Consider adding an optional strict mode or a small set of doctests that verify output types/shapes explicitly (without the checker's normalization).

---

## Finding 16

**Severity:** LOW
**Category:** qol
**Location:** `.github/workflows/workflow.yml:47-52`
**Description:** The ruff linting step is entirely commented out, including a stale `--target-version=py37` flag (the project's `requires_python` is `>=3.8`). If re-enabled, it should target the project's actual minimum Python version.

**Suggested fix:** If linting is desired, uncomment and update `--target-version=py38`.

---

## Finding 17

**Severity:** LOW
**Category:** improvement
**Location:** `pygad/tracing/gtracepy3:208`
**Description:** `if phy_units == True:` uses `== True` for a boolean comparison instead of just `if phy_units:`. The variable `phy_units` is defined at module level as `False`. While functionally correct, `== True` is an anti-pattern that ruff flags.

**Suggested fix:** Use `if phy_units:`.

---

## Finding 18

**Severity:** LOW
**Category:** qol
**Location:** `pygad/tracing/gtracepy3:843`
**Description:** Debug print statement `print(f"starting!!! {args.start=}, {args.end=}, {-args.step=}")` appears to be a leftover from development. It is immediately followed by a similar but different print on line 845, suggesting copy-paste iteration. These debug prints produce noisy output in production use.

**Suggested fix:** Remove or gate behind `--verbose`.

---

## Finding 19 (Informational)

**Severity:** LOW
**Category:** improvement
**Location:** `setup.py:89`
**Description:** `BuildCtypesLibrary.get_ext_filename` returns `ext_name.replace(".", os.sep) + ".so"`. The `os.sep`-based path construction produces `pygad/C/cpygad.so` on Unix but `pygad\C\cpygad.so` on Windows. Since Python extensions on Windows use `.pyd`, this path would be incorrect on Windows. The project likely does not support Windows, but there is no explicit guard.

**Suggested fix:** Add a platform check or document that only Unix-like platforms are supported.
