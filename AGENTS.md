# AGENTS.md

Guidance for AI agents (and humans) working in this repository. Everything here was verified against the tree on 2026-07-16.

## What this is

**PyGAD** — a Python + small C++/ctextension framework for analyzing Gadget/Gizmo/Arepo SPH simulation snapshots (cosmology, gas physics, absorption spectra, plotting). ~25k LOC. Dist package name is **`pygadmpa`**; you import it as **`pygad`**. Upstream: github.com/Migelo/pygad (formerly bitbucket.org/broett/pygad).

This is **not** the genetic-algorithm `pygad` package — do not confuse the two.

## Hard environment constraints

- **Python 3.10–3.13 only.** `setup.cfg` `requires_python = >=3.8.0` but the CI matrix and classifiers are 3.10–3.13. **Python 3.14 is too new** — several pins (`numpy<2.6`, `scipy<=1.18.0`, `matplotlib<3.12.0`, `astropy<=8.0.1`, `h5py<=3.16.0`) and the in-tree code (`tarfile.extractall` without `filter=`, etc.) will break. Do not use the system 3.14; create a 3.12 venv.
- **System libs required to build the C extension:** GSL (`libgsl-dev` / `brew install gsl`), a C++11 compiler (`g++`/`gcc`/`build-essential`).
- **macOS extra:** Apple clang has no `-fopenmp`/`-lgomp`. `setup.py` expects **libomp** — `brew install libomp` (it searches `/opt/homebrew/opt/libomp` and `/usr/local/opt/libomp`). Without it `pip install -e .` fails with `RuntimeError: libomp was not found`.

## Setup (verified working recipe)

```bash
# 1. venv on a supported Python (uv is available in this env)
uv venv --python 3.12 .venv

# 2. install deps + build the C extension (setup.py compiles pygad/C/cpygad.so
#    itself via the Extension — you do NOT run `make` manually)
VIRTUAL_ENV="$PWD/.venv" uv pip install -e .

# 3. smoke check (first import auto-downloads ~400MB+ of auxiliary data — see below)
.venv/bin/python -c "import pygad; print(pygad.__version__)"
```

The canonical scripted setup is `.devcontainer/setup.sh` (used by the devcontainer `postCreateCommand`); it does `apt-get install libgsl-dev g++ gcc build-essential clang-format`, then `python -m venv .venv && pip install -e . && pip install ipython ipykernel jupyter`.

### Known setup landmines
- **`pygad/C/Makefile` hardcodes `CC = g++-13` on Darwin.** `g++-13` usually does not exist on macOS. If you build via the Makefile instead of `setup.py`, override: `make -C pygad/C CC=g++`. (`setup.py`'s own Extension build is what `pip install` uses and does not hit this.)
- **First `import pygad` downloads auxiliary data** (see next section) — expect ~400 MB and network access. It is slow but must succeed for import to work.
- **GSL_HOME** defaults in the Makefile can emit a spurious `-I/include` flag if the env var is unset; harmless but noisy.

## Auxiliary data (required at runtime)

`pygad/__init__.py::_ensure_auxiliary_data()` runs on first import and downloads + extracts four tarballs into `pygad/`:

| Archive | Extracts to | Purpose |
|---|---|---|
| `z_0.000_highres.tar.gz` | `pygad/CoolingTables/` | cooling-function tables |
| `iontbls.tar.gz` | `pygad/iontbls/` | ionisation tables |
| `snaps.tar.gz` | `pygad/snaps/` | test snapshots (~388 MB) |
| `bc03.tar.gz` | `pygad/bc03/` | Bruzual & Charlot 2003 SSP model |

- **Base URL** defaults to `https://github.com/Migelo/pygad/releases/download/pygad-data`. Override with env var **`PYGAD_DATA_BASE_URL`**.
- **Skip bootstrap** with **`PYGAD_SKIP_DATA_BOOTSTRAP=1`** (devcontainer script only; the Python import path always checks).
- For CI, archives are fetched under the `pygad-data` release tag and extracted with `find data/*.tar.gz | xargs tar xzf -C pygad`.

## Verification gates

| Gate | Command | What it checks | Notes |
|---|---|---|---|
| **Doctest suite (primary)** | `.venv/bin/python runTestsAll.py` | doctests across every submodule | Takes a few minutes. **Exit code = number of failed doctests** (0 = pass). Needs auxiliary data + the C extension. |
| **Regression tests** | `.venv/bin/python tests/test_bugfixes.py` | 3 historical bug fixes (units div, derive cache, cfg undefine) | Fast (~1 s). No auxiliary data needed beyond import. |
| **Plotting smoke** | `cp tests/QuickStart.py . && .venv/bin/python QuickStart.py` | end-to-end plotting pipeline | Needs a non-interactive backend; heavy. |
| **Lint** | `ruff check pygad/` | static checks | **Currently 754 findings.** ruff is available but the CI ruff step is **commented out**. Don't treat ruff cleanliness as a gate today. |
| **Import health** | `.venv/bin/python -c "import pygad"` | bootstrap + data + C ext load | The devcontainer script's final check. |

### Doctest infra caveat (important)
Tests are **doctests**, run through a **custom numeric checker** in `pygad/doctest/doctest.py` (`NumericOutputChecker`). It **normalizes away** array `shape=(...)` metadata and numpy scalar wrappers (`np.float64`, `np.void`) before comparison, and compares floats with `rel_tol=2e-6`/`abs_tol=1e-12`. Consequence: doctests will **not** catch shape regressions or dtype changes (e.g. `float32` vs `float64`). See `DOCTEST_FAILURES.md` for the history of why each normalization was added. When you add doctests, be aware they are lenient on shape/type.

## Repo layout

```
pygad/
  __init__.py          package bootstrap: gc tuning, auxiliary-data download/extract, star-imports of submodules
  environment.py       verbosity levels, gc_full_collect, module_dir
  tools.py             large misc collection (traced-gas readers, zoom prep, info files) ~49KB
  mem_debug.py         memory-tracking helpers
  units/               Unit / UnitArr / UnitQty / UnitScalar — the unit system (units.py parser, unit_arr.py array subclass)
  utils/               utils.py, safe_eval.py (sandboxed AST evaluator), geo.py, term.py
  snapshot/            THE CORE: snapshot.py (~75KB) + snapshotcache.py (~95KB), derived.py, derive_rules.py, masks.py, sim_arr.py
  gadget/              binary+HDF5 IO: lowlevel_file.py (~37KB format 1/2/3 parsing), handler.py, config.py
  config/              *.cfg — gadget.cfg (block names), units.cfg (~152 units), derived.cfg (derived-quantity rules)
  binning/             SPH binning: core.py, cbinning.py (C-backed), mapping.py, oneDbinning.py
  octree/              octree.py (pure py) + coctree.py (C-backed via ctypes)
  kernels/             SPH kernel definitions.py + integral.py; uses pygad/C/cpygad.so
  C/                   C++ extension (src/, include/) → cpygad.so; kernels + Voigt + octree; loaded via ctypes in C/__init__.py
  physics/             cosmology.py, cooling.py, quantities.py
  ssp/                 bc_ssp.py — Bruzual & Charlot stellar populations
  transformation/      rotation/translation matrices
  cloudy/              treecol.py (HEALPix tree column density), cloudy_tables.py
  analysis/            absorption_spectra.py (~84KB, largest file), fit_profiles.py (~60KB Voigt fitting), halo.py, properties.py, sph_eval.py, vpfit.py
  plotting/            maps.py, general.py, profiles.py, colormaps.py (matplotlib; lazy-import contract — see below)
  doctest/             the custom doctest runner + numeric checker
  cmdtool/             gCache.py — snapshot caching CLI backend
tracing/               gtracepy3, gtracegaspy3 — gas-tracing pipeline scripts
commands/              CmdCatalog.py, CmdStarform.py, CmdTraceGas.py
bin/                   ginsp, gconv, gCache3, gCatalog3, gStarform3 — installed console scripts
tests/                 test_bugfixes.py, QuickStart.py
runTestsAll.py         doctest runner entry point
setup.py               builds cpygad.so via Extension; Darwin libomp detection; versioneer
```

## Conventions

- **`from ... import *` is pervasive** (every `__init__.py`, `tools.py`, the big analysis files). This is why ruff reports ~474 F405 / ~89 F403. Names are exported at package level — `import pygad` exposes `Snapshot`, `UnitArr`, ` cosmology`, etc. directly. Follow this pattern in-module; do **not** add a second, narrower import style alongside it.
- **Submodule import order matters.** `pygad/__init__.py` imports submodules in a specific sequence (utils → environment → tools → snapshot → transformation → units → plotting → cloudy → analysis → binning → ssp → octree → gadget → kernels → physics) to avoid circular imports. A new submodule that depends on another must be imported after its dependency. The docstring in `__init__.py` documents this contract.
- **The unit system is load-bearing.** `UnitArr`/`UnitQty`/`UnitScalar` are `ndarray` subclasses that carry `.units`. Arithmetic propagates units; many bugs in this codebase come from silently dropping units or comparing unitful to unitless. Preserve units in new code.
- **Particle types** are indexed 0–5 (gas, dm, …). Derived quantities come from rules in `config/derived.cfg` + `snapshot/derive_rules.py`; `_host_derive_block` computes/caches them (note the `cache=True` parameter — see `tests/test_bugfixes.py` fix 1).
- **C extension via ctypes**, not a CPython extension module. `pygad/C/__init__.py` does `cdll.LoadLibrary` on `cpygad*.so` and sets `argtypes`/`restype` by hand. If you change a C signature in `pygad/C/src/`, you **must** update the ctypes declarations in `C/__init__.py`, `binning/cbinning.py`, and `octree/coctree.py`. The `.so` is built by `setup.py`'s `BuildCtypesLibrary` and forced to the `.so` name on all platforms (intentional, for `dlopen`).
- **Plotting** (`pygad/plotting`) must be imported explicitly (`import pygad.plotting`) for use, though `__init__.py:85` does import it unconditionally at present (the "lazy/interactive-only" docstring is stale — noted in the review).
- **Doctests are the test suite.** Most functions carry `>>>` examples; the numeric checker makes them tolerant. When changing a function's numeric output, update its doctest expected values and re-run `runTestsAll.py`.

## Known issues & where to look

- **`DOCTEST_FAILURES.md`** — 19 historical failures and their fixes (NumPy-2 `.ptp`, Py3.12 `collections.Iterable`, SciPy 1.14 `interp2d`→`RegularGridInterpolator`, a 15.7 TiB allocation in `flow_rates`, etc.). Read before touching numerics.
- **`TODO.md`** — acknowledged, intentionally-open limitations (EAGLE block names, `__array_wrap__` vs `__numpy_ufunc__`, halo-finder interface). Don't "fix" these without discussion.
- **`review/REVIEW.md`** — a full code review (136 findings: 13 CRITICAL / 29 HIGH / 54 MEDIUM / 40 LOW) with file:line locations and suggested fixes, plus per-slice `review/01-…09-…`. **Read this before editing** — it documents latent crash bugs (e.g. `exit(1)` in `snapshotcache.py`, `eval()` in the galaxy setter, `==`-instead-of-`=` sites, off-by-ones in `vpfit.py`/`fit_profiles.py`) and NumPy-2 compat hazards (`.ptp()` in `binning/core.py:371`).

## Working in this repo — quick rules

1. **Always use the `.venv` (Python 3.12).** Never the system Python.
2. After any source change to `pygad/`, re-run `.venv/bin/python runTestsAll.py` and `.venv/bin/python tests/test_bugfixes.py`. A change to a C signature requires a rebuild (`pip install -e .` again).
3. Do not add `print()` diagnostics to library code — several existing ones are flagged as bugs to remove.
4. Prefer specific `except SomeError:` over bare `except:` / `except Exception:` — bare-except swallowing is the single biggest debuggability problem here.
5. When you see `==` where an assignment is intended, or `not a==b and c==d` without parentheses, treat it as a bug, not style.
6. The `reviewer` agent specialist is **unavailable in this environment** (404s); use the general-purpose `task` agent for review/research work.
