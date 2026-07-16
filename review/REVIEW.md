# PyGAD — Consolidated Code Review

**Scope:** Full read-only review of the PyGAD astrophysics Gadget-snapshot analysis framework (~25k LOC Python + a small C++/ctypes extension), partitioned into 9 disjoint slices and reviewed by 9 parallel `task` review agents, corroborated by a `ruff` static-analysis pass.

**Baseline (green at review time):**
- Environment bootstrapped with `uv` — Python 3.12.13, numpy 2.5.1, scipy 1.18.0, matplotlib 3.11.0, astropy 8.0.1, h5py 3.16.0. C extension `pygad/C/cpygad.so` built and loads.
- `tests/test_bugfixes.py` → **PASS**; full doctest suite `runTestsAll.py` → **return code 0** (all 118 snapshot + 6 analysis + ... doctests pass). This is the regression baseline.
- `ruff check pygad/` → **754 findings** (see §5).

**Findings totals:** **13 CRITICAL · 29 HIGH · 54 MEDIUM · 40 LOW = 136 findings**

> Items already documented as fixed in `DOCTEST_FAILURES.md` (1–19) and known limitations in `TODO.md` were excluded from re-reporting. Detailed per-slice reports live in `review/01-…` through `review/09-…`.

---

## 1. Executive summary

| Slice | Files | C | H | M | L | Report |
|-------|-------|--:|--:|--:|--:|--------|
| 1 — core / units / utils | `__init__.py`, `environment.py`, `tools.py`, `mem_debug.py`, `units/*`, `utils/*` | 1 | 2 | 6 | 5 | [01](01-core-units-utils.md) |
| 2 — snapshot framework | `snapshot/{snapshot,snapshotcache,derived,derive_rules,masks,sim_arr}.py` | 2 | 3 | 6 | 5 | [02](02-snapshot.md) |
| 3 — gadget IO + config | `gadget/*`, `config/*.cfg` | 2 | 4 | 5 | 4 | [03](03-gadget-io.md) |
| 4 — binning / octree / kernels | `binning/*`, `octree/*`, `kernels/*`, `C/__init__.py` | 2 | 3 | 5 | 3 | [04](04-binning-octree-kernels.md) |
| 5 — analysis: spectra + fitting | `analysis/{absorption_spectra,fit_profiles,profiles}.py` | 1 | 3 | 5 | 4 | [05](05-analysis-spectra.md) |
| 6 — analysis: halo + properties + vpfit | `analysis/{halo,properties,sph_eval,vpfit,analysis}.py` | 1 | 2 | 7 | 5 | [06](06-analysis-halo.md) |
| 7 — physics / ssp / transformation / cloudy | `physics/*`, `ssp/*`, `transformation/*`, `cloudy/*` | 2 | 4 | 6 | 4 | [07](07-physics-ssp-trans-cloudy.md) |
| 8 — plotting | `plotting/*` | 2 | 3 | 6 | 6 | [08](08-plotting.md) |
| 9 — scripts / build / CI / doctest | `cmdtool/`, `doctest/`, `tracing/`, `commands/`, `bin/`, `tests/`, `setup.py`, `Dockerfile`, CI | 0 | 5 | 8 | 4 | [09](09-scripts-build-ci-doctest.md) |

The codebase is mature and largely correct (the doctest suite passing is meaningful), but it carries a cluster of **latent crash bugs** that the doctests don't exercise, several **silent-correctness bugs** in numerical paths, and broad **error-swallowing** that makes the library hard to debug. NumPy-2 / SciPy-1.14 / Python-3.14 compatibility hazards are the most forward-looking risk.

---

## 2. CRITICAL findings (13) — fix first

These are confirmed crashes or silent data corruption.

1. **`pygad/snapshot/snapshotcache.py:1549-1550`** — `exit(1)` inside an `except Exception:` handler hard-kills the entire Python process on any failure during traced-gas reading; an unreachable `return` follows. Replace with `raise`. *(Slice 2)*
2. **`pygad/snapshot/snapshotcache.py:315`** — `galaxy` setter runs `eval('gx.' + str(value), globals(), locals())` on user-supplied input → arbitrary code execution; also raises a confusing `AttributeError` for any typo. Replace with `getattr(gx, value)` after validating against `gadget.families`. *(Slice 2)*
3. **`pygad/tools.py:693`** — `e[-4]` instead of `e[-4:]` (missing colon) appends a single scalar where every other branch appends a 4-element array. Corrupts traced-gas star records and crashes downstream shape assumptions. *(Slice 1)*
4. **`pygad/gadget/config.py:83`** — `global std_nameto_HDF5` (missing underscore) vs the real module name `std_name_to_HDF5`. The `global` declaration is a no-op for the intended name. *(Slice 3)*
5. **`pygad/gadget/lowlevel_file.py:724`** — `ptype = int(name[-1])` takes only the last char of `PartTypeN`; breaks for `PartType10+` (parsed as type `0`). Use `int(name[len('PartType'):])`. *(Slice 3)*
6. **`pygad/binning/cbinning.py:180-187` (+ 317-322, 444-449, 577-585)** — non-contiguous arrays passed to the C extension: `if pos.base is not None: pos.copy()` discards the returned contiguous copy, so the original non-contiguous buffer is read by C (undefined behavior / wrong data). Affects all four C-binning functions for `hsml`/`dV`/`qty`. Fix: `pos = pos.copy()`. *(Slice 4)*
7. **`pygad/octree/coctree.py:188`** — `len(s)` where the parameter is `pos` → `NameError` whenever an octree is built with verbose ≥ TALKY (ruff F821). Fix: `len(pos)`. *(Slice 4)*
8. **`pygad/analysis/fit_profiles.py:1058-1080`** — after jiggle+BFGS, `chisq_new` is evaluated on the *jiggled* `params` (mutated in-place) not `soln.x`, and on rejection `params` is left jiggled with no rollback; `cov = soln.hess_inv` then mismatches `params`. Save `params_best`, evaluate on `soln.x`, restore on reject. *(Slice 5)*
9. **`pygad/analysis/vpfit.py:476-477`** — `periodic_wrap` uses `flux[starting_pixel:-1]`, dropping the last pixel and duplicating `flux[starting_pixel]`. Verified: `[0.5,0.9,0.7,0.3,0.8]` → `[0.9,0.7,0.3,0.5,0.9]` (loses 0.8). Corrupts every wrapped spectrum (flux + noise). *(Slice 6)*
10. **`pygad/physics/cosmology.py:195-201`** — `FLRWCosmo.__eq__` compares 4 of 6 fields to **`self`** (`self._Omega_m == self._Omega_m`, …) instead of `other`. Any two cosms equal in `h_0`+`Omega_Lambda` compare equal regardless of `Omega_m`/`Omega_b`/`sigma_8`/`n_s`. *(Slice 7)*
11. **`pygad/transformation/transformation.py:312,417`** — operator-precedence bug: `if not len(block.shape)==2 and block.shape[1]==...:` parses as `(not 2D) and (cols match)`, so a 2D block with the **wrong** column count passes validation silently, and a 1D block crashes with `IndexError` instead of a clean `ValueError`. Add parentheses. *(Slice 7)*
12. **`pygad/plotting/profiles.py:206`** — `history()` appends to `av_hist` which is never defined → `NameError` for any weighted-history plot. *(Slice 8)*
13. **`pygad/plotting/profiles.py:168-176`** — `AV = s.get(av)` only assigned inside `if isinstance(av, str)`; a non-string `av` (e.g. a `UnitArr`) makes the unconditional `AV` reference on the next line `NameError`. Add an `else: AV = av`. *(Slice 8)*

---

## 3. HIGH findings (29)

### Correctness / crash
- **`lowlevel_file.py:250,254`** — `write_header` packs `N_part`/`N_part_all` with signed `'6i'` but `read_header` unpacks unsigned `'6I'`; writing >2.1e9 particles of a type → `struct.error`. Use `'6I'`. *(3)*
- **`lowlevel_file.py:170`** — SIMBA detection keyed on hardcoded `N_part_all[0] == 1016261591` (one specific run). *(3)*
- **`lowlevel_file.py:508`** — `el_size == block.dimension * ...` is a comparison not assignment; latent `UnboundLocalError` masked by an early return (dead branch). *(3)*
- **`config/gadget.cfg:96` + `derived.cfg:35`** — known TODO: block `Z` (scalar `Metallicity`) is treated as the full `elements` array; EAGLE element abundances silently skipped. *(3)*
- **`core.py:371`** — `Map.vol_tot()` uses removed ndarray `.ptp()` method (NumPy 2.0); use `np.ptp(self.extent, axis=1)`. *(4)*
- **`core.py:444`** — `smooth()` always reshapes the conv kernel to 2D `(pxs,pxs)` → crashes on 1D/line grids; use `(pxs,)*D`. *(4)*
- **`coctree.py:135`** — `restypes = c_int` typo (should be `restype`) silently leaves the C return type default. *(4)*
- **`fit_profiles.py:466,472`** — reduced χ² denominator is `np.count_nonzero(residuals)` (excludes zero-residual pixels, inflating the statistic) instead of `len − n_params`. *(5)*
- **`fit_profiles.py:1504-1507`** — `EquivalentWidth` off-by-one: `dEW[i-1]` overwrites the 3rd-to-last element and never writes `dEW[-1]`; last pixel's EW silently dropped (~15% undercount in the verified case). *(5)* (Same bug independently found at `vpfit.py:323-327` — Slice 6.)
- **`fit_profiles.py:1270-1283`** — `find_regions` merges with a fixed `range(len-1)` while `np.delete`-ing in place → adjacent regions stay un-merged. Use a `while` loop. *(5)*
- **`analysis.py:8,41`** — module is dead-broken on import: bare `from units import UnitArr` (not relative) + `NameError` on undefined `profile`. Unloaded today only because `__init__` doesn't import it. *(6)*
- **`cosmology.py:278-281`** — `sigma_8` setter asserts bare `sigma_8` (not `value`) → `NameError` whenever set. *(7)*
- **`transformation.py:134,140`** — `prost`/`self._prost` typos (`post`/`self._post`); `__init__(post=...)` and base `__copy__` are dead-broken (subclasses override). *(7)*
- **`maps.py:866`** — `correlation_chart` rcParam restore writes `useoffset` (the param, default `False`) instead of the saved `useoffset_sys` → silently clobbers the user's matplotlib setting. *(8)*
- **`maps.py:892-893`** — `sharex=axs[N-1,y]` references not-yet-created (`None`) axes → columns silently fail to share x-ranges. *(8)*
- **`colormaps.py:40,54,66,324`** — `mpl.colormaps.register(...)` without `force=True` → `ValueError` on reload/reimport (breaks test/multiprocess). *(8)*

### Build / CI / packaging
- **`bin/gconv:16`** — `--format default=470` (int) with string `choices`; argparse doesn't validate the default, so `470` flows silently to the snapshot writer. *(9)*
- **`pygad/cmdtool/gCache.py:257`** — `.py` extension check uses `[:-3]` (everything *but* last 3 chars) compared to `'.py'` → always appends `.py` → `script.py.py`. *(9)*
- **`bin/gCache3,gCatalog3,gStarform3` + `gCache.py:380,419`** — bare `exit(1)` instead of `sys.exit(1)`. *(9)*
- **`.github/workflows/workflow.yml:14,16`** — `actions/checkout@v7` / `actions/setup-python@v6` tags **do not exist** (latest are v4 / v5) → CI fails on every push. *(9)*

---

## 4. Cross-cutting themes

These recur across slices and are the highest-leverage cleanup targets.

1. **`==` instead of `=` (assignment-via-comparison).** Appears **3×**: `absorption_spectra.py:883` (`spatial_res == ...`), `lowlevel_file.py:508` (`el_size == ...`), and the operator-precedence variant in `transformation.py`. The boolean is discarded and the intended assignment silently does nothing.
2. **Bare `except:` / `except Exception:` swallowing (ruff: 23 E722 + many more).** Worst offenders: `snapshotcache.py` (4 handlers silently returning `None`/defaults, masking corruption/OOM), `absorption_spectra.py` (4 bare `except:`), `C/__init__.py:14` (masks missing/wrong-arch `.so`). None log or warn.
3. **`exit(1)` in library code** — `snapshotcache.py:1550` kills the host process from inside a data-reading path.
4. **Latent `NameError`/`AttributeError` in untested branches** — `av_hist`, `AV`, `sigma_8` setter, `prost`/`_prost`, `len(s)`, `e.message` (Py2-only at `halo.py:560`). The doctest suite passes because these paths are never exercised.
5. **Off-by-one / last-pixel-drop in EW/spectrum math** — `vpfit.py:476` (periodic_wrap) and `fit_profiles.py:1504` / `vpfit.py:323` (EquivalentWidth). Numerical results are silently wrong.
6. **NumPy-2 / SciPy compat** — remaining `.ptp()` method (`core.py:371`), `scipy.ndimage.filters` (`core.py:92`, removed-target), legacy `interp1d` (`kernels/integral.py`), `tarfile.extractall` without `filter=` (Py3.14-breaking, `__init__.py:168`).
7. **Dead/broken code left behind** — disabled `_correct_virial_info` with a `$$$Korrekturfaktor` debug print (`snapshotcache.py:814`/907), `if True or rs == 0.0` dead guard (`snapshotcache.py:532`), infinite-recursion `copy()` shadowed by a real one (`transformation.py:284,358`), commented-out blocks in `tools.py:1087` and `snapshot.py:891-951`.
8. **Mutable default args** — `fit_profiles` `logN_bounds=[...]`/`b_bounds=[...]` (`vpfit.py:64`, `fit_profiles.py:180`).
9. **ctypes correctness** — `restype` truncation (`coctree.py` size_t↔c_int), discarded `.copy()` before passing pointers to C (the CRITICAL #6), duplicate `restype`/`argtypes` lines.
10. **Personal/hardcoded paths & noise** — `commands/Cmd*.py` hardcode `/mnt/hgfs/Astro/...`; debug `print`s in `maps.py:124`, `halo.py:1133`, `tracing/gtracepy3:843`.

---

## 5. ruff static corroboration (`ruff check pygad/` → 754)

Top rules (mechanical, fast to clear):

| Rule | Count | Note |
|------|------:|------|
| F405 import-star usage | 474 | consequence of `from … import *` everywhere |
| F403 import-star | 89 | — |
| E701 multiple-statements-on-one-line | 47 | |
| E741 ambiguous name `l` | 45 | wavelength variable; style only |
| **E722 bare `except`** | **23** | real risk — see theme #2 |
| **F401 unused import** | **22** | e.g. 8 in `C/__init__.py`, 12 in `__init__.py` |
| **F841 unused variable** | **21** | mostly in `fit_profiles.py` (dead locals) |
| E402 import not at top | 12 | |
| E731 lambda assignment | 7 | |
| **F821 undefined name** | **3** | `analysis.py:41 profile`, `coctree.py:188 s` — both confirmed real |
| F823 ref-before-assign | 1 | `fit_profiles.py:495 b_bounds` |
| E721 / E711 / E743 / F507 / F402 / F811 | ~10 | |

The **F821/F823/E722** rows are the subset that overlaps with reviewer-flagged real bugs; the F405/F403/E741 rows are mostly stylistic debt from pervasive star-imports. **Uncommenting ruff in CI (currently commented) and fixing the E722/F821/F823 set would catch a class of these regressions automatically.**

---

## 6. Recommended fix priority

1. **Immediate (data-integrity / crash):** CRITICAL #1, #3, #6, #8, #9, #10, #11 and the EW off-by-ones (#3 HIGH set). These silently corrupt results or kill processes in normal use.
2. **Short-term (compat / CI):** `core.py:371` `.ptp()` (already broken on the installed NumPy 2.5.1), CI action tags (`workflow.yml`), `__init__.py:168` tarfile `filter=`, `scipy.ndimage` import.
3. **Hygiene sweep (high leverage, low risk):** clear the bare-`except`/`exit(1)` cluster, the `==`-vs-`=` trio, dead/disabled code blocks, `e.message`, mutable defaults. Re-enable ruff in CI on at least E722/F821/F823/F401.
4. **Numerical correctness review:** the χ² denominator, `find_regions` merge loop, cooling per-particle loop (`cooling.py:291`), metallicity bin-edge discontinuity (`bc_ssp.py:307`).

---

## 7. Verification status at handoff

- Reviewers were **read-only**; **no source files were modified** by this pass — only these report files under `review/`.
- The working tree is unchanged except for the `review/` directory and `.venv/`. `git status` was clean before the review; baseline tests remain green.
- Every CRITICAL/HIGH line number above was grounded by the reviewers' `read`/`grep` of the actual files; many were additionally reproduced with `.venv/bin/python` snippets (periodic_wrap, `__eq__`, EW off-by-one, `e[-4]`, `exit` semantics).
