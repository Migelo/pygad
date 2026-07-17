"""Regression test for CRITICAL #8 (review/REVIEW.md).

pygad/analysis/fit_profiles.py (fit_profiles_sat) jiggles the best-fit
params in place (``params += 0.02 * (2*np.random.rand(...) - 1)``) and
re-optimises with BFGS to obtain a covariance estimate.  The unfixed code
then

1. evaluates ``chisq_new`` at the *jiggled* params instead of the
   re-optimised ``soln.x`` (so a genuine BFGS improvement is compared
   against the wrong point), and
2. leaves ``params`` jiggled when the jiggled chisq is rejected, with no
   rollback to the pre-jiggle best -- while ``cov = soln.hess_inv``
   describes ``soln.x``.  The reported N/b/l are hence silently shifted by
   the random jiggle and no longer match the reported chisq/covariance.

The test drives the real ``fit_profiles_sat`` on a synthetic Voigt
absorption line twice: once with the jiggle forced to zero (reference) and
once with a maximal positive jiggle (+0.02 on every parameter).  On the
unfixed code the maximal jiggle is rejected everywhere and the returned
params are all left shifted by exactly +0.02 with an inflated chisq; the
fix restores/evaluates the true optimum so both runs agree.

Run with the project venv:

    .venv/bin/python -m pytest tests/test_critical_jiggle.py
"""
import sys
import types

import numpy as np

import pygad  # noqa: F401  (ensures package init before submodule import)

# pygad/analysis/fit_profiles.py is legacy code whose module-level bare
# imports (``from physics import tau_to_flux, wave_to_vel`` and ``from utils
# import read_h5_into_dict``) reference helpers that no longer exist anywhere
# in the package, so the module is not importable as-is.  Those names are
# only used by the Spectrum class, never by fit_profiles_sat/model_tau, so
# provide minimal stand-ins purely to make the module loadable; every code
# path exercised below is the real one.
if "physics" not in sys.modules:
    _physics_stub = types.ModuleType("physics")
    _physics_stub.tau_to_flux = lambda tau: np.exp(-tau)
    _physics_stub.wave_to_vel = lambda *args, **kwargs: None
    sys.modules["physics"] = _physics_stub
if "utils" not in sys.modules:
    _utils_stub = types.ModuleType("utils")
    _utils_stub.read_h5_into_dict = lambda *args, **kwargs: {}
    sys.modules["utils"] = _utils_stub

from pygad.analysis.absorption_spectra import lines
from pygad.analysis.fit_profiles import fit_profiles_sat, model_tau

LINE = "H1215"
L0 = float(lines[LINE]["l"].split()[0])  # 1215.6701 Angstrom
P_TRUE = np.array([12.5, 20.0, L0])  # logN, b [km/s], line center [Angstrom]
SNR = 100.0
CHISQ_LIM = 2.5


def _make_spectrum():
    """Single Voigt line from known truth plus seeded Gaussian noise."""
    waves = np.linspace(L0 - 4.0, L0 + 4.0, 321)
    flux = np.exp(-model_tau(lines[LINE], P_TRUE, waves, "Voigt"))
    noise = np.full(len(waves), 1.0 / SNR)
    rng = np.random.default_rng(42)
    return waves, flux + rng.normal(0.0, noise), noise


def _fit_with_forced_jiggle(monkeypatch, waves, flux, noise, rand_value):
    """Run fit_profiles_sat with np.random.rand pinned to ``rand_value``.

    The only np.random.rand call on the fitting path is the param jiggle
    ``0.02 * (2*np.random.rand(len(params)) - 1)``:
      rand_value=0.5 -> zero jiggle (reference fit at the found optimum),
      rand_value=1.0 -> +0.02 on every parameter; on the unfixed code this
    displaces chisq above the best value, so the jiggle is rejected and the
    params are left corrupted.
    """
    monkeypatch.setattr(
        np.random, "rand", lambda *shape: np.full(shape, rand_value)
    )
    return fit_profiles_sat(
        LINE, waves, flux, noise, chisq_lim=CHISQ_LIM, max_lines=10
    )


def _main_region_mask(res):
    """Mask selecting the lines of the region containing the true line."""
    ireg = res["region"][np.argmin(np.abs(res["l"] - L0))]
    return res["region"] == ireg


def test_rejected_jiggle_keeps_best_params(monkeypatch):
    waves, flux, noise = _make_spectrum()
    ref = _fit_with_forced_jiggle(monkeypatch, waves, flux, noise, 0.5)
    jig = _fit_with_forced_jiggle(monkeypatch, waves, flux, noise, 1.0)

    m_ref, m_jig = _main_region_mask(ref), _main_region_mask(jig)
    assert m_ref.sum() >= 1 and m_jig.sum() == m_ref.sum()
    for key in ("N", "b", "l"):
        # the reported best-fit params must not depend on the random jiggle:
        # pre-fix every value in the jiggled run is left shifted by +0.02
        assert np.allclose(jig[key][m_jig], ref[key][m_ref], atol=1e-2, rtol=0), (
            "returned %s shifted by the rejected jiggle: %s vs %s"
            % (key, jig[key][m_jig], ref[key][m_ref])
        )


def test_rejected_jiggle_keeps_chisq_consistent(monkeypatch):
    waves, flux, noise = _make_spectrum()
    ref = _fit_with_forced_jiggle(monkeypatch, waves, flux, noise, 0.5)
    jig = _fit_with_forced_jiggle(monkeypatch, waves, flux, noise, 1.0)

    # the returned chisq must describe the returned (best) params, not the
    # jiggled ones: pre-fix it is inflated (0.62 -> 2.78 in the main region)
    assert len(jig["Chisq"]) == len(ref["Chisq"])
    assert np.all(jig["Chisq"] <= ref["Chisq"] + 1e-2), (
        "returned chisq inflated by the rejected jiggle: %s vs %s"
        % (jig["Chisq"], ref["Chisq"])
    )


def test_reference_fit_quality(monkeypatch):
    # sanity of the synthetic problem: the un-jiggled reference fit of the
    # region containing the true line must reach the acceptance threshold
    waves, flux, noise = _make_spectrum()
    ref = _fit_with_forced_jiggle(monkeypatch, waves, flux, noise, 0.5)
    assert np.all(ref["Chisq"][_main_region_mask(ref)] < CHISQ_LIM)
