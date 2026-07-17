"""Regression tests for CRITICAL #9: periodic_wrap dropping the last pixel.

pygad/analysis/vpfit.py:476-477 used `flux[starting_pixel:-1]` concatenated
with `flux[0:starting_pixel+1]`, which drops the last pixel of the array and
duplicates `flux[starting_pixel]`.  The wrap must be a true rotation (same
length, same multiset, np.roll-equivalent) for flux AND noise alike.
"""
import numpy as np

from pygad.analysis import periodic_wrap


def test_periodic_wrap_documented_example():
    # Reviewer's example: argmax is at index 1; pre-fix output was
    # [0.9, 0.7, 0.3, 0.5, 0.9] -- 0.8 lost, 0.9 duplicated.
    l = np.array([4000.0, 4001.0, 4002.0, 4003.0, 4004.0])
    flux = np.array([0.5, 0.9, 0.7, 0.3, 0.8])
    noise = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    wflux, wnoise, sp = periodic_wrap(l, flux, noise)

    assert sp == 1
    expected = np.roll(flux, -sp)  # [0.9, 0.7, 0.3, 0.8, 0.5]
    assert len(wflux) == len(flux)
    np.testing.assert_array_equal(wflux, expected)
    # same multiset: no element dropped or duplicated
    np.testing.assert_array_equal(np.sort(wflux), np.sort(flux))


def test_periodic_wrap_starting_pixel_zero_is_identity():
    # argmax at index 0: wrapping must be the identity rotation.
    l = np.arange(5.0)
    flux = np.array([0.9, 0.5, 0.7, 0.3, 0.8])
    noise = np.array([0.05, 0.01, 0.03, 0.04, 0.02])

    wflux, wnoise, sp = periodic_wrap(l, flux, noise)

    assert sp == 0
    np.testing.assert_array_equal(wflux, flux)
    np.testing.assert_array_equal(wnoise, noise)


def test_periodic_wrap_argmax_at_last_pixel():
    # Boundary case: argmax at the final pixel; rotation moves only that
    # pixel to the front.
    l = np.arange(5.0)
    flux = np.array([0.5, 0.3, 0.7, 0.8, 0.9])
    noise = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

    wflux, wnoise, sp = periodic_wrap(l, flux, noise)

    assert sp == len(flux) - 1
    np.testing.assert_array_equal(wflux, np.roll(flux, -sp))
    np.testing.assert_array_equal(np.sort(wflux), np.sort(flux))


def test_periodic_wrap_noise_rotated_consistently():
    # The noise array must receive exactly the same rotation as the flux.
    l = np.arange(6.0)
    flux = np.array([0.4, 0.6, 0.95, 0.8, 0.3, 0.7])
    noise = np.array([0.11, 0.12, 0.13, 0.14, 0.15, 0.16])

    wflux, wnoise, sp = periodic_wrap(l, flux, noise)

    assert sp == 2
    np.testing.assert_array_equal(wflux, np.roll(flux, -sp))
    np.testing.assert_array_equal(wnoise, np.roll(noise, -sp))
    assert len(wnoise) == len(noise)
