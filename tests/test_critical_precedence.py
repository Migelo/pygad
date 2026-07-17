"""Regression tests for CRITICAL #11: operator-precedence bug in the
``_apply_to_block`` shape validation of ``Translation`` and ``Rotation``.

``if not len(block.shape)==2 and block.shape[1]==N:`` parses as
``(not 2D) and (cols == N)`` instead of ``not (2D and cols == N)``, so

* a 2D block with the WRONG column count passed validation silently
  (for ``Translation`` it even broadcasts a (N,1) block to (N,3)), and
* a 1D block crashed with ``IndexError`` on ``block.shape[1]`` instead of
  the intended ``ValueError``.
"""
import numpy as np
import pytest

from pygad.transformation import Rotation, Translation


def _translation():
    return Translation([1.0, 2.0, 3.0])


def _rotation():
    # proper cyclic permutation (det == +1): [x,y,z] -> [y,z,x]
    return Rotation([[0, 1, 0], [0, 0, 1], [1, 0, 0]])


# (a) 2D block with the wrong number of columns must be rejected by the
#     validator with a ValueError. Pre-fix the guard is skipped: Translation
#     silently broadcasts (raises nothing), Rotation fails later inside numpy
#     with an unrelated message.
def test_translation_wrong_columns_rejected():
    with pytest.raises(ValueError, match='The block has to have shape'):
        _translation().apply_to_block(np.ones((4, 1)))


def test_rotation_wrong_columns_rejected():
    with pytest.raises(ValueError, match='The block has to have shape'):
        _rotation().apply_to_block(np.ones((4, 2)))


# (b) 1D block must raise ValueError, not IndexError from block.shape[1].
def test_translation_1d_block_rejected():
    with pytest.raises(ValueError, match='The block has to have shape'):
        _translation().apply_to_block(np.ones(3))


def test_rotation_1d_block_rejected():
    with pytest.raises(ValueError, match='The block has to have shape'):
        _rotation().apply_to_block(np.ones(3))


# (c) Correctly shaped 2D blocks are still accepted and transformed as before.
def test_translation_correct_block_unchanged():
    block = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    out = _translation().apply_to_block(block)
    assert np.allclose(out, [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]])


def test_rotation_correct_block_unchanged():
    block = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = _rotation().apply_to_block(block)
    assert np.allclose(out, [[2.0, 3.0, 1.0], [5.0, 6.0, 4.0]])
