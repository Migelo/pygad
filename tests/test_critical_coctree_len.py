"""Regression test for CRITICAL #7: NameError in cOctree.__init__ verbose path.

`cOctree.__init__` printed ``len(s)`` instead of ``len(pos)`` when
``environment.verbose >= environment.VERBOSE_TALKY``, raising
``NameError: name 's' is not defined`` on every verbose tree construction.
"""
import numpy as np

from pygad import environment
from pygad.octree.coctree import cOctree

POS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)


def _build(pos, verbose):
    old = environment.verbose
    try:
        environment.verbose = verbose
        return cOctree(pos)
    finally:
        environment.verbose = old


def test_coctree_construction_talky_verbose():
    # Pre-fix this raised NameError: name 's' is not defined.
    tree = _build(POS, environment.VERBOSE_TALKY)
    assert tree.tot_num_part == len(POS)


def test_coctree_talky_tree_is_functional():
    tree = _build(POS, environment.VERBOSE_TALKY)
    ngbs = tree.find_ngbs_within([0.0, 0.0, 0.0], 0.1, POS)
    assert sorted(ngbs.tolist()) == [0]
    ngbs = tree.find_ngbs_within([0.0, 0.0, 0.0], 1.01, POS)
    assert sorted(ngbs.tolist()) == [0, 1, 2, 3, 5]
    assert tree.is_in_node([0.25, 0.25, 0.25])


def test_coctree_construction_quiet_verbose():
    tree = _build(POS, environment.VERBOSE_QUIET)
    assert tree.tot_num_part == len(POS)
    ngbs = tree.find_ngbs_within([0.0, 0.0, 0.0], 0.1, POS)
    assert sorted(ngbs.tolist()) == [0]
