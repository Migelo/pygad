"""Regression test for CRITICAL #3: `e[-4]` slice typo in
`pygad.tools.read_traced_gas` (type-4 branch: left region, turned into a star).

The buggy branch appended the scalar `e[-4]` instead of the 4-element sequence
`e[-4:]`, corrupting traced-gas star records and breaking downstream shape
assumptions.
"""

import pickle

import numpy as np

from pygad.tools import read_traced_gas


def _write_trace(tmp_path, records):
    path = tmp_path / "gtracegas.pkl"
    with open(str(path), "wb") as f:
        pickle.dump(records, f)
    return str(path)


def _build_records():
    # event layouts (see read_traced_gas docstring):
    #   (re-)enter: [a, mass, metals, j_z, T]          -> 5 elements
    #   leave:      [a, mass, metals, j_z, T, vel]     -> 6 elements
    #   out:        [(a, r_max), (a, z_max)]           -> 4 elements
    #   star form:  [a, mass, metals, j_z]             -> 4 elements
    enter = [0.10, 1.0, 0.010, 100.0, 1.0e4]
    leave = [0.20, 1.0, 0.011, 110.0, 1.1e4, 42.0]
    out = [0.30, 12.0, 0.40, 5.0]
    star = [0.50, 0.9, 0.020, 120.0]
    return {
        101: enter,  # len  5 -> type 1: gas in region
        102: enter + leave + out,  # len 15 -> type 2: gas out of region
        103: enter + star,  # len  9 -> type 3: star formed in region
        104: enter + leave + out + star,  # len 19 -> type 4: star outside
    }, star


def test_read_traced_gas_type4_star_event_is_4element_sequence(tmp_path):
    records, star = _build_records()
    tr = read_traced_gas(_write_trace(tmp_path, records))

    assert set(tr.keys()) == {101, 102, 103, 104}
    assert [tr[ID][0][0] for ID in (101, 102, 103, 104)] == [1, 2, 3, 4]

    # Every event entry (after the [type, n_cycles] header) must be a
    # 1-D sequence of 4-6 elements -- never a bare scalar. Pre-fix, the
    # type-4 record's final (star formation) entry is the scalar e[-4].
    for ID, rec in tr.items():
        for event in rec[1:]:
            event = np.asarray(event)
            assert event.ndim == 1, "ID %d: scalar event %r" % (ID, event)
            assert event.shape[0] in (4, 5, 6)

    # The final star-formation event of each star record must be exactly
    # the 4-element [a, mass, metals, j_z] sequence.
    for ID in (103, 104):
        assert list(tr[ID][-1]) == star


def test_read_traced_gas_star_records_stack_to_regular_2d_array(tmp_path):
    records, _ = _build_records()
    tr = read_traced_gas(_write_trace(tmp_path, records))

    # Stacking the star-formation events of the type-3 and type-4 records
    # must yield a regular (2, 4) float array. Pre-fix the type-4 entry is a
    # scalar, making the stack ragged -> ValueError (or object dtype).
    star_events = [tr[103][-1], tr[104][-1]]
    stacked = np.array(star_events, dtype=float)
    assert stacked.shape == (2, 4)
    assert stacked.dtype.kind == "f"
    np.testing.assert_allclose(stacked[0], stacked[1])
