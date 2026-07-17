"""Regression test for CRITICAL #1 (review/REVIEW.md).

``SnapshotCache._fill_star_from_info`` (pygad/snapshot/snapshotcache.py)
wrapped the star-formation-file read in ``except Exception: ... exit(1)``,
hard-killing the whole Python process on any failure (e.g. a missing or
corrupt file), followed by an unreachable ``return``.  The fix re-raises the
original exception instead.

Because the bug terminates the interpreter, the failure path is exercised in
a subprocess: pre-fix the subprocess exits with code 1 and an *empty* stderr
(no traceback); post-fix it still exits non-zero, but with the real exception
traceback on stderr.
"""
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SNIPPET = r'''
import pygad as pg
from pygad.snapshot.snapshotcache import SnapshotCache

snap = pg.Snapshot("pygad/snaps/snap_M1196_4x_470", load_double_prec=False)
cache = SnapshotCache("pygad/snaps/snap_M1196_4x_470")
cache.snapshot = snap
# np.loadtxt raises OSError for this missing file; the handler must let the
# original exception propagate instead of calling exit(1).
cache._fill_star_from_info("/nonexistent/star_form.ascii", 1.0)
print("UNREACHABLE: _fill_star_from_info returned despite the failed read")
'''


def test_failed_star_form_read_raises_instead_of_exit1():
    result = subprocess.run(
        [sys.executable, "-c", SNIPPET],
        capture_output=True,
        cwd=REPO,
    )
    # The read failure must still be fatal (the handler keeps re-raising) ...
    assert result.returncode != 0
    # ... but because the genuine exception propagated, not because of a bare
    # exit(1): stderr must carry the traceback naming the real exception.
    assert b"Traceback" in result.stderr, (
        "no traceback on stderr -> the process was hard-killed by exit(1); "
        "stdout was: %r" % result.stdout
    )
    last_line = result.stderr.strip().splitlines()[-1]
    assert last_line.startswith((b"OSError", b"FileNotFoundError")), last_line
    # The handler's diagnostic output is still produced (semantics kept) ...
    assert b"error importing star_form file" in result.stdout
    # ... and the function did not silently fall through and return.
    assert b"UNREACHABLE" not in result.stdout
