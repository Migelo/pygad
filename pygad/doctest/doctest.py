import doctest as _doctest
import re
from doctest import (BLANKLINE_MARKER, DONT_ACCEPT_BLANKLINE,
                     DONT_ACCEPT_TRUE_FOR_1, ELLIPSIS, NORMALIZE_WHITESPACE,
                     _ellipsis_match)
from math import isclose

NUMERIC_REL_TOL = 2e-6
NUMERIC_ABS_TOL = 1e-12


def string_to_numbers(string: str):
    """
    Convert a string to a list of numbers.
    Use refex to find numbers, integers and floats.
    https://stackoverflow.com/a/29581287/633093
    """
    numbers = re.findall(
        r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", string)
    return numbers


def _without_array_shape(string: str):
    """Remove NumPy's optional array-shape suffix from a representation."""
    return re.sub(r",\s*shape=\([^)]*\)", "", string)


def _without_numpy_scalar_wrapper(string: str):
    """Normalize NumPy scalar reprs, which vary between NumPy releases."""
    string = re.sub(r"np\.(?:float|int|uint)\d+\(([^()]*)\)", r"\1", string)
    return re.sub(r"np\.void\((\([^)]*\)), dtype=.*\)", r"\1", string)


def check_output_numbers(self, want, got, optionflags):
    """
    Return True iff the actual output from an example (`got`)
    matches the expected output (`want`).  These strings are
    always considered to match if they are identical; but
    depending on what option flags the test runner is using,
    several non-exact match types are also possible.  See the
    documentation for `TestRunner` for more information about
    option flags.
    """

    # If `want` contains hex-escaped character such as "\u1234",
    # then `want` is a string of six characters(e.g. [\,u,1,2,3,4]).
    # On the other hand, `got` could be another sequence of
    # characters such as [\u1234], so `want` and `got` should
    # be folded to hex-escaped ASCII string to compare.
    got = self._toAscii(got)
    want = self._toAscii(want)

    # Handle the common case first, for efficiency:
    # if they're string-identical, always return true.
    if got == want:
        return True

    # The values True and False replaced 1 and 0 as the return
    # value for boolean comparisons in Python 2.3.
    if not (optionflags & DONT_ACCEPT_TRUE_FOR_1):
        if (got, want) == ("True\n", "1\n"):
            return True
        if (got, want) == ("False\n", "0\n"):
            return True

    # <BLANKLINE> can be used as a special sequence to signify a
    # blank line, unless the DONT_ACCEPT_BLANKLINE flag is used.
    if not (optionflags & DONT_ACCEPT_BLANKLINE):
        # Replace <BLANKLINE> in want with a blank line.
        want = re.sub(r"(?m)^%s\s*?$" % re.escape(BLANKLINE_MARKER), "", want)
        # If a line in got contains only spaces, then remove the
        # spaces.
        got = re.sub(r"(?m)^[^\S\n]+$", "", got)
        if got == want:
            return True

    # Normalize representation-only differences before applying the standard
    # textual checks.  These are not differences in the computed result.
    got = _without_numpy_scalar_wrapper(_without_array_shape(got))
    want = _without_numpy_scalar_wrapper(_without_array_shape(want))

    # This flag causes doctest to ignore any differences in the
    # contents of whitespace strings.  Note that this can be used
    # in conjunction with the ELLIPSIS flag.
    if optionflags & NORMALIZE_WHITESPACE:
        got = " ".join(got.split())
        want = " ".join(want.split())
        if got == want:
            return True

    # The ELLIPSIS flag says to let the sequence "..." in `want`
    # match any substring in `got`.
    if optionflags & ELLIPSIS:
        if _ellipsis_match(want, got):
            return True

    # Compare the two strings number by number
    # print("using the homemade check_output_numbers function")
    # print(f"got = {got}")
    # print(f"want = {want}")
    # NumPy versions differ in whether they append ``shape=(...)`` to array
    # representations.  It is metadata, not part of the numerical result.
    got_numbers = string_to_numbers(got)
    want_numbers = string_to_numbers(want)
    # print(f"got_numbers = {got_numbers}")
    # print(f"want_numbers = {want_numbers}")
    if len(got_numbers) == len(want_numbers) and all(
        isclose(float(got_number), float(want_number),
                rel_tol=NUMERIC_REL_TOL, abs_tol=NUMERIC_ABS_TOL)
        for got_number, want_number in zip(got_numbers, want_numbers)
    ):
        return True

    # We didn't find any match; return false.
    return False


class NumericOutputChecker(_doctest.OutputChecker):
    """Output checker that also accepts numerically close values."""

    def check_output(self, want, got, optionflags):
        return check_output_numbers(self, want, got, optionflags)


def testmod(m=None, name=None, globs=None, verbose=None, report=True,
            optionflags=0, extraglobs=None, raise_on_error=False,
            exclude_empty=True):
    """Run a module's doctests with :class:`NumericOutputChecker`.

    This mirrors the standard :func:`doctest.testmod` API closely, while
    avoiding a process-wide replacement of ``OutputChecker.check_output``.
    """
    if m is None:
        m = _doctest._normalize_module(None)
    if name is None:
        name = getattr(m, "__name__", None)
    if globs is None:
        globs = getattr(m, "__dict__", {})
    if extraglobs:
        globs = globs.copy()
        globs.update(extraglobs)

    finder = _doctest.DocTestFinder(exclude_empty=exclude_empty)
    runner = _doctest.DocTestRunner(
        checker=NumericOutputChecker(),
        verbose=verbose,
        optionflags=optionflags,
    )

    for test in finder.find(m, name=name, globs=globs):
        runner.run(test, clear_globs=False)

    if report:
        runner.summarize(verbose=verbose)
    return _doctest.TestResults(runner.failures, runner.tries)
