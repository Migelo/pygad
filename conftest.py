# conftest.py – monkey-patch doctest to use pygad's numeric checker, but only
# when pytest is actually collecting doctests. Importing pygad unconditionally
# at startup triggers pygad/__init__.py, which runs the auxiliary-data
# bootstrap: a slow, network-dependent download when the data dirs are
# missing. Delaying the import keeps plain `pytest` runs fast and offline-safe.
import doctest


def _doctests_enabled(config) -> bool:
    # Covers --doctest-modules / --doctest-glob, including via addopts.
    # getattr guards against the doctest plugin being disabled (-p no:doctest).
    option = config.option
    return bool(
        getattr(option, "doctestmodules", False)
        or getattr(option, "doctestglob", None)
    )


def pytest_configure(config):
    if not _doctests_enabled(config):
        return
    # pytest lazily subclasses doctest.OutputChecker at collection time, so
    # patching here (before collection) is early enough.
    from pygad.doctest import NumericOutputChecker

    doctest.OutputChecker = NumericOutputChecker
