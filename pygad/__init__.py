'''
The pygad module is a light-weighted analysis module for Gadget files /
snapshots.

All sub-modules are imported along with a "import pygad" except the
plotting sub-module, that is only imported automatically when in
interactive mode (for more details see the documentation about the
plotting module.

doctests:
    >>> import doctest
    >>> import sys
    >>> print("ATTENTION: full doctest takes a few minutes!!!", file=sys.stderr)
    >>> print('testing module utils...', file=sys.stderr)
    >>> doctest.testmod(utils)
    TestResults(failed=0, attempted=5)
    >>> print('testing module environment...', file=sys.stderr)
    >>> doctest.testmod(environment)
    TestResults(failed=0, attempted=0)
    >>> print('testing module units...', file=sys.stderr)
    >>> doctest.testmod(units)
    TestResults(failed=0, attempted=38)
    >>> print('testing module physics...', file=sys.stderr)
    >>> doctest.testmod(physics)
    TestResults(failed=0, attempted=4)
    >>> print('testing module kernels...', file=sys.stderr)
    >>> doctest.testmod(kernels)
    TestResults(failed=0, attempted=3)
    >>> print('testing module gadget...', file=sys.stderr)
    >>> doctest.testmod(gadget)
    TestResults(failed=0, attempted=3)
    >>> print('testing module transformation...', file=sys.stderr)
    >>> doctest.testmod(transformation)
    TestResults(failed=0, attempted=54)
    >>> print('testing module snapshot...', file=sys.stderr)
    >>> doctest.testmod(snapshot)
    TestResults(failed=0, attempted=118)
    >>> print('testing module octree...', file=sys.stderr)
    >>> doctest.testmod(octree)
    TestResults(failed=0, attempted=3)
    >>> print('testing module ssp...', file=sys.stderr)
    >>> doctest.testmod(ssp)
    TestResults(failed=0, attempted=2)
    >>> print('testing module binning...', file=sys.stderr)
    >>> doctest.testmod(binning)
    TestResults(failed=0, attempted=5)
    >>> print('testing module analysis...', file=sys.stderr)
    >>> doctest.testmod(analysis)
    TestResults(failed=0, attempted=6)

    #>>> print('testing module tools...', file=sys.stderr)
    #>>> doctest.testmod(.tools)
    #TestResults(failed=0, attempted=0)

    >>> print('testing module plotting...', file=sys.stderr)
    >>> doctest.testmod(plotting)
    TestResults(failed=0, attempted=2)

For development:
    In order to make the include of pygad work:
    The submodules shall only depent globally on submodules that are listed
    previuously. That means, for instance, physics can depend on units but not
    vice versa. Global dependence on another module is when it is imported in the
    global namespace. Than implies, that including other modules within certain
    functions is allowed.
    Similar is required for the submodules.
'''
import gc
import os
import sys
import tarfile
import tempfile
import time
import urllib.request
from . import utils
from . import environment
from .environment import gc_full_collect, module_dir

_DATA_BASE_URL = os.getenv(
    'PYGAD_DATA_BASE_URL',
    'https://github.com/Migelo/pygad/releases/download/pygad-data',
).rstrip('/')


def _human_size(nbytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if nbytes < 1024 or unit == 'GB':
            return f'{nbytes:.1f} {unit}'
        nbytes /= 1024.0
    return f'{nbytes:.1f} GB'


def _download_with_progress(url, archive_path, data_file):
    req = urllib.request.Request(url, headers={'User-Agent': 'pygad'})
    with urllib.request.urlopen(req, timeout=120) as response, \
            open(archive_path, 'wb') as out:
        total_hdr = response.headers.get('Content-Length')
        total = int(total_hdr) if total_hdr and total_hdr.isdigit() else None

        downloaded = 0
        last_pct = -5
        last_time = 0.0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            now = time.time()

            if total:
                pct = int(downloaded * 100 / total)
                if pct >= last_pct + 5 or now - last_time >= 1.5:
                    print(f'pygad: downloading {data_file}: {pct:3d}% '
                          f'({_human_size(downloaded)}/{_human_size(total)})',
                          file=sys.stderr, flush=True)
                    last_pct = pct
                    last_time = now
            elif now - last_time >= 1.5:
                print(f'pygad: downloading {data_file}: {_human_size(downloaded)}',
                      file=sys.stderr, flush=True)
                last_time = now


def _download_and_extract(data_file):
    url = f'{_DATA_BASE_URL}/{data_file}'
    print(f'pygad: downloading {data_file} from {url}',
          file=sys.stderr, flush=True)

    with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp:
        archive_path = tmp.name

    try:
        last_err = None
        for attempt in range(1, 4):
            try:
                _download_with_progress(url, archive_path, data_file)
                break
            except Exception as err:
                last_err = err
                if attempt < 3:
                    print(f'pygad: retry {attempt}/2 for {data_file} after '
                          f'error: {err}',
                          file=sys.stderr, flush=True)
                time.sleep(2)
        else:
            raise RuntimeError(f'failed to download {url}: {last_err}')

        print(f'pygad: extracting {data_file} ...', file=sys.stderr, flush=True)
        with tarfile.open(archive_path, mode='r:gz') as archive:
            archive.extractall(module_dir)
        print(f'pygad: extracted {data_file}', file=sys.stderr, flush=True)
    finally:
        if os.path.exists(archive_path):
            os.remove(archive_path)


def _ensure_auxiliary_data():
    required_data = [
        (module_dir + './CoolingTables/z_0.000.hdf5', 'z_0.000_highres.tar.gz'),
        (module_dir + './iontbls', 'iontbls.tar.gz'),
        (module_dir + './bc03', 'bc03.tar.gz'),
        (module_dir + './snaps', 'snaps.tar.gz'),
    ]
    missing = [archive for path, archive in required_data if not os.path.exists(path)]
    if not missing:
        return

    print('pygad: missing auxiliary data detected. This can take a few minutes '
          'on first import.',
          file=sys.stderr, flush=True)
    for archive in missing:
        _download_and_extract(archive)
    print('pygad: auxiliary data ready.', file=sys.stderr, flush=True)


_ensure_auxiliary_data()

# import all modules
#from . import units            made visible directly below
from . import physics
from . import kernels
from . import gadget
#from . import transformation   made visible directly below
#from . import snapshot         made visible directly below
from . import octree
from . import ssp
from . import binning
from . import analysis
#from . import tools            prepare zoom moved to SnapshotCache, others -> Cmd-scripts
from . import cloudy
from . import plotting

# the version of this pygad
# this line gets changed by setup.py during installation, and hence is different
# for the repo and the installed module
#   - in the git repository:
#       simply calls `git_descr(module_dir, PEP440=True)`
#   - in the installed version
#       a string holding the value of above's function call at the moment of
#       installation via setup.py

# make some chosen elements directly visible
# from .units import Unit, Units, UnitArr, UnitQty, UnitScalar, Fraction
# from .physics import cosmology
# from .physics import G, m_p, solar
# from .transformation import Translation, Rotation
# from .snapshot import Snapshot, SubSnapshot, BallMask, BoxMask, DiscMask, IDMask, ExprMask, SimArr, SnapMask
# from .binning import gridbin2d, gridbin, smooth
# from .tools import prepare_zoom, read_info_file   # prepare_zoom moved to SnapshotCache, read_info_file to SnapshotProperty
# ################################################
# only import core objects to be visible on package level
# functions to be used from sub-modules only
# ################################################
from .units import *
from .transformation import *
from .snapshot import *
from .tools import prepare_zoom, read_info_file

# default seems to be (700, 10, 10)
# pygad should more often collect garbage, since it has huge objects (SimArr and
# _Snaps), in critical cases, call `gc_full_collect`.
gc.set_threshold(50, 3, 3)
gc_full_collect()

from ._version import get_versions
__version__ = get_versions()['version']
if environment.verbose > environment.VERBOSE_QUIET:
    print(('imported pygad', __version__))
del get_versions
